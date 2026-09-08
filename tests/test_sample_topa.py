"""sample_topa (v1.8): fused top-a sampling.

The nucleus is every token with probability >= top_a * p_max^2 - a
value threshold like min-p, but driven by the SQUARED peak (top-a rule
from the open-source sampling stack). In the max-normalized exp column
the cutoff is exp >= top_a / total, so it is a PREFIX cut. Cases:

- nucleus membership vs a numpy reference (peaked / midtail / flat)
- exact draw parity against a numpy mirror of the CPU reference
- top_a = 1.0 keeps tokens whose probability reaches the squared peak
  (for a two-horse distribution that is BOTH leaders, not the argmax
  alone - distinguishes top-a from min-p at the same setting)
- tiny top_a ~ whole-vocabulary nucleus (widening ladder + the
  full-vocabulary fast path)
- per-seed determinism, cross-path (CPU / staged / zero-copy) parity
- batched parity (per-row == single-row), chunk boundary at 33 rows
- error contract (top_a bounds, temperature, wrong dtype / 1-D input)
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

needs_gpu = pytest.mark.skipif(
    not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")


def _logits(rng, kind, n):
    """Raw logits of the requested shape (the op softmaxes these)."""
    out = {
        "peaked": rng.standard_normal(n).astype(np.float32) * 0.5,
        "midtail": rng.standard_normal(n).astype(np.float32),
        "flat": rng.standard_normal(n).astype(np.float32) * 1e-3,
    }[kind]
    if kind == "peaked":
        out[7] += 6.0
    return out


def _probs(logits):
    e = np.exp((logits - logits.max()).astype(np.float32))
    return e / e.sum()


def _reference(logits, top_a, seed):
    """numpy mirror of sample_topa_cpu: softmax -> filter
    p >= top_a * p_max^2 -> renormalize -> splitmix inverse-CDF in
    descending order, replicating the float accumulation order."""
    e = np.exp((logits - logits.max()).astype(np.float32))
    total = np.float64(e.astype(np.float64).sum())
    order = np.argsort(-e, kind="stable")   # value desc, tie earliest
    cutoff = np.float64(top_a) / (total * total)   # top_a * p_max^2
    nucleus = []
    mass = np.float32(0)
    for t in order:
        p = np.float64(e[t]) / total
        if p < cutoff:
            break
        nucleus.append(int(t))
        mass = np.float32(mass + e[t])
    if not nucleus:
        nucleus = [int(order[0])]
        mass = np.float32(e[order[0]])
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    u = (z >> 11) * (1.0 / 9007199254740992.0)
    target = np.float32(u) * mass
    cum = np.float32(0)
    for t in nucleus:
        cum = np.float32(cum + e[t])
        if cum >= target:
            return t
    return nucleus[-1]


@pytest.mark.parametrize("kind", ["peaked", "midtail", "flat"])
@pytest.mark.parametrize("top_a", [0.5, 0.2, 0.05])
def test_membership_matches_reference_cpu(kind, top_a):
    rng = np.random.default_rng(90)
    logits = _logits(rng, kind, 4096)
    probs = _probs(logits)
    p_max = probs.max()
    nucleus = set(np.flatnonzero(
        probs >= top_a * p_max * p_max).tolist())
    for seed in range(8):
        tok = fusedtok.sample_topa(logits, top_a, seed=seed)
        assert tok in nucleus, (kind, top_a, seed)


def test_cpu_matches_reference_distribution():
    rng = np.random.default_rng(91)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(16):
        assert fusedtok.sample_topa(logits, 0.2, seed=seed) == \
            _reference(logits, 0.2, seed)


def test_topa_one_keeps_the_squared_peak_leaders():
    # top_a = 1 cutoffs at p_max^2. With two strong tokens (p ~ 0.58 /
    # 0.39, square of the peak ~ 0.34) BOTH stay in the nucleus - this
    # is where top-a deliberately differs from min_p = 1 (which would
    # keep only the argmax)
    x = np.array([0.0, 5.0, 4.6, 2.0], dtype=np.float32)
    probs = _probs(x)
    assert probs[2] >= probs[1] ** 2          # runner-up survives
    assert probs[0] < probs[1] ** 2           # the tail does not
    drawn = {fusedtok.sample_topa(x, 1.0, seed=s) for s in range(32)}
    assert drawn <= {1, 2} and 1 in drawn and 2 in drawn
    # ...while min_p = 1 collapses to the argmax alone on the same row
    assert fusedtok.sample_minp(x, 1.0, seed=0) == 1


def test_topa_one_unique_dominant_peak_is_greedy():
    # one dominant token: p_max^2 exceeds every other probability, so
    # top_a = 1 degenerates to the argmax for every seed
    x = np.array([1.0, 5.0, 4.0, 2.0], dtype=np.float32)
    probs = _probs(x)
    assert probs[2] < probs[1] ** 2
    for seed in range(8):
        assert fusedtok.sample_topa(x, 1.0, seed=seed) == 1


def test_low_temperature_collapse():
    x = np.array([0.1, 3.0, 2.9], dtype=np.float32)
    assert fusedtok.sample_topa(x, 0.5, temperature=1e-4, seed=0) == 1


def test_tiny_topa_spans_vocabulary():
    # top_a below every probability keeps the whole vocab: the nucleus
    # is everything and the draw is a plain softmax sample
    rng = np.random.default_rng(92)
    logits = _logits(rng, "peaked", 3000)
    tok = fusedtok.sample_topa(logits, 1e-9, seed=3)
    assert 0 <= tok < 3000
    assert fusedtok.sample_topa(logits, 1e-9, seed=3) == tok


def test_nucleus_shape_sanity():
    # top-a is a VALUE threshold: the nucleus is a small prefix of the
    # descending order (never empty - exps[0] == 1.0 passes any valid
    # cutoff - and strictly smaller than the vocabulary at a healthy
    # threshold). Unlike top-p it carries no mass guarantee, by design.
    rng = np.random.default_rng(93)
    logits = _logits(rng, "peaked", 8192)
    probs = _probs(logits)
    mask = probs >= 0.1 * probs.max() ** 2
    assert mask.any() and mask.sum() < 8192


def test_flat_distribution_keeps_almost_everything():
    # the defining top-a property: on a flat distribution p_max is only
    # marginally above 1/n, so p_max^2 ~ p_max / n collapses the cutoff
    # toward "keep everything" - the adaptive widening must reach the
    # full vocabulary without a "not covered" error
    rng = np.random.default_rng(94)
    logits = _logits(rng, "flat", 20000)
    probs = _probs(logits)
    keep = (probs >= 0.5 * probs.max() ** 2).sum()
    assert keep > 19000       # the nucleus is (nearly) the whole vocab
    for seed in (0, 1, 2):
        tok = fusedtok.sample_topa(logits, 0.5, seed=seed)
        assert 0 <= tok < 20000
        assert tok == fusedtok.sample_topa(logits, 0.5, seed=seed)


def test_determinism_and_seed_coverage_cpu():
    rng = np.random.default_rng(95)
    logits = _logits(rng, "midtail", 1024)
    seen = set()
    for seed in range(64):
        tok = fusedtok.sample_topa(logits, 0.2, seed=seed)
        assert tok == fusedtok.sample_topa(logits, 0.2, seed=seed)
        seen.add(tok)
    assert len(seen) > 1          # seeds must not all collapse to one


def test_error_contract_cpu():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_topa(x, 0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_topa(x, 1.5)
    with pytest.raises(ValueError):
        fusedtok.sample_topa(x, 0.1, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_topa(np.ones((2, 2), dtype=np.float32), 0.1)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="staged needs a GPU")
def test_staged_matches_cpu():
    rng = np.random.default_rng(96)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(6):
        assert fusedtok.sample_topa(logits, 0.2, seed=seed) == \
            fusedtok.sample_topa(logits, 0.2, seed=seed, cuda=True)


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu_all_regimes(self):
        rng = np.random.default_rng(97)
        for kind in ("peaked", "midtail", "flat"):
            logits = _logits(rng, kind, 8192)
            probs = _probs(logits)
            dev = torch.from_numpy(logits).cuda()
            for top_a in (0.5, 0.2, 0.02):
                for seed in range(6):
                    host = fusedtok.sample_topa(logits, top_a, seed=seed)
                    got = int(fusedtok.sample_topa(dev, top_a, seed=seed))
                    # exact-exp CPU vs __expf GPU can neighbor on the
                    # CDF boundary: same nucleus, at most a neighbor rank
                    order = np.argsort(-probs, kind="stable")
                    rank = {int(t): i for i, t in enumerate(order)}
                    assert host in rank and got in rank
                    assert abs(rank[host] - rank[got]) <= 1

    def test_full_vocabulary_fast_path(self):
        # near-uniform logits + tiny threshold -> nucleus = whole vocab
        # via the widening ladder + k==n parallel-pack path
        rng = np.random.default_rng(98)
        logits = _logits(rng, "flat", 131072)
        dev = torch.from_numpy(logits).cuda()
        tok = int(fusedtok.sample_topa(dev, 1e-6, seed=5))
        assert 0 <= tok < 131072
        assert tok == int(fusedtok.sample_topa(dev, 1e-6, seed=5))

    def test_wide_nucleus_adaptive_jump_matches_cpu(self):
        # wide-nucleus regimes at full-vocabulary scale, where the
        # adaptive jump (divisor top_a / total) takes DIFFERENT window
        # schedules than the plain x8 ladder. Tokens are
        # schedule-independent by construction - this pins that against
        # the CPU reference (__expf boundary drift keeps the usual
        # rank-window tolerance, generous for a ~70k-wide nucleus of
        # serially accumulated mass)
        rng = np.random.default_rng(99)
        logits = rng.standard_normal(131072).astype(np.float32)
        probs = _probs(logits)
        order = np.argsort(-probs, kind="stable")
        rank = {int(t): i for i, t in enumerate(order)}
        dev = torch.from_numpy(logits).cuda()
        for top_a in (0.05, 0.01):
            for seed in (0, 3, 7):
                host = fusedtok.sample_topa(logits, top_a, seed=seed)
                got = int(fusedtok.sample_topa(dev, top_a, seed=seed))
                assert host in rank and got in rank
                assert abs(rank[host] - rank[got]) <= 64
                assert got == int(fusedtok.sample_topa(dev, top_a,
                                                       seed=seed))

    def test_adaptive_jump_mixed_rows(self):
        # a heavy spike in one row keeps its nucleus inside the first
        # window while a plain-randn row widens - one call each must
        # equal the row sampled alone (schedule + lazy-total paths
        # interleaved across calls on one stream)
        rng = np.random.default_rng(100)
        plain = rng.standard_normal(32768).astype(np.float32)
        spiky = plain.copy()
        spiky[11] += 8.0
        d_plain = torch.from_numpy(plain).cuda()
        d_spiky = torch.from_numpy(spiky).cuda()
        for top_a in (0.1, 0.02):
            for seed in (1, 4):
                assert int(fusedtok.sample_topa(d_plain, top_a,
                                                seed=seed)) == \
                    int(fusedtok.sample_topa(d_plain, top_a, seed=seed))
                assert int(fusedtok.sample_topa(d_spiky, top_a,
                                                seed=seed)) == \
                    fusedtok.sample_topa(spiky, top_a, seed=seed)

    def test_batched_matches_single_rows_mixed(self):
        # per-row parity on mixed peaked / flat rows: the batched
        # pipeline must return each row's single-row token (rank-window
        # tolerance for the documented __expf / atomic-order ulp)
        rng = np.random.default_rng(101)
        b, n = 8, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(0, b, 3):
            x[r, 7] += 10.0
        for r in range(1, b, 3):
            x[r] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        for top_a in (0.1, 0.5):
            got = fusedtok.sample_topa_batched(dev, top_a, seeds=seeds)
            for r in range(b):
                want = int(fusedtok.sample_topa(dev[r], top_a,
                                                seed=int(seeds[r])))
                if got[r] == want:
                    continue
                order = np.argsort(-x[r], kind="stable")
                rank = {int(t): i for i, t in enumerate(order)}
                assert abs(rank[int(got[r])] - rank[want]) <= 2, \
                    (top_a, r, int(got[r]), want)

    def test_b33_chunk_boundary(self):
        rng = np.random.default_rng(102)
        b, n = 33, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(0, b, 3):
            x[r, 7] += 10.0
        for r in range(1, b, 3):
            x[r] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_topa_batched(dev, 0.3, seeds=seeds)
        for r in (0, 1, 16, 31, 32):   # both chunks covered
            want = int(fusedtok.sample_topa(dev[r], 0.3,
                                            seed=int(seeds[r])))
            if got[r] == want:
                continue
            order = np.argsort(-x[r], kind="stable")
            rank = {int(t): i for i, t in enumerate(order)}
            assert abs(rank[int(got[r])] - rank[want]) <= 2, (r,)

    def test_batched_determinism_and_types(self):
        rng = np.random.default_rng(103)
        b, n = 8, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[3, 5] += 8.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        first = fusedtok.sample_topa_batched(dev, 0.3, seeds=seeds)
        for _ in range(3):
            assert (fusedtok.sample_topa_batched(
                dev, 0.3, seeds=seeds).tolist() == first.tolist())
        assert isinstance(first, torch.Tensor)
        assert first.dtype == torch.int64 and first.is_cpu
        out_np = fusedtok.sample_topa_batched(x, 0.3, seeds=seeds)
        assert isinstance(out_np, np.ndarray)
        assert out_np.dtype == np.int64

    def test_batched_single_token_rows(self):
        # n = 1 rows through the batched pipeline: every row returns 0
        x = np.zeros((3, 1), dtype=np.float32)
        got = fusedtok.sample_topa_batched(
            torch.from_numpy(x).cuda(), 0.5)
        assert [int(v) for v in got] == [0, 0, 0]

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_topa(x, 0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_topa(x, 1.5)
        with pytest.raises(TypeError):
            fusedtok.sample_topa(x.to(torch.bfloat16), 0.1)
        with pytest.raises(ValueError):
            fusedtok.sample_topa(torch.ones(2, 2, device="cuda"), 0.1)
        with pytest.raises(ValueError):
            fusedtok.sample_topa_batched(x.unsqueeze(0), 0.0)


def test_batched_cpu_paths_match_single_rows():
    # the numpy-input route (the *_batched_cpu references) with real
    # rows: each row must equal the single-row CPU reference
    # bit-for-bit - that is the batched contract by construction
    rng = np.random.default_rng(104)
    b, n = 4, 2048
    x = rng.standard_normal((b, n)).astype(np.float32)
    x[1] *= 1e-3
    x[2, 5] += 9.0
    seeds = np.arange(b, dtype=np.int64)
    got = fusedtok.sample_topa_batched(x, 0.2, seeds=seeds)
    for r in range(b):
        want = int(fusedtok.sample_topa(x[r], 0.2, seed=int(seeds[r])))
        assert int(got[r]) == want


def test_batched_cpu_direct_surface_shape_contract():
    # the direct _fusedtok surface rejects mis-shaped host buffers
    # instead of trusting rows * n (same contract as the older
    # siblings; the C++-level rows * n guard fires here)
    from fusedtok import _fusedtok
    with pytest.raises(ValueError):
        _fusedtok.sample_topa_batched_cpu(
            np.zeros(8, dtype=np.float32), 2, 8, 0.2, 1.0,
            np.zeros(2, dtype=np.int64))
    # a right-shaped call draws deterministically (the flat row is an
    # 8-way tie, so only in-range + repeat-stability can be pinned)
    x = np.zeros((2, 4), dtype=np.float32)
    seeds = np.zeros(2, dtype=np.int64)
    out = _fusedtok.sample_topa_batched_cpu(x, 2, 4, 0.5, 1.0, seeds)
    again = _fusedtok.sample_topa_batched_cpu(x, 2, 4, 0.5, 1.0, seeds)
    assert [int(v) for v in out] == [int(v) for v in again]
    assert all(0 <= int(v) < 4 for v in out)
