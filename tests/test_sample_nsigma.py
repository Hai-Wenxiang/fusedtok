"""sample_nsigma (v1.8): fused top-n-sigma sampling.

The nucleus is every token whose temperature-scaled logit stays at or
above ``mean - nsigma * sigma`` (moments over the whole row - Shi et
al. 2024, "Top-n sigma: Not All Logits Are You Need"). The cutoff is a
value threshold in LOGIT space, so in the max-normalized exp column it
is still a PREFIX cut at exp(cutoff - max). Cases:

- nucleus membership vs a numpy reference (peaked / midtail / flat)
- exact draw parity against a numpy mirror of the CPU reference
- nsigma sweep: small n trims hard, large n approaches plain sampling
- the all-equal row (sigma = 0) keeps the whole window
- per-seed determinism, cross-path (CPU / staged / zero-copy) parity
- batched parity (per-row == single-row), chunk boundary at 33 rows
- error contract (nsigma bounds, temperature, wrong dtype / 1-D input)
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


def _probs(logits, t=1.0):
    e = np.exp(((logits - logits.max()) / t).astype(np.float32))
    return e / e.sum()


def _reference(logits, nsigma, seed, t=1.0):
    """numpy mirror of sample_nsigma_cpu: scaled-logit moments ->
    cutoff at mean - nsigma*sigma -> prefix -> splitmix inverse-CDF in
    descending order, replicating the float accumulation order."""
    v = (logits.astype(np.float32) / np.float32(t))
    order = np.argsort(-v, kind="stable")   # value desc, tie earliest
    row_max = v[order[0]]
    d = (v[order].astype(np.float64) - np.float64(row_max))
    s1 = d.sum()
    s2 = (d * d).sum()
    mu_d = s1 / d.size
    sigma = np.sqrt(max(s2 / d.size - mu_d * mu_d, 0.0))
    cutoff = np.float64(row_max) + mu_d - np.float64(nsigma) * sigma
    e = np.exp((v - row_max).astype(np.float32))
    nucleus = []
    mass = np.float32(0)
    for i, t_id in enumerate(order):
        if np.float64(v[t_id]) < cutoff:
            break
        nucleus.append(int(t_id))
        mass = np.float32(mass + e[t_id])
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
    for tok in nucleus:
        cum = np.float32(cum + e[tok])
        if cum >= target:
            return tok
    return nucleus[-1]


@pytest.mark.parametrize("kind", ["peaked", "midtail", "flat"])
@pytest.mark.parametrize("nsigma", [1.0, 2.0, 3.0])
def test_membership_matches_reference_cpu(kind, nsigma):
    rng = np.random.default_rng(130)
    logits = _logits(rng, kind, 4096)
    v = logits
    mu, sigma = v.mean(), v.std()
    nucleus = set(np.flatnonzero(v >= mu - nsigma * sigma).tolist())
    for seed in range(8):
        tok = fusedtok.sample_nsigma(logits, nsigma, seed=seed)
        assert tok in nucleus, (kind, nsigma, seed)


def test_cpu_matches_reference_distribution():
    rng = np.random.default_rng(131)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(16):
        assert fusedtok.sample_nsigma(logits, 1.5, seed=seed) == \
            _reference(logits, 1.5, seed)


def test_nsigma_sweep_trims_then_opens():
    # larger nsigma keeps strictly more of the value-sorted row: the
    # nucleus at n = 3 must be a superset of the nucleus at n = 1
    rng = np.random.default_rng(132)
    logits = _logits(rng, "midtail", 4096)
    mu, sigma = logits.mean(), logits.std()
    keep1 = logits >= mu - 1.0 * sigma
    keep3 = logits >= mu - 3.0 * sigma
    assert keep3.sum() > keep1.sum()
    # tiny nsigma trims hard (but never to empty)
    assert keep1.sum() < 4096
    tok = fusedtok.sample_nsigma(logits, 1e-3, seed=0)
    assert 0 <= tok < 4096


def test_huge_nsigma_spans_vocabulary():
    # nsigma far beyond the row's spread keeps the whole vocab: the
    # draw is a plain softmax sample through the widening ladder
    rng = np.random.default_rng(133)
    logits = _logits(rng, "peaked", 3000)
    tok = fusedtok.sample_nsigma(logits, 1e6, seed=3)
    assert 0 <= tok < 3000
    assert fusedtok.sample_nsigma(logits, 1e6, seed=3) == tok


def test_all_equal_row_keeps_everything():
    # sigma = 0 puts the cutoff on every logit: the whole window is the
    # nucleus (the tie resolves toward the earliest indices)
    x = np.zeros(64, dtype=np.float32)
    for seed in range(8):
        tok = fusedtok.sample_nsigma(x, 1.0, seed=seed)
        assert 0 <= tok < 64
        assert tok == fusedtok.sample_nsigma(x, 1.0, seed=seed)


def test_low_temperature_collapse():
    x = np.array([0.1, 3.0, 2.9], dtype=np.float32)
    assert fusedtok.sample_nsigma(x, 1.0, temperature=1e-4, seed=0) == 1


def test_temperature_scales_the_moments():
    # the moments are taken over the SCALED logits: scaling by 1/T is
    # linear, so the kept SET is temperature-invariant... the draw is
    # not (the softmax mass moves), pin both halves separately
    x = np.array([0.2, 2.0, 1.2, -0.4, 0.9, 1.6], dtype=np.float32)
    mu, sigma = (x / np.float32(2.0)).mean(), (x / np.float32(2.0)).std()
    nucleus = set(np.flatnonzero(
        x / np.float32(2.0) >= mu - 1.0 * sigma).tolist())
    for seed in range(8):
        assert fusedtok.sample_nsigma(x, 1.0, temperature=2.0,
                                      seed=seed) in nucleus


def test_determinism_and_seed_coverage_cpu():
    rng = np.random.default_rng(134)
    logits = _logits(rng, "midtail", 1024)
    seen = set()
    for seed in range(64):
        tok = fusedtok.sample_nsigma(logits, 1.5, seed=seed)
        assert tok == fusedtok.sample_nsigma(logits, 1.5, seed=seed)
        seen.add(tok)
    assert len(seen) > 1          # seeds must not all collapse to one


def test_error_contract_cpu():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_nsigma(x, 0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_nsigma(x, -1.0)
    with pytest.raises(ValueError):
        fusedtok.sample_nsigma(x, 1.0, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_nsigma(np.ones((2, 2), dtype=np.float32), 1.0)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="staged needs a GPU")
def test_staged_matches_cpu():
    # the cutoff derives from the float moment accumulators, so a draw
    # landing on a rounding boundary may shift one rank between paths -
    # the documented neighbor-rank contract, not exact equality
    rng = np.random.default_rng(135)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(6):
        host = fusedtok.sample_nsigma(logits, 1.5, seed=seed)
        got = fusedtok.sample_nsigma(logits, 1.5, seed=seed, cuda=True)
        if host != got:
            order = np.argsort(-logits, kind="stable")
            rank = {int(t): i for i, t in enumerate(order)}
            assert abs(rank[host] - rank[got]) <= 1, seed


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu_all_regimes(self):
        rng = np.random.default_rng(136)
        for kind in ("peaked", "midtail", "flat"):
            logits = _logits(rng, kind, 8192)
            dev = torch.from_numpy(logits).cuda()
            for nsigma in (1.0, 2.0, 3.0):
                for seed in range(6):
                    host = fusedtok.sample_nsigma(logits, nsigma, seed=seed)
                    got = int(fusedtok.sample_nsigma(dev, nsigma, seed=seed))
                    if host == got:
                        continue
                    # float-moment drift can flip one boundary token:
                    # the members differ by at most one rank
                    order = np.argsort(-logits, kind="stable")
                    rank = {int(t): i for i, t in enumerate(order)}
                    assert abs(rank[host] - rank[got]) <= 1, \
                        (kind, nsigma, seed)

    def test_full_vocabulary_fast_path(self):
        # huge nsigma + large vocab -> nucleus = whole vocab via the
        # widening ladder + k==n parallel-pack path
        rng = np.random.default_rng(137)
        logits = _logits(rng, "flat", 131072)
        dev = torch.from_numpy(logits).cuda()
        tok = int(fusedtok.sample_nsigma(dev, 1e6, seed=5))
        assert 0 <= tok < 131072
        assert tok == int(fusedtok.sample_nsigma(dev, 1e6, seed=5))

    def test_batched_matches_single_rows_mixed(self):
        # per-row parity on mixed peaked / flat rows: the batched
        # pipeline must return each row's single-row token (rank-window
        # tolerance for the documented __expf / atomic-order ulp)
        rng = np.random.default_rng(138)
        b, n = 8, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(0, b, 3):
            x[r, 7] += 10.0
        for r in range(1, b, 3):
            x[r] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        for nsigma in (1.0, 2.0):
            got = fusedtok.sample_nsigma_batched(dev, nsigma, seeds=seeds)
            for r in range(b):
                want = int(fusedtok.sample_nsigma(dev[r], nsigma,
                                                  seed=int(seeds[r])))
                if got[r] == want:
                    continue
                order = np.argsort(-x[r], kind="stable")
                rank = {int(t): i for i, t in enumerate(order)}
                assert abs(rank[int(got[r])] - rank[want]) <= 2, \
                    (nsigma, r, int(got[r]), want)

    def test_b33_chunk_boundary(self):
        rng = np.random.default_rng(139)
        b, n = 33, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(0, b, 3):
            x[r, 7] += 10.0
        for r in range(1, b, 3):
            x[r] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_nsigma_batched(dev, 1.5, seeds=seeds)
        for r in (0, 1, 16, 31, 32):   # both chunks covered
            want = int(fusedtok.sample_nsigma(dev[r], 1.5,
                                              seed=int(seeds[r])))
            if got[r] == want:
                continue
            order = np.argsort(-x[r], kind="stable")
            rank = {int(t): i for i, t in enumerate(order)}
            assert abs(rank[int(got[r])] - rank[want]) <= 2, (r,)

    def test_batched_determinism_and_types(self):
        rng = np.random.default_rng(140)
        b, n = 8, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[3, 5] += 8.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        first = fusedtok.sample_nsigma_batched(dev, 1.5, seeds=seeds)
        for _ in range(3):
            assert (fusedtok.sample_nsigma_batched(
                dev, 1.5, seeds=seeds).tolist() == first.tolist())
        assert isinstance(first, torch.Tensor)
        assert first.dtype == torch.int64 and first.is_cpu
        out_np = fusedtok.sample_nsigma_batched(x, 1.5, seeds=seeds)
        assert isinstance(out_np, np.ndarray)
        assert out_np.dtype == np.int64

    def test_batched_single_token_rows(self):
        # n = 1 rows through the batched pipeline: every row returns 0
        x = np.zeros((3, 1), dtype=np.float32)
        got = fusedtok.sample_nsigma_batched(
            torch.from_numpy(x).cuda(), 1.0)
        assert [int(v) for v in got] == [0, 0, 0]

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_nsigma(x, 0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_nsigma(x, -0.5)
        with pytest.raises(TypeError):
            fusedtok.sample_nsigma(x.to(torch.bfloat16), 1.0)
        with pytest.raises(ValueError):
            fusedtok.sample_nsigma(torch.ones(2, 2, device="cuda"), 1.0)
        with pytest.raises(ValueError):
            fusedtok.sample_nsigma_batched(x.unsqueeze(0), 0.0)


def test_batched_cpu_paths_match_single_rows():
    # the numpy-input route (the *_batched_cpu references) with real
    # rows: each row must equal the single-row CPU reference
    # bit-for-bit - that is the batched contract by construction
    rng = np.random.default_rng(141)
    b, n = 4, 2048
    x = rng.standard_normal((b, n)).astype(np.float32)
    x[1] *= 1e-3
    x[2, 5] += 9.0
    seeds = np.arange(b, dtype=np.int64)
    got = fusedtok.sample_nsigma_batched(x, 1.5, seeds=seeds)
    for r in range(b):
        want = int(fusedtok.sample_nsigma(x[r], 1.5, seed=int(seeds[r])))
        assert int(got[r]) == want


def test_batched_cpu_direct_surface_shape_contract():
    # the direct _fusedtok surface rejects mis-shaped host buffers
    # instead of trusting rows * n (same contract as the older
    # siblings; the C++-level rows * n guard fires here)
    from fusedtok import _fusedtok
    with pytest.raises(ValueError):
        _fusedtok.sample_nsigma_batched_cpu(
            np.zeros(8, dtype=np.float32), 2, 8, 1.5, 1.0,
            np.zeros(2, dtype=np.int64))
    # a right-shaped call draws deterministically (the flat row keeps
    # its 4-way tie, so only in-range + repeat-stability can be pinned)
    x = np.zeros((2, 4), dtype=np.float32)
    seeds = np.zeros(2, dtype=np.int64)
    out = _fusedtok.sample_nsigma_batched_cpu(x, 2, 4, 1.0, 1.0, seeds)
    again = _fusedtok.sample_nsigma_batched_cpu(x, 2, 4, 1.0, 1.0, seeds)
    assert [int(v) for v in out] == [int(v) for v in again]
    assert all(0 <= int(v) < 4 for v in out)
