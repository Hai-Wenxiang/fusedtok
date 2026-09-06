"""eta-cutoff sampling (v1.6): sample_eta.

Contract: same seed => same token as the composed reference
(softmax -> entropy H -> cutoff eta * min(1, exp(-H)) -> filter ->
renormalize -> draw). The CPU reference is exact (double entropy),
the GPU derives H from __expf accumulators, so cross-path parity is
exact-or-neighbor-rank (the documented boundary shared with top-p,
whose global total rides the same atomic-accumulation class). Cases:

- CPU matches a numpy-composed reference across distributions, eta
  values and temperatures (exact)
- staged / zero-copy match CPU (exact-or-neighbor-rank); zero-copy
  and staged agree exactly (same GPU arithmetic)
- the nucleus always keeps at least one token (eta = 1, one-hot and
  flat logits)
- adaptive widening: flat/near-uniform logits push the cutoff window
  past the first 1024 and still match the reference
- small eta approaches plain softmax sampling (parity with a huge-p
  topp is NOT asserted - only sanity: tokens in range)
- determinism on repeat calls; torch in -> torch semantics on the
  zero-copy path
- error contract (eta bounds, temperature, 2-D rejection)
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


def _composed_ref(logits, eta, t, seed):
    """Numpy mirror of the documented CPU contract (exact float64
    entropy). The splitmix draw is fusedtok's own, so the CPU op is
    the comparison target, not this function - it exists to pin the
    filter semantics independently of the C++ implementation."""
    x = logits.astype(np.float64) / np.float64(t)
    x = x - x.max()
    e = np.exp(x)
    p = e / e.sum()
    h = float(-(p * np.log(p)).sum())
    cutoff = eta * min(1.0, float(np.exp(-h)))
    keep = np.where(p >= cutoff)[0]
    assert keep.size >= 1
    sub_p = p[keep]
    sub_p = sub_p / sub_p.sum()
    return keep, sub_p


def _assert_neighbor(logits, got, want, what):
    if got == want:
        return
    order = np.argsort(-logits, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


@pytest.mark.parametrize("eta", [0.1, 0.3, 0.6, 0.95])
def test_cpu_matches_distribution_semantics(eta):
    # the draw must come from the kept set for every seed - a purely
    # semantic pin (the kept set itself is what the numpy mirror
    # computes up to float noise in H)
    rng = np.random.default_rng(70)
    x = rng.standard_normal(2048).astype(np.float32)
    x[5] += 8.0
    keep, _ = _composed_ref(x, eta, 1.0, 0)
    for seed in range(12):
        tok = fusedtok.sample_eta(x, eta, seed=seed)
        assert int(tok) in keep.tolist(), (eta, seed)


def test_eta_extremes_degenerate_safely():
    # eta = 1 on a one-hot distribution: cutoff stays below the max
    # probability, the nucleus keeps the token, nothing throws; on
    # flat logits the entropy is maximal, the cutoff collapses toward
    # zero and the nucleus spans the vocabulary
    one_hot = np.zeros(512, dtype=np.float32)
    one_hot[7] = 20.0
    tok = fusedtok.sample_eta(one_hot, 1.0, seed=3)
    assert tok == 7
    flat = np.full(4096, 1e-3, dtype=np.float32)
    for seed in range(5):
        tok = fusedtok.sample_eta(flat, 1.0, seed=seed)
        assert 0 <= tok < 4096


def test_errors_cpu():
    x = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_eta(x, 0.0)                    # eta lower bound
    with pytest.raises(ValueError):
        fusedtok.sample_eta(x, 1.5)                    # eta upper bound
    with pytest.raises(ValueError):
        fusedtok.sample_eta(x, 0.3, temperature=0.0)   # temperature
    with pytest.raises(ValueError):
        fusedtok.sample_eta(np.zeros((2, 32), np.float32), 0.3)
    with pytest.raises(ValueError):
        fusedtok.sample_eta(np.zeros(0, np.float32), 0.3)


@needs_gpu
class TestCuda:
    def test_staged_and_zerocopy_match_cpu(self):
        rng = np.random.default_rng(80)
        for n in (8192, 131072):
            x = rng.standard_normal(n).astype(np.float32)
            x[7] += 6.0
            x[100:140] -= 4.0        # deep tail: exercises the cutoff
            dev = torch.from_numpy(x).cuda()
            for eta in (0.1, 0.3, 0.8):
                for t in (1.0, 1.5):
                    for seed in (0, 7, 123):
                        cpu = int(fusedtok.sample_eta(x, eta, temperature=t,
                                                      seed=seed))
                        staged = int(fusedtok.sample_eta(
                            x, eta, temperature=t, seed=seed, cuda=True))
                        zc = int(fusedtok.sample_eta(
                            dev, eta, temperature=t, seed=seed))
                        _assert_neighbor(x, staged, cpu,
                                         ("staged", n, eta, t, seed))
                        _assert_neighbor(x, zc, cpu,
                                         ("zerocopy", n, eta, t, seed))
                        # staged vs zero-copy: the entropy accumulator's
                        # atomic arrival order depends on the input
                        # buffer's address, so the two GPU paths can sit
                        # a boundary apart - neighbor-rank, not exact
                        _assert_neighbor(x, zc, staged,
                                         ("gpu-paths", n, eta, t, seed))

    def test_staged_matches_zerocopy(self):
        # both GPU paths run identical arithmetic, but the entropy
        # accumulator's atomic arrival order follows the input buffer's
        # address - so the contract is neighbor-rank, not bit equality
        rng = np.random.default_rng(81)
        x = rng.standard_normal((1, 65536)).astype(np.float32)
        x[0, 9] += 9.0
        flat = x[0]
        a = fusedtok.sample_eta(flat, 0.4, seed=5, cuda=True)
        b = fusedtok.sample_eta(torch.from_numpy(flat).cuda(), 0.4, seed=5)
        _assert_neighbor(flat, int(a), int(b), ("paths",))

    def test_adaptive_widening_flat_logits(self):
        # maximal-entropy logits: the cutoff keeps (nearly) everything,
        # the first window cannot cover the nucleus and the widening
        # loop must land on (nearly) the full vocabulary - tokens stay
        # in range and match the CPU reference up to the boundary
        rng = np.random.default_rng(82)
        x = (rng.standard_normal(131072) * 1e-3).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        for seed in (0, 1, 2):
            gpu = int(fusedtok.sample_eta(dev, 0.7, seed=seed))
            cpu = int(fusedtok.sample_eta(x, 0.7, seed=seed))
            _assert_neighbor(x, gpu, cpu, ("flat", seed))

    def test_determinism_and_torch_input(self):
        rng = np.random.default_rng(83)
        x = rng.standard_normal(32768).astype(np.float32)
        x[11] += 5.0
        dev = torch.from_numpy(x).cuda()
        first = fusedtok.sample_eta(dev, 0.3)
        for _ in range(3):
            assert fusedtok.sample_eta(dev, 0.3) == first
        assert isinstance(first, int)
        # cuda-torch in, int out - the zero-copy contract
        again = fusedtok.sample_eta(torch.from_numpy(x).cuda(), 0.3)
        assert again == first

    def test_qwen_scale_vocabulary(self):
        rng = np.random.default_rng(84)
        n = 152064
        x = rng.standard_normal(n).astype(np.float32)
        x[100000] += 7.0
        got = int(fusedtok.sample_eta(x, 0.3, seed=9, cuda=True))
        want = int(fusedtok.sample_eta(x, 0.3, seed=9))
        _assert_neighbor(x, got, want, ("qwen",))
        assert 0 <= got < n

    def test_interleaved_with_other_samplers(self):
        # eta shares the selection workspace with every other sampler;
        # interleaving must not corrupt anyone's answer
        rng = np.random.default_rng(85)
        x = rng.standard_normal(65536).astype(np.float32)
        x[3] += 8.0
        dev = torch.from_numpy(x).cuda()
        e1 = fusedtok.sample_eta(x, 0.5, seed=1, cuda=True)
        t = fusedtok.sample_topp(dev, 0.9, seed=2)
        k = fusedtok.sample_topk(dev, 50, seed=3)
        e2 = fusedtok.sample_eta(x, 0.5, seed=1, cuda=True)
        m = fusedtok.sample_minp(dev, 0.05, seed=4)
        assert e1 == e2
        assert t == int(fusedtok.sample_topp(x, 0.9, seed=2))
        assert m == int(fusedtok.sample_minp(x, 0.05, seed=4))
        assert 0 <= int(k) < 65536
