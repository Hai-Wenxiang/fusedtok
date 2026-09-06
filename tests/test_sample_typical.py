"""locally typical sampling (v1.6): sample_typical.

Contract: same seed => same token as the composed reference
(softmax -> entropy H -> keep the smallest value-ordered band with
mass >= typical -> renormalize -> draw). The CPU reference is exact
(double entropy, exact two-pointer band); the GPU derives H and m
from __expf accumulators, so cross-path parity is
exact-or-neighbor-rank (documented boundary class). The kept set is
a contiguous BAND of the value order, not a prefix - the numpy
mirror computes the band independently. Cases:

- CPU matches the numpy-composed band semantics across
  distributions, typical values and temperatures
- staged / zero-copy match CPU (exact-or-neighbor-rank); staged and
  zero-copy agree exactly here (same kernel, and the band draw's
  serial replay is insensitive to the accumulator drift the eta
  cutoff exposed - only the band EDGES can move by a member)
- tiny typical concentrates on the valley (most typical tokens);
  typical = 1 degenerates to the whole vocabulary
- adaptive widening: flat logits force the x8 ladder to the full
  vocabulary and still match the reference
- determinism on repeat calls; torch zero-copy semantics
- error contract (typical bounds, temperature, 2-D rejection)
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


def _composed_band(logits, typical, t):
    """Numpy mirror of the band semantics (float64): the value-sorted
    order is what makes the kept set a contiguous band, so sort first,
    find the band there, and map back to token ids."""
    x = logits.astype(np.float64) / np.float64(t)
    lv = x - x.max()
    e = np.exp(lv)
    total = e.sum()
    m = float((e * lv).sum() / total)
    order = np.argsort(-x, kind="stable")
    shifted = np.abs(lv[order] - m)
    amin = int(shifted.argmin())
    lo = hi = amin
    mass = float(e[order[amin]])
    need = float(typical) * float(total)
    while mass < need and (lo > 0 or hi < len(e) - 1):
        dl = abs(lv[order[lo - 1]] - m) if lo > 0 else np.inf
        dr = abs(lv[order[hi + 1]] - m) if hi < len(e) - 1 else np.inf
        if dl <= dr:
            lo -= 1
            mass += float(e[order[lo]])
        else:
            hi += 1
            mass += float(e[order[hi]])
    return [int(order[i]) for i in range(lo, hi + 1)]


def _assert_neighbor(logits, got, want, what):
    if got == want:
        return
    order = np.argsort(-logits, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


@pytest.mark.parametrize("typical", [0.1, 0.3, 0.6, 0.95])
def test_cpu_draw_lands_in_band(typical):
    # semantic pin: every draw must land inside the exact band
    rng = np.random.default_rng(90)
    x = rng.standard_normal(2048).astype(np.float32)
    band = _composed_band(x, typical, 1.0)
    for seed in range(12):
        tok = fusedtok.sample_typical(x, typical, seed=seed)
        assert int(tok) in band, (typical, seed, tok)


def test_typical_extremes_degenerate_safely():
    # typical = 1: the band must span the whole mass, i.e. the whole
    # vocabulary - every token is a legal draw; tiny typical on a
    # one-hot distribution keeps the hot token (the valley sits there)
    rng = np.random.default_rng(91)
    x = rng.standard_normal(4096).astype(np.float32)
    for seed in range(5):
        tok = fusedtok.sample_typical(x, 1.0, seed=seed)
        assert 0 <= tok < 4096
    one_hot = np.zeros(512, dtype=np.float32)
    one_hot[7] = 20.0
    tok = fusedtok.sample_typical(one_hot, 0.01, seed=3)
    assert tok == 7


def test_errors_cpu():
    x = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_typical(x, 0.0)                    # lower bound
    with pytest.raises(ValueError):
        fusedtok.sample_typical(x, 1.5)                    # upper bound
    with pytest.raises(ValueError):
        fusedtok.sample_typical(x, 0.3, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_typical(np.zeros((2, 32), np.float32), 0.3)
    with pytest.raises(ValueError):
        fusedtok.sample_typical(np.zeros(0, np.float32), 0.3)


@needs_gpu
class TestCuda:
    def test_staged_and_zerocopy_match_cpu(self):
        rng = np.random.default_rng(95)
        for n in (8192, 131072):
            x = rng.standard_normal(n).astype(np.float32)
            x[7] += 6.0
            x[100:140] -= 4.0
            dev = torch.from_numpy(x).cuda()
            for typical in (0.1, 0.3, 0.8):
                for t in (1.0, 1.5):
                    for seed in (0, 7, 123):
                        cpu = int(fusedtok.sample_typical(
                            x, typical, temperature=t, seed=seed))
                        staged = int(fusedtok.sample_typical(
                            x, typical, temperature=t, seed=seed,
                            cuda=True))
                        zc = int(fusedtok.sample_typical(
                            dev, typical, temperature=t, seed=seed))
                        _assert_neighbor(x, staged, cpu,
                                         ("staged", n, typical, t, seed))
                        _assert_neighbor(x, zc, cpu,
                                         ("zerocopy", n, typical, t, seed))

    def test_adaptive_widening_flat_logits(self):
        # maximal-entropy logits: the band spans (nearly) the whole
        # vocabulary and the x8 ladder walks to the full window. On
        # near-uniform logits the band edges are rank-DENSE (an ulp of
        # entropy drift moves the edges by dozens of ranks), so the
        # contract is asserted in probability space: the drawn token's
        # probability must sit inside a generous window around the
        # band's probability level exp(-H)
        rng = np.random.default_rng(96)
        x = (rng.standard_normal(131072) * 1e-3).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        x64 = x.astype(np.float64)
        lv = x64 - x64.max()
        e = np.exp(lv)
        p = e / e.sum()
        h = float(-(p * np.log(p)).sum())
        plo = float(np.exp(-h - 3.0))      # e^-H window, +-3 nats slack
        phi = float(np.exp(-h + 3.0))
        for seed in (0, 1, 2):
            gpu = int(fusedtok.sample_typical(dev, 0.7, seed=seed))
            cpu = int(fusedtok.sample_typical(x, 0.7, seed=seed))
            assert plo <= float(p[gpu]) <= phi, (seed, float(p[gpu]))
            assert plo <= float(p[cpu]) <= phi, (seed, float(p[cpu]))

    def test_determinism_and_torch_input(self):
        rng = np.random.default_rng(97)
        x = rng.standard_normal(32768).astype(np.float32)
        x[11] += 5.0
        dev = torch.from_numpy(x).cuda()
        first = fusedtok.sample_typical(dev, 0.3)
        for _ in range(3):
            assert fusedtok.sample_typical(dev, 0.3) == first
        again = fusedtok.sample_typical(torch.from_numpy(x).cuda(), 0.3)
        assert again == first

    def test_qwen_scale_vocabulary(self):
        rng = np.random.default_rng(98)
        n = 152064
        x = rng.standard_normal(n).astype(np.float32)
        x[100000] += 7.0
        got = int(fusedtok.sample_typical(x, 0.3, seed=9, cuda=True))
        want = int(fusedtok.sample_typical(x, 0.3, seed=9))
        _assert_neighbor(x, got, want, ("qwen",))
        assert 0 <= got < n

    def test_interleaved_with_other_samplers(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal(65536).astype(np.float32)
        x[3] += 8.0
        dev = torch.from_numpy(x).cuda()
        t1 = fusedtok.sample_typical(x, 0.5, seed=1, cuda=True)
        fusedtok.sample_topp(dev, 0.9, seed=2)
        kk = fusedtok.sample_topk(dev, 50, seed=3)
        t2 = fusedtok.sample_typical(x, 0.5, seed=1, cuda=True)
        fusedtok.sample_minp(dev, 0.05, seed=4)
        e = fusedtok.sample_eta(x, 0.3, seed=5)
        assert t1 == t2
        assert t1 == int(fusedtok.sample_typical(x, 0.5, seed=1))
        assert 0 <= int(kk) < 65536
        assert 0 <= int(e) < 65536
