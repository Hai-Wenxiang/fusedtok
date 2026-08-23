"""Fused nucleus sampling (sample_topp): determinism, distribution, edges."""

import math

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def ref_probs(logits, t=1.0):
    z = logits.astype(np.float64) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def test_deterministic_per_seed():
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal(500) * 2).astype(np.float32)
    first = fusedtok.sample_topp(logits, 0.9, seed=123)
    for _ in range(5):
        assert fusedtok.sample_topp(logits, 0.9, seed=123) == first
    # different seeds eventually produce different draws (not guaranteed
    # per pair, but across 32 seeds at least two outcomes with flat logits)
    flat = np.zeros(64, dtype=np.float32)
    outcomes = {fusedtok.sample_topp(flat, 1.0, seed=s) for s in range(32)}
    assert len(outcomes) > 1


def test_peaked_logits_sample_the_peak():
    logits = np.array([10.0] + [0.0] * 49, dtype=np.float32)
    # low temperature: the peak dominates softmax; nucleus = {0}
    draws = {fusedtok.sample_topp(logits, 0.9, temperature=0.1, seed=s)
             for s in range(16)}
    assert draws == {0}


def test_temperature_flattens():
    logits = np.array([10.0] + [0.0] * 49, dtype=np.float32)
    # very high temperature ~ uniform over the p=1.0 nucleus (all tokens)
    draws = {fusedtok.sample_topp(logits, 1.0, temperature=1e4, seed=s)
             for s in range(64)}
    assert len(draws) > 10     # spread over the vocab, not stuck at the peak


def test_distribution_flat_logits():
    # flat logits, p=1: samples should be ~uniform across the vocab
    logits = np.zeros(32, dtype=np.float32)
    n = 4000
    counts = np.zeros(32, dtype=int)
    for s in range(n):
        counts[fusedtok.sample_topp(logits, 1.0, seed=s)] += 1
    expected = n / 32
    # 4-sigma band on the binomial count
    band = 4 * math.sqrt(n * (1 / 32) * (31 / 32))
    assert counts.max() - counts.min() < 2 * band
    assert (np.abs(counts - expected) < band).all()


def test_distribution_matches_softmax_probs():
    # skewed logits: empirical frequency tracks softmax probabilities
    rng = np.random.default_rng(3)
    logits = (rng.standard_normal(8) * 2).astype(np.float32)
    probs = ref_probs(logits)
    n = 6000
    counts = np.zeros(8, dtype=int)
    for s in range(n):
        counts[fusedtok.sample_topp(logits, 0.95, seed=s)] += 1
    freq = counts / n
    # nucleus truncation keeps the top tokens; compare against renormalized
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    keep = order[:np.searchsorted(cum, 0.95) + 1]
    keep_probs = probs[keep] / probs[keep].sum()
    for tok, p in zip(keep, keep_probs):
        sigma = math.sqrt(p * (1 - p) / n)
        assert abs(freq[tok] - p) < 5 * sigma + 1e-4, (tok, freq[tok], p)


def test_never_outside_nucleus():
    rng = np.random.default_rng(4)
    logits = (rng.standard_normal(100) * 3).astype(np.float32)
    probs = ref_probs(logits)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    nucleus = set(order[:np.searchsorted(cum, 0.7) + 1].tolist())
    for s in range(200):
        tok = fusedtok.sample_topp(logits, 0.7, seed=s)
        assert tok in nucleus


def test_errors():
    x = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_topp(x, 0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_topp(x, 1.5)
    with pytest.raises(ValueError):
        fusedtok.sample_topp(x, 0.9, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_topp(np.ones((2, 2), dtype=np.float32), 0.9)
    with pytest.raises(ValueError):
        fusedtok.sample_topp(np.array([], dtype=np.float32), 0.9)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_matches_cpu_across_seeds(self):
        # same seed must give the same token on CPU and GPU (identical
        # algorithm and float32 accumulation order; exact exp vs __expf can
        # only differ when a draw lands exactly on a mass boundary)
        rng = np.random.default_rng(5)
        logits = (rng.standard_normal(2000) * 2).astype(np.float32)
        for seed in (0, 1, 7, 42, 12345):
            assert fusedtok.sample_topp(logits, 0.9, seed=seed, cuda=True) == \
                fusedtok.sample_topp(logits, 0.9, seed=seed)

    def test_deterministic_gpu(self):
        rng = np.random.default_rng(6)
        logits = (rng.standard_normal(1000) * 2).astype(np.float32)
        first = fusedtok.sample_topp(logits, 0.9, seed=9, cuda=True)
        assert all(fusedtok.sample_topp(logits, 0.9, seed=9, cuda=True) == first
                   for _ in range(3))

    def test_first_call_in_process_growth(self):
        # sample as the very first selection call exercises workspace growth
        # ordering (the token slot must survive the buffer reallocation)
        rng = np.random.default_rng(7)
        logits = (rng.standard_normal(3000) * 2).astype(np.float32)
        tok = fusedtok.sample_topp(logits, 0.8, seed=3, cuda=True)
        assert 0 <= tok < 3000


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_cuda_tensor_matches_cpu(self):
        rng = np.random.default_rng(8)
        logits = (rng.standard_normal(1000) * 2).astype(np.float32)
        t = torch.from_numpy(logits).cuda()
        assert fusedtok.sample_topp(t, 0.9, seed=11) == \
            fusedtok.sample_topp(logits, 0.9, seed=11)

    def test_wrong_dtype_raises(self):
        t = torch.randn(8, dtype=torch.float64, device="cuda")
        with pytest.raises(TypeError):
            fusedtok.sample_topp(t, 0.9)
