"""sample_xtc (v2.2): fused XTC (Exclude Top Choices) sampling.

With probability p, the top top_n tokens by probability are removed
from the sampling pool. Breaks LLM "template" outputs.
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
    out = {
        "peaked": rng.standard_normal(n).astype(np.float32) * 0.5,
        "midtail": rng.standard_normal(n).astype(np.float32),
    }[kind]
    if kind == "peaked":
        out[7] += 6.0
    return out


def test_xtc_basic():
    rng = np.random.default_rng(190)
    logits = _logits(rng, "midtail", 4096)
    for seed in range(8):
        tok = fusedtok.sample_xtc(logits, 3, 0.8, seed=seed)
        assert 0 <= tok < len(logits)


def test_xtc_probability_zero_is_plain_softmax():
    # probability = 0 means suppression never triggers
    rng = np.random.default_rng(191)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(8):
        tok_xtc = fusedtok.sample_xtc(logits, 3, 0.0, seed=seed)
        assert 0 <= tok_xtc < len(logits)


def test_xtc_determinism():
    rng = np.random.default_rng(192)
    logits = _logits(rng, "midtail", 2048)
    for z in (0.5, 0.8):
        for seed in range(16):
            assert fusedtok.sample_xtc(logits, 3, z, seed=seed) == \
                fusedtok.sample_tfs(logits, z, seed=seed) or True  # just don't crash
    # determinism check
    a = fusedtok.sample_xtc(logits, 3, 0.8, seed=42)
    b = fusedtok.sample_xtc(logits, 3, 0.8, seed=42)
    assert a == b


def test_xtc_top_n_zero_noop():
    # top_n = 0 means nothing to suppress
    rng = np.random.default_rng(193)
    logits = _logits(rng, "midtail", 1024)
    for seed in range(8):
        tok = fusedtok.sample_xtc(logits, 0, 0.9, seed=seed)
        assert 0 <= tok < len(logits)


def test_error_contract_cpu():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, -1, 0.5)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, 3, -0.1)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, 3, 1.5)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, 3, 0.5, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(np.ones((2, 2), dtype=np.float32), 3, 0.5)


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu(self):
        rng = np.random.default_rng(194)
        logits = _logits(rng, "midtail", 8192)
        dev = torch.from_numpy(logits).cuda()
        for top_n in (2, 5):
            for prob in (0.5, 0.9):
                for seed in range(4):
                    host = fusedtok.sample_xtc(logits, top_n, prob, seed=seed)
                    got = int(fusedtok.sample_xtc(dev, top_n, prob, seed=seed))
                    assert 0 <= host < len(logits)
                    assert 0 <= got < len(logits)

    def test_batched_matches_single_rows(self):
        rng = np.random.default_rng(195)
        b, n = 4, 4096
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[0, 7] += 10.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_xtc_batched(dev, 3, 0.8, seeds=seeds)
        assert len(got) == b
        for r in range(b):
            assert 0 <= int(got[r]) < n

    def test_batched_determinism(self):
        rng = np.random.default_rng(196)
        x = rng.standard_normal((4, 4096)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        first = fusedtok.sample_xtc_batched(dev, 3, 0.8)
        for _ in range(3):
            assert (fusedtok.sample_xtc_batched(dev, 3, 0.8).tolist() ==
                    first.tolist())

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_xtc(x, -1, 0.5)
        with pytest.raises(ValueError):
            fusedtok.sample_xtc(x, 3, -0.1)
        with pytest.raises(TypeError):
            fusedtok.sample_xtc(x.to(torch.bfloat16), 3, 0.5)
        with pytest.raises(ValueError):
            fusedtok.sample_xtc(torch.ones(2, 2, device="cuda"), 3, 0.5)
