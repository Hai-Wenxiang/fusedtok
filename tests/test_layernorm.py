"""LayerNorm correctness: y = (x - mean) / sqrt(biased_var + eps) * w + b."""

import math

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def ref_layernorm(x, w, b, eps):
    mu = x.mean(axis=-1, keepdims=True, dtype=np.float64)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True, dtype=np.float64)
    return ((x - mu) / np.sqrt(var + eps) * w + b).astype(np.float32)


def make_case(rows, cols, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)).astype(np.float32),
            rng.uniform(0.5, 1.5, cols).astype(np.float32),
            rng.uniform(-0.5, 0.5, cols).astype(np.float32))


def test_hand_checkable():
    # Row [1, 2, 3]: mean = 2, var = 2/3; unit affine -> (x-2)/sqrt(2/3)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    w = np.ones(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    y = fusedtok.layernorm(x, w, b, eps=1e-12)
    scale = 1.0 / math.sqrt(2.0 / 3.0)
    assert y[0] == pytest.approx([-scale, 0.0, scale], abs=1e-5)


def test_constant_row_is_pure_bias():
    # variance 0: every output equals b regardless of w
    x = np.full((2, 5), 3.0, dtype=np.float32)
    w = np.full(5, 2.0, dtype=np.float32)
    b = np.arange(5, dtype=np.float32)
    y = fusedtok.layernorm(x, w, b)
    assert y == pytest.approx(np.stack([b, b]), abs=1e-5)


def test_matches_reference():
    for rows, cols in [(4, 65), (1, 128), (7, 3)]:
        x, w, b = make_case(rows, cols, seed=rows * 100 + cols)
        y = fusedtok.layernorm(x, w, b)
        assert y == pytest.approx(ref_layernorm(x, w, b, 1e-6), rel=1e-4, abs=1e-5)


def test_errors():
    x, w, b = make_case(2, 4, seed=1)
    with pytest.raises(ValueError):
        fusedtok.layernorm(x, w[:3], b)                       # weight too short
    with pytest.raises(ValueError):
        fusedtok.layernorm(x, w, np.zeros(5, dtype=np.float32))  # bias too long


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_staged_matches_cpu(self):
        x, w, b = make_case(4, 513, seed=2)
        cpu = fusedtok.layernorm(x, w, b)
        gpu = fusedtok.layernorm(x, w, b, cuda=True)
        assert gpu == pytest.approx(cpu, abs=1e-5)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_gpu_matches_torch_reference(self):
        # Cross-check against torch's own nn.functional layernorm
        x, w, b = make_case(4, 256, seed=3)
        out = fusedtok.layernorm(torch.from_numpy(x).cuda(),
                                 torch.from_numpy(w).cuda(),
                                 torch.from_numpy(b).cuda())
        torch.cuda.synchronize()
        ref = torch.nn.functional.layer_norm(
            torch.from_numpy(x), (256,), torch.from_numpy(w), torch.from_numpy(b), 1e-6)
        assert out.cpu().numpy() == pytest.approx(ref.numpy(), abs=1e-4)
