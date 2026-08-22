"""RMSNorm correctness across all execution paths.

Reference formula (computed independently in Python):
    y = (x + r) * rsqrt(mean((x + r)^2) + eps) * w
"""

import math

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

RTOL_CASES = [((4, 65), 1e-6), ((1, 2), 1e-12), ((8, 257), 1e-6), ((3, 1), 1e-6)]


def ref_rmsnorm(x, w, eps, residual=None):
    """Vectorized pure-numpy reference over the last dimension."""
    v = x if residual is None else x + residual
    inv = 1.0 / np.sqrt((v.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)
    return (v * inv * w).astype(np.float32)


def make_case(rows, cols, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, cols).astype(np.float32)
    return x, w


def test_hand_checkable():
    # One row [3, 4], unit weight, eps ~ 0:
    #   rms = sqrt((9 + 16) / 2) = sqrt(12.5), y = x / rms
    y = fusedtok.rmsnorm(np.array([[3.0, 4.0]], dtype=np.float32),
                         np.array([1.0, 1.0], dtype=np.float32), eps=1e-12)
    inv = 1.0 / math.sqrt(12.5)
    assert y[0] == pytest.approx([3.0 * inv, 4.0 * inv], abs=1e-5)


@pytest.mark.parametrize(("shape", "eps"), RTOL_CASES)
def test_matches_reference(shape, eps):
    x, w = make_case(*shape, seed=1)
    y = fusedtok.rmsnorm(x, w, eps=eps)
    assert y.shape == x.shape
    assert y == pytest.approx(ref_rmsnorm(x, w, eps), rel=1e-4, abs=1e-5)


def test_residual_equals_shifted_input():
    # rmsnorm(x, r) must equal rmsnorm(x + r) without residual
    rng = np.random.default_rng(2)
    x, w = make_case(3, 32, seed=3)
    r = rng.uniform(-1, 1, x.shape).astype(np.float32)
    with_r = fusedtok.rmsnorm(x, w, residual=r)
    plain = fusedtok.rmsnorm(x + r, w)
    assert with_r == pytest.approx(plain, abs=1e-5)


def test_1d_input_is_single_row():
    x, w = make_case(1, 8, seed=4)
    y_1d = fusedtok.rmsnorm(x[0], w)
    y_2d = fusedtok.rmsnorm(x, w)
    assert y_1d.shape == (8,)
    assert y_1d == pytest.approx(y_2d[0], abs=1e-6)


def test_shape_errors():
    x, w = make_case(1, 4, seed=5)
    with pytest.raises(ValueError):
        fusedtok.rmsnorm(x, w[:3])           # weight length != cols
    with pytest.raises(ValueError):
        fusedtok.rmsnorm(x, w, residual=np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        fusedtok.rmsnorm(np.zeros((2, 2, 2), dtype=np.float32), w)  # 3-D


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_staged_matches_cpu(self):
        rng = np.random.default_rng(6)
        x, w = make_case(8, 257, seed=7)
        r = rng.uniform(-1, 1, x.shape).astype(np.float32)
        cpu = fusedtok.rmsnorm(x, w, residual=r)
        gpu = fusedtok.rmsnorm(x, w, residual=r, cuda=True)
        assert gpu == pytest.approx(cpu, abs=1e-5)

    def test_single_col(self):
        # cols = 1 edge case: every row normalizes to +-w[0]
        y = fusedtok.rmsnorm(np.array([[2.0], [-3.0]], dtype=np.float32),
                             np.array([1.5], dtype=np.float32), cuda=True)
        assert np.allclose(y, [[1.5], [-1.5]], atol=1e-5)

    def test_empty_rows(self):
        y = fusedtok.rmsnorm(np.zeros((0, 4), dtype=np.float32),
                             np.ones(4, dtype=np.float32), cuda=True)
        assert y.shape == (0, 4)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_gpu_matches_cpu(self):
        rng = np.random.default_rng(8)
        x, w = make_case(8, 257, seed=9)
        r = rng.uniform(-1, 1, x.shape).astype(np.float32)
        cpu = fusedtok.rmsnorm(x, w, residual=r)
        out = fusedtok.rmsnorm(torch.from_numpy(x).cuda(),
                               torch.from_numpy(w).cuda(),
                               residual=torch.from_numpy(r).cuda())
        torch.cuda.synchronize()
        assert out.cpu().numpy() == pytest.approx(cpu, abs=1e-5)

    def test_wrong_dtype_raises(self):
        x = torch.randn(2, 8, dtype=torch.float64, device="cuda")
        w = torch.ones(8, dtype=torch.float32, device="cuda")
        with pytest.raises(TypeError):
            fusedtok.rmsnorm(x, w)

    def test_weight_must_be_on_gpu(self):
        x = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        w = torch.ones(8, dtype=torch.float32)
        with pytest.raises(TypeError):
            fusedtok.rmsnorm(x, w)
