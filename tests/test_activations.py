"""Elementwise activations and binary ops.

Each op is checked against a pure-Python/numpy formula, on extreme values,
and for dtype/shape handling. Note float32 tolerance: CPU reference is also
float32, so exact equality mostly holds; approx is used for transcendentals.
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

EXTREMES = np.array([-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0], dtype=np.float32)


def test_silu_formula():
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    ref = [v / (1 + math.exp(-v)) for v in x]
    assert fusedtok.silu(x) == pytest.approx(ref, rel=1e-5, abs=1e-6)


def test_silu_extremes():
    y = fusedtok.silu(EXTREMES)
    # limits: silu(-inf) -> 0-, silu(+inf) -> x
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert y[-1] == pytest.approx(100.0, rel=1e-6)


def test_gelu_erf_formula():
    x = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    ref = [0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in x]
    assert fusedtok.gelu(x) == pytest.approx(ref, rel=1e-5, abs=1e-6)
    assert fusedtok.gelu(np.array([0.0], dtype=np.float32))[0] == 0.0


def test_gelu_tanh_close_to_erf():
    # tanh approximation tracks the exact form within ~1e-3
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    diff = np.abs(fusedtok.gelu_tanh(x) - fusedtok.gelu(x)).max()
    assert diff < 1e-3


def test_relu():
    x = np.array([-5.0, 0.0, 3.5], dtype=np.float32)
    assert fusedtok.relu(x) == pytest.approx([0.0, 0.0, 3.5], abs=1e-7)


def test_tanh_formula():
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    ref = np.tanh(x.astype(np.float64))
    assert fusedtok.tanh(x) == pytest.approx(ref, abs=1e-6)


def test_sigmoid_formula_and_extremes():
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    ref = [1 / (1 + math.exp(-v)) for v in x]
    assert fusedtok.sigmoid(x) == pytest.approx(ref, rel=1e-5)
    y = fusedtok.sigmoid(EXTREMES)
    assert y[3] == pytest.approx(0.5, abs=1e-7)   # EXTREMES[3] = 0.0
    assert y[0] == pytest.approx(0.0, abs=1e-7)
    assert y[-1] == pytest.approx(1.0, abs=1e-7)


def test_add_and_mul():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([10.0, 20.0], dtype=np.float32)
    assert fusedtok.add(a, b) == pytest.approx([11.0, 22.0], abs=1e-6)
    assert fusedtok.mul(a, b) == pytest.approx([10.0, 40.0], abs=1e-6)


def test_binary_shape_mismatch():
    a = np.ones(4, dtype=np.float32)
    b = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.add(a, b)
    with pytest.raises(ValueError):
        fusedtok.mul(a, b)


def test_2d_shapes_preserved():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5)).astype(np.float32)
    y = fusedtok.relu(x)
    assert y.shape == (3, 5)
    assert (y == np.maximum(x, 0)).all()


def test_float64_input_is_cast():
    y = fusedtok.relu(np.array([-1.0, 2.0]))
    assert y.dtype == np.float32
    assert y == pytest.approx([0.0, 2.0], abs=1e-7)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    @pytest.mark.parametrize("op,ref", [
        ("silu", lambda x: x / (1 + np.exp(-x))),
        ("gelu", None),
        ("gelu_tanh", None),
        ("relu", lambda x: np.maximum(x, 0)),
        ("tanh", None),
        ("sigmoid", lambda x: 1 / (1 + np.exp(-x))),
    ])
    def test_staged_matches_cpu(self, op, ref):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((7, 33)).astype(np.float32)
        cpu = getattr(fusedtok, op)(x)
        gpu = getattr(fusedtok, op)(x, cuda=True)
        assert gpu == pytest.approx(cpu, abs=1e-5)

    def test_binary_staged(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(100).astype(np.float32)
        b = rng.standard_normal(100).astype(np.float32)
        assert fusedtok.add(a, b, cuda=True) == pytest.approx(a + b, abs=1e-5)
        assert fusedtok.mul(a, b, cuda=True) == pytest.approx(a * b, abs=1e-4)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    @pytest.mark.parametrize("op", ["silu", "gelu", "gelu_tanh", "relu", "tanh", "sigmoid"])
    def test_gpu_matches_torch_reference(self, op):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((5, 17)).astype(np.float32)
        out = getattr(fusedtok, op)(torch.from_numpy(x).cuda())
        torch.cuda.synchronize()
        cpu = getattr(fusedtok, op)(x)
        assert out.cpu().numpy() == pytest.approx(cpu, abs=1e-5)

    def test_add_gpu(self):
        a = torch.randn(64, device="cuda", dtype=torch.float32)
        b = torch.randn(64, device="cuda", dtype=torch.float32)
        out = fusedtok.add(a, b)
        torch.cuda.synchronize()
        assert out.cpu().numpy() == pytest.approx((a + b).cpu().numpy(), abs=1e-5)
