"""SwiGLU: out = silu(gate) * up, across all execution paths."""

import math

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def ref_swiglu(gate, up):
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up).astype(np.float32)


def make_case(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n).astype(np.float32),
            rng.standard_normal(n).astype(np.float32))


def test_hand_checkable():
    # gate = 0 -> silu(0) = 0; up = anything -> 0
    # gate = 1, up = 2 -> silu(1) * 2
    gate = np.array([0.0, 1.0], dtype=np.float32)
    up = np.array([5.0, 2.0], dtype=np.float32)
    y = fusedtok.swiglu(gate, up)
    assert y[0] == pytest.approx(0.0, abs=1e-7)
    assert y[1] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)) * 2.0, rel=1e-5)


def test_matches_reference():
    gate, up = make_case(1000, seed=0)
    assert fusedtok.swiglu(gate, up) == pytest.approx(ref_swiglu(gate, up), rel=1e-4, abs=1e-5)


def test_extreme_gate():
    gate = np.array([-100.0, 100.0], dtype=np.float32)
    up = np.array([3.0, 3.0], dtype=np.float32)
    y = fusedtok.swiglu(gate, up)
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert y[1] == pytest.approx(300.0, rel=1e-5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fusedtok.swiglu(np.ones(4, dtype=np.float32), np.ones(5, dtype=np.float32))


def test_empty():
    y = fusedtok.swiglu(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    assert y.size == 0


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_staged_matches_cpu(self):
        gate, up = make_case(5000, seed=1)
        cpu = fusedtok.swiglu(gate, up)
        gpu = fusedtok.swiglu(gate, up, cuda=True)
        assert gpu == pytest.approx(cpu, abs=1e-4)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_gpu_matches_torch_reference(self):
        # Compare against torch's own silu for confidence
        gate, up = make_case(4096, seed=2)
        out = fusedtok.swiglu(torch.from_numpy(gate).cuda(), torch.from_numpy(up).cuda())
        torch.cuda.synchronize()
        ref = torch.nn.functional.silu(torch.from_numpy(gate)) * torch.from_numpy(up)
        assert out.cpu().numpy() == pytest.approx(ref.numpy(), abs=1e-4)
