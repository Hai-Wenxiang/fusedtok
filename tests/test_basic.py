"""Basic module behavior: import, version, cuda_available, and the axpy
skeleton operator across numpy / staged-CUDA / torch paths."""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def test_import_and_version():
    assert fusedtok.__version__
    assert isinstance(fusedtok.cuda_available(), bool)


def test_axpy_cpu_numpy():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = fusedtok.axpy(x, 2.0, 1.0)
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y == pytest.approx([3.0, 5.0, 7.0], abs=1e-6)


def test_axpy_empty():
    y = fusedtok.axpy(np.array([], dtype=np.float32), 1.0, 2.0)
    assert y.size == 0


def test_axpy_accepts_lists_and_casts_dtype():
    y = fusedtok.axpy([1, 2, 3], 1.0, 0.0)
    assert y.dtype == np.float32
    assert y == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
def test_axpy_cuda_staged():
    x = np.array([1.0, -2.0, 4.0], dtype=np.float32)
    y = fusedtok.axpy(x, 0.5, 2.0, cuda=True)
    assert y == pytest.approx([2.5, 1.0, 4.0], abs=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_axpy_torch_cpu_roundtrip():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = fusedtok.axpy(x, 3.0, 1.0)
    assert torch.is_tensor(y)
    assert y.dtype == torch.float32
    assert y.tolist() == pytest.approx([4.0, 7.0], abs=1e-6)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
def test_axpy_torch_cuda_zero_copy():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")
    y = fusedtok.axpy(x, 3.0, 1.0)
    torch.cuda.synchronize()
    assert y.is_cuda
    assert y.tolist() == pytest.approx([4.0, 7.0], abs=1e-6)
