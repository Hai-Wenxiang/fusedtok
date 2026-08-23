"""Row-wise softmax correctness, including overflow protection."""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def ref_softmax(x):
    z = x.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def test_hand_checkable():
    x = np.array([[0.0, 1.0]], dtype=np.float32)
    y = fusedtok.softmax(x)
    e = np.exp(1.0)
    assert y[0] == pytest.approx([1 / (1 + e), e / (1 + e)], abs=1e-6)


def test_rows_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 100)).astype(np.float32)
    y = fusedtok.softmax(x)
    assert y.sum(axis=-1) == pytest.approx(np.ones(5), abs=1e-5)


def test_large_logits_do_not_overflow():
    # naive exp(1000) would be inf; max-subtraction must keep it finite
    x = np.array([[1000.0, 1000.5, 999.0]], dtype=np.float32)
    y = fusedtok.softmax(x)
    assert np.isfinite(y).all()
    assert y.argmax() == 1


def test_matches_reference_random():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 33)).astype(np.float32) * 3
    assert fusedtok.softmax(x) == pytest.approx(ref_softmax(x), abs=1e-6)


def test_1d_row():
    y = fusedtok.softmax(np.zeros(4, dtype=np.float32))
    assert y == pytest.approx(np.full(4, 0.25), abs=1e-6)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_staged_matches_cpu(self):
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((8, 129)) * 5).astype(np.float32)
        assert fusedtok.softmax(x, cuda=True) == pytest.approx(
            fusedtok.softmax(x), abs=1e-5)

    @pytest.mark.parametrize("cols", [8191, 8192, 8193, 16384])
    def test_kernel_cutover_widths(self, cols):
        # 8192 = exactly kSmPerThread * kSmBlock (register path boundary);
        # 8193+ switches to the online streaming kernel. Both must match
        # the CPU reference within the fast-exp tolerance.
        rng = np.random.default_rng(cols)
        x = (rng.standard_normal((4, cols)) * 3).astype(np.float32)
        gpu = fusedtok.softmax(x, cuda=True)
        cpu = fusedtok.softmax(x)
        # __expf is a fast approximation; worst-case relative drift vs the
        # exact-exp CPU reference is ~2.4e-5 on rows dominated by one
        # extreme element (both float32 accumulation orders contribute).
        # Typical rows match far below 1e-6.
        assert gpu == pytest.approx(cpu, rel=3e-5, abs=1e-6)
        assert np.isfinite(gpu).all()


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_gpu_matches_torch_reference(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((8, 64)).astype(np.float32)
        out = fusedtok.softmax(torch.from_numpy(x).cuda())
        torch.cuda.synchronize()
        ref = torch.softmax(torch.from_numpy(x), dim=-1)
        assert out.cpu().numpy() == pytest.approx(ref.numpy(), abs=1e-6)
