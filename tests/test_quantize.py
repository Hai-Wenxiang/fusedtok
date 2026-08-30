"""INT8 quantization utilities: roundtrip error bounds, scale semantics,
fused qadd, dtype/shape contracts, CPU/GPU/zero-copy parity."""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def test_quantize_scale_semantics():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    q, s = fusedtok.quantize_int8(x)
    assert s == pytest.approx(2.0 / 127.0)
    assert q.dtype == np.int8
    assert q[0] == -127 and q[4] == 127
    assert q[2] == 0


def test_roundtrip_error_bounded_by_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000).astype(np.float32)
    q, s = fusedtok.quantize_int8(x)
    back = fusedtok.dequantize_int8(q, s)
    # each element is within half a quantization step (plus rounding)
    assert np.abs(back - x).max() <= s * 0.51


def test_zero_input_scale_is_one():
    # degenerate all-zero input: scale 1.0 avoids div-by-zero
    x = np.zeros(8, dtype=np.float32)
    q, s = fusedtok.quantize_int8(x)
    assert s == 1.0
    assert (q == 0).all()


def test_clamping_never_overflow():
    x = np.array([1e30, -1e30], dtype=np.float32)
    q, s = fusedtok.quantize_int8(x)
    assert q[0] == 127 and q[1] == -127


def test_qadd_matches_explicit_pipeline():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(1000).astype(np.float32)
    b = rng.standard_normal(1000).astype(np.float32)
    qa, sa = fusedtok.quantize_int8(a)
    qb, sb = fusedtok.quantize_int8(b)
    # explicit: dequant, add, requant
    ref, sref = fusedtok.quantize_int8(
        fusedtok.dequantize_int8(qa, sa) + fusedtok.dequantize_int8(qb, sb))
    if HAS_TORCH and fusedtok.cuda_available():
        import torch
        tq = fusedtok.qadd_int8(torch.from_numpy(qa).cuda(), sa,
                                torch.from_numpy(qb).cuda(), sb)
        y, sy = tq[0].cpu().numpy(), float(tq[1])
        # same scale (same absmax math); codes may differ by 1 from
        # rintf-vs-rint rounding - compare dequantized values
        assert sy == pytest.approx(sref, rel=1e-6)
        assert np.abs(y.astype(np.float32) * sy - ref * sref).max() <= sy * 1.01


def test_qadd_requires_cuda_int8():
    with pytest.raises(TypeError):
        fusedtok.qadd_int8(np.ones(4, dtype=np.int8), 1.0,
                           np.ones(4, dtype=np.int8), 1.0)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestCuda:
    def test_zero_copy_roundtrip(self):
        x = torch.randn(1000, device="cuda", dtype=torch.float32)
        q, s = fusedtok.quantize_int8(x)
        assert q.dtype is torch.int8 and q.is_cuda
        back = fusedtok.dequantize_int8(q, s)
        assert back.dtype is torch.float32
        assert (back - x).abs().max().item() <= s * 0.51

    def test_matches_cpu(self):
        x = np.random.default_rng(2).standard_normal(777).astype(np.float32)
        q_cpu, s_cpu = fusedtok.quantize_int8(x)
        q_gpu, s_gpu = fusedtok.quantize_int8(torch.from_numpy(x).cuda())
        assert float(s_gpu) == pytest.approx(s_cpu, rel=1e-6)
        assert (q_gpu.cpu().numpy() == q_cpu).all()
