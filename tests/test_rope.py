"""RoPE correctness: interleaved and NeoX layouts, position offsets."""

import math

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def angles(m, j, dim, theta):
    return m * theta ** (-2.0 * j / dim)


def ref_rope_interleaved(x, theta, pos_offset):
    seq, dim = x.shape
    y = np.zeros_like(x)
    for r in range(seq):
        for j in range(dim // 2):
            a = angles(r + pos_offset, j, dim, theta)
            c, s = math.cos(a), math.sin(a)
            e, o = 2 * j, 2 * j + 1
            y[r, e] = x[r, e] * c - x[r, o] * s
            y[r, o] = x[r, e] * s + x[r, o] * c
    return y


def ref_rope_neox(x, theta, pos_offset):
    seq, dim = x.shape
    y = np.zeros_like(x)
    for r in range(seq):
        for j in range(dim // 2):
            a = angles(r + pos_offset, j, dim, theta)
            c, s = math.cos(a), math.sin(a)
            y[r, j] = x[r, j] * c - x[r, dim // 2 + j] * s
            y[r, dim // 2 + j] = x[r, j] * s + x[r, dim // 2 + j] * c
    return y


def make_qk(seq, dim, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((seq, dim)).astype(np.float32),
            rng.standard_normal((seq, dim)).astype(np.float32))


def test_position_zero_is_identity():
    q, _ = make_qk(3, 8, seed=0)
    q2, k2 = fusedtok.rope(q, None)
    # angle = 0 -> cos=1, sin=0 for every pair at position 0
    assert q2[0] == pytest.approx(q[0], abs=1e-6)
    assert k2 is None


def test_matches_reference_both_layouts():
    for neox, ref in [(False, ref_rope_interleaved), (True, ref_rope_neox)]:
        q, k = make_qk(4, 16, seed=1)
        q2, k2 = fusedtok.rope(q, k, neox=neox)
        assert q2 == pytest.approx(ref(q, 10000.0, 0), abs=1e-4)
        assert k2 == pytest.approx(ref(k, 10000.0, 0), abs=1e-4)


def test_pos_offset_matches_shifted_reference():
    q, _ = make_qk(2, 16, seed=2)
    for neox, ref in [(False, ref_rope_interleaved), (True, ref_rope_neox)]:
        q2, _ = fusedtok.rope(q, None, pos_offset=7, neox=neox)
        assert q2 == pytest.approx(ref(q, 10000.0, 7), abs=1e-4)


def test_pos_offset_equals_longer_sequence_slice():
    # rope(seq=2, offset=5) must equal rows 5..6 of rope(seq=8, offset=0)
    q, _ = make_qk(8, 16, seed=3)
    short, _ = fusedtok.rope(q[5:7].copy(), None, pos_offset=5)
    full, _ = fusedtok.rope(q, None)
    assert short == pytest.approx(full[5:7], abs=1e-4)


def test_norm_is_conserved():
    # rotation preserves the per-pair norm
    q, _ = make_qk(3, 8, seed=4)
    q2, _ = fusedtok.rope(q, None)
    norm_before = np.sqrt(q[:, 0::2] ** 2 + q[:, 1::2] ** 2)
    norm_after = np.sqrt(q2[:, 0::2] ** 2 + q2[:, 1::2] ** 2)
    assert norm_after == pytest.approx(norm_before, abs=1e-5)


def test_custom_theta():
    q, _ = make_qk(2, 8, seed=5)
    q2, _ = fusedtok.rope(q, None, theta=500000.0)
    assert q2 == pytest.approx(ref_rope_interleaved(q, 500000.0, 0), abs=1e-4)


def test_odd_dim_raises():
    with pytest.raises(ValueError):
        fusedtok.rope(np.zeros((2, 7), dtype=np.float32), None)
    with pytest.raises(ValueError):
        fusedtok.rope(np.zeros((2, 8), dtype=np.float32), None, pos_offset=-1)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    @pytest.mark.parametrize("neox", [False, True])
    def test_staged_matches_cpu(self, neox):
        q, k = make_qk(4, 64, seed=6)
        cpu_q, cpu_k = fusedtok.rope(q, k, neox=neox, pos_offset=3)
        gpu_q, gpu_k = fusedtok.rope(q, k, neox=neox, pos_offset=3, cuda=True)
        assert gpu_q == pytest.approx(cpu_q, abs=1e-4)
        assert gpu_k == pytest.approx(cpu_k, abs=1e-4)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_gpu_matches_cpu(self):
        q, k = make_qk(4, 64, seed=7)
        cpu_q, cpu_k = fusedtok.rope(q, k, neox=True, pos_offset=2)
        out_q, out_k = fusedtok.rope(torch.from_numpy(q).cuda(),
                                     torch.from_numpy(k).cuda(),
                                     neox=True, pos_offset=2)
        torch.cuda.synchronize()
        assert out_q.cpu().numpy() == pytest.approx(cpu_q, abs=1e-4)
        assert out_k.cpu().numpy() == pytest.approx(cpu_k, abs=1e-4)
