"""INT8 matmul with per-channel weight scales (qgemm_perchannel).

The W8A8 layout real INT8 inference uses: activations quantized
per-tensor, weights quantized per output channel (one scale per row of
B_q). The kernel composes the output scale as float32(sa * sb[j]) with
a single rounding and applies it once to the exact int32 accumulator,
so CPU / staged / zero-copy results are BIT-IDENTICAL - these tests
assert exact equality across paths (no tolerance games), and against a
numpy float32 reference the only allowed difference is the scale
rounding.
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def ref_qgemm_pc(a, b, sa, sb):
    # numpy mirror of the documented operation order: exact int32
    # matmul, f32(sa * sb[j]) per column (one rounding), one product
    acc = a.astype(np.int32) @ b.astype(np.int32).T
    scales = np.float32(sa) * sb           # float32 scalar x float32 vec
    return acc.astype(np.float32) * scales


def rand_q(rng, rows, k):
    q = rng.integers(-127, 128, size=(rows, k), dtype=np.int64)
    return q.astype(np.int8)


def rand_scales(rng, n):
    # spread over ~2 decades so per-channel vs per-tensor differences
    # actually show up in the numerics
    return (10.0 ** rng.uniform(-2.0, 0.3, size=n)).astype(np.float32)


@pytest.mark.parametrize("m,n,k", [
    (1, 8, 16),           # GEMV path, tiny
    (1, 1000, 4096),      # GEMV path, decode-like
    (3, 5, 7),            # everything non-tiled, odd K (scalar fallback)
    (65, 63, 33),         # one tile over, partial slabs
    (300, 500, 1024),     # multi-tile
    (17, 4096, 8192),     # tall N, wide K (exercise the tuning sweep)
])
def test_qgemm_pc_cpu_matches_numpy(m, n, k):
    rng = np.random.default_rng(m * 1000 + n + k)
    a = rand_q(rng, m, k)
    b = rand_q(rng, n, k)
    sb = rand_scales(rng, n)
    y = fusedtok.qgemm_perchannel(a, 0.03, b, sb)
    assert y.shape == (m, n)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, ref_qgemm_pc(a, b, 0.03, sb), rtol=1e-6)


def test_qgemm_pc_extreme_values():
    # int8 extremes with adversarial scales: exact integers times exact
    # f32 scale products - no overflow before the float conversion
    a = np.full((4, 100), -127, dtype=np.int8)
    b = np.full((3, 100), -127, dtype=np.int8)
    sb = np.array([0.001, 1.0, 1000.0], dtype=np.float32)
    y = fusedtok.qgemm_perchannel(a, 1.0, b, sb)
    np.testing.assert_allclose(y, ref_qgemm_pc(a, b, 1.0, sb), rtol=1e-6)


def test_qgemm_pc_equal_scales_matches_per_tensor():
    # the consistency anchor: with all b_scales == sb, the per-channel
    # kernel must produce EXACTLY the per-tensor result (same f32 scale
    # composition f32(sa*sb), same single product)
    rng = np.random.default_rng(5)
    a = rand_q(rng, 33, 129)
    b = rand_q(rng, 47, 129)
    sb = np.float32(0.02)
    sb_vec = np.full(47, sb, dtype=np.float32)
    y_pc = fusedtok.qgemm_perchannel(a, 0.03, b, sb_vec)
    y_pt = fusedtok.qgemm(a, 0.03, b, float(sb))
    np.testing.assert_array_equal(y_pc, y_pt)


def test_qgemm_pc_k0_zero_fill():
    # K == 0 -> zeros on every path (same contract as per-tensor qgemm)
    a = np.zeros((4, 0), dtype=np.int8)
    b = np.zeros((3, 0), dtype=np.int8)
    sb = np.ones(3, dtype=np.float32)
    y = fusedtok.qgemm_perchannel(a, 0.5, b, sb)
    np.testing.assert_array_equal(y, np.zeros((4, 3), np.float32))


def test_qgemm_pc_errors():
    a = np.zeros((4, 8), dtype=np.int8)
    b = np.zeros((2, 8), dtype=np.int8)
    with pytest.raises(ValueError):
        fusedtok.qgemm_perchannel(a, 1.0, b, np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        fusedtok.qgemm_perchannel(a, 1.0, np.zeros((2, 7), dtype=np.int8),
                                  np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError):
        fusedtok.qgemm_perchannel(np.zeros((4,), dtype=np.int8), 1.0, b,
                                  np.ones(2, dtype=np.float32))


def test_qgemm_pc_end_to_end_w8a8():
    # THE W8A8 selling point, verified end to end on spiky weights:
    # every 4th output row carries a 12x outlier, so the per-tensor
    # scale is owned by those rows and the remaining rows quantize to a
    # handful of levels - per-channel scales quantize each row to its
    # own grid and must land dramatically closer to the float matmul.
    rng = np.random.default_rng(9)
    k, n, m = 256, 64, 8
    # activations scaled small so the WEIGHT quantization error dominates
    # the total (the regime where per-channel scales matter in practice)
    x = (0.01 * rng.standard_normal((m, k))).astype(np.float32)
    spikes = (np.arange(n) % 4 == 0).astype(np.float32) * 12.0
    w = (rng.standard_normal((n, k)) * 0.1 + spikes[:, None]).astype(np.float32)
    ref = x.astype(np.float64) @ w.T.astype(np.float64)

    xq, sx = fusedtok.quantize_int8(x.ravel())
    xq = xq.reshape(m, k)
    sw = np.float32(fusedtok.quantize_int8(w.ravel())[1])
    wq_pt = fusedtok.quantize_int8(w.ravel())[0].reshape(n, k)

    wq_rows = np.zeros((n, k), dtype=np.int8)
    sw_rows = np.zeros(n, dtype=np.float32)
    for r in range(n):
        q, s = fusedtok.quantize_int8(w[r])
        wq_rows[r] = q
        sw_rows[r] = float(s)

    y_pt = fusedtok.qgemm(xq, float(sx), wq_pt, float(sw))
    y_pc = fusedtok.qgemm_perchannel(xq, float(sx), wq_rows, sw_rows)

    # The W8A8 claim, checked on the rows where the schemes differ:
    # NON-spike rows own their scale under per-channel (grid ~0.002) but
    # are crushed to a few levels under the spike-dominated per-tensor
    # scale (~0.097). The max error there must drop by a wide factor.
    flat = spikes == 0
    err_pt_ns = np.abs((y_pt.astype(np.float64) - ref)[:, flat]).max()
    err_pc_ns = np.abs((y_pc.astype(np.float64) - ref)[:, flat]).max()
    assert err_pc_ns < err_pt_ns / 5.0, (
        f"non-spike rows: per-channel error {err_pc_ns} not far below "
        f"per-tensor {err_pt_ns}")

    # Spike rows are weight-quantization-equivalent in BOTH schemes
    # (their own outlier sets the scale either way); the error there is
    # the shared activation-noise floor:
    #   sqrt(k) * (0.5 * sa) * rms(weight) * tail slack
    err_pc_all = np.abs(y_pc.astype(np.float64) - ref).max()
    act_floor = (np.sqrt(k) * 0.5 * float(sx)
                 * np.sqrt((w ** 2).max()) * 5.0)
    assert err_pc_all < act_floor, (
        f"per-channel error {err_pc_all} exceeds activation-noise "
        f"floor {act_floor}")


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    @pytest.mark.parametrize("m,n,k", [
        (1, 8, 16),
        (1, 4096, 4096),    # GEMV kernel
        (65, 63, 33),
        (300, 500, 1024),   # multi-tile GEMM
        (17, 4096, 129),    # odd K (scalar fallback path)
    ])
    def test_staged_matches_cpu_bitexact(self, m, n, k):
        rng = np.random.default_rng(m + n * 7 + k)
        a = rand_q(rng, m, k)
        b = rand_q(rng, n, k)
        sb = rand_scales(rng, n)
        y = fusedtok.qgemm_perchannel(a, 0.05, b, sb, cuda=True)
        ycpu = fusedtok.qgemm_perchannel(a, 0.05, b, sb)
        # cross-path: GPU must be bit-identical to the CPU path
        np.testing.assert_array_equal(y, ycpu)

    def test_tuned_repeats_bitidentical(self):
        # tuning runs the real kernel into the same output before the
        # caller ever sees a result; repeats must be bit-stable
        rng = np.random.default_rng(23)
        a = rand_q(rng, 96, 512)
        b = rand_q(rng, 77, 512)
        sb = rand_scales(rng, 77)
        y = fusedtok.qgemm_perchannel(a, 0.03, b, sb, cuda=True)
        ycpu = fusedtok.qgemm_perchannel(a, 0.03, b, sb)
        np.testing.assert_array_equal(y, ycpu)
        for _ in range(3):
            np.testing.assert_array_equal(
                fusedtok.qgemm_perchannel(a, 0.03, b, sb, cuda=True), ycpu)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_matches_cpu_bitexact(self):
        rng = np.random.default_rng(24)
        a = rand_q(rng, 129, 500)
        b = rand_q(rng, 260, 500)
        sb = rand_scales(rng, 260)
        at = torch.from_numpy(a).cuda()
        bt = torch.from_numpy(b).cuda()
        sbt = torch.from_numpy(sb).cuda()
        y = fusedtok.qgemm_perchannel(at, 0.03, bt, sbt)
        np.testing.assert_array_equal(
            y.cpu().numpy(), fusedtok.qgemm_perchannel(a, 0.03, b, sb))

    def test_gemv_decode_shape(self):
        # realistic W8A8 decode step: [1, hidden] @ [out, hidden]^T with
        # per-channel weight scales
        rng = np.random.default_rng(25)
        hidden, out = 1024, 8192
        x = rand_q(rng, 1, hidden)
        w = rand_q(rng, out, hidden)
        sb = rand_scales(rng, out)
        xt = torch.from_numpy(x).cuda()
        wt = torch.from_numpy(w).cuda()
        sbt = torch.from_numpy(sb).cuda()
        y = fusedtok.qgemm_perchannel(xt, 0.06, wt, sbt)
        assert y.shape == (1, out)
        np.testing.assert_array_equal(
            y.cpu().numpy(), fusedtok.qgemm_perchannel(x, 0.06, w, sb))

    def test_bscale_broadcast_and_device_hops(self):
        # zero-copy is selected by the ACTIVATIONS tensor; b_scales may
        # arrive as a CPU list / CPU tensor / strided tensor and must be
        # validated, normalized and moved without changing the result
        rng = np.random.default_rng(26)
        a = rand_q(rng, 32, 128)
        b = rand_q(rng, 48, 128)
        sb = rand_scales(rng, 48)
        at = torch.from_numpy(a).cuda()
        bt = torch.from_numpy(b).cuda()
        y_ref = fusedtok.qgemm_perchannel(a, 0.05, b, sb)
        y_list = fusedtok.qgemm_perchannel(at, 0.05, bt, sb.tolist())
        np.testing.assert_array_equal(y_list.cpu().numpy(), y_ref)
        y_cpu_t = fusedtok.qgemm_perchannel(at, 0.05, bt,
                                            torch.from_numpy(sb))
        np.testing.assert_array_equal(y_cpu_t.cpu().numpy(), y_ref)
        # a strided (non-contiguous) scale vector must be normalized too
        sbt_strided = torch.from_numpy(np.repeat(sb, 2)).cuda()[::2]
        assert sbt_strided.is_cuda and not sbt_strided.is_contiguous()
        y_strided = fusedtok.qgemm_perchannel(at, 0.05, bt, sbt_strided)
        np.testing.assert_array_equal(y_strided.cpu().numpy(), y_ref)

    def test_mixed_device_family_rejected(self):
        # a CUDA activation with a CPU weight is the zero-copy path with
        # a CPU second operand - the same TypeError contract as qgemm
        a = torch.randint(-127, 128, (8, 32), dtype=torch.int8)
        b = torch.randint(-127, 128, (8, 32), dtype=torch.int8).cuda()
        with pytest.raises(TypeError):
            fusedtok.qgemm_perchannel(a, 0.05, b, np.ones(8, np.float32))

    def test_graph_capture_replay(self):
        # per-channel qgemm is graph-capturable; replay must recompute
        # through the baked sb pointer with mutated int8 inputs
        m, n, k = 64, 64, 256
        a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        sb = torch.rand(n, device="cuda") + 0.5
        y = torch.empty((m, n), device="cuda", dtype=torch.float32)

        def run():
            from fusedtok import _fusedtok
            _fusedtok.qgemm_perchannel_launch(
                a.data_ptr(), b.data_ptr(), sb.data_ptr(), y.data_ptr(),
                m, n, k, 0.05, torch.cuda.current_stream().cuda_stream)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        a.mul_(-1)                       # replay must recompute
        g.replay()
        torch.cuda.synchronize()
        ref = torch.from_numpy(fusedtok.qgemm_perchannel(
            a.cpu().numpy(), 0.05, b.cpu().numpy(), sb.cpu().numpy())).cuda()
        assert torch.allclose(y, ref, rtol=1e-6)
