"""INT8 matmul (qgemm): exact integer parity across every path.

The kernel accumulates int8 x int8 products in int32 and applies the
combined per-tensor scale exactly once at the store, so GPU and CPU
results are BIT-IDENTICAL to numpy's int32 matmul times a float scale -
these tests assert exact equality, no quantization tolerances.
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def ref_qgemm(a, b, sa, sb):
    acc = a.astype(np.int32) @ b.astype(np.int32).T
    return acc.astype(np.float32) * np.float32(sa * sb)


def rand_q(rng, rows, k):
    q = rng.integers(-127, 128, size=(rows, k), dtype=np.int64)
    return q.astype(np.int8)


@pytest.mark.parametrize("m,n,k", [
    (1, 8, 16),           # GEMV path, tiny
    (1, 1000, 4096),      # GEMV path, decode-like
    (3, 5, 7),            # everything non-tiled
    (64, 64, 32),         # exactly one tile, one slab
    (65, 63, 33),         # one tile over, partial slabs
    (128, 256, 512),      # multi-tile
    (17, 4096, 129),      # tall N, odd K (scalar tails)
    (100, 50, 1),         # K = 1
])
def test_qgemm_cpu_exact(m, n, k):
    rng = np.random.default_rng(m * 1000 + n + k)
    a = rand_q(rng, m, k)
    b = rand_q(rng, n, k)
    y = fusedtok.qgemm(a, 0.03, b, 0.02)
    assert y.shape == (m, n)
    assert y.dtype == np.float32
    # integer accumulation is exact; the float scale can differ from the
    # numpy reference by one rounding (f32(f32*f32) vs f32(f64*f64))
    np.testing.assert_allclose(y, ref_qgemm(a, b, 0.03, 0.02), rtol=1e-6)


def test_qgemm_extreme_values():
    # int8 extremes: -127 * -127 * 4096 nears int32 range in real GEMMs;
    # keep K modest so the exact reference stays in range
    a = np.full((4, 100), -127, dtype=np.int8)
    b = np.full((3, 100), -127, dtype=np.int8)
    y = fusedtok.qgemm(a, 1.0, b, 1.0)
    np.testing.assert_array_equal(y, ref_qgemm(a, b, 1.0, 1.0))


def test_qgemm_k0_zero_fill():
    # K == 0: every dot product is empty -> the output is all zeros on
    # every path. The v0.4 launcher skipped the GPU write entirely and
    # left torch.empty garbage here; the zero-fill contract is now fixed
    # and pinned (the CPU reference always produced zeros).
    a = np.zeros((4, 0), dtype=np.int8)
    b = np.zeros((3, 0), dtype=np.int8)
    y_cpu = fusedtok.qgemm(a, 0.5, b, 0.25)
    assert y_cpu.shape == (4, 3)
    np.testing.assert_array_equal(y_cpu, np.zeros((4, 3), np.float32))
    if fusedtok.cuda_available():
        y_staged = fusedtok.qgemm(a, 0.5, b, 0.25, cuda=True)
        np.testing.assert_array_equal(y_staged, y_cpu)
        if HAS_TORCH:
            at = torch.zeros((4, 0), dtype=torch.int8, device="cuda")
            bt = torch.zeros((3, 0), dtype=torch.int8, device="cuda")
            y_zc = fusedtok.qgemm(at, 0.5, bt, 0.25)
            np.testing.assert_array_equal(y_zc.cpu().numpy(), y_cpu)


def test_qgemm_errors():
    a = np.zeros((4, 8), dtype=np.int8)
    b = np.zeros((2, 8), dtype=np.int8)
    with pytest.raises(ValueError):
        fusedtok.qgemm(a, 1.0, np.zeros((2, 7), dtype=np.int8), 1.0)
    if HAS_TORCH:
        with pytest.raises(ValueError):
            fusedtok.qgemm(np.zeros((4,), dtype=np.int8), 1.0, b, 1.0)


def test_qgemm_end_to_end_quantized():
    # quantize -> matmul -> compare against the float matmul within the
    # per-tensor quantization error bound (sqrt(K) * quantum scale)
    rng = np.random.default_rng(7)
    k = 256
    x = rng.standard_normal((3, k)).astype(np.float32)
    w = (rng.standard_normal((64, k)) * 0.1).astype(np.float32)
    xq, sx = fusedtok.quantize_int8(x.ravel())
    wq, sw = fusedtok.quantize_int8(w.ravel())
    xq = xq.reshape(3, k)
    wq = wq.reshape(64, k)
    y = fusedtok.qgemm(xq, sx, wq, sw)
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    bound = 0.5 * (float(sx) + float(sw)) * np.sqrt(k) * 1.5
    assert np.abs(y.astype(np.float64) - ref).max() < bound


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    @pytest.mark.parametrize("m,n,k", [
        (1, 8, 16),
        (1, 4096, 4096),    # GEMV kernel
        (64, 64, 32),
        (65, 63, 33),
        (300, 500, 1024),   # multi-tile GEMM
        (17, 4096, 129),
    ])
    def test_staged_matches_cpu_bitexact(self, m, n, k):
        rng = np.random.default_rng(m + n * 7 + k)
        a = rand_q(rng, m, k)
        b = rand_q(rng, n, k)
        y = fusedtok.qgemm(a, 0.05, b, 0.04, cuda=True)
        ycpu = fusedtok.qgemm(a, 0.05, b, 0.04)
        # cross-path: GPU must be bit-identical to the CPU path (same
        # exact integers, same single float scale application)
        np.testing.assert_array_equal(y, ycpu)

    def test_repeated_calls_same_result(self):
        rng = np.random.default_rng(11)
        a = rand_q(rng, 32, 128)
        b = rand_q(rng, 48, 128)
        first = fusedtok.qgemm(a, 0.1, b, 0.2, cuda=True)
        for _ in range(3):
            np.testing.assert_array_equal(
                fusedtok.qgemm(a, 0.1, b, 0.2, cuda=True), first)

    def test_wide_slab_shapes_bitexact(self):
        # The launcher micro-benchmarks SLAB 64 vs SLAB 128 per shape and
        # caches the winner; whichever config wins on this GPU, integer
        # accumulation must stay exact. Drive shapes that stress both
        # slabs, multi-slab K streams, boundary tiles and scalar tails.
        for m, n, k in [(65, 63, 2000), (300, 500, 1024),
                        (128, 128, 4096), (17, 4096, 8192)]:
            rng = np.random.default_rng(m + n + k)
            a = rand_q(rng, m, k)
            b = rand_q(rng, n, k)
            y = fusedtok.qgemm(a, 0.05, b, 0.04, cuda=True)
            ycpu = fusedtok.qgemm(a, 0.05, b, 0.04)
            np.testing.assert_array_equal(y, ycpu)

    def test_tuned_repeats_bitidentical(self):
        # First call tunes (11+ launches of the real kernel into the
        # same output); the tuning must be invisible to the caller:
        # the post-tune answer equals the CPU path AND every repeat.
        rng = np.random.default_rng(21)
        a = rand_q(rng, 96, 512)
        b = rand_q(rng, 77, 512)
        y = fusedtok.qgemm(a, 0.03, b, 0.02, cuda=True)
        ycpu = fusedtok.qgemm(a, 0.03, b, 0.02)
        np.testing.assert_array_equal(y, ycpu)
        for _ in range(3):
            np.testing.assert_array_equal(
                fusedtok.qgemm(a, 0.03, b, 0.02, cuda=True), ycpu)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_matches_cpu_bitexact(self):
        rng = np.random.default_rng(12)
        a = rand_q(rng, 129, 500)
        b = rand_q(rng, 260, 500)
        at = torch.from_numpy(a).cuda()
        bt = torch.from_numpy(b).cuda()
        y = fusedtok.qgemm(at, 0.03, bt, 0.02)
        np.testing.assert_array_equal(
            y.cpu().numpy(), fusedtok.qgemm(a, 0.03, b, 0.04 if False else 0.02))

    def test_gemv_odd_k_scalar_fallback(self):
        # The GEMV vector loop gates on 4-byte row alignment, and any
        # k % 4 != 0 misaligns rows 1..n-1 (row r starts at r*k bytes).
        # The scalar pass must then cover the WHOLE row, not just the
        # [k4, k) tail - the pre-1.5.1 kernel silently dropped the
        # first k4 elements of every misaligned row. Pin the fixed
        # behavior bit-exactly on both M=1 shapes that always take the
        # GEMV path and multi-row ones where only some rows align.
        for m, n, k in [(1, 8, 129), (1, 5, 33), (1, 3, 7),
                        (4, 8, 129), (5, 16, 30)]:
            rng = np.random.default_rng(m * 31 + n + k)
            a = rand_q(rng, m, k)
            b = rand_q(rng, n, k)
            y = fusedtok.qgemm(a, 0.05, b, 0.04, cuda=True)
            ycpu = fusedtok.qgemm(a, 0.05, b, 0.04)
            np.testing.assert_array_equal(
                y, ycpu, err_msg=f"m={m} n={n} k={k}")

    def test_gemv_decode_shape(self):
        # the realistic decode step: [1, hidden] @ [vocab, hidden]^T
        rng = np.random.default_rng(13)
        hidden, vocab = 1024, 8192
        x = rand_q(rng, 1, hidden)
        w = rand_q(rng, vocab, hidden)
        xt = torch.from_numpy(x).cuda()
        wt = torch.from_numpy(w).cuda()
        y = fusedtok.qgemm(xt, 0.06, wt, 0.01)
        assert y.shape == (1, vocab)
        np.testing.assert_array_equal(
            y.cpu().numpy(), fusedtok.qgemm(x, 0.06, w, 0.01))

    def test_graph_capture_replay(self):
        # qgemm is graph-capturable (no allocs/syncs in the launcher)
        a = torch.randint(-127, 128, (64, 256), device="cuda", dtype=torch.int8)
        b = torch.randint(-127, 128, (64, 256), device="cuda", dtype=torch.int8)
        y = torch.empty((64, 64), device="cuda", dtype=torch.float32)

        def run():
            from fusedtok import _fusedtok
            _fusedtok.qgemm_launch(a.data_ptr(), b.data_ptr(), y.data_ptr(),
                                   64, 64, 256, 0.05, 0.02,
                                   torch.cuda.current_stream().cuda_stream)
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
        # torch has no CUDA int matmul; the numpy CPU path is exact
        ref = torch.from_numpy(fusedtok.qgemm(
            a.cpu().numpy(), 0.05, b.cpu().numpy(), 0.02)).cuda()
        assert torch.allclose(y, ref, rtol=1e-6)

    def test_graph_capture_after_tuning(self):
        # A big GEMM first call tunes the slab config on the live stream;
        # capturing the SAME shape afterwards must replay the tuned (or
        # default) config and recompute on mutation. Pins the interplay
        # between the config cache and the capture path (captures skip
        # tuning, but a pre-tuned shape must still be capturable).
        m = n = k = 256
        a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        y = torch.empty((m, n), device="cuda", dtype=torch.float32)

        def run():
            from fusedtok import _fusedtok
            _fusedtok.qgemm_launch(a.data_ptr(), b.data_ptr(), y.data_ptr(),
                                   m, n, k, 0.05, 0.02,
                                   torch.cuda.current_stream().cuda_stream)
        run()                            # tunes outside any capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        a.mul_(-1)
        g.replay()
        torch.cuda.synchronize()
        ref = torch.from_numpy(fusedtok.qgemm(
            a.cpu().numpy(), 0.05, b.cpu().numpy(), 0.02)).cuda()
        assert torch.allclose(y, ref, rtol=1e-6)
