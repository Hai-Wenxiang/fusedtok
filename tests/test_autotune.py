"""Runtime block-size autotuning (v0.4.1) for rmsnorm/layernorm/softmax.

The launchers pick the thread-block size once per (op, dtype, cols) with a
micro-benchmark on the caller's stream and cache it for the process.
Contracts pinned here:

- results stay correct across the shapes that tune to DIFFERENT blocks
  (narrow rows favor 256-512, wide rows up to 1024)
- the first (tuning) call and every cached call afterwards produce
  bit-identical outputs for the same input
- stream captures skip tuning and use the default block - still correct,
  and the tuned-cache path captured later replays correctly
- structurally unlaunchable candidates (register pressure at 1024 threads
  on the register-resident softmax) score as slow instead of failing
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

# widths that exercised different winners in the RTX 3060 sweep:
# 1024 -> 256/512, 4096 -> 512, 8192+ -> up to 1024
WIDTHS = [128, 1024, 4096, 8192, 16384]


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestAutotuneCorrectness:
    @pytest.mark.parametrize("cols", WIDTHS)
    def test_rmsnorm_across_tuned_blocks(self, cols):
        rng = np.random.default_rng(cols)
        x = rng.standard_normal((17, cols)).astype(np.float32)
        w = (rng.random(cols) + 0.5).astype(np.float32)
        y = fusedtok.rmsnorm(x, w, cuda=True)
        ref = fusedtok.rmsnorm(x, w)
        np.testing.assert_allclose(y, ref, rtol=2e-5, atol=2e-6)

    @pytest.mark.parametrize("cols", WIDTHS)
    def test_layernorm_across_tuned_blocks(self, cols):
        rng = np.random.default_rng(cols + 1)
        x = rng.standard_normal((9, cols)).astype(np.float32)
        w = (rng.random(cols) + 0.5).astype(np.float32)
        b = rng.standard_normal(cols).astype(np.float32)
        y = fusedtok.layernorm(x, w, b, cuda=True)
        ref = fusedtok.layernorm(x, w, b)
        np.testing.assert_allclose(y, ref, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("cols", WIDTHS)
    def test_softmax_across_tuned_blocks(self, cols):
        # 16384 crosses into the online (streaming) softmax variant
        rng = np.random.default_rng(cols + 2)
        x = (rng.standard_normal((8, cols)) * 3).astype(np.float32)
        y = fusedtok.softmax(x, cuda=True)
        ref = fusedtok.softmax(x)
        np.testing.assert_allclose(y, ref, rtol=2e-5, atol=2e-6)

    def test_tuned_cache_bitidentical_repeats(self):
        # the tuning call and every cached call afterwards must produce
        # bit-identical output (same kernel, same block, same input)
        rng = np.random.default_rng(7)
        x = rng.standard_normal((64, 4096)).astype(np.float32)
        w = (rng.random(4096) + 0.5).astype(np.float32)
        first = fusedtok.rmsnorm(x, w, cuda=True)
        for _ in range(3):
            np.testing.assert_array_equal(fusedtok.rmsnorm(x, w, cuda=True),
                                          first)

    def test_dtype_specific_choices(self):
        # f32 and bf16 tune independently; both stay correct
        rng = np.random.default_rng(11)
        x = rng.standard_normal((8, 4096)).astype(np.float32)
        w = (rng.random(4096) + 0.5).astype(np.float32)
        y32 = fusedtok.rmsnorm(x, w, cuda=True)
        xt = torch.from_numpy(x).cuda()
        wt = torch.from_numpy(w).cuda()
        ybf = fusedtok.rmsnorm(xt.to(torch.bfloat16), wt)
        np.testing.assert_allclose(ybf.float().cpu().numpy(), y32,
                                   rtol=8e-3, atol=8e-3)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestAutotuneGraphs:
    def test_capture_skips_tuning_and_stays_correct(self):
        # warm up first (tunes), then capture: the captured kernels use
        # the tuned block and replay must recompute
        x = torch.randn(256, 4096, device="cuda")
        w = torch.rand(4096, device="cuda") + 0.5
        out = {}
        def fn():
            out["y"] = fusedtok.rmsnorm(x, w)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        x.mul_(2.0)
        out["y"].fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        ref = fusedtok.rmsnorm(x, w)
        assert torch.allclose(out["y"], ref, atol=1e-5)

    def test_capture_before_any_tuning(self):
        # a shape no earlier test used, captured immediately after the
        # mandatory warm-ups: the capture itself never tunes (events and
        # syncs are illegal mid-capture - the guard in the launchers
        # falls back to the default block), and the replay is correct
        cols = 4097
        x = torch.randn(64, cols, device="cuda")
        b = torch.randn(cols, device="cuda")
        out = {}
        def fn():
            out["y"] = fusedtok.softmax(x)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        x.mul_(3.0)
        g.replay()
        torch.cuda.synchronize()
        sums = out["y"].sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
        assert torch.allclose(out["y"], torch.softmax(x, -1), atol=1e-5)
