"""CUDA graph capture compatibility of the zero-copy launchers.

The _launch entries must be capturable: no cudaMalloc/cudaFree, no host
syncs, no blocking readbacks inside capture. This requires launching on
the CALLER'S stream (the python layer passes
torch.cuda.current_stream().cuda_stream) - kernels launched on the legacy
default stream invalidate the capture and silently produce an empty
graph. The replay assertions below mutate the input between replays: a
vacuously-captured (empty) graph would keep returning the warm-up
result. sample_topp is NOT capturable by design (it returns a host int
and widens its window based on results) - documented limitation.
"""

import pytest

import fusedtok

try:
    import torch
    HAS_TORCH = True
except ImportError:          # torch is optional; CI runs without it
    torch = None
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and fusedtok.cuda_available()),
    reason="no torch / no GPU")

HAS_GRAPH = HAS_TORCH and hasattr(torch.cuda, "CUDAGraph")


def _capture(fn):
    """Warm up on a side stream (required), then capture on the current
    stream and return the graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


@pytest.mark.skipif(not HAS_GRAPH, reason="torch.cuda.CUDAGraph unavailable")
class TestGraphCapture:
    def test_elementwise_capture_replay(self):
        x = torch.randn(1024, 512, device="cuda")
        out = {}
        def fn():
            out["y"] = fusedtok.silu(x)
        g = _capture(fn)
        x.mul_(2.0)                      # replay must recompute
        ref = torch.nn.functional.silu(x)
        out["y"].fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out["y"], ref, atol=1e-5)

    def test_norm_capture_replay(self):
        x = torch.randn(256, 1024, device="cuda")
        r = torch.randn(256, 1024, device="cuda")
        w = torch.rand(1024, device="cuda") + 0.5
        out = {}
        def fn():
            out["y"] = fusedtok.rmsnorm(x, w, residual=r)
        g = _capture(fn)
        x.mul_(2.0)                      # replay must recompute
        v = x + r
        ref = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True)) * w
        out["y"].fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out["y"], ref, atol=1e-4)

    def test_softmax_capture_replay(self):
        x = torch.randn(128, 4096, device="cuda")
        out = {}
        def fn():
            out["y"] = fusedtok.softmax(x)
        g = _capture(fn)
        x.mul_(3.0)
        ref = torch.softmax(x, dim=-1)
        out["y"].fill_(0.0)
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out["y"], ref, atol=1e-5)
        sums = out["y"].sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_topk_capture_replay(self):
        x = torch.randn(50_000, device="cuda")
        vals = torch.empty(10, device="cuda", dtype=torch.float32)
        idxs = torch.empty(10, device="cuda", dtype=torch.int64)

        def run():
            from fusedtok import _fusedtok
            _fusedtok.topk_launch(x.data_ptr(), vals.data_ptr(),
                                  idxs.data_ptr(), x.numel(), 10,
                                  torch.cuda.current_stream().cuda_stream)
        g = _capture(run)
        ref1 = torch.topk(x, 10).values.clone()
        g.replay(); torch.cuda.synchronize()
        assert vals.cpu().numpy() == pytest.approx(ref1.cpu().numpy(), abs=1e-5)
        x.mul_(10.0)                     # replay must recompute
        ref2 = torch.topk(x, 10).values.clone()
        vals.fill_(float("nan"))
        g.replay(); torch.cuda.synchronize()
        assert vals.cpu().numpy() == pytest.approx(ref2.cpu().numpy(), abs=1e-4)

    def test_topp_capture_replay(self):
        # full big path: rounds + emit + chunk sort + merge ladder +
        # decode + nucleus count, all in one graph
        p = torch.rand(20_000, device="cuda")
        p = p / p.sum()
        vals = torch.empty_like(p)
        idxs = torch.empty(20_000, device="cuda", dtype=torch.int64)
        cnt = torch.empty(1, device="cuda", dtype=torch.int32)

        def run():
            from fusedtok import _fusedtok
            _fusedtok.topp_select_launch(p.data_ptr(), vals.data_ptr(),
                                         idxs.data_ptr(), p.numel(), 0.9,
                                         cnt.data_ptr(),
                                         torch.cuda.current_stream().cuda_stream)
        g = _capture(run)
        g.replay(); torch.cuda.synchronize()
        c = int(cnt.item())
        assert 1 <= c <= p.numel()
        cum = vals[:c].cumsum(0)
        assert cum[-1].item() >= 0.9 - 1e-4
        assert (vals.diff() <= 1e-6).all().item()

    def test_argmax_capture_replay(self):
        # argmax is a single self-resetting kernel (v1.2): replays must
        # stay correct because each replay's finalize re-zeros the
        # workspace slots (the pre-1.2 version replayed a memset node
        # alongside the kernel instead)
        x = torch.randn(100_000, device="cuda")
        out = torch.empty(1, dtype=torch.int32, device="cuda")

        def run():
            from fusedtok import _fusedtok
            _fusedtok.argmax_launch(x.data_ptr(), out.data_ptr(),
                                    x.numel(),
                                    torch.cuda.current_stream().cuda_stream)
        g = _capture(run)
        g.replay(); torch.cuda.synchronize()
        assert int(out.item()) == int(x.argmax())
        x[7] = 1e6                       # move the argmax to a known index
        g.replay(); torch.cuda.synchronize()
        assert int(out.item()) == 7
        g.replay(); torch.cuda.synchronize()   # replay again: slots reset
        assert int(out.item()) == 7

    def test_rope_capture_replay(self):
        q = torch.randn(64, 128, device="cuda")
        out = {}
        def fn():
            out["qr"], _ = fusedtok.rope(q, None, neox=True, pos_offset=7)
        g = _capture(fn)
        q.mul_(2.0)
        ref, _ = fusedtok.rope(q, None, neox=True, pos_offset=7)
        out["qr"].fill_(0.0)
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out["qr"], ref, atol=1e-5)
