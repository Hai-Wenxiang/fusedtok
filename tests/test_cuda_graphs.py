"""CUDA graph capture compatibility of the zero-copy launchers.

The _launch entries must be capturable: no cudaMalloc/cudaFree, no host
syncs, no blocking readbacks inside capture. The selection ops satisfy
this via the process-cached workspace and async memsets. sample_topp is
NOT capturable by design (it returns a host int and widens its window
based on results) - documented limitation.
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


@pytest.mark.skipif(not HAS_GRAPH, reason="torch.cuda.CUDAGraph unavailable")
class TestGraphCapture:
    def test_elementwise_capture_replay(self):
        x = torch.randn(1024, 512, device="cuda")
        out = torch.empty_like(x)
        # warm-up in a side stream (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fusedtok.silu_launch = None  # noqa: F841 (sanity that pkg loaded)
                y = fusedtok.silu(x)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fusedtok.silu(x)
        out.copy_(torch.nn.functional.silu(x))
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(y, out, atol=1e-5)

    def test_norm_capture_replay(self):
        x = torch.randn(256, 1024, device="cuda")
        r = torch.randn(256, 1024, device="cuda")
        w = torch.rand(1024, device="cuda") + 0.5
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                y = fusedtok.rmsnorm(x, w, residual=r)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fusedtok.rmsnorm(x, w, residual=r)
        v = x + r
        ref = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True)) * w
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(y, ref, atol=1e-4)

    def test_softmax_capture_replay(self):
        x = torch.randn(128, 4096, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                y = fusedtok.softmax(x)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fusedtok.softmax(x)
        g.replay()
        torch.cuda.synchronize()
        sums = y.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_topk_capture_replay(self):
        # selection ops use async workspace memsets - capturable
        x = torch.randn(50_000, device="cuda")
        vals = torch.empty(10, device="cuda", dtype=torch.float32)
        idxs = torch.empty(10, device="cuda", dtype=torch.int64)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _fusedtok_topk(x, vals, idxs)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _fusedtok_topk(x, vals, idxs)
        g.replay()
        torch.cuda.synchronize()
        assert (vals.cpu().numpy() ==
                pytest.approx(torch.topk(x, 10).values.cpu().numpy(), abs=1e-5))

    def test_rope_capture_replay(self):
        q = torch.randn(64, 128, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                qr, _ = fusedtok.rope(q, None, neox=True, pos_offset=7)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            qr, _ = fusedtok.rope(q, None, neox=True, pos_offset=7)
        ref, _ = fusedtok.rope(q, None, neox=True, pos_offset=7)
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(qr, ref, atol=1e-5)


def _fusedtok_topk(x, vals, idxs):
    from fusedtok import _fusedtok
    _fusedtok.topk_launch(x.data_ptr(), vals.data_ptr(), idxs.data_ptr(),
                          x.numel(), 10)
