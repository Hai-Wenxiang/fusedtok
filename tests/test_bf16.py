"""bf16 support: torch tensors in / torch tensors out on the zero-copy
path, with float32 references (bf16 has ~3 decimal digits; tolerances
account for the 8-bit mantissa)."""

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


class TestBf16:
    def _pair(self, *shape, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        x32 = torch.randn(*shape, generator=g).cuda()
        return x32, x32.to(torch.bfloat16)

    def test_rmsnorm_matches_f32_within_bf16_quantum(self):
        x32, x16 = self._pair(64, 1024)
        w = torch.rand(1024, device="cuda") + 0.5
        r16 = torch.randn(64, 1024, device="cuda").to(torch.bfloat16)
        y32 = fusedtok.rmsnorm(x32, w)
        y16 = fusedtok.rmsnorm(x16, w, residual=r16)
        assert y16.dtype is torch.bfloat16
        # residual path differs from plain path by design; compare against
        # torch reference instead
        v = (x16.float() + r16.float())
        ref = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True)) * w
        assert torch.allclose(y16.float(), ref, rtol=2e-2, atol=2e-2)

    def test_layernorm_bf16(self):
        x32, x16 = self._pair(32, 768, seed=1)
        w = torch.rand(768, device="cuda") + 0.5
        b = torch.zeros(768, device="cuda")
        ref = torch.nn.functional.layer_norm(x16.float(), (768,), w, b)
        y = fusedtok.layernorm(x16, w, b)
        assert y.dtype is torch.bfloat16
        assert torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2)

    def test_softmax_bf16_rows_sum_to_one(self):
        _, x16 = self._pair(16, 4096, seed=2)
        y = fusedtok.softmax(x16)
        assert y.dtype is torch.bfloat16
        sums = y.float().sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=5e-3)

    def test_elementwise_ops_bf16(self):
        for op, torch_ref in [
            ("silu", lambda v: torch.nn.functional.silu(v)),
            ("gelu", lambda v: torch.nn.functional.gelu(v)),
            ("relu", lambda v: torch.relu(v)),
            ("tanh", lambda v: torch.tanh(v)),
            ("sigmoid", lambda v: torch.sigmoid(v)),
        ]:
            x32, x16 = self._pair(256, 512, seed=3)
            y = getattr(fusedtok, op)(x16)
            assert y.dtype is torch.bfloat16, op
            assert torch.allclose(y.float(), torch_ref(x16.float()),
                                  rtol=2e-2, atol=2e-2), op

    def test_binary_ops_bf16(self):
        x32, x16 = self._pair(128, 333, seed=4)
        _, z16 = self._pair(128, 333, seed=5)
        assert fusedtok.add(x16, z16).dtype is torch.bfloat16
        assert torch.allclose(fusedtok.add(x16, z16).float(),
                              x16.float() + z16.float(), rtol=2e-2, atol=2e-2)
        assert torch.allclose(fusedtok.mul(x16, z16).float(),
                              x16.float() * z16.float(), rtol=2e-2, atol=2e-2)
        sw = fusedtok.swiglu(x16, z16)
        ref = torch.nn.functional.silu(x16.float()) * z16.float()
        assert torch.allclose(sw.float(), ref, rtol=2e-2, atol=2e-2)

    def test_rope_bf16_matches_f32_layout(self):
        q32 = torch.randn(4, 128, device="cuda")
        q16 = q32.to(torch.bfloat16)
        for neox in (False, True):
            r32, _ = fusedtok.rope(q32, None, neox=neox, pos_offset=5)
            r16, _ = fusedtok.rope(q16, None, neox=neox, pos_offset=5)
            assert r16.dtype is torch.bfloat16
            assert torch.allclose(r16.float(), r32, rtol=2e-2, atol=2e-2), neox

    def test_mixed_dtype_rejected(self):
        x16 = torch.randn(8, 64, device="cuda").to(torch.bfloat16)
        z32 = torch.randn(8, 64, device="cuda")
        with pytest.raises(TypeError):
            fusedtok.add(x16, z32)
        q16 = torch.randn(4, 64, device="cuda").to(torch.bfloat16)
        k32 = torch.randn(4, 64, device="cuda")
        with pytest.raises(TypeError):
            fusedtok.rope(q16, k32)

    def test_unsupported_dtype_rejected(self):
        xh = torch.randn(8, 64, device="cuda", dtype=torch.float64)
        with pytest.raises(TypeError):
            fusedtok.silu(xh)

    def test_bf16x4_tail_and_unaligned(self):
        # v0.3 elementwise bf16 vectorizes 4-per-thread when 8B aligned;
        # sizes not divisible by 4 take the scalar tail, and storage-offset
        # views (odd element offset -> odd byte offset) fall back to the
        # scalar kernel. All paths must agree with the f32 reference.
        base = torch.randn(4, 1027, device="cuda")   # 1027 = 4k+3 tail
        x16 = base.to(torch.bfloat16)
        ref = torch.nn.functional.silu(base)
        # aligned whole tensor (vector + tail)
        y = fusedtok.silu(x16)
        assert torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2)
        # unaligned view: storage_offset 1 -> 2-byte misalignment
        v16 = x16.flatten()[1:].view(4, 1026)
        r16 = fusedtok.silu(v16)
        assert torch.allclose(r16.float(),
                              torch.nn.functional.silu(base.flatten()[1:].view(4, 1026)),
                              rtol=2e-2, atol=2e-2)
        # binary op with tail
        z16 = torch.randn(4, 1027, device="cuda").to(torch.bfloat16)
        a = fusedtok.add(x16, z16)
        assert torch.allclose(a.float(), x16.float() + z16.float(),
                              rtol=2e-2, atol=2e-2)

    def test_bf16_weight_upcasted_for_norms(self):
        # norm weights may arrive as bf16; the layer upcasts the [cols]
        # vector (small copy) and stays correct
        x16 = torch.randn(16, 256, device="cuda").to(torch.bfloat16)
        w16 = torch.rand(256, device="cuda").to(torch.bfloat16)
        w32 = w16.float()
        v = x16.float()
        ref = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True)) * w32
        y = fusedtok.rmsnorm(x16, w16)
        assert torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2)
