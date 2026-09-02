"""kv_append (v1.3): the cache-write side of the CONTIGUOUS decode loop.

One fresh token's k/v rows per sequence are scattered into the
[B, Hkv, T, D] caches at row lens[b] - the contiguous twin of
kv_append_paged. Cases:

- exact-position writes (varied per-sequence lengths, first and last
  cache row)
- repeated appends building a cache that then decodes identically to a
  one-shot cache (the end-to-end contiguous loop, CPU and GPU)
- CPU / staged / torch-path parity and the f32/bf16/fp16 dtype matrix
- graph capture of the append launch
- the in-place contract (float32 host arrays) and the error contract
  (lens out of range host-side, dtype mismatch, device-lens trust)
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

needs_gpu = pytest.mark.skipif(
    not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")


def _host_append(k_cache, v_cache, k_new, v_new, lens):
    b, hkv, d = k_new.shape
    for bi in range(b):
        pos = int(lens[bi])
        k_cache[bi, :, pos] = k_new[bi]
        v_cache[bi, :, pos] = v_new[bi]


def test_append_positions_cpu():
    rng = np.random.default_rng(70)
    b, hkv, t, d = 3, 2, 8, 16
    kc = np.zeros((b, hkv, t, d), dtype=np.float32)
    vc = np.zeros_like(kc)
    # per-sequence write positions, including 0 and the last row
    lens = np.array([0, t - 1, 3], dtype=np.int32)
    kn = rng.standard_normal((b, hkv, d)).astype(np.float32)
    vn = rng.standard_normal((b, hkv, d)).astype(np.float32)
    fusedtok.kv_append(kc, vc, kn, vn, lens)
    rk, rv = np.zeros_like(kc), np.zeros_like(vc)
    _host_append(rk, rv, kn, vn, lens)
    np.testing.assert_array_equal(kc, rk)
    np.testing.assert_array_equal(vc, rv)


def test_append_loop_builds_working_cache_cpu():
    # append step by step, then decode and compare against a decode over
    # a one-shot cache holding the same tokens (the end-to-end loop)
    rng = np.random.default_rng(71)
    b, hkv, hq, d, steps = 2, 2, 4, 16, 9
    t = steps + 4                       # slack rows must stay zero
    kc = np.zeros((b, hkv, t, d), dtype=np.float32)
    vc = np.zeros_like(kc)
    ks = rng.standard_normal((steps, b, hkv, d)).astype(np.float32)
    vs = rng.standard_normal((steps, b, hkv, d)).astype(np.float32)
    lens = np.zeros(b, dtype=np.int32)
    for s in range(steps):
        fusedtok.kv_append(kc, vc, ks[s], vs[s], lens)
        lens += 1
    q = rng.standard_normal((b, hq, d)).astype(np.float32)
    out = fusedtok.attention_decode(q, kc, vc, lens)
    one_shot_k = np.transpose(ks, (1, 2, 0, 3)).copy()
    one_shot_v = np.transpose(vs, (1, 2, 0, 3)).copy()
    ref = fusedtok.attention_decode(q, one_shot_k, one_shot_v)
    np.testing.assert_allclose(out, ref, atol=1e-5)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="staged needs a GPU")
def test_staged_matches_cpu():
    rng = np.random.default_rng(72)
    b, hkv, t, d = 2, 3, 6, 8
    kc = rng.standard_normal((b, hkv, t, d)).astype(np.float32)
    vc = rng.standard_normal((b, hkv, t, d)).astype(np.float32)
    kn = rng.standard_normal((b, hkv, d)).astype(np.float32)
    vn = rng.standard_normal((b, hkv, d)).astype(np.float32)
    lens = np.array([4, 0], dtype=np.int32)
    kc2, vc2 = kc.copy(), vc.copy()
    fusedtok.kv_append(kc, vc, kn, vn, lens)
    fusedtok.kv_append(kc2, vc2, kn, vn, lens, cuda=True)
    np.testing.assert_array_equal(kc, kc2)
    np.testing.assert_array_equal(vc, vc2)


def test_lens_bounds_and_shape_errors_cpu():
    kc = np.zeros((1, 2, 4, 8), dtype=np.float32)
    vc = np.zeros_like(kc)
    kn = np.zeros((1, 2, 8), dtype=np.float32)
    vn = np.zeros_like(kn)
    with pytest.raises(ValueError):
        fusedtok.kv_append(kc, vc, kn, vn, [4])     # == T: not a row
    with pytest.raises(ValueError):
        fusedtok.kv_append(kc, vc, kn, vn, [-1])
    with pytest.raises(ValueError):
        fusedtok.kv_append(kc, vc, kn, vn, [1, 1])  # wrong entry count
    with pytest.raises(TypeError):
        # in-place op: float64 caches would drop the writes on a view
        fusedtok.kv_append(kc.astype(np.float64),
                           vc.astype(np.float64), kn, vn, [0])


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu_and_dtype_matrix(self):
        rng = np.random.default_rng(73)
        b, hkv, t, d = 2, 2, 8, 16
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            kc = torch.zeros((b, hkv, t, d), dtype=dt, device="cuda")
            vc = torch.zeros_like(kc)
            kn = torch.randn(b, hkv, d, device="cuda").to(dt)
            vn = torch.randn(b, hkv, d, device="cuda").to(dt)
            lens = torch.tensor([5, 0], dtype=torch.int32, device="cuda")
            fusedtok.kv_append(kc, vc, kn, vn, lens)
            # host lists route through validation then upload
            kc2 = torch.zeros((b, hkv, t, d), dtype=dt, device="cuda")
            vc2 = torch.zeros_like(kc2)
            fusedtok.kv_append(kc2, vc2, kn, vn, [5, 0])
            assert torch.equal(kc, kc2)
            assert torch.equal(vc, vc2)
            # f32 spot-check against the host reference
            if dt is torch.float32:
                rk = kc.cpu().numpy()
                ref = np.zeros_like(rk)
                _host_append(ref, np.zeros_like(rk),
                             kn.cpu().numpy(), vn.cpu().numpy(),
                             np.array([5, 0]))
                np.testing.assert_array_equal(rk, ref)

    def test_append_then_decode_end_to_end_gpu(self):
        # build a cache append-by-append, decode, and match a decode over
        # a one-shot cache holding the SAME tokens (the end-to-end loop)
        rng = np.random.default_rng(74)
        b, hkv, hq, d, steps = 2, 4, 8, 64, 33
        t = 40
        kc = torch.zeros((b, hkv, t, d), device="cuda")
        vc = torch.zeros_like(kc)
        tokens_k, tokens_v = [], []
        lens = torch.zeros(b, dtype=torch.int32, device="cuda")
        for _ in range(steps):
            kn = torch.randn(b, hkv, d, device="cuda")
            vn = torch.randn(b, hkv, d, device="cuda")
            tokens_k.append(kn)
            tokens_v.append(vn)
            fusedtok.kv_append(kc, vc, kn, vn, lens)
            lens += 1
        q = torch.randn(b, hq, d, device="cuda")
        out = fusedtok.attention_decode(q, kc, vc, lens)
        one_shot_k = torch.stack(tokens_k, dim=2).contiguous()  # [b,hkv,S,d]
        one_shot_v = torch.stack(tokens_v, dim=2).contiguous()
        ref = fusedtok.attention_decode(q, one_shot_k, one_shot_v)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_graph_capture_and_replay(self):
        # dedicated capture test: replay after mutating k_new recomputes
        rng = np.random.default_rng(75)
        b, hkv, t, d = 1, 2, 8, 32
        kc = torch.zeros((b, hkv, t, d), device="cuda")
        vc = torch.zeros_like(kc)
        kn = torch.randn(b, hkv, d, device="cuda")
        vn = torch.randn(b, hkv, d, device="cuda")
        lens = torch.tensor([3], dtype=torch.int32, device="cuda")
        fusedtok.kv_append(kc, vc, kn, vn, lens)    # warm-up
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fusedtok.kv_append(kc, vc, kn, vn, lens)
        kc.zero_(), vc.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(kc[0, 0, 3], kn[0, 0])
        assert torch.equal(vc[0, 0, 3], vn[0, 0])
        # mutate the input, replay again: the cache follows
        kn2 = torch.randn(b, hkv, d, device="cuda")
        kn.copy_(kn2)
        kc.zero_(), vc.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(kc[0, 0, 3], kn2[0, 0])

    def test_error_contract_cuda(self):
        kc = torch.zeros((1, 2, 4, 8), device="cuda")
        vc = torch.zeros_like(kc)
        kn = torch.zeros((1, 2, 8), device="cuda")
        vn = torch.zeros_like(kn)
        with pytest.raises(ValueError):
            fusedtok.kv_append(kc, vc, kn, vn, [4])    # == T
        with pytest.raises(TypeError):
            fusedtok.kv_append(kc, vc, kn.to(torch.bfloat16), vn, [0])
        with pytest.raises(TypeError):
            fusedtok.kv_append(kc, vc, kn.cpu(), vn, [0])
