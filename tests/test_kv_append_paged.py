"""kv_append_paged (v1.2): the cache-write side of the paged decode loop.

One fresh token's k/v rows per sequence are scattered into the pool at
position lens[b] (block table[b, lens/P], offset lens%P). Cases:

- exact-position writes across page boundaries (including the very
  first token of a new block)
- repeated appends building a cache that then decodes identically to a
  contiguous cache (the end-to-end paged loop)
- CPU / staged / torch-path parity and the dtype matrix
- the in-place contract (float32 host arrays; torch CPU views) and the
  error contract
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

GPU_TORCH = HAS_TORCH and fusedtok.cuda_available()
PAGE = 16


def _pool_case(rng, b, hkv, d, nb, width, steps):
    """Zero pools + a table; simulate `steps` appends host-side."""
    table = rng.integers(0, nb, (b, width)).astype(np.int32)
    k_pool = np.zeros((nb, hkv, PAGE, d), dtype=np.float32)
    v_pool = np.zeros((nb, hkv, PAGE, d), dtype=np.float32)
    ks = rng.standard_normal((steps, b, hkv, d)).astype(np.float32)
    vs = rng.standard_normal((steps, b, hkv, d)).astype(np.float32)
    return table, k_pool, v_pool, ks, vs


def _host_append(k_pool, v_pool, table, k_new, v_new, lens):
    b, hkv, d = k_new.shape
    for bi in range(b):
        pos = int(lens[bi])
        blk = int(table[bi, pos // PAGE])
        k_pool[blk, :, pos % PAGE] = k_new[bi]
        v_pool[blk, :, pos % PAGE] = v_new[bi]


def test_append_positions_and_page_boundaries_cpu():
    rng = np.random.default_rng(60)
    table, kp, vp, ks, vs = _pool_case(rng, b=2, hkv=2, d=32, nb=6, width=3,
                                       steps=PAGE + 2)
    lens = np.zeros(2, dtype=np.int32)
    for step in range(PAGE + 2):
        fusedtok.kv_append_paged(kp, vp, table, ks[step], vs[step], lens)
        lens += 1                     # caller tracks lengths (scheduler)
    ref_k = np.zeros_like(kp)
    ref_v = np.zeros_like(vp)
    rl = np.zeros(2, dtype=np.int32)
    for step in range(PAGE + 2):
        _host_append(ref_k, ref_v, table, ks[step], vs[step], rl)
        rl += 1
    np.testing.assert_array_equal(kp, ref_k)
    np.testing.assert_array_equal(vp, ref_v)


def test_append_then_paged_decode_end_to_end_cpu():
    # the whole point: build a cache append-by-append, decode, and match
    # a contiguous reference built the same way
    rng = np.random.default_rng(61)
    b, hkv, hq, d, steps = 2, 2, 4, 32, 33        # crosses a page boundary
    width = (steps + PAGE - 1) // PAGE
    table, kp, vp, ks, vs = _pool_case(rng, b, hkv, d, nb=12, width=width,
                                       steps=steps)
    lens = np.zeros(b, dtype=np.int32)
    for s in range(steps):
        fusedtok.kv_append_paged(kp, vp, table, ks[s], vs[s], lens)
        lens += 1
    kc = np.zeros((b, hkv, width * PAGE, d), dtype=np.float32)
    vc = np.zeros_like(kc)
    for bi in range(b):
        for p in range(width):
            kc[bi, :, p * PAGE:(p + 1) * PAGE] = kp[table[bi, p]]
            vc[bi, :, p * PAGE:(p + 1) * PAGE] = vp[table[bi, p]]
    q = rng.standard_normal((b, hq, d)).astype(np.float32)
    out_p = fusedtok.attention_decode_paged(q, kp, vp, table,
                                            np.full(b, steps, np.int32))
    out_c = fusedtok.attention_decode(q, kc, vc,
                                      np.full(b, steps, np.int32))
    np.testing.assert_array_equal(out_p, out_c)


def test_append_staged_matches_cpu():
    if not fusedtok.cuda_available():
        pytest.skip("no GPU")
    rng = np.random.default_rng(62)
    table, kp, vp, ks, vs = _pool_case(rng, b=2, hkv=2, d=32, nb=6, width=2,
                                       steps=3)
    lens = np.array([5, PAGE], dtype=np.int32)     # 0-based write slots
    kp2, vp2 = kp.copy(), vp.copy()
    fusedtok.kv_append_paged(kp, vp, table, ks[0], vs[0], lens)
    fusedtok.kv_append_paged(kp2, vp2, table, ks[0], vs[0], lens,
                             cuda=True)
    np.testing.assert_array_equal(kp, kp2)
    np.testing.assert_array_equal(vp, vp2)


def test_append_errors_cpu():
    rng = np.random.default_rng(63)
    table, kp, vp, ks, vs = _pool_case(rng, b=1, hkv=2, d=32, nb=4, width=2,
                                       steps=1)
    with pytest.raises(ValueError):
        # write position beyond the table capacity
        fusedtok.kv_append_paged(kp, vp, table, ks[0], vs[0],
                                 np.array([2 * PAGE], dtype=np.int32))
    bad = table.copy()
    bad[0, 0] = 99
    with pytest.raises(ValueError):
        fusedtok.kv_append_paged(kp, vp, bad, ks[0], vs[0],
                                 np.array([0], dtype=np.int32))
    with pytest.raises(TypeError):
        # in-place op: float64 pools would drop the writes
        fusedtok.kv_append_paged(kp.astype(np.float64),
                                 vp.astype(np.float64), table,
                                 ks[0], vs[0], np.array([0], np.int32))


@pytest.mark.skipif(not GPU_TORCH, reason="no torch/GPU")
class TestAppendTorch:
    def test_append_torch_f32_inplace(self):
        rng = np.random.default_rng(64)
        table, kp, vp, ks, vs = _pool_case(rng, b=2, hkv=2, d=32, nb=6,
                                           width=2, steps=2)
        kpt = torch.from_numpy(kp).cuda()
        vpt = torch.from_numpy(vp).cuda()
        tblt = torch.from_numpy(table).cuda()
        lens = torch.tensor([PAGE - 1, 0], dtype=torch.int32,
                            device="cuda")          # boundary + fresh
        kn = torch.from_numpy(ks[0]).cuda()
        vn = torch.from_numpy(vs[0]).cuda()
        fusedtok.kv_append_paged(kpt, vpt, tblt, kn, vn, lens)
        torch.cuda.synchronize()
        kp_ref, vp_ref = kp.copy(), vp.copy()
        _host_append(kp_ref, vp_ref, table, ks[0], vs[0],
                     np.array([PAGE - 1, 0]))
        np.testing.assert_array_equal(kpt.cpu().numpy(), kp_ref)
        np.testing.assert_array_equal(vpt.cpu().numpy(), vp_ref)

    @pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16]
                             if HAS_TORCH else [])
    def test_append_half_dtype(self, dt):
        rng = np.random.default_rng(65)
        table, kp, vp, ks, vs = _pool_case(rng, b=1, hkv=2, d=32, nb=4,
                                           width=1, steps=1)
        kpt = torch.from_numpy(kp).cuda().to(dt)
        vpt = torch.from_numpy(vp).cuda().to(dt)
        tblt = torch.from_numpy(table).cuda()
        lens = torch.tensor([3], dtype=torch.int32, device="cuda")
        kn = torch.from_numpy(ks[0]).cuda().to(dt)
        vn = torch.from_numpy(vs[0]).cuda().to(dt)
        fusedtok.kv_append_paged(kpt, vpt, tblt, kn, vn, lens)
        torch.cuda.synchronize()
        got = kpt.float().cpu().numpy()
        blk = int(table[0, 0])
        ref = kp.copy()
        ref[blk, :, 3] = ks[0][0]
        np.testing.assert_allclose(got, ref, atol=2e-2)

    def test_append_then_paged_decode_gpu_loop(self):
        # the realistic loop: append -> decode -> append ...
        rng = np.random.default_rng(66)
        b, hkv, hq, d, steps = 1, 2, 4, 64, 40
        width = (steps + PAGE - 1) // PAGE
        table, kp, vp, ks, vs = _pool_case(rng, b, hkv, d, nb=8,
                                           width=width, steps=steps)
        kpt = torch.from_numpy(kp).cuda()
        vpt = torch.from_numpy(vp).cuda()
        tblt = torch.from_numpy(table).cuda()
        q = torch.from_numpy(
            rng.standard_normal((b, hq, d)).astype(np.float32)).cuda()
        for s in range(steps):
            lens = torch.tensor([s], dtype=torch.int32, device="cuda")
            kn = torch.from_numpy(ks[s]).cuda()
            vn = torch.from_numpy(vs[s]).cuda()
            fusedtok.kv_append_paged(kpt, vpt, tblt, kn, vn, lens)
            out = fusedtok.attention_decode_paged(
                q, kpt, vpt, tblt,
                torch.tensor([s + 1], dtype=torch.int32, device="cuda"))
            torch.cuda.synchronize()
            assert out.shape == (b, hq, d)
            assert torch.isfinite(out).all().item()
