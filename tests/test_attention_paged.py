"""Paged kv-cache decode attention (v1.2).

The paged op walks the same online-softmax split pipeline as the
contiguous one, but every token row is addressed through a block table:
row(t) = pool[table[b, t // P], kv, t % P, :]. These cases pin the
behaviors that are easiest to break:

- parity against the contiguous op when the pools are materialized into
  a contiguous cache (the paging indirection must change nothing but the
  address math)
- the GQA mapping through the constant-V probe
- partial last pages, exact page multiples, single pages, empty rows
- non-monotonic block tables (the indirection itself)
- the dtype matrix (f32 exact vs contiguous; bf16/fp16 vs float64 on the
  half-rounded inputs)
- CUDA-graph capture with replay-recompute
- the error contract, including CPU-side block-table value validation
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
# resolved at import time for parametrize; empty without torch so CI
# (no torch, no GPU) collects zero cases instead of raising (the 1.1-A
# HALF_DTYPES lesson)
HALF_DTYPES = ([(torch.bfloat16, 2e-2), (torch.float16, 5e-3)]
               if HAS_TORCH else [])


def _make_case(rng, b, hq, hkv, d, nb, lens):
    """Pools + a random (non-monotonic) block table + the contiguous
    cache the table materializes to."""
    nseq = len(lens)
    width = max((l + PAGE - 1) // PAGE for l in lens) if nseq else 1
    k_pool = rng.standard_normal((nb, hkv, PAGE, d)).astype(np.float32)
    v_pool = rng.standard_normal((nb, hkv, PAGE, d)).astype(np.float32)
    # fully random (non-monotonic, possibly repeating) block ids: the
    # paging indirection must handle any valid table
    table = rng.integers(0, nb, (b, width)).astype(np.int32)
    t_rows = width * PAGE
    k_cache = np.zeros((b, hkv, t_rows, d), dtype=np.float32)
    v_cache = np.zeros((b, hkv, t_rows, d), dtype=np.float32)
    for bi in range(b):
        for s in range(width):
            k_cache[bi, :, s * PAGE:(s + 1) * PAGE] = k_pool[table[bi, s]]
            v_cache[bi, :, s * PAGE:(s + 1) * PAGE] = v_pool[table[bi, s]]
    q = rng.standard_normal((b, hq, d)).astype(np.float32)
    lens_arr = np.asarray(lens, dtype=np.int32)
    return q, k_pool, v_pool, table, k_cache, v_cache, lens_arr


def test_paged_matches_contiguous_cpu():
    rng = np.random.default_rng(41)
    q, kp, vp, tbl, kc, vc, lens = _make_case(
        rng, b=3, hq=8, hkv=2, d=64, nb=20, lens=[37, 16, 48])
    out_p = fusedtok.attention_decode_paged(q, kp, vp, tbl, lens)
    out_c = fusedtok.attention_decode(q, kc, vc, lens)
    # both are float32 CPU two-pass walks with the SAME row order, so
    # the paged gather must be bit-exact against the materialized cache
    np.testing.assert_array_equal(out_p, out_c)


def test_paged_partial_page_and_zero_len_cpu():
    rng = np.random.default_rng(42)
    # 37 = 2 pages + 5 tokens (partial last page), 16 = exactly one page,
    # 0 = empty row
    q, kp, vp, tbl, kc, vc, lens = _make_case(
        rng, b=3, hq=4, hkv=4, d=32, nb=8, lens=[37, 16, 0])
    out_p = fusedtok.attention_decode_paged(q, kp, vp, tbl, lens)
    out_c = fusedtok.attention_decode(q, kc, vc, lens)
    np.testing.assert_array_equal(out_p, out_c)
    assert np.all(out_p[2] == 0.0)          # empty row convention


def test_paged_single_token_and_single_page_cpu():
    rng = np.random.default_rng(43)
    q, kp, vp, tbl, kc, vc, lens = _make_case(
        rng, b=2, hq=2, hkv=2, d=128, nb=4, lens=[1, PAGE])
    out_p = fusedtok.attention_decode_paged(q, kp, vp, tbl, lens)
    out_c = fusedtok.attention_decode(q, kc, vc, lens)
    np.testing.assert_array_equal(out_p, out_c)
    # one token: out == v of that token exactly
    blk = tbl[0, 0]
    np.testing.assert_allclose(out_p[0, 0], vp[blk, 0, 0], atol=1e-6)


def test_paged_constant_v_gqa_probe_cpu():
    # every kv head carries a constant V: each q head's output must be
    # exactly its OWN kv head's constant (the GQA map h -> h // group)
    rng = np.random.default_rng(44)
    b, hq, hkv, d, nb, width = 2, 8, 2, 32, 6, 2
    k_pool = rng.standard_normal((nb, hkv, PAGE, d)).astype(np.float32)
    v_pool = np.zeros((nb, hkv, PAGE, d), dtype=np.float32)
    for kv in range(hkv):
        v_pool[:, kv] = (kv + 1) * 1.0
    tbl = rng.integers(0, nb, (b, width)).astype(np.int32)
    q = rng.standard_normal((b, hq, d)).astype(np.float32)
    lens = np.full(b, width * PAGE, dtype=np.int32)
    out = fusedtok.attention_decode_paged(q, k_pool, v_pool, tbl, lens)
    group = hq // hkv
    for h in range(hq):
        assert np.allclose(out[:, h], (h // group) + 1.0, atol=1e-5)


def test_paged_len_none_uses_full_table_width():
    rng = np.random.default_rng(45)
    width = 3
    lens = [width * PAGE] * 2
    q, kp, vp, tbl, kc, vc, _ = _make_case(
        rng, b=2, hq=4, hkv=2, d=32, nb=10, lens=lens)
    out_none = fusedtok.attention_decode_paged(q, kp, vp, tbl)
    out_lens = fusedtok.attention_decode_paged(q, kp, vp, tbl,
                                               np.asarray(lens, np.int32))
    np.testing.assert_array_equal(out_none, out_lens)


def test_paged_errors_cpu():
    rng = np.random.default_rng(46)
    q, kp, vp, tbl, kc, vc, lens = _make_case(
        rng, b=2, hq=4, hkv=2, d=32, nb=8, lens=[16, 32])
    with pytest.raises(ValueError):
        fusedtok.attention_decode_paged(q, kp, vp, tbl,
                                        np.array([2 * PAGE + 1, 3],
                                                 dtype=np.int32))
    bad = tbl.copy()
    bad[0, 0] = 999                          # out-of-range block id
    with pytest.raises(ValueError):
        fusedtok.attention_decode_paged(q, kp, vp, bad, lens)
    if fusedtok.cuda_available():
        # the staged GPU path validates host-visible table values too
        with pytest.raises(ValueError):
            fusedtok.attention_decode_paged(q, kp, vp, bad, lens, cuda=True)
    with pytest.raises(ValueError):
        # GQA group 3 is not in {1,2,4,8,16}
        q3 = rng.standard_normal((1, 3, 32)).astype(np.float32)
        fusedtok.attention_decode_paged(q3, kp, vp, tbl[:1, :1],
                                        np.array([16], dtype=np.int32))
    with pytest.raises(ValueError):
        fusedtok.attention_decode_paged(q, kp, vp, tbl[:1], lens)  # table rows
    with pytest.raises(ValueError):
        fusedtok.attention_decode_paged(q[:, :, :16], kp, vp, tbl, lens)


@pytest.mark.skipif(not GPU_TORCH, reason="no torch/GPU")
class TestPagedTorch:
    def _case(self, seed, dt=None):
        if dt is None:
            dt = torch.float32
        rng = np.random.default_rng(seed)
        q, kp, vp, tbl, kc, vc, lens = _make_case(
            rng, b=3, hq=8, hkv=2, d=64, nb=20, lens=[511, 100, 1024])
        qt = torch.from_numpy(q).cuda().to(dt)
        kpt = torch.from_numpy(kp).cuda().to(dt)
        vpt = torch.from_numpy(vp).cuda().to(dt)
        tblt = torch.from_numpy(tbl).cuda()
        lenst = torch.from_numpy(lens).cuda()
        kct = torch.from_numpy(kc).cuda().to(dt)
        vct = torch.from_numpy(vc).cuda().to(dt)
        return qt, kpt, vpt, tblt, lenst, kct, vct

    def test_paged_vs_contiguous_gpu_f32(self):
        qt, kpt, vpt, tblt, lenst, kct, vct = self._case(47)
        out = fusedtok.attention_decode_paged(qt, kpt, vpt, tblt, lenst)
        ref = fusedtok.attention_decode(qt, kct, vct, lenst)
        torch.cuda.synchronize()
        assert out.dtype is torch.float32
        # split boundaries can differ between the two launchers; both
        # are float32 online-softmax walks of the same rows
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dt,tol", HALF_DTYPES)
    def test_paged_half_precision_parity(self, dt, tol):
        qt, kpt, vpt, tblt, lenst, kct, vct = self._case(48, dt)
        out = fusedtok.attention_decode_paged(qt, kpt, vpt, tblt, lenst)
        torch.cuda.synchronize()
        assert out.dtype is dt
        # reference: float64 math on the half-rounded inputs
        q64 = qt.double().cpu().numpy()
        k64 = kpt.double().cpu().numpy()
        v64 = vpt.double().cpu().numpy()
        tbl = tblt.cpu().numpy()
        lens = lenst.cpu().numpy()
        b, hq, d = q64.shape
        hkv = k64.shape[1]
        page = k64.shape[2]
        group = hq // hkv
        scale = 1.0 / np.sqrt(d)
        ref = np.zeros((b, hq, d))
        for bi in range(b):
            for h in range(hq):
                kv = h // group
                idx = [((int(tbl[bi, t // page]) * hkv + kv) * page
                        + t % page) for t in range(lens[bi])]
                K = k64.reshape(-1, d)[idx]
                V = v64.reshape(-1, d)[idx]
                s = (K @ q64[bi, h]) * scale
                p = np.exp(s - s.max())
                ref[bi, h] = p @ V / p.sum()
        err = np.abs(out.float().cpu().numpy() - ref).max()
        assert err < tol, f"{dt} max err {err}"

    def test_paged_staged_matches_torch_path(self):
        rng = np.random.default_rng(49)
        q, kp, vp, tbl, kc, vc, lens = _make_case(
            rng, b=2, hq=8, hkv=2, d=64, nb=12, lens=[100, 512])
        out_gpu = fusedtok.attention_decode_paged(q, kp, vp, tbl, lens,
                                                  cuda=True)
        qt = torch.from_numpy(q).cuda()
        out_t = fusedtok.attention_decode_paged(
            qt, torch.from_numpy(kp).cuda(), torch.from_numpy(vp).cuda(),
            torch.from_numpy(tbl).cuda(), torch.from_numpy(lens).cuda())
        torch.cuda.synchronize()
        np.testing.assert_allclose(out_gpu, out_t.cpu().numpy(), atol=1e-5)

    def test_paged_deterministic_repeat(self):
        qt, kpt, vpt, tblt, lenst, kct, vct = self._case(50)
        a = fusedtok.attention_decode_paged(qt, kpt, vpt, tblt, lenst)
        b = fusedtok.attention_decode_paged(qt, kpt, vpt, tblt, lenst)
        torch.cuda.synchronize()
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)

    def test_paged_graph_capture_replay(self):
        # warm the workspace OUTSIDE the capture (documented contract),
        # then capture and assert replays recompute after mutation
        qt, kpt, vpt, tblt, lenst, kct, vct = self._case(51)
        for _ in range(2):
            fusedtok.attention_decode_paged(qt, kpt, vpt, tblt, lenst)
        out = torch.empty_like(qt)

        def run():
            from fusedtok import _fusedtok
            b, hq, d = qt.shape
            nb, hkv, page, _ = kpt.shape
            _fusedtok.attention_decode_paged_launch(
                qt.data_ptr(), kpt.data_ptr(), vpt.data_ptr(),
                tblt.data_ptr(), lenst.data_ptr(), out.data_ptr(),
                b, hq, hkv, page, tblt.shape[1], d,
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
        g.replay()
        torch.cuda.synchronize()
        ref1 = fusedtok.attention_decode_paged(qt, kpt, vpt, tblt, lenst)
        torch.testing.assert_close(out, ref1, rtol=1e-5, atol=1e-5)
        kpt.mul_(2.0)                       # replay must recompute
        g.replay()
        torch.cuda.synchronize()
        ref2 = fusedtok.attention_decode_paged(qt, kpt, vpt, tblt, lenst)
        torch.testing.assert_close(out, ref2, rtol=1e-5, atol=1e-5)

    def test_paged_torch_errors(self):
        qt, kpt, vpt, tblt, lenst, kct, vct = self._case(52)
        with pytest.raises(TypeError):
            fusedtok.attention_decode_paged(qt.double(), kpt, vpt, tblt)
        with pytest.raises(TypeError):
            fusedtok.attention_decode_paged(qt, kpt.bfloat16(), vpt, tblt)
        with pytest.raises(ValueError):
            fusedtok.attention_decode_paged(qt, kpt, vpt, tblt[:1])
