"""Decode-step attention (attention_decode): GQA mapping, per-sequence
lengths, and numerical parity across every execution path.

The kernel keeps a per-warp online softmax (8 warps stride the key rows)
and merges partials in shared memory, so GPU summation order differs
from the float64 numpy reference - parity is asserted within float32
tolerances. Conventions pinned here: q head h uses kv head h // group
(contiguous GQA groups), rows past lens[b] are ignored, and a sequence
with length 0 (or an empty cache) yields a zero output row.
"""

import math

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def ref_decode(q, k, v, lens=None):
    """float64 eager reference with the documented conventions."""
    b, hq, d = q.shape
    _, hkv, t, _ = k.shape
    group = hq // hkv
    out = np.zeros((b, hq, d), dtype=np.float64)
    for bi in range(b):
        length = t if lens is None else int(lens[bi])
        if length == 0:
            continue
        for h in range(hq):
            kv = h // group
            qd = q[bi, h].astype(np.float64)
            kd = k[bi, kv, :length].astype(np.float64)
            vd = v[bi, kv, :length].astype(np.float64)
            s = kd @ qd / math.sqrt(d)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[bi, h] = p @ vd
    return out


def make_case(rng, b, hq, hkv, t, d, with_lens):
    q = rng.standard_normal((b, hq, d)).astype(np.float32)
    k = rng.standard_normal((b, hkv, t, d)).astype(np.float32)
    v = rng.standard_normal((b, hkv, t, d)).astype(np.float32)
    lens = None
    if with_lens:
        lens = rng.integers(0, t + 1, size=b).astype(np.int32)
    return q, k, v, lens


SHAPES = [
    (1, 8, 2, 1, 64, False),        # single key row
    (1, 32, 8, 517, 128, False),    # GQA group 4, odd T, LLaMA-ish
    (1, 32, 32, 100, 64, False),    # MHA (group 1)
    (3, 12, 4, 64, 32, True),       # variable-length batch
    (2, 6, 3, 33, 8, True),         # odd T, tiny D, group 2
    (1, 4, 1, 16, 4, True),         # extreme GQA: all q heads on one kv
    (1, 8, 8, 0, 64, False),        # empty cache -> all rows zero
    (2, 4, 4, 0, 32, True),         # empty cache through the lens path
    (1, 2, 2, 4096, 256, False),    # long cache, max supported dim
]


@pytest.mark.parametrize("b,hq,hkv,t,d,with_lens", SHAPES)
def test_decode_cpu_matches_reference(b, hq, hkv, t, d, with_lens):
    rng = np.random.default_rng(b * 1000 + hq * 31 + hkv * 7 + t + d)
    q, k, v, lens = make_case(rng, b, hq, hkv, t, d, with_lens)
    y = fusedtok.attention_decode(q, k, v, lens)
    assert y.shape == (b, hq, d)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, ref_decode(q, k, v, lens),
                               rtol=1e-4, atol=1e-5)


def test_decode_gqa_mapping_is_contiguous_groups():
    # q head h must read kv head h // group: make every kv head's values
    # a distinct constant so a wrong mapping shifts the output visibly
    rng = np.random.default_rng(5)
    b, hq, hkv, t, d = 1, 6, 3, 4, 8
    q = np.ones((b, hq, d), dtype=np.float32)
    k = np.zeros((b, hkv, t, d), dtype=np.float32)
    v = rng.standard_normal((b, hkv, d)).astype(np.float32)
    v = np.broadcast_to(v[:, :, None, :], (b, hkv, t, d)).copy()
    y = fusedtok.attention_decode(q, k, v)     # uniform scores -> mean of V
    group = hq // hkv
    for h in range(hq):
        np.testing.assert_allclose(y[0, h], v[0, h // group].mean(axis=0),
                                   rtol=1e-5, atol=1e-6)


def test_decode_lens_zero_rows_and_padding_ignored():
    # rows past lens[b] must not leak: poisoning them changes nothing
    rng = np.random.default_rng(6)
    b, hq, hkv, t, d = 2, 8, 4, 16, 32
    q, k, v, _ = make_case(rng, b, hq, hkv, t, d, with_lens=False)
    lens = np.array([t, 5], dtype=np.int32)
    y = fusedtok.attention_decode(q, k, v, lens)
    ref = ref_decode(q, k, v, lens)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)
    k2, v2 = k.copy(), v.copy()
    k2[1, :, 5:] = 1e6          # poison the padding region
    v2[1, :, 5:] = 1e6
    y2 = fusedtok.attention_decode(q, k2, v2, lens)
    np.testing.assert_allclose(y2[1], y[1], rtol=1e-6, atol=1e-7)
    # zero-length sequence -> zero rows
    lens0 = np.array([0, 5], dtype=np.int32)
    y0 = fusedtok.attention_decode(q, k, v, lens0)
    assert np.all(y0[0] == 0)


def test_decode_errors():
    q = np.zeros((1, 6, 32), np.float32)
    k = np.zeros((1, 2, 8, 32), np.float32)
    v = np.zeros_like(k)
    # Hq not a multiple of Hkv
    with pytest.raises(ValueError):
        fusedtok.attention_decode(np.zeros((1, 5, 32), np.float32), k, v)
    # v shape mismatch
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, np.zeros((1, 2, 7, 32), np.float32))
    # dim mismatch between q and the caches
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, np.zeros((1, 2, 8, 30), np.float32), v)
    # dim not a multiple of 4
    bad = (np.zeros((1, 6, 30), np.float32),
           np.zeros((1, 2, 8, 30), np.float32),
           np.zeros((1, 2, 8, 30), np.float32))
    with pytest.raises(ValueError):
        fusedtok.attention_decode(*bad)
    # dim over the shared-memory budget
    big = (np.zeros((1, 2, 516), np.float32),
           np.zeros((1, 1, 4, 516), np.float32),
           np.zeros((1, 1, 4, 516), np.float32))
    with pytest.raises(ValueError):
        fusedtok.attention_decode(*big)
    # lens out of range / wrong length
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, v, np.array([9], np.int32))
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, v, np.array([-1], np.int32))
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, v, np.array([1, 2], np.int32))


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


def ref_prefill(q, k, v, causal=True):
    """float64 eager reference with the causal prefill diagonal."""
    b, hq, s, d = q.shape
    _, hkv, _, _ = k.shape
    group = hq // hkv
    out = np.zeros((b, hq, s, d), dtype=np.float64)
    for bi in range(b):
        for h in range(hq):
            kv = h // group
            qd = q[bi, h].astype(np.float64)               # [s, d]
            kd = k[bi, kv].astype(np.float64)
            vd = v[bi, kv].astype(np.float64)
            scores = qd @ kd.T / np.sqrt(d)                # [s, s]
            for i in range(s):
                lim = i + 1 if causal else s
                p = np.exp(scores[i, :lim] - scores[i, :lim].max())
                p /= p.sum()
                out[bi, h, i] = p @ vd[:lim]
    return out


PREFILL_SHAPES = [
    (1, 4, 2, 1, 32),          # S=1 degenerates to a decode step
    (1, 8, 8, 5, 64),          # MHA, tiny
    (2, 6, 3, 17, 16),         # odd S, group 2, batch
    (1, 32, 8, 40, 128),       # LLaMA-ish GQA
    (1, 4, 1, 33, 8),          # ragged S over the 16-row tiles
    (1, 8, 2, 64, 256),        # LPR=8 band's largest dim
    (1, 4, 2, 9, 512),         # LPR=16 band + the dim <= 512 boundary
]


@pytest.mark.parametrize("b,hq,hkv,s,d", PREFILL_SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_prefill_cpu_matches_reference(b, hq, hkv, s, d, causal):
    rng = np.random.default_rng(b * 100 + hq * 17 + s * 3 + d + causal)
    q = rng.standard_normal((b, hq, s, d)).astype(np.float32)
    k = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
    v = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
    y = fusedtok.attention_prefill(q, k, v, causal=causal)
    assert y.shape == (b, hq, s, d)
    np.testing.assert_allclose(y, ref_prefill(q, k, v, causal),
                               rtol=1e-4, atol=1e-5)


def test_prefill_first_row_attends_only_to_itself():
    # causal row 0: softmax over a single score -> output equals v[0]
    rng = np.random.default_rng(50)
    q = rng.standard_normal((1, 4, 9, 32)).astype(np.float32)
    k = rng.standard_normal((1, 2, 9, 32)).astype(np.float32)
    v = rng.standard_normal((1, 2, 9, 32)).astype(np.float32)
    y = fusedtok.attention_prefill(q, k, v, causal=True)
    for h in range(4):                      # GQA: q head h -> kv head h//2
        np.testing.assert_allclose(y[0, h, 0], v[0, h // 2, 0],
                                   rtol=1e-5, atol=1e-6)


def test_prefill_s1_equals_decode():
    # S=1 causal prefill IS the decode step with len=1
    rng = np.random.default_rng(51)
    q = rng.standard_normal((2, 8, 4, 64)).astype(np.float32)[:, :, :1, :]
    k = rng.standard_normal((2, 2, 4, 64)).astype(np.float32)[:, :, :1, :]
    v = rng.standard_normal((2, 2, 4, 64)).astype(np.float32)[:, :, :1, :]
    pre = fusedtok.attention_prefill(q, k, v, causal=True)
    dec = fusedtok.attention_decode(q.reshape(2, 8, 64),
                                    k.reshape(2, 2, 1, 64),
                                    v.reshape(2, 2, 1, 64))
    np.testing.assert_allclose(pre.reshape(2, 8, 64), dec,
                               rtol=1e-5, atol=1e-6)


def test_prefill_errors():
    q = np.zeros((1, 4, 8, 32), np.float32)
    k = np.zeros((1, 2, 8, 32), np.float32)
    v = np.zeros_like(k)
    with pytest.raises(ValueError):
        fusedtok.attention_prefill(np.zeros((1, 5, 8, 32), np.float32), k, v)
    with pytest.raises(ValueError):
        fusedtok.attention_prefill(q, k, np.zeros((1, 2, 7, 32), np.float32))
    with pytest.raises(ValueError):
        fusedtok.attention_prefill(q, np.zeros((1, 2, 8, 30), np.float32), v)
    with pytest.raises(ValueError):
        fusedtok.attention_prefill(q, k, v[:, :, :, :31])


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    @pytest.mark.parametrize("b,hq,hkv,t,d,with_lens", SHAPES)
    def test_staged_matches_reference(self, b, hq, hkv, t, d, with_lens):
        rng = np.random.default_rng(b * 1000 + hq * 31 + hkv * 7 + t + d)
        q, k, v, lens = make_case(rng, b, hq, hkv, t, d, with_lens)
        y = fusedtok.attention_decode(q, k, v, lens, cuda=True)
        np.testing.assert_allclose(y, ref_decode(q, k, v, lens),
                                   rtol=1e-4, atol=1e-5)

    def test_staged_matches_cpu(self):
        # GPU vs CPU parity beyond the float64 reference: both are the
        # same math in different summation orders - keep them tight
        rng = np.random.default_rng(21)
        q, k, v, lens = make_case(rng, 3, 12, 4, 200, 128, True)
        y = fusedtok.attention_decode(q, k, v, lens, cuda=True)
        ycpu = fusedtok.attention_decode(q, k, v, lens)
        np.testing.assert_allclose(y, ycpu, rtol=1e-5, atol=1e-6)

    def test_repeated_calls_same_result(self):
        rng = np.random.default_rng(22)
        q, k, v, lens = make_case(rng, 2, 8, 2, 100, 64, True)
        first = fusedtok.attention_decode(q, k, v, lens, cuda=True)
        for _ in range(3):
            np.testing.assert_array_equal(
                fusedtok.attention_decode(q, k, v, lens, cuda=True), first)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestSplitPath:
    """Long caches take the flash-decoding split path (per-slice partials
    in a cached workspace + reduce). The heuristic slices at >= 512 rows
    for every templated GQA group width; these pin it against the float64
    reference with lens crossing slice boundaries."""

    @pytest.mark.parametrize("group", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("t", [2048, 5000])       # 2^k and ragged
    def test_long_cache_matches_reference(self, group, t):
        rng = np.random.default_rng(group * 977 + t)
        hkv, d = 4, 128
        hq = hkv * group
        q, k, v, _ = make_case(rng, 1, hq, hkv, t, d, False)
        y = fusedtok.attention_decode(q, k, v, cuda=True)
        np.testing.assert_allclose(y, ref_decode(q, k, v, None),
                                   rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("len_after_first_slice", [0, 1, 293])
    def test_lens_across_slice_boundaries(self, len_after_first_slice):
        # slice boundaries fall at ~len/14 on this shape; a length just
        # past the first boundary leaves one dense slice, one stub and
        # empty remainder slices - all must merge to the reference
        rng = np.random.default_rng(4000 + len_after_first_slice)
        t, hq, hkv, d = 4096, 32, 8, 64
        q, k, v, _ = make_case(rng, 2, hq, hkv, t, d, False)
        lens = np.array([t, 400 + len_after_first_slice], dtype=np.int32)
        y = fusedtok.attention_decode(q, k, v, lens, cuda=True)
        np.testing.assert_allclose(y, ref_decode(q, k, v, lens),
                                   rtol=1e-4, atol=1e-5)

    def test_batched_long_cache(self):
        # a bigger batch shrinks the slice count toward 1; both regimes
        # must produce the same math as the reference
        rng = np.random.default_rng(77)
        b, hq, hkv, t, d = 6, 32, 8, 4096, 128
        q, k, v, lens = make_case(rng, b, hq, hkv, t, d, True)
        y = fusedtok.attention_decode(q, k, v, lens, cuda=True)
        np.testing.assert_allclose(y, ref_decode(q, k, v, lens),
                                   rtol=1e-4, atol=1e-5)

    def test_16k_cache(self):
        rng = np.random.default_rng(88)
        q, k, v, _ = make_case(rng, 1, 32, 8, 16384, 128, False)
        y = fusedtok.attention_decode(q, k, v, cuda=True)
        np.testing.assert_allclose(y, ref_decode(q, k, v, None),
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestPrefillCuda:
    @pytest.mark.parametrize("b,hq,hkv,s,d", PREFILL_SHAPES)
    @pytest.mark.parametrize("causal", [True, False])
    def test_staged_matches_reference(self, b, hq, hkv, s, d, causal):
        rng = np.random.default_rng(b * 100 + hq * 17 + s * 3 + d + causal)
        q = rng.standard_normal((b, hq, s, d)).astype(np.float32)
        k = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
        v = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
        y = fusedtok.attention_prefill(q, k, v, causal=causal, cuda=True)
        np.testing.assert_allclose(y, ref_prefill(q, k, v, causal),
                                   rtol=1e-4, atol=1e-5)

    def test_long_sequence_matches_reference(self):
        # multi-tile S with the causal diagonal crossing every tile
        rng = np.random.default_rng(99)
        b, hq, hkv, s, d = 1, 8, 2, 300, 64
        q = rng.standard_normal((b, hq, s, d)).astype(np.float32)
        k = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
        v = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
        y = fusedtok.attention_prefill(q, k, v, cuda=True)
        np.testing.assert_allclose(y, ref_prefill(q, k, v, True),
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestPrefillTorch:
    def test_zero_copy_matches_reference(self):
        rng = np.random.default_rng(61)
        q = rng.standard_normal((2, 12, 100, 64)).astype(np.float32)
        k = rng.standard_normal((2, 4, 100, 64)).astype(np.float32)
        v = rng.standard_normal((2, 4, 100, 64)).astype(np.float32)
        y = fusedtok.attention_prefill(
            torch.from_numpy(q).cuda(), torch.from_numpy(k).cuda(),
            torch.from_numpy(v).cuda(), causal=False)
        torch.cuda.synchronize()
        np.testing.assert_allclose(y.cpu().numpy(), ref_prefill(q, k, v, False),
                                   rtol=1e-4, atol=1e-5)

    def test_sdpa_crosscheck(self):
        # independent implementation: torch SDPA with is_causal=True
        # (heads expanded with repeat_interleave)
        rng = np.random.default_rng(62)
        b, hq, hkv, s, d = 1, 32, 8, 256, 128
        q = rng.standard_normal((b, hq, s, d)).astype(np.float32)
        k = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
        v = rng.standard_normal((b, hkv, s, d)).astype(np.float32)
        y = fusedtok.attention_prefill(
            torch.from_numpy(q).cuda(), torch.from_numpy(k).cuda(),
            torch.from_numpy(v).cuda(), causal=True)
        torch.cuda.synchronize()
        group = hq // hkv
        kk = torch.from_numpy(k).cuda().repeat_interleave(group, dim=1)
        vv = torch.from_numpy(v).cuda().repeat_interleave(group, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q).cuda(), kk, vv, is_causal=True)
        np.testing.assert_allclose(y.cpu().numpy(), ref.cpu().numpy(),
                                   rtol=1e-3, atol=1e-4)

    def test_graph_capture_replay(self):
        from fusedtok import _fusedtok

        b, hq, hkv, s, d = 1, 8, 2, 64, 64
        q = torch.randn(b, hq, s, d, device="cuda")
        k = torch.randn(b, hkv, s, d, device="cuda")
        v = torch.randn(b, hkv, s, d, device="cuda")
        y = torch.empty(b, hq, s, d, device="cuda")

        def run():
            _fusedtok.attention_prefill_launch(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
                b, hq, hkv, s, d, True,
                torch.cuda.current_stream().cuda_stream)

        s_side = torch.cuda.Stream()
        s_side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_side):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s_side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        q.mul_(1.5)                       # replay must recompute
        y.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        ref = fusedtok.attention_prefill(q, k, v).cpu().numpy()
        np.testing.assert_allclose(y.cpu().numpy(), ref, rtol=1e-5,
                                   atol=1e-6)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_matches_reference(self):
        rng = np.random.default_rng(31)
        q, k, v, lens = make_case(rng, 3, 16, 4, 333, 128, True)
        y = fusedtok.attention_decode(
            torch.from_numpy(q).cuda(), torch.from_numpy(k).cuda(),
            torch.from_numpy(v).cuda(), torch.from_numpy(lens).cuda())
        assert y.is_cuda and y.dtype is torch.float32
        torch.cuda.synchronize()
        np.testing.assert_allclose(y.cpu().numpy(), ref_decode(q, k, v, lens),
                                   rtol=1e-4, atol=1e-5)

    def test_lens_as_list_and_host_tensor(self):
        rng = np.random.default_rng(32)
        q, k, v, _ = make_case(rng, 2, 8, 4, 64, 32, False)
        lens = [64, 10]
        qt, kt, vt = (torch.from_numpy(x).cuda() for x in (q, k, v))
        y1 = fusedtok.attention_decode(qt, kt, vt, lens)
        y2 = fusedtok.attention_decode(qt, kt, vt,
                                       torch.tensor(lens, dtype=torch.int64))
        torch.cuda.synchronize()
        np.testing.assert_array_equal(y1.cpu().numpy(), y2.cpu().numpy())

    def test_torch_eager_crosscheck(self):
        # independent torch implementation (repeat_interleave expansion)
        rng = np.random.default_rng(33)
        b, hq, hkv, t, d = 2, 32, 8, 512, 128
        q, k, v, lens = make_case(rng, b, hq, hkv, t, d, True)
        y = fusedtok.attention_decode(
            torch.from_numpy(q).cuda(), torch.from_numpy(k).cuda(),
            torch.from_numpy(v).cuda(), torch.from_numpy(lens).cuda())
        torch.cuda.synchronize()
        group = hq // hkv
        for bi in range(b):
            length = int(lens[bi])
            if length == 0:
                assert torch.count_nonzero(y[bi]) == 0
                continue
            kk = torch.from_numpy(k[bi, :, :length]).cuda() \
                .repeat_interleave(group, dim=0).double()
            vv = torch.from_numpy(v[bi, :, :length]).cuda() \
                .repeat_interleave(group, dim=0).double()
            s = torch.einsum("hd,htd->ht",
                             torch.from_numpy(q[bi]).cuda().double(), kk)
            s = s / math.sqrt(d)
            p = torch.softmax(s, dim=-1)
            ref = torch.einsum("ht,htd->hd", p, vv).float().cpu().numpy()
            np.testing.assert_allclose(y[bi].cpu().numpy(), ref,
                                       rtol=1e-4, atol=1e-5)

    def test_bf16_accepted_since_1_1(self):
        # bf16/fp16 storage was REJECTED before v1.1 and is now the
        # supported half-precision path: same inputs, same GQA answer as
        # the f32 path on the half-rounded inputs (out dtype = input)
        q = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
        out = fusedtok.attention_decode(q, k, k)
        assert out.dtype is torch.bfloat16
        ref = fusedtok.attention_decode(q.float(), k.float(), k.float())
        np.testing.assert_allclose(out.float().cpu().numpy(),
                                   ref.cpu().numpy(), rtol=2e-2, atol=2e-2)
        with pytest.raises(TypeError):
            # int8 was never an attention dtype
            fusedtok.attention_decode(
                torch.zeros(1, 8, 64, device="cuda", dtype=torch.int8),
                torch.randn(1, 2, 32, 64, device="cuda"),
                torch.randn(1, 2, 32, 64, device="cuda"))

    def test_graph_capture_replay(self):
        # the launcher is capture-safe (no allocs/syncs/readbacks); the
        # replay must recompute after the inputs mutate
        from fusedtok import _fusedtok

        q = torch.randn(2, 8, 64, device="cuda")
        k = torch.randn(2, 2, 100, 64, device="cuda")
        v = torch.randn(2, 2, 100, 64, device="cuda")
        lens = torch.tensor([100, 40], dtype=torch.int32, device="cuda")
        y = torch.empty(2, 8, 64, device="cuda")

        def run():
            _fusedtok.attention_decode_launch(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), lens.data_ptr(),
                y.data_ptr(), 2, 8, 2, 100, 64,
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
        q.mul_(2.0)                       # replay must recompute
        y.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        ref = fusedtok.attention_decode(q, k, v, lens).cpu().numpy()
        np.testing.assert_allclose(y.cpu().numpy(), ref, rtol=1e-5,
                                   atol=1e-6)

    def test_graph_capture_replay_split_path(self):
        # long caches capture the TWO-KERNEL split path with its cached
        # workspace (pointers baked into the graph stay valid because
        # the workspace is process-cached per shape)
        from fusedtok import _fusedtok

        b, hq, hkv, t, d = 1, 32, 8, 4096, 128
        q = torch.randn(b, hq, d, device="cuda")
        k = torch.randn(b, hkv, t, d, device="cuda")
        v = torch.randn(b, hkv, t, d, device="cuda")
        y = torch.empty(b, hq, d, device="cuda")

        def run():
            _fusedtok.attention_decode_launch(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), 0,
                y.data_ptr(), b, hq, hkv, t, d,
                torch.cuda.current_stream().cuda_stream)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()                      # populates the workspace cache
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        k.mul_(1.5)                        # replay must recompute
        v.mul_(-0.5)
        y.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        ref = fusedtok.attention_decode(q, k, v).cpu().numpy()
        np.testing.assert_allclose(y.cpu().numpy(), ref, rtol=1e-5,
                                   atol=1e-6)

if HAS_TORCH:
    HALF_DTYPES = [pytest.param(torch.bfloat16, 2e-2, 2e-2, id="bf16"),
                   pytest.param(torch.float16, 2e-3, 2e-3, id="fp16")]
    HALF_DTYPE_OBJECTS = [torch.bfloat16, torch.float16]
else:
    # empty lists keep the class importable on torch-less machines
    HALF_DTYPES = []
    HALF_DTYPE_OBJECTS = []


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestHalfStorage:
    """bfloat16 / float16 storage (v1.1): q/k/v/out in half width,
    softmax and accumulators in float32.

    Reference protocol: compute the float64 eager attention on the
    HALF-ROUNDED inputs (the kernel never sees the original f32 bits),
    so the only tolerated error is softmax accumulation + the final
    store rounding - bf16 keeps ~3 significant digits, fp16 ~3.5.
    """



    @pytest.mark.parametrize("dt,rtol,atol", HALF_DTYPES)
    def test_decode_matches_f32reference(self, dt, rtol, atol):
        rng = np.random.default_rng(101)
        b, hq, hkv, t, d = 2, 8, 2, 512, 128
        q = torch.randn(b, hq, d, device="cuda").to(dt)
        k = torch.randn(b, hkv, t, d, device="cuda").to(dt)
        v = torch.randn(b, hkv, t, d, device="cuda").to(dt)
        out = fusedtok.attention_decode(q, k, v)
        assert out.dtype is dt                       # out matches storage
        ref = ref_decode(q.float().cpu().numpy(),
                         k.float().cpu().numpy(),
                         v.float().cpu().numpy())
        np.testing.assert_allclose(out.float().cpu().numpy(), ref,
                                   rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dt,rtol,atol", HALF_DTYPES)
    def test_decode_gqa_mapping_and_lens(self, dt, rtol, atol):
        # constant-V probe: V is IDENTICAL across the rows of each kv
        # head, so every output row of a GQA group must equal that
        # head's constant vector (softmax over any subset of identical
        # values is uniform) - pins the h -> h // group mapping without
        # any score math
        b, hq, hkv, t, d = 1, 8, 2, 300, 128
        q = torch.randn(b, hq, d, device="cuda").to(dt)
        k = torch.randn(b, hkv, t, d, device="cuda").to(dt)
        v0 = torch.randn(hkv, d, device="cuda").to(dt)
        v = v0[None, :, None, :].expand(b, hkv, t, d).contiguous()
        lens = torch.tensor([300], dtype=torch.int32, device="cuda")
        out = fusedtok.attention_decode(q, k, v, lens)
        for h in range(hq):
            np.testing.assert_allclose(out[0, h].float().cpu().numpy(),
                                       v0[h // 4].float().cpu().numpy(),
                                       rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dt,rtol,atol", HALF_DTYPES)
    def test_decode_padding_and_zero_len(self, dt, rtol, atol):
        # rows past lens[b] must be ignored even when poisoned with huge
        # values, and lens[b] == 0 must yield a zero output row in the
        # same dtype
        b, hq, hkv, t, d = 2, 8, 2, 256, 128
        q = torch.randn(b, hq, d, device="cuda").to(dt)
        k = torch.randn(b, hkv, t, d, device="cuda").to(dt)
        v = torch.randn(b, hkv, t, d, device="cuda").to(dt)
        lens = torch.tensor([100, 0], dtype=torch.int32, device="cuda")
        k[0, :, 100:] = 30000.0                      # poison the padding
        v[0, :, 100:] = 30000.0
        out = fusedtok.attention_decode(q, k, v, lens)
        ref0 = ref_decode(q[0:1].float().cpu().numpy(),
                          k[0:1, :, :100].float().cpu().numpy(),
                          v[0:1, :, :100].float().cpu().numpy())
        np.testing.assert_allclose(out[0].float().cpu().numpy(), ref0[0],
                                   rtol=rtol, atol=atol)
        np.testing.assert_array_equal(out[1].float().cpu().numpy(),
                                      np.zeros((hq, d), np.float32))

    @pytest.mark.parametrize("dt,rtol,atol", HALF_DTYPES)
    def test_prefill_causal_and_bidirectional(self, dt, rtol, atol):
        b, hq, hkv, s, d = 1, 4, 2, 96, 128
        q = torch.randn(b, hq, s, d, device="cuda").to(dt)
        k = torch.randn(b, hkv, s, d, device="cuda").to(dt)
        v = torch.randn(b, hkv, s, d, device="cuda").to(dt)

        def ref(qn, kn, vn, causal):
            group = hq // hkv
            out = np.zeros_like(qn, dtype=np.float64)
            for h in range(hq):
                kvh = h // group
                scores = qn[0, h].astype(np.float64) @ \
                    kn[0, kvh].astype(np.float64).T / math.sqrt(d)
                for i in range(s):
                    lim = i + 1 if causal else s
                    p = np.exp(scores[i, :lim] - scores[i, :lim].max())
                    p /= p.sum()
                    out[0, h, i] = p @ vn[0, kvh, :lim].astype(np.float64)
            return out

        out = fusedtok.attention_prefill(q, k, v, causal=True)
        assert out.dtype is dt
        np.testing.assert_allclose(
            out.float().cpu().numpy(),
            ref(q.float().cpu().numpy(), k.float().cpu().numpy(),
                v.float().cpu().numpy(), True), rtol=rtol, atol=atol)
        outb = fusedtok.attention_prefill(q, k, v, causal=False)
        np.testing.assert_allclose(
            outb.float().cpu().numpy(),
            ref(q.float().cpu().numpy(), k.float().cpu().numpy(),
                v.float().cpu().numpy(), False), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dt", HALF_DTYPE_OBJECTS)
    def test_split_and_single_paths_match(self, dt):
        # the same shape across the split threshold: both kernel paths
        # must agree within half-precision tolerance
        b, hq, hkv, t, d = 1, 8, 2, 2000, 128
        q = (np.random.default_rng(105).standard_normal((b, hq, d)) *
             .5).astype(np.float32)
        k = (np.random.default_rng(106).standard_normal(
            (b, hkv, t, d)) * .5).astype(np.float32)
        v = (np.random.default_rng(107).standard_normal(
            (b, hkv, t, d)) * .5).astype(np.float32)
        qt = torch.from_numpy(q).cuda().to(dt)
        kt = torch.from_numpy(k).cuda().to(dt)
        vt = torch.from_numpy(v).cuda().to(dt)
        y = fusedtok.attention_decode(qt, kt, vt)
        # f32 path on the same (half-rounded) inputs is the reference
        y32 = fusedtok.attention_decode(qt.float(), kt.float(), vt.float())
        np.testing.assert_allclose(y.float().cpu().numpy(),
                                   y32.cpu().numpy(), rtol=2e-2, atol=2e-2)

    def test_mixed_dtypes_rejected(self):
        q = torch.randn(1, 8, 64, device="cuda")
        k16 = torch.randn(1, 2, 32, 64, device="cuda").to(torch.float16)
        with pytest.raises(TypeError):
            fusedtok.attention_decode(q, k16, k16)
        with pytest.raises(TypeError):
            fusedtok.attention_decode(
                torch.zeros(1, 8, 64, device="cuda", dtype=torch.int8),
                torch.randn(1, 2, 32, 64, device="cuda"),
                torch.randn(1, 2, 32, 64, device="cuda"))

    def test_graph_capture_bf16(self):
        # capture-replay with mutation on the bf16 split path
        from fusedtok import _fusedtok
        b, hq, hkv, t, d = 1, 8, 2, 1024, 128
        q = torch.randn(b, hq, d, device="cuda").to(torch.bfloat16)
        k = torch.randn(b, hkv, t, d, device="cuda").to(torch.bfloat16)
        v = torch.randn(b, hkv, t, d, device="cuda").to(torch.bfloat16)
        y = torch.empty(b, hq, d, device="cuda", dtype=torch.bfloat16)

        def run():
            _fusedtok.attention_decode_launch_bf16(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), 0, y.data_ptr(),
                b, hq, hkv, t, d, torch.cuda.current_stream().cuda_stream)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        k.mul_(1.25)
        y.fill_(0)
        g.replay()
        torch.cuda.synchronize()
        ref = fusedtok.attention_decode(q, k, v)
        np.testing.assert_allclose(y.float().cpu().numpy(),
                                   ref.float().cpu().numpy(),
                                   rtol=2e-2, atol=2e-2)