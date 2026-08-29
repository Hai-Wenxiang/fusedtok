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

    def test_bf16_rejected(self):
        q = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(TypeError):
            fusedtok.attention_decode(q, k, k)

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
