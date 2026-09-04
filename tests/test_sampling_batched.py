"""Batched sampling (v1.4): sample_topp/minp/topk_batched.

Every row runs the single-row pipeline verbatim (same kernels, same
accumulation order), so the core contract is per-row parity with the
single-row API. Cases:

- per-row parity vs the singles on the GPU (mixed peaked/midtail/flat
  rows - the widening loop skips finished rows while wide-nucleus rows
  retry under a uniform window)
- per-row parity on CPU/staged vs the row-wise singles (bit-identical
  by construction)
- self-determinism (identical repeated calls), seed semantics
  (default arange; explicit seeds honored; per-row independence)
- B = 1 matches the single call exactly; B = 0 returns empty; B = 33
  crosses the device-side chunk boundary
- Qwen-scale vocabulary (152064) and k = 1 greedy / k >= vocab
- torch input returns a torch tensor, numpy input a numpy array
- error contract (parameter bounds, 1-D/3-D rejection, seeds shape /
  dtype / negativity, non-contiguous GPU input)
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

BATCHED = {
    "topp": (fusedtok.sample_topp_batched, fusedtok.sample_topp, 0.9),
    "minp": (fusedtok.sample_minp_batched, fusedtok.sample_minp, 0.05),
    "topk": (fusedtok.sample_topk_batched, fusedtok.sample_topk, 50),
}


def _assert_row_close(logits_row, got, want, what):
    """Exact per-row parity, with the documented ulp fallback: the
    global-softmax total is accumulated with per-block float atomics
    whose arrival order differs between launch shapes (and across
    processes - a 1.3.1-era property of the single-row API itself), so
    a draw landing exactly on a CDF boundary may pick a neighbor."""
    if got == want:
        return
    order = np.argsort(-logits_row, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


def _rows(rng, b, n):
    """Mixed-shape batch: spiked / plain randn / flattened rows."""
    x = rng.standard_normal((b, n)).astype(np.float32)
    for r in range(0, b, 3):
        x[r, 7] += 8.0                    # heavily peaked
    for r in range(1, b, 3):
        x[r] *= 1e-3                       # near-uniform (wide nucleus)
    return x


def test_cpu_matches_rowwise_singles():
    rng = np.random.default_rng(90)
    x = _rows(rng, 8, 4096)
    seeds = np.arange(8, dtype=np.int64)
    for name, (batched, single, arg) in BATCHED.items():
        got = batched(x, arg, seeds=seeds)
        want = [int(single(x[r], arg, seed=int(s)))
                for r, s in enumerate(seeds)]
        assert got.tolist() == want, name


def test_staged_matches_rowwise_singles():
    if not fusedtok.cuda_available():
        pytest.skip("staged needs a GPU")
    rng = np.random.default_rng(91)
    x = _rows(rng, 6, 4096)
    seeds = np.arange(6, dtype=np.int64)
    for name, (batched, single, arg) in BATCHED.items():
        got = batched(x, arg, seeds=seeds, cuda=True)
        want = [int(single(x[r], arg, seed=int(s)))
                for r, s in enumerate(seeds)]
        assert got.tolist() == want, name


def test_default_seeds_are_arange():
    rng = np.random.default_rng(92)
    x = np.tile(rng.standard_normal(2048).astype(np.float32), (4, 1))
    # identical rows + default seeds -> distinct streams -> the draws
    # must equal the singles called with seeds 0..3
    got = fusedtok.sample_topp_batched(x, 0.9)
    want = [int(fusedtok.sample_topp(x[0], 0.9, seed=s)) for s in range(4)]
    assert got.tolist() == want


def test_explicit_seeds_honored():
    rng = np.random.default_rng(93)
    x = rng.standard_normal((5, 3000)).astype(np.float32)
    seeds = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    got = fusedtok.sample_topk_batched(x, 32, seeds=seeds)
    want = [int(fusedtok.sample_topk(x[r], 32, seed=int(s)))
            for r, s in enumerate(seeds)]
    assert got.tolist() == want


def test_empty_batch():
    for batched, _, arg in BATCHED.values():
        out = batched(np.empty((0, 128), dtype=np.float32), arg)
        assert out.shape == (0,)
        assert out.dtype == np.int64


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_singles_mixed_rows(self):
        rng = np.random.default_rng(94)
        x = _rows(rng, 8, 131072)
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(8, dtype=np.int64)
        for name, (batched, single, arg) in BATCHED.items():
            got = batched(dev, arg, seeds=seeds)
            assert isinstance(got, torch.Tensor)
            assert got.dtype == torch.int64 and got.is_cpu
            for r in range(8):
                want = int(single(dev[r], arg, seed=int(seeds[r])))
                _assert_row_close(x[r], int(got[r]), want, (name, r))

    def test_self_determinism_repeat_call(self):
        rng = np.random.default_rng(95)
        x = _rows(rng, 8, 65536)
        dev = torch.from_numpy(x).cuda()
        for batched, _, arg in BATCHED.values():
            a = batched(dev, arg)
            b = batched(dev, arg)
            assert a.tolist() == b.tolist()

    def test_single_row_matches_single_api(self):
        # B = 1 launches the same grid shape as a standalone call, so
        # even the atomic-order-sensitive totals coincide - exact match
        rng = np.random.default_rng(96)
        x = rng.standard_normal((1, 32768)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        for name, (batched, single, arg) in BATCHED.items():
            got = batched(dev, arg, seeds=np.array([7], dtype=np.int64))
            assert int(got[0]) == int(single(dev[0], arg, seed=7)), name

    def test_chunked_large_batch(self):
        # 33 rows crosses the 32-row device chunk boundary; every row
        # must still match its standalone call
        rng = np.random.default_rng(97)
        x = _rows(rng, 33, 8192)
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(33, dtype=np.int64)
        for name, (batched, single, arg) in BATCHED.items():
            got = batched(dev, arg, seeds=seeds)
            for r in (0, 1, 2, 16, 31, 32):
                want = int(single(dev[r], arg, seed=int(seeds[r])))
                _assert_row_close(x[r], int(got[r]), want, (name, r))

    def test_qwen_scale_vocabulary(self):
        rng = np.random.default_rng(98)
        n = 152064
        x = rng.standard_normal((4, n)).astype(np.float32)
        x[1] += 5.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.array([3, 4, 5, 6], dtype=np.int64)
        got = fusedtok.sample_topp_batched(dev, 0.95, seeds=seeds)
        for r in range(4):
            want = int(fusedtok.sample_topp(dev[r], 0.95,
                                            seed=int(seeds[r])))
            _assert_row_close(x[r], int(got[r]), want, r)

    def test_topk_k1_greedy_and_full_vocab(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal((6, 5000)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        greedy = fusedtok.sample_topk_batched(dev, 1)
        for r in range(6):
            assert int(greedy[r]) == int(dev[r].argmax())
        full = fusedtok.sample_topk_batched(dev, 9999)
        again = fusedtok.sample_topk_batched(dev, 9999)
        assert full.tolist() == again.tolist()
        assert ((full >= 0) & (full < 5000)).all()

    def test_flat_rows_mixed_with_peaked(self):
        # the harshest divergence: one flat row forces the full
        # vocabulary while the peaked rows finished two windows ago
        rng = np.random.default_rng(100)
        n = 32768
        x = rng.standard_normal((4, n)).astype(np.float32) * 1e-3
        x[0, 11] += 6.0
        x[2, 5] += 6.0
        dev = torch.from_numpy(x).cuda()
        got = fusedtok.sample_minp_batched(dev, 0.02)
        for r in range(4):
            want = int(fusedtok.sample_minp(dev[r], 0.02, seed=r))
            _assert_row_close(x[r], int(got[r]), want, r)

    def test_error_contract_cuda(self):
        x = torch.ones(4, 64, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_topp_batched(x, 0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_minp_batched(x, 1.5)
        with pytest.raises(ValueError):
            fusedtok.sample_topk_batched(x, 0)
        with pytest.raises(ValueError):
            fusedtok.sample_topp_batched(x, 0.9, temperature=0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_topp_batched(torch.ones(64, device="cuda"),
                                         0.9)
        with pytest.raises(ValueError):
            fusedtok.sample_topp_batched(torch.ones(2, 2, 2, device="cuda"),
                                         0.9)
        with pytest.raises(TypeError):
            fusedtok.sample_topp_batched(x.to(torch.bfloat16), 0.9)
        with pytest.raises(ValueError):      # non-contiguous 2-D view
            fusedtok.sample_topp_batched(
                torch.ones(64, 4, device="cuda").t(), 0.9)
        with pytest.raises(ValueError):      # wrong seed count
            fusedtok.sample_topp_batched(x, 0.9,
                                         seeds=np.zeros(3, dtype=np.int64))
        with pytest.raises(TypeError):       # float seeds
            fusedtok.sample_topp_batched(x, 0.9,
                                         seeds=np.zeros(4, dtype=np.float64))
        with pytest.raises(ValueError):      # negative seeds
            fusedtok.sample_topp_batched(
                x, 0.9, seeds=-np.ones(4, dtype=np.int64))

    def test_error_contract_cpu(self):
        x = np.ones((4, 64), dtype=np.float32)
        with pytest.raises(ValueError):
            fusedtok.sample_topp_batched(x, 0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_topp_batched(np.ones(64, dtype=np.float32), 0.9)
        with pytest.raises(ValueError):
            fusedtok.sample_topk_batched(x, -1)
        with pytest.raises(ValueError):
            fusedtok.sample_minp_batched(x, 2.0)
        with pytest.raises(ValueError):
            fusedtok.sample_topk_batched(x, 8, seeds=[1, 2, 3])

    def test_torch_input_returns_torch_cpu_tensor(self):
        if not HAS_TORCH:
            pytest.skip("torch")
        x = torch.randn(4, 1000)
        out = fusedtok.sample_topp_batched(x, 0.9)
        assert torch.is_tensor(out) and out.dtype == torch.int64
        ref = fusedtok.sample_topp_batched(x.numpy(), 0.9)
        assert out.numpy().tolist() == ref.tolist()

    def test_large_batches_multiple_chunks(self):
        # 64 rows = two full 32-row device chunks with mixed shapes;
        # 32 rows = the exact chunk boundary; spot-check each row
        # against its standalone call
        rng = np.random.default_rng(101)
        x = _rows(rng, 64, 4096)
        dev = torch.from_numpy(x).cuda()
        for b in (32, 64):
            d = dev[:b]
            for name, (batched, single, arg) in BATCHED.items():
                got = batched(d, arg)
                for r in (0, 1, 31, b - 1):
                    want = int(single(d[r], arg, seed=r))
                    _assert_row_close(x[r], int(got[r]), want,
                                      (name, b, r))

    def test_seeds_as_torch_and_list_containers(self):
        # _batch_seeds accepts CUDA tensors (moved to host), CPU
        # tensors and plain lists - all must give the same stream as
        # the numpy array form
        rng = np.random.default_rng(102)
        x = rng.standard_normal((4, 2048)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        ref = fusedtok.sample_topp_batched(dev, 0.9,
                                           seeds=[5, 6, 7, 8])
        same = [
            fusedtok.sample_topp_batched(
                dev, 0.9, seeds=np.array([5, 6, 7, 8], dtype=np.int64)),
            fusedtok.sample_topp_batched(
                dev, 0.9, seeds=torch.tensor([5, 6, 7, 8])),
            fusedtok.sample_topp_batched(
                dev, 0.9,
                seeds=torch.tensor([5, 6, 7, 8], device="cuda")),
        ]
        for got in same:
            assert got.tolist() == ref.tolist()

    def test_temperature_extremes(self):
        # tiny temperature collapses every row onto its argmax; a huge
        # temperature flattens the distribution and forces the
        # widening loop (the widest T - C mass the lazy totals cache
        # ever sees)
        rng = np.random.default_rng(103)
        x = rng.standard_normal((4, 8192)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        cold = fusedtok.sample_topp_batched(dev, 0.9, temperature=1e-4)
        for r in range(4):
            assert int(cold[r]) == int(dev[r].argmax())
        hot = fusedtok.sample_minp_batched(dev, 0.05, temperature=1e4)
        again = fusedtok.sample_minp_batched(dev, 0.05, temperature=1e4)
        assert hot.tolist() == again.tolist()

    def test_interleaved_with_argmax_and_singles(self):
        # the batched family shares the process-wide selection
        # workspace with argmax and the single-row samplers; alternate
        # them (small after large, so the workspace REALLOCS between
        # calls) to pin the reset invariants the widening loop relies on
        rng = np.random.default_rng(104)
        big = rng.standard_normal((8, 131072)).astype(np.float32)
        big[:, 3] += 20.0
        dbig = torch.from_numpy(big).cuda()
        small = rng.standard_normal((3, 1024)).astype(np.float32)
        dsmall = torch.from_numpy(small).cuda()
        for _ in range(3):
            toks = fusedtok.sample_topp_batched(dbig, 0.9)
            assert all(0 <= t < 131072 for t in toks.tolist())
            for r in range(8):
                assert int(fusedtok.argmax(dbig[r])) == 3
            got = fusedtok.sample_minp_batched(dsmall, 0.1)
            for r in range(3):
                want = int(fusedtok.sample_minp(dsmall[r], 0.1, seed=r))
                assert int(got[r]) == want
            assert fusedtok.sample_topk(dsmall[0], 10, seed=1) == \
                fusedtok.sample_topk(dsmall[0], 10, seed=1)

    def test_empty_batch_on_gpu_paths(self):
        if not fusedtok.cuda_available():
            pytest.skip("GPU")
        z = torch.empty(0, 64, device="cuda")
        for batched, _, arg in BATCHED.values():
            out = batched(z, arg)
            assert out.shape == (0,)
        # staged empty batch (host staging path)
        for batched, _, arg in BATCHED.values():
            out = batched(np.empty((0, 64), dtype=np.float32), arg,
                          cuda=True)
            assert out.shape == (0,)

    def test_parameter_edges_p_and_minp_one(self):
        # p = 1.0: the whole distribution is the nucleus (forced
        # coverage); min_p = 1.0: only the row maxima survive - both
        # per-row equal to the singles
        rng = np.random.default_rng(105)
        x = rng.standard_normal((4, 3000)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        got = fusedtok.sample_topp_batched(dev, 1.0)
        for r in range(4):
            want = int(fusedtok.sample_topp(dev[r], 1.0, seed=r))
            _assert_row_close(x[r], int(got[r]), want, ("topp1", r))
        got = fusedtok.sample_minp_batched(dev, 1.0)
        for r in range(4):
            want = int(fusedtok.sample_minp(dev[r], 1.0, seed=r))
            _assert_row_close(x[r], int(got[r]), want, ("minp1", r))

    def test_default_seeds_cover_all_three_samplers(self):
        rng = np.random.default_rng(106)
        x = np.tile(rng.standard_normal(2048).astype(np.float32), (4, 1))
        for batched, single, arg in BATCHED.values():
            got = batched(x, arg)
            want = [int(single(x[0], arg, seed=s)) for s in range(4)]
            assert got.tolist() == want
