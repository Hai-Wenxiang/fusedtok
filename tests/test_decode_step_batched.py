"""Batched fused decode step (v1.5): decode_step_batched.

Every row runs the single-row decode_step pipeline (repetition
penalty on the RAW logit -> temperature -> nucleus sampling) with its
own ragged history, so the core contract is per-row parity with the
single-row API - exact on CPU (same composed reference, bit-identical
by construction) and on the GPU up to the documented exptotal
arrival-order ulp boundary (the same exact-or-neighbor-rank fallback
the batched samplers use). Cases:

- per-row parity vs the singles on every path (CPU, staged,
  torch zero-copy), ragged histories including empty rows
- B = 1 matches the single call exactly (same grid shape); B = 0
  returns empty; B = 33 crosses the device-side chunk boundary
- the ids container contract: ragged list of lists, flat ids +
  ids_offsets, 2-D array (equal-length rows), numpy and torch forms
- penalty = 1.0 / all-empty histories reduce exactly to
  sample_topp_batched (the bitmap traffic vanishes)
- a generation loop grows the histories across steps (bitmap growth)
  and breaks greedy repetition under a heavy penalty
- temperature extremes (1e-4 collapses to the penalized argmax, 1e4
  forces the widening loop), p = 1.0 full-nucleus rows
- determinism on repeat calls; interleaving with the batched samplers
  and argmax across workspace reallocations
- Qwen-scale vocabulary (152064) with the bitmap at its worst
- error contract (penalty/p/temperature bounds, ids out of vocab,
  wrong offset shape, non-decreasing offsets, ragged row count,
  non-integer ids, 1-D/3-D logits)
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


def _histories(rng, b, n, max_len=64):
    """Ragged histories: empty rows, single-id rows, duplicates, and a
    long row (the bitmap's interesting shapes)."""
    out = []
    for r in range(b):
        kind = r % 4
        if kind == 0:
            out.append([])                          # no penalty at all
        elif kind == 1:
            out.append([int(rng.integers(0, n))])   # single id
        elif kind == 2:
            out.append([int(rng.integers(0, n))] * 7)   # duplicates
        else:
            out.append([int(v) for v in
                        rng.integers(0, n, size=max_len)])
    return out


def _logits(rng, b, n):
    x = rng.standard_normal((b, n)).astype(np.float32)
    for r in range(0, b, 3):
        x[r, 3] += 12.0                             # heavily peaked row
    for r in range(1, b, 3):
        x[r] *= 1e-3                                # near-uniform row
    return x


def _assert_row_close(logits_row, got, want, what):
    """Exact parity with the documented ulp fallback (same contract as
    the batched samplers: the global total's atomic accumulation order
    differs between launch shapes)."""
    if got == want:
        return
    order = np.argsort(-logits_row, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


def test_cpu_matches_rowwise_singles():
    rng = np.random.default_rng(300)
    b, n = 8, 4096
    x = _logits(rng, b, n)
    ids = _histories(rng, b, n)
    seeds = np.arange(b, dtype=np.int64)
    for penalty in (1.0, 1.3, 0.7):
        got = fusedtok.decode_step_batched(x, ids, penalty, p=0.9,
                                           seeds=seeds)
        want = [int(fusedtok.decode_step(x[r], ids[r], penalty, p=0.9,
                                         seed=int(s)))
                for r, s in enumerate(seeds)]
        assert got.tolist() == want, penalty


def test_cpu_flat_offsets_and_2d_agree_with_ragged():
    rng = np.random.default_rng(301)
    b, n = 6, 2048
    x = rng.standard_normal((b, n)).astype(np.float32)
    ids = _histories(rng, b, n)
    flat = [i for row in ids for i in row]
    offs = [0]
    for row in ids:
        offs.append(offs[-1] + len(row))
    base = fusedtok.decode_step_batched(x, ids, 1.2, seeds=np.arange(b))
    got_flat = fusedtok.decode_step_batched(
        x, np.array(flat, dtype=np.int64), 1.2,
        seeds=np.arange(b),
        ids_offsets=np.array(offs, dtype=np.int64))
    ids2d = np.array([[r * 10 + j for j in range(5)] for r in range(b)],
                     dtype=np.int64)
    got_2d = fusedtok.decode_step_batched(x, ids2d, 1.2,
                                          seeds=np.arange(b))
    want_2d = [int(fusedtok.decode_step(x[r], ids2d[r], 1.2, seed=r))
               for r in range(b)]
    assert got_flat.tolist() == base.tolist()
    assert got_2d.tolist() == want_2d


def test_disabled_penalty_reduces_to_topp_batched():
    # penalty 1.0 and all-empty histories both skip the bitmap
    # entirely - the call IS sample_topp_batched
    rng = np.random.default_rng(302)
    b, n = 8, 4096
    x = _logits(rng, b, n)
    ids = _histories(rng, b, n)
    seeds = np.arange(b, dtype=np.int64)
    a = fusedtok.decode_step_batched(x, ids, 1.0, seeds=seeds)
    e = fusedtok.decode_step_batched(x, [[] for _ in range(b)], 1.7,
                                     seeds=seeds)
    t = fusedtok.sample_topp_batched(x, 0.9, seeds=seeds)
    assert a.tolist() == t.tolist()
    assert e.tolist() == t.tolist()


def test_empty_batch():
    out = fusedtok.decode_step_batched(np.empty((0, 128), dtype=np.float32),
                                       [])
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_errors_cpu():
    x = np.zeros((4, 64), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(x, [[0]], 0.0)          # penalty
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(x, [[0]], 1.1, p=0.0)   # p bound
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(x, [[0]], 1.1, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(x, [[64]], 1.1)         # id == vocab
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(x, [[-1]], 1.1)         # id < 0
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(x, [[0], [0]], 1.1)     # row count
    with pytest.raises(ValueError):
        # offsets of the wrong length for the flat form
        fusedtok.decode_step_batched(
            x, np.array([0, 1], dtype=np.int64), 1.1,
            ids_offsets=np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError):
        # non-monotone offsets
        fusedtok.decode_step_batched(
            x, np.array([1, 0, 1], dtype=np.int64), 1.1,
            ids_offsets=np.array([0, 2, 1, 3], dtype=np.int64))
    with pytest.raises(TypeError):
        fusedtok.decode_step_batched(
            x, [[0.5], [], [1], [2]], 1.1)                    # float ids
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(np.zeros(64, dtype=np.float32),
                                     [[0]])                  # 1-D logits
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(
            np.zeros((2, 2, 8), dtype=np.float32), [[0], [0]])  # 3-D
    with pytest.raises(ValueError):
        fusedtok.decode_step_batched(x, [[0]], 1.1,
                                     seeds=np.arange(3, dtype=np.int64))


def test_validation_binding_surface():
    # the _fusedtok staged binding carries the same contract (the
    # 1.4.1 hardening pattern): shape mismatch, ids range, offsets
    ft = fusedtok._fusedtok
    x = np.ones((2, 8), dtype=np.float32)
    seeds = np.zeros(2, dtype=np.int64)
    good_ids = np.array([0, 3, 5], dtype=np.int64)
    good_offs = np.array([0, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        ft.decode_step_batched(x, 3, 8, good_ids, good_offs, 1.1, 0.9,
                               1.0, seeds)                    # rows*8 != 16
    with pytest.raises(ValueError):
        ft.decode_step_batched(np.ones((2, 8), dtype=np.float32), 2, 0,
                               good_ids, good_offs, 1.1, 0.9, 1.0, seeds)
    with pytest.raises(ValueError):
        ft.decode_step_batched(x, 2, 8, np.array([8], dtype=np.int64),
                               np.array([0, 1, 1], dtype=np.int64),
                               1.1, 0.9, 1.0, seeds)          # id == vocab
    with pytest.raises(ValueError):
        ft.decode_step_batched(x, 2, 8, good_ids,
                               np.array([0, 2], dtype=np.int64),
                               1.1, 0.9, 1.0, seeds)          # offs len


@needs_gpu
class TestCuda:
    def test_staged_and_zerocopy_match_singles(self):
        rng = np.random.default_rng(310)
        b, n = 8, 131072
        x = _logits(rng, b, n)
        ids = _histories(rng, b, n)
        seeds = np.arange(b, dtype=np.int64)
        for penalty in (1.0, 1.3, 0.7, 100.0):
            got = fusedtok.decode_step_batched(x, ids, penalty, p=0.9,
                                               seeds=seeds, cuda=True)
            dev = torch.from_numpy(x).cuda()
            zc = fusedtok.decode_step_batched(dev, ids, penalty, p=0.9,
                                              seeds=seeds)
            assert isinstance(zc, torch.Tensor) and zc.dtype == torch.int64
            assert zc.is_cpu
            for r in range(b):
                want = int(fusedtok.decode_step(
                    dev[r], ids[r], penalty, p=0.9, seed=int(seeds[r])))
                _assert_row_close(x[r], int(got[r]), want,
                                  ("staged", penalty, r))
                _assert_row_close(x[r], int(zc[r]), want,
                                  ("zerocopy", penalty, r))

    def test_single_row_matches_single_api_exactly(self):
        # B = 1 sees the same grid shape as the standalone call, so
        # even the atomic-order totals coincide - exact equality
        rng = np.random.default_rng(311)
        n = 131072
        x = _logits(rng, 1, n)
        ids = [[7, 12, 12, 900]]
        got = fusedtok.decode_step_batched(
            x, ids, 1.4, seeds=np.zeros(1, dtype=np.int64), cuda=True)
        want = int(fusedtok.decode_step(x[0], ids[0], 1.4, seed=0,
                                        cuda=True))
        assert int(got[0]) == want

    def test_b33_chunk_boundary(self):
        rng = np.random.default_rng(312)
        b, n = 33, 131072
        x = _logits(rng, b, n)
        ids = _histories(rng, b, n)
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.decode_step_batched(x, ids, 1.25, seeds=seeds,
                                           cuda=True)
        dev = torch.from_numpy(x).cuda()
        for r in (0, 1, 2, 3, 16, 31, 32):
            want = int(fusedtok.decode_step(dev[r], ids[r], 1.25,
                                            seed=int(seeds[r])))
            _assert_row_close(x[r], int(got[r]), want, ("b33", r))

    def test_qwen_vocabulary(self):
        rng = np.random.default_rng(313)
        b, n = 4, 152064
        x = _logits(rng, b, n)
        ids = _histories(rng, b, n)
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.decode_step_batched(x, ids, 1.2, seeds=seeds,
                                           cuda=True)
        assert all(0 <= int(t) < n for t in got)
        dev = torch.from_numpy(x).cuda()
        for r in range(b):
            want = int(fusedtok.decode_step(dev[r], ids[r], 1.2,
                                            seed=int(seeds[r])))
            _assert_row_close(x[r], int(got[r]), want, ("qwen", r))

    def test_generation_loop_histories_grow(self):
        # 24 steps: every row appends its token; the heavy penalty must
        # break greedy repetition on the peaked rows
        rng = np.random.default_rng(314)
        b, n = 4, 8192
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(b):
            x[r, 5] += 10.0
        dev = torch.from_numpy(x).cuda()
        hist = [[] for _ in range(b)]
        distinct = {r: set() for r in range(b)}
        for step in range(24):
            toks = fusedtok.decode_step_batched(dev, hist, 8.0,
                                                seeds=np.arange(b))
            for r in range(b):
                hist[r].append(int(toks[r]))
                distinct[r].add(int(toks[r]))
        # a penalty of 8 pushes draws off the repeated top token often
        assert all(len(distinct[r]) >= 3 for r in range(b))

    def test_temperature_extremes_and_p_one(self):
        rng = np.random.default_rng(315)
        b, n = 6, 131072
        x = _logits(rng, b, n)
        ids = _histories(rng, b, n)
        seeds = np.arange(b, dtype=np.int64)
        dev = torch.from_numpy(x).cuda()
        # T -> 0 collapses each row to the penalized argmax: exact
        got = fusedtok.decode_step_batched(x, ids, 1.3, p=0.9,
                                           temperature=1e-4,
                                           seeds=seeds, cuda=True)
        for r in range(b):
            want = int(fusedtok.decode_step(
                dev[r], ids[r], 1.3, p=0.9, temperature=1e-4,
                seed=int(seeds[r])))
            assert int(got[r]) == want, r
        # T huge: every row widens to the full vocabulary
        got = fusedtok.decode_step_batched(x, ids, 1.3, p=0.9,
                                           temperature=1e4,
                                           seeds=seeds, cuda=True)
        assert all(0 <= int(t) < n for t in got)
        # p = 1.0: the whole distribution is the nucleus
        got = fusedtok.decode_step_batched(x, ids, 1.3, p=1.0,
                                           seeds=seeds, cuda=True)
        for r in range(b):
            want = int(fusedtok.decode_step(dev[r], ids[r], 1.3, p=1.0,
                                            seed=int(seeds[r])))
            _assert_row_close(x[r], int(got[r]), want, ("p1", r))

    def test_self_determinism_and_interleaving(self):
        rng = np.random.default_rng(316)
        b, n = 8, 131072
        x = _logits(rng, b, n)
        ids = _histories(rng, b, n)
        seeds = np.arange(b, dtype=np.int64)
        first = fusedtok.decode_step_batched(x, ids, 1.2, seeds=seeds,
                                             cuda=True)
        for _ in range(3):
            assert fusedtok.decode_step_batched(
                x, ids, 1.2, seeds=seeds, cuda=True).tolist() \
                == first.tolist()
        # interleave across workspace reallocations: the batched
        # samplers, argmax and singles all share the selection
        # workspace grower
        dev = torch.from_numpy(x).cuda()
        wide = fusedtok.sample_topp_batched(dev, 0.9, seeds=seeds)
        assert int(fusedtok.argmax(dev[0])) == 3
        again = fusedtok.decode_step_batched(x, ids, 1.2, seeds=seeds,
                                             cuda=True)
        assert again.tolist() == first.tolist()
        assert fusedtok.sample_topk_batched(dev, 50,
                                            seeds=seeds).shape == (b,)

    def test_cuda_ids_tensor_and_torch_output(self):
        rng = np.random.default_rng(317)
        b, n = 4, 4096
        x = rng.standard_normal((b, n)).astype(np.float32)
        ids_np = np.arange(b * 3, dtype=np.int64).reshape(b, 3) % n
        dev = torch.from_numpy(x).cuda()
        ids_t = torch.from_numpy(ids_np).cuda()
        got = fusedtok.decode_step_batched(dev, ids_t, 1.3,
                                           seeds=np.arange(b))
        assert isinstance(got, torch.Tensor)
        want = [int(fusedtok.decode_step(dev[r], ids_np[r], 1.3, seed=r))
                for r in range(b)]
        for r in range(b):
            _assert_row_close(x[r], int(got[r]), want[r], ("torchin", r))
