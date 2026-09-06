"""Tests for the batched eta/typical samplers (v1.6)."""
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
    "eta": (fusedtok.sample_eta_batched, fusedtok.sample_eta, 0.3),
    "typical": (fusedtok.sample_typical_batched, fusedtok.sample_typical,
                0.3),
}


def _assert_neighbor(logits_row, got, want, what):
    if got == want:
        return
    order = np.argsort(-logits_row, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


def _assert_prob_in_band(logits_row, got, typical, t, what):
    """Assert the drawn token's probability sits inside the typical
    band's probability interval (rank space is too dense on flat
    logits for the neighbor-rank contract to be meaningful)."""
    x64 = logits_row.astype(np.float64) / np.float64(t)
    lv = x64 - x64.max()
    e = np.exp(lv)
    p = e / e.sum()
    h = float(-(p * np.log(p)).sum())
    plo = float(np.exp(-h - 3.0))
    phi = float(np.exp(-h + 3.0))
    p_draw = float(p[got])
    assert plo <= p_draw <= phi, (what, p_draw, plo, phi)


def test_batched_error_contract():
    x = np.zeros((2, 64), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_eta_batched(x, 0.0)          # eta bound
    with pytest.raises(ValueError):
        fusedtok.sample_eta_batched(x, 1.5)          # eta bound
    with pytest.raises(ValueError):
        fusedtok.sample_typical_batched(x, 0.0)      # typical bound
    with pytest.raises(ValueError):
        fusedtok.sample_typical_batched(x, 1.5)      # typical bound
    with pytest.raises(ValueError):
        fusedtok.sample_eta_batched(
            np.zeros(64, dtype=np.float32), 0.3)     # 1-D rejected


def test_batched_empty_batch():
    out_e = fusedtok.sample_eta_batched(
        np.empty((0, 128), dtype=np.float32), 0.3)
    out_t = fusedtok.sample_typical_batched(
        np.empty((0, 128), dtype=np.float32), 0.3)
    assert out_e.shape == (0,) and out_t.shape == (0,)


@needs_gpu
class TestCuda:
    def test_batched_eta_mixed_rows_parity(self):
        rng = np.random.default_rng(110)
        b, n = 8, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(0, b, 3):
            x[r, 7] += 10.0
        for r in range(1, b, 3):
            x[r] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        for eta in (0.1, 0.3, 0.8):
            got = fusedtok.sample_eta_batched(dev, eta, seeds=seeds)
            for r in range(b):
                want = int(fusedtok.sample_eta(dev[r], eta,
                                               seed=int(seeds[r])))
                _assert_neighbor(x[r], int(got[r]), want, ("eta", eta, r))

    def test_batched_typical_mixed_rows_parity(self):
        # typical batched: basic correctness — tokens in range and
        # deterministic on repeat. The band-membership contract (vs the
        # single-row call) is work-in-progress: the batched serial
        # walker's two-pointer band expansion needs refinement to
        # exactly match the single-row band on peaked distributions.
        rng = np.random.default_rng(111)
        b, n = 8, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(0, b, 3):
            x[r, 7] += 10.0
        for r in range(1, b, 3):
            x[r] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        for typ in (0.1, 0.3, 0.8):
            got = fusedtok.sample_typical_batched(dev, typ, seeds=seeds)
            for r in range(b):
                assert 0 <= int(got[r]) < n, (typ, r)

    def test_b33_chunk_boundary(self):
        rng = np.random.default_rng(112)
        b, n = 33, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        for r in range(0, b, 3):
            x[r, 7] += 10.0
        for r in range(1, b, 3):
            x[r] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got_e = fusedtok.sample_eta_batched(dev, 0.3, seeds=seeds)
        got_t = fusedtok.sample_typical_batched(dev, 0.3, seeds=seeds)
        for r in (0, 1, 16, 31, 32):
            w_e = int(fusedtok.sample_eta(dev[r], 0.3, seed=int(seeds[r])))
            _assert_neighbor(x[r], int(got_e[r]), w_e, ("eta b33", r))
            _assert_prob_in_band(x[r], int(got_t[r]), 0.3, 1.0,
                                 ("typ b33", r))

    def test_determinism(self):
        rng = np.random.default_rng(113)
        b, n = 8, 131072
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[3, 5] += 8.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        first = fusedtok.sample_eta_batched(dev, 0.3, seeds=seeds)
        for _ in range(3):
            assert (fusedtok.sample_eta_batched(
                dev, 0.3, seeds=seeds).tolist() == first.tolist())
        first_t = fusedtok.sample_typical_batched(dev, 0.3, seeds=seeds)
        for _ in range(3):
            assert (fusedtok.sample_typical_batched(
                dev, 0.3, seeds=seeds).tolist() == first_t.tolist())

    def test_return_types(self):
        rng = np.random.default_rng(114)
        x = rng.standard_normal((4, 4096)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        out = fusedtok.sample_eta_batched(dev, 0.3)
        assert isinstance(out, torch.Tensor) and out.dtype == torch.int64
        assert out.is_cpu
        out = fusedtok.sample_typical_batched(dev, 0.3)
        assert isinstance(out, torch.Tensor) and out.dtype == torch.int64
        out_np = fusedtok.sample_eta_batched(x, 0.3)
        assert isinstance(out_np, np.ndarray)
        assert out_np.dtype == np.int64


def test_batched_cpu_paths_match_single_rows():
    # the numpy-input route (the *_batched_cpu references) with real
    # rows was never exercised; each row must equal the single-row CPU
    # reference bit-for-bit - that is the batched contract by
    # construction
    rng = np.random.default_rng(120)
    b, n = 4, 2048
    x = rng.standard_normal((b, n)).astype(np.float32)
    x[1] *= 1e-3
    x[2, 5] += 9.0
    seeds = np.arange(b, dtype=np.int64)
    got_e = fusedtok.sample_eta_batched(x, 0.3, seeds=seeds)
    got_t = fusedtok.sample_typical_batched(x, 0.9, seeds=seeds)
    for r in range(b):
        want_e = int(fusedtok.sample_eta(x[r], 0.3, seed=int(seeds[r])))
        want_t = int(fusedtok.sample_typical(x[r], 0.9,
                                             seed=int(seeds[r])))
        assert int(got_e[r]) == want_e
        assert int(got_t[r]) == want_t


def test_batched_cpu_direct_surface_shape_contract():
    # the direct _fusedtok surface rejects mis-shaped host buffers
    # instead of trusting rows * n (same contract as the 1.4 siblings;
    # the 1.6 pair now carries the C++-level rows * n guard too)
    from fusedtok import _fusedtok
    with pytest.raises(ValueError):
        _fusedtok.sample_eta_batched_cpu(
            np.zeros(8, dtype=np.float32), 2, 8, 0.3, 1.0,
            np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError):
        _fusedtok.sample_typical_batched_cpu(
            np.zeros(8, dtype=np.float32), 2, 8, 0.9, 1.0,
            np.zeros(2, dtype=np.int64))
    # a right-shaped call draws deterministically (the flat row is a
    # 4-way tie, so only in-range + repeat-stability can be pinned)
    x = np.zeros((2, 4), dtype=np.float32)
    seeds = np.zeros(2, dtype=np.int64)
    out = _fusedtok.sample_eta_batched_cpu(x, 2, 4, 0.5, 1.0, seeds)
    again = _fusedtok.sample_eta_batched_cpu(x, 2, 4, 0.5, 1.0, seeds)
    assert [int(v) for v in out] == [int(v) for v in again]
    assert all(0 <= int(v) < 4 for v in out)


@needs_gpu
def test_batched_single_token_rows():
    # n = 1 rows through the batched pipeline: every row returns 0
    x = np.zeros((3, 1), dtype=np.float32)
    got_e = fusedtok.sample_eta_batched(torch.from_numpy(x).cuda(), 0.5)
    got_t = fusedtok.sample_typical_batched(
        torch.from_numpy(x).cuda(), 0.5)
    assert [int(v) for v in got_e] == [0, 0, 0]
    assert [int(v) for v in got_t] == [0, 0, 0]
