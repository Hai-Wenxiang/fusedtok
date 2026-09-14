"""sample_dry (v2.3): fused DRY (Don't Repeat Yourself) sampling.

Sequence-aware repeat penalty + temperature + full-softmax draw.
Contract (documented in the wrapper docstring and docs/*/sampling.md):

- scan window = the most recent kDryMaxScan = 64 history tokens
- for every suffix length L in [allowed_length, window): each earlier
  in-window occurrence of the current L-suffix contributes the token
  that followed it; the token keeps the MAX exponent
  (L - allowed_length + 1)
- application: positive logits divide by multiplier ** exponent,
  negative logits multiply (repetition_penalty convention)
- draw: plain softmax inverse-CDF (the min_p=1e-9 pipeline),
  deterministic per seed

Cases: hand-computed penalty tables on crafted histories, the
no-history and multiplier=1.0 no-ops, negative-logit handling,
per-seed determinism, CPU/staged/zero-copy parity (with the documented
neighbor-rank fallback), batched parity incl. ragged + 2-D histories,
the B=33 chunk boundary, empty batch, and the error contract.
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


def _assert_neighbor_rank(logits_row, got, want, what):
    """Exact parity with the documented ulp fallback: the GPU walk's
    total differs by float rounding from the CPU's, so a draw landing
    exactly on a CDF boundary may pick a neighbor rank."""
    if got == want:
        return
    order = np.argsort(-logits_row, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


def _composed_dry(logits, history, allowed_length, multiplier, t, seed):
    """The documented contract, written out with numpy: penalize, then
    draw from the full softmax (the sample_xtc(top_n=0) reference
    path). Used as the composed reference for parity checks."""
    y = logits.copy()
    W = min(len(history), 64)
    exponent = np.zeros(logits.shape[0], dtype=np.int64)
    for L in range(allowed_length, W):
        suffix = history[len(history) - L:]
        for j in range(0, W - L):
            if list(history[j:j + L]) == list(suffix):
                c = history[j + L]
                exponent[c] = max(exponent[c], L - allowed_length + 1)
    for c in np.flatnonzero(exponent):
        pf = float(multiplier) ** int(exponent[c])
        y[c] = y[c] / pf if y[c] > 0 else y[c] * pf
    return fusedtok.sample_xtc(y, 0, 0.0, temperature=t, seed=seed)


def test_dry_matches_composed_reference():
    rng = np.random.default_rng(230)
    logits = (rng.standard_normal(2048) * 2).astype(np.float32)
    history = [7, 8, 9, 7, 8, 9, 7]
    for seed in range(8):
        want = _composed_dry(logits, history, 2, 1.75, 0.8, seed)
        got = fusedtok.sample_dry(logits, history, 2, 1.75,
                                  temperature=0.8, seed=seed)
        _assert_neighbor_rank(logits, got, want, seed)


def test_dry_penalty_table_exact():
    # hand-computed exponents on a crafted history:
    # window = [4, 5, 4, 5] (allowed_length=2):
    #   L=2: suffix [4,5] occurs at j=0 with follower 4  -> exp(4) = 1
    #   L=3: suffix [5,4,5] has no earlier occurrence with room
    # so ONLY token 4 is penalized with exponent 1. Verify through a
    # two-candidate distribution where the draw must flip when 4's
    # logit drops below its rival's.
    # temperature 0.01 makes each comparison deterministic (the
    # runner-up is ~24 softmax-logits behind at this scale)
    t = 0.01
    logits = np.full(64, -20.0, dtype=np.float32)
    logits[4] = 1.0
    logits[5] = 0.5
    hist = [4, 5, 4, 5]
    # multiplier 1.0: no penalty -> token 4 (logit 1.0) always wins
    draws = {fusedtok.sample_dry(logits, hist, 2, 1.0, temperature=t,
                                 seed=s) for s in range(16)}
    assert draws == {4}
    # exponent 1 on token 4: multiplier 3.9 -> logit 1/3.9 = 0.26 < 0.5,
    # so token 5 always wins
    draws = {fusedtok.sample_dry(logits, hist, 2, 3.9, temperature=t,
                                 seed=s) for s in range(16)}
    assert draws == {5}
    # and with multiplier 1.5 -> 1/1.5 = 0.67 > 0.5, token 4 holds
    draws = {fusedtok.sample_dry(logits, hist, 2, 1.5, temperature=t,
                                 seed=s) for s in range(16)}
    assert draws == {4}


def test_dry_exponent_grows_with_match_length():
    # window = [7, 7, 7, 7], allowed_length=1: token 7 keeps the max
    # exponent over L = 1..3 -> e = 3, penalty = multiplier ** 3. With
    # logit(7) = ln(4) and multiplier 2 the penalized logit is
    # ln(4) - ln(8) = ln(0.5): a uniform 4-way draw now never picks 7
    # deterministically at these seeds... instead verify the flip
    # threshold: multiplier just below 4^(1/3) must still pick 7.
    # temperature 0.01: each seed's draw is deterministic once the
    # penalized logit ordering is fixed (the -20 background sits ~2000
    # logits behind and never draws)
    t = 0.01
    logits = np.full(64, -20.0, dtype=np.float32)
    logits[7] = np.log(4.0)                 # 1.386
    for r in (1, 2, 3, 4):
        logits[r] = 0.3                     # the field, tightly packed
    hist = [7, 7, 7, 7]
    # max exponent is e = 3 (L = 3), so the penalized logit is
    # ln(4) / multiplier**3: 1.5**3 = 3.375 -> 0.41 > 0.3, 7 keeps the
    # lead on every seed; 1.8**3 = 5.832 -> 0.24 < 0.3, the field wins
    # the overwhelming majority
    draws = [fusedtok.sample_dry(logits, hist, 1, 1.5, temperature=t,
                                 seed=s) for s in range(48)]
    assert set(draws) == {7}
    draws = [fusedtok.sample_dry(logits, hist, 1, 1.8, temperature=t,
                                 seed=s) for s in range(48)]
    assert sum(1 for d in draws if d != 7) > 24
    assert set(draws) <= {1, 2, 3, 4, 7}


def test_dry_noop_paths():
    rng = np.random.default_rng(231)
    logits = rng.standard_normal(1024).astype(np.float32)
    # no history -> plain full-softmax draw (the xtc top_n=0 reference)
    for seed in range(8):
        want = fusedtok.sample_xtc(logits, 0, 0.0, seed=seed)
        got = fusedtok.sample_dry(logits, [], 2, 1.75, seed=seed)
        assert got == want, seed
    # multiplier = 1.0 -> penalty factor 1 -> identical draws
    history = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    for seed in range(8):
        want = fusedtok.sample_dry(logits, history, 2, 1.0, seed=seed)
        got = fusedtok.sample_dry(logits, [], 2, 1.75, seed=seed)
        assert got == want, seed
    # allowed_length beyond the window -> no suffix lengths to scan
    got = fusedtok.sample_dry(logits, history, 64, 1.75, seed=3)
    want = fusedtok.sample_xtc(logits, 0, 0.0, seed=3)
    assert got == want


def test_dry_negative_logit_penalty():
    # the repetition_penalty convention on a negative logit: multiply
    # (magnitude shrinks), which RAISES it toward the field - the
    # documented behavior must match the composed reference exactly.
    rng = np.random.default_rng(232)
    logits = (-np.abs(rng.standard_normal(512)) - 0.5).astype(np.float32)
    history = [11, 12, 11, 12]
    for seed in range(6):
        want = _composed_dry(logits, history, 2, 2.0, 1.0, seed)
        got = fusedtok.sample_dry(logits, history, 2, 2.0, seed=seed)
        _assert_neighbor_rank(logits, got, want, seed)


def test_dry_determinism():
    rng = np.random.default_rng(233)
    logits = rng.standard_normal(2048).astype(np.float32)
    history = [9, 9, 9, 1, 2, 3, 1, 2, 3]
    for seed in range(12):
        a = fusedtok.sample_dry(logits, history, 2, 1.75, seed=seed)
        b = fusedtok.sample_dry(logits, history, 2, 1.75, seed=seed)
        assert a == b, seed


def test_dry_error_contract():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_dry(x, [], 0, 1.75)          # allowed_length < 1
    with pytest.raises(ValueError):
        fusedtok.sample_dry(x, [], 65, 1.75)         # beyond the window cap
    with pytest.raises(ValueError):
        fusedtok.sample_dry(x, [], 2, 0.99)          # multiplier < 1
    with pytest.raises(ValueError):
        fusedtok.sample_dry(x, [], 2, 1.75, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_dry(x, [8], 2, 1.75)         # id out of range
    with pytest.raises(ValueError):
        fusedtok.sample_dry(x, [-1], 2, 1.75)
    with pytest.raises(ValueError):
        fusedtok.sample_dry(np.ones((2, 2), dtype=np.float32), [], 2, 1.75)
    b = np.ones((2, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_dry_batched(b, [[0], [8]], 2, 1.75)  # id out of range
    with pytest.raises(ValueError):
        fusedtok.sample_dry_batched(b, [[0]], 2, 1.75)       # rows mismatch


def test_batched_cpu_matches_rowwise_singles():
    rng = np.random.default_rng(234)
    x = rng.standard_normal((5, 4096)).astype(np.float32)
    x[1, 7] += 8.0
    hists = [[5, 9, 5, 9], [], [7] * 8, [1, 2, 3], [4, 4, 4, 4, 4, 4]]
    seeds = np.arange(5, dtype=np.int64)
    got = fusedtok.sample_dry_batched(x, hists, 2, 1.75, seeds=seeds)
    for r in range(5):
        want = fusedtok.sample_dry(x[r], hists[r], 2, 1.75,
                                   seed=int(seeds[r]))
        assert int(got[r]) == want, r


@needs_gpu
class TestCuda:
    def test_staged_and_zerocopy_match_cpu(self):
        rng = np.random.default_rng(235)
        logits = (rng.standard_normal(4096) * 2).astype(np.float32)
        history = [7, 8, 9, 7, 8, 9, 7, 8, 9]
        dev = torch.from_numpy(logits).cuda()
        ht = torch.tensor(history, dtype=torch.int64).cuda()
        for seed in range(10):
            want = fusedtok.sample_dry(logits, history, 3, 1.6, seed=seed)
            staged = fusedtok.sample_dry(logits, history, 3, 1.6, seed=seed,
                                         cuda=True)
            zero = int(fusedtok.sample_dry(dev, ht, 3, 1.6, seed=seed))
            _assert_neighbor_rank(logits, staged, want, ("staged", seed))
            _assert_neighbor_rank(logits, zero, want, ("zero", seed))

    def test_batched_matches_single_rows(self):
        rng = np.random.default_rng(236)
        b, n = 6, 4096
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[0, 7] += 10.0
        hists = [[5, 9, 5, 9], [], [7] * 8, [1, 2, 3], [], [6, 6, 6]]
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_dry_batched(dev, hists, 2, 1.75, seeds=seeds)
        for r in range(b):
            want = int(fusedtok.sample_dry(x[r], hists[r], 2, 1.75,
                                           seed=int(seeds[r])))
            _assert_neighbor_rank(x[r], int(got[r]), want, r)

    def test_batched_chunk_boundary_33(self):
        # 33 rows crosses the kBMaxBatch = 32 device chunk boundary of
        # the batched draw; the rewrite pass itself is chunk-free, so
        # both halves must track the row-wise singles.
        rng = np.random.default_rng(237)
        b, n = 33, 2048
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[0, 5] += 8.0
        hists = [[3, 1, 4, 1, 5] if r % 3 else [9, 9, 9, 9] for r in range(b)]
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_dry_batched(dev, hists, 2, 1.75, seeds=seeds)
        assert got.shape[0] == b
        for r in (0, 1, 16, 31, 32):
            want = int(fusedtok.sample_dry(x[r], hists[r], 2, 1.75,
                                           seed=r))
            _assert_neighbor_rank(x[r], int(got[r]), want, r)

    def test_batched_determinism_and_types(self):
        rng = np.random.default_rng(238)
        x = rng.standard_normal((4, 2048)).astype(np.float32)
        hists = [[5], [5, 5, 5], [], [1, 2, 1, 2]]
        dev = torch.from_numpy(x).cuda()
        first = fusedtok.sample_dry_batched(dev, hists, 2, 1.75)
        for _ in range(3):
            assert (fusedtok.sample_dry_batched(dev, hists, 2, 1.75)
                    .tolist() == first.tolist())
        out_np = fusedtok.sample_dry_batched(x, hists, 2, 1.75)
        assert isinstance(out_np, np.ndarray)
        assert isinstance(first, torch.Tensor)

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_dry(x, [], 0, 1.75)
        with pytest.raises(ValueError):
            fusedtok.sample_dry(x, [], 2, 0.5)
        with pytest.raises(TypeError):
            fusedtok.sample_dry(x.to(torch.bfloat16), [], 2, 1.75)
        with pytest.raises(ValueError):
            fusedtok.sample_dry(torch.ones(2, 2, device="cuda"), [], 2,
                                1.75)
        # device-resident ids are trusted (documented boundary); host
        # ids are validated
        with pytest.raises(ValueError):
            fusedtok.sample_dry(x, [8], 2, 1.75)
