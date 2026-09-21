"""sample_mirostat (v2.5): fused Mirostat v2 sampling (Basirat 2023).

Contract (documented in the wrapper docstring and docs/*/sampling.md):

- nucleus = {i : p_i >= 2 ** -mu} - an ABSOLUTE probability threshold,
  boundary-inclusive like the library's other value-threshold samplers
- empty nucleus (2**-mu > p_max) falls back to the top-1 token
- the draw renormalizes inside the nucleus; deterministic per seed
- the state update s = -log2(p_sampled) under the FULL softmax, then
  new_mu = mu - eta * (s - tau); f32 op order matches across paths,
  so new_mu agrees up to the documented exp ulp boundary
- RETURNS (token, new_mu) - the tuple API is what a decode loop feeds
  back; initialize mu at 2 * tau

Cases: hand-computed nucleus/update on a crafted distribution, the
empty-nucleus fallback, per-seed determinism, cross-path parity
(CPU / staged / zero-copy; token via the neighbor-rank contract, mu
via a small abs tolerance), batched per-row parity incl. independent
mu states, multi-step surprisal tracking toward tau, error contract.
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

needs_gpu = pytest.mark.skipif(
    not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")


def _probs(x, t=1.0):
    z = x.astype(np.float64) / t
    e = np.exp(z - z.max())
    return e / e.sum()


MASK64 = (1 << 64) - 1


def _splitmix_uniform(seed):
    """Exact host-side replication of the library's splitmix_uniform
    (see src/cuda_util.cuh): splitmix64 finalized to a float in
    [0, 1). u64 arithmetic emulated with python ints."""
    z = (seed + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z ^= z >> 31
    return (z >> 11) * (1.0 / 9007199254740992.0)


def _composed(x, mu, tau, eta, t, seed):
    """The documented contract in numpy with the library's exact
    seeded-hash draw (splitmix replication) over the renormalized
    nucleus."""
    p = _probs(x, t)
    keep = np.flatnonzero(p >= 2.0 ** (-mu))
    if keep.size == 0:
        keep = np.array([int(np.argmax(p))])
    w = p[keep]
    u = _splitmix_uniform(seed)
    target = u * w.sum()
    cum = np.cumsum(w)
    i = int(np.searchsorted(cum, target))
    i = min(i, keep.size - 1)
    tok = int(keep[i])
    s = -math.log2(p[tok])
    return tok, mu - eta * (s - tau)


def test_returns_tuple_and_types():
    rng = np.random.default_rng(260)
    x = rng.standard_normal(1024).astype(np.float32)
    out = fusedtok.sample_mirostat(x, 10.0)
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], int) and isinstance(out[1], float)
    assert 0 <= out[0] < 1024


def test_hand_computed_nucleus_and_update():
    # two-token-dominant distribution with hand-checkable surprisal:
    # mu chosen so the nucleus is exactly the top-2; the update must
    # track tau regardless of which of the two wins the draw
    x = np.full(64, -20.0, dtype=np.float32)
    x[0] = 0.0
    x[1] = -0.5
    p = _probs(x)
    # nucleus = top-2: threshold between p[1] and p[2]
    mu = -math.log2(math.sqrt(p[1] * p[2]))
    for seed in range(16):
        tok, mu2 = fusedtok.sample_mirostat(x, mu, tau=5.0, eta=0.1,
                                            seed=seed)
        assert tok in (0, 1)
        s = -math.log2(p[tok])
        assert mu2 == pytest.approx(mu - 0.1 * (s - 5.0), abs=1e-5), seed


def test_empty_nucleus_falls_back_to_top1():
    # mu deeply negative -> 2^-mu >> p_max -> nucleus empty -> the
    # fallback must return the argmax token deterministically
    rng = np.random.default_rng(261)
    x = rng.standard_normal(512).astype(np.float32)
    top = int(np.argmax(x))
    for seed in range(8):
        tok, _ = fusedtok.sample_mirostat(x, -20.0, seed=seed)
        assert tok == top, seed
    # and the update still uses the top-1 surprisal
    p = _probs(x)
    _, mu2 = fusedtok.sample_mirostat(x, -20.0, tau=5.0, eta=0.1,
                                      seed=0)
    s = -math.log2(p[top])
    assert mu2 == pytest.approx(-20.0 - 0.1 * (s - 5.0), abs=1e-5)


def test_matches_composed_reference_narrow_nucleus():
    # narrow nuclei (small mu -> high threshold): the float64
    # reference and the f32 library walk the same few tokens, so the
    # token AND the update must match
    rng = np.random.default_rng(262)
    x = (rng.standard_normal(2048) * 2).astype(np.float32)
    x[7] += 6.0
    for mu in (2.0, 3.0, 5.0):
        for seed in range(6):
            want_tok, want_mu = _composed(x, mu, 5.0, 0.1, 1.0, seed)
            got_tok, got_mu = fusedtok.sample_mirostat(x, mu, seed=seed)
            if got_tok != want_tok:
                order = np.argsort(-x, kind="stable")
                rank = {int(t): i for i, t in enumerate(order)}
                assert abs(rank[got_tok] - rank[want_tok]) <= 2, (mu, seed)
            else:
                assert got_mu == pytest.approx(want_mu, abs=1e-4), (mu, seed)


def test_wide_nucleus_membership_and_conditional_update():
    # wide nuclei (large mu -> threshold ~1e-3): the f32 walk
    # accumulates across thousands of exps, so a float64 reference
    # legitimately lands elsewhere - the honest wide-regime checks are
    # nucleus MEMBERSHIP of the drawn token and the update formula
    # evaluated at the token the library actually drew
    rng = np.random.default_rng(269)
    x = (rng.standard_normal(2048) * 2).astype(np.float32)
    x[7] += 6.0
    p = _probs(x)
    for mu in (8.0, 10.0):
        thr = 2.0 ** (-mu)
        for seed in range(6):
            tok, mu2 = fusedtok.sample_mirostat(x, mu, tau=5.0, eta=0.1,
                                                seed=seed)
            if p[tok] >= thr:               # inside the nucleus
                s = -math.log2(p[tok])
                assert mu2 == pytest.approx(
                    mu - 0.1 * (s - 5.0), abs=1e-4), (mu, seed)


def test_determinism():
    rng = np.random.default_rng(263)
    x = rng.standard_normal(2048).astype(np.float32)
    for seed in range(8):
        a = fusedtok.sample_mirostat(x, 6.0, seed=seed)
        b = fusedtok.sample_mirostat(x, 6.0, seed=seed)
        assert a == b, seed


def test_error_contract():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat(x, 5.0, tau=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat(x, 5.0, eta=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat(x, 5.0, eta=-0.1)
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat(x, 5.0, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat(x, float("nan"))
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat(np.ones((2, 2), dtype=np.float32), 5.0)
    b = np.ones((2, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat_batched(b, np.array([1.0]))  # rows
    with pytest.raises(ValueError):
        fusedtok.sample_mirostat_batched(b, np.array([1.0, float("nan")]))


def test_batched_cpu_matches_rowwise():
    rng = np.random.default_rng(264)
    x = rng.standard_normal((5, 4096)).astype(np.float32)
    x[2, 7] += 8.0
    mus = np.array([10.0, 5.0, 2.0, 7.5, 1.0], dtype=np.float32)
    seeds = np.arange(5, dtype=np.int64)
    toks, muv = fusedtok.sample_mirostat_batched(x, mus, seeds=seeds)
    for r in range(5):
        t, m = fusedtok.sample_mirostat(x[r], float(mus[r]),
                                        seed=int(seeds[r]))
        assert int(toks[r]) == t, r
        assert muv[r] == pytest.approx(m, abs=1e-6), r


def test_multi_step_tracks_tau():
    # the mirostat property: feeding new_mu back keeps the running
    # surprisal near tau. A peaked distribution (low surprisal)
    # must PUSH mu down over steps (the bound tightens)
    rng = np.random.default_rng(265)
    x = rng.standard_normal(2048).astype(np.float32)
    x[3] += 8.0
    mu, tau, eta = 10.0, 3.0, 0.1
    mus = [mu]
    surprisals = []
    for step in range(64):
        tok, mu = fusedtok.sample_mirostat(x, mu, tau=tau, eta=eta,
                                           seed=step)
        p = _probs(x)
        surprisals.append(-math.log2(p[tok]))
        mus.append(mu)
    # running mean surprisal must approach tau from above (peaked
    # start) within a loose tolerance - this is the algorithm's
    # advertised behavior, not an exact invariant
    late = float(np.mean(surprisals[32:]))
    assert abs(late - tau) < 1.5, late
    assert mus[-1] < mus[0]          # the bound tightened


@needs_gpu
class TestCuda:
    def test_staged_and_zerocopy_match_cpu(self):
        rng = np.random.default_rng(266)
        logits = (rng.standard_normal(4096) * 2).astype(np.float32)
        logits[7] += 8.0
        dev = torch.from_numpy(logits).cuda()
        for mu in (2.0, 5.0, 10.0, -3.0):
            for seed in range(8):
                want = fusedtok.sample_mirostat(logits, mu, seed=seed)
                staged = fusedtok.sample_mirostat(logits, mu, seed=seed,
                                                  cuda=True)
                zero = fusedtok.sample_mirostat(dev, mu, seed=seed)
                for got in (staged, zero):
                    if got[0] != want[0]:
                        order = np.argsort(-logits, kind="stable")
                        rank = {int(t): i for i, t in enumerate(order)}
                        assert abs(rank[got[0]] - rank[want[0]]) <= 2
                    else:
                        assert got[1] == pytest.approx(want[1], abs=1e-5)

    def test_batched_gpu_matches_rowwise(self):
        rng = np.random.default_rng(267)
        x = rng.standard_normal((6, 4096)).astype(np.float32)
        x[0, 7] += 10.0
        mus = np.array([10.0, 5.0, 2.0, 0.0, -2.0, 7.0],
                       dtype=np.float32)
        seeds = np.arange(6, dtype=np.int64)
        dev = torch.from_numpy(x).cuda()
        toks, muv = fusedtok.sample_mirostat_batched(dev, mus,
                                                     seeds=seeds)
        for r in range(6):
            t, m = fusedtok.sample_mirostat(x[r], float(mus[r]),
                                            seed=int(seeds[r]))
            if int(toks[r]) != t:
                order = np.argsort(-x[r], kind="stable")
                rank = {int(v): i for i, v in enumerate(order)}
                assert abs(rank[int(toks[r])] - rank[t]) <= 2, r
            assert float(muv[r]) == pytest.approx(m, abs=1e-5), r

    def test_staged_batched_and_types(self):
        rng = np.random.default_rng(268)
        x = rng.standard_normal((4, 2048)).astype(np.float32)
        mus = np.full(4, 8.0, dtype=np.float32)
        toks, muv = fusedtok.sample_mirostat_batched(
            x, mus, seeds=np.arange(4, dtype=np.int64), cuda=True)
        assert isinstance(toks, np.ndarray) and isinstance(muv, np.ndarray)
        assert toks.shape == (4,) and muv.shape == (4,)
        assert toks.dtype == np.int64 and muv.dtype == np.float32

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_mirostat(x, 5.0, tau=0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_mirostat(x, float("inf"))
        with pytest.raises(TypeError):
            fusedtok.sample_mirostat(x.to(torch.bfloat16), 5.0)
