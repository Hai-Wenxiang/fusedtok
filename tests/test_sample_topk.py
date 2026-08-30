"""Fused top-k sampling (sample_topk): semantics, determinism, parity.

Top-k sampling keeps the k highest-probability tokens of
softmax(logits / temperature), renormalizes WITHIN them, and inverse-CDF
draws with a splitmix hash of the seed. Contract points pinned here:

- the sampled token ALWAYS lies in the top-k set (k >= 1, any seed),
- k = 1 is exactly greedy (argmax of logits / temperature),
- the same seed reproduces the same token on every path and every call,
- seeds spread draws across the top-k set in proportion to mass,
- CPU / staged / zero-copy agree per seed (same algorithm, same order;
  exact-exp vs fast-exp can only move draws sitting exactly on a
  floating-point boundary, which the test distributions avoid).
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

HAS_GPU = HAS_TORCH and fusedtok.cuda_available()


def ref_topk_set(logits, k, t=1.0):
    """The top-k index set in the packed-key order (value desc, index asc)."""
    inv_t = 1.0 / t
    order = sorted(range(len(logits)),
                   key=lambda i: (-(logits[i] * inv_t), i))
    return set(order[:k])


@pytest.mark.parametrize("n,k", [(131072, 50), (1024, 100), (64, 8),
                                 (17, 17), (8, 1)])
def test_sample_topk_in_set(n, k):
    rng = np.random.default_rng(n * 31 + k)
    logits = (rng.standard_normal(n) * 3.0).astype(np.float32)
    for seed in range(8):
        token = fusedtok.sample_topk(logits, k, seed=seed)
        assert token in ref_topk_set(logits, k), (
            f"seed {seed}: token {token} outside the top-{k} set")


def test_sample_topk_k1_is_greedy():
    # k = 1 must be the argmax for every seed and temperature
    rng = np.random.default_rng(7)
    logits = (rng.standard_normal(2048) * 5.0).astype(np.float32)
    greedy = int(np.argmax(logits))
    for seed in range(8):
        assert fusedtok.sample_topk(logits, 1, seed=seed) == greedy
    # temperature only rescales - the argmax order is unchanged
    assert fusedtok.sample_topk(logits, 1, temperature=7.5, seed=3) == greedy


def test_sample_topk_deterministic_per_seed():
    rng = np.random.default_rng(11)
    logits = (rng.standard_normal(4096) * 4.0).astype(np.float32)
    first = [fusedtok.sample_topk(logits, 64, seed=s) for s in range(16)]
    again = [fusedtok.sample_topk(logits, 64, seed=s) for s in range(16)]
    assert first == again


def test_sample_topk_seeds_cover_the_set():
    # many seeds must draw several DISTINCT tokens from the top-k set
    # (a sampler stuck on one token would pass the in-set check)
    rng = np.random.default_rng(13)
    logits = (rng.standard_normal(4096) * 3.0).astype(np.float32)
    draws = {fusedtok.sample_topk(logits, 32, seed=s) for s in range(64)}
    assert len(draws) >= 8, f"64 seeds produced only {len(draws)} tokens"


def test_sample_topk_full_vocab():
    # k >= n samples the whole distribution - still in-set trivially
    # (the set is everything), still deterministic
    rng = np.random.default_rng(15)
    logits = (rng.standard_normal(256) * 2.0).astype(np.float32)
    a = fusedtok.sample_topk(logits, 256, seed=42)
    b = fusedtok.sample_topk(logits, 1000, seed=42)   # clamped to n
    assert a == b and 0 <= a < 256


def test_sample_topk_temperature_orders():
    # a hot temperature flattens; the draw set stays the top-k set, but a
    # cold temperature must collapse draws toward the greedy token
    rng = np.random.default_rng(17)
    logits = (rng.standard_normal(2048) * 3.0).astype(np.float32)
    cold = {fusedtok.sample_topk(logits, 64, temperature=0.05, seed=s)
            for s in range(32)}
    greedy = int(np.argmax(logits))
    assert cold == {greedy}


def test_sample_topk_errors():
    logits = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_topk(logits, 0)
    with pytest.raises(ValueError):
        fusedtok.sample_topk(logits, 8, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_topk(np.zeros(0, dtype=np.float32), 4)


def test_sample_topk_mass_concentration():
    # a distribution with a dominant token must draw it almost always:
    # mass concentration is the actual sampling semantics, not just set
    # membership
    rng = np.random.default_rng(19)
    logits = (rng.standard_normal(1024) * 0.1).astype(np.float32)
    dominant = 100
    logits[dominant] = 12.0
    logits[rng.choice(1024, size=32, replace=False)] += 2.0   # decoys
    draws = [fusedtok.sample_topk(logits, 32, seed=s) for s in range(128)]
    assert draws.count(dominant) >= 120


@pytest.mark.skipif(not (HAS_TORCH and HAS_GPU), reason="no torch/GPU")
class TestCuda:
    def test_staged_matches_cpu(self):
        rng = np.random.default_rng(21)
        logits = (rng.standard_normal(8192) * 4.0).astype(np.float32)
        for seed in range(8):
            y = fusedtok.sample_topk(logits, 128, seed=seed, cuda=True)
            ycpu = fusedtok.sample_topk(logits, 128, seed=seed)
            assert y == ycpu

    def test_zero_copy_matches_cpu(self):
        rng = np.random.default_rng(23)
        logits = (rng.standard_normal(131072) * 4.0).astype(np.float32)
        lt = torch.from_numpy(logits).cuda()
        for seed in range(8):
            y = fusedtok.sample_topk(lt, 256, temperature=0.8, seed=seed)
            ycpu = fusedtok.sample_topk(logits, 256, temperature=0.8,
                                        seed=seed)
            assert y == ycpu

    def test_bigger_than_early_out(self):
        # k past the in-block-sort threshold rides the chunk+merge tail
        rng = np.random.default_rng(25)
        logits = (rng.standard_normal(131072) * 3.0).astype(np.float32)
        lt = torch.from_numpy(logits).cuda()
        top = ref_topk_set(logits, 3000)
        for seed in range(4):
            y = fusedtok.sample_topk(lt, 3000, seed=seed)
            assert y in top

    def test_repeated_calls_stable(self):
        logits = torch.randn(4096, device="cuda")
        a = [fusedtok.sample_topk(logits, 32, seed=9) for _ in range(4)]
        assert len(set(a)) == 1
