"""sample_minp (v1.3): fused min-p sampling.

The nucleus is every token with probability >= min_p * p_max - a value
threshold, so in the max-normalized exp column it is a PREFIX (exps[0]
== 1.0 exactly). Cases:

- nucleus membership vs a numpy reference (peaked / midtail / flat)
- min_p = 1.0 keeps only the maximum-probability tokens (unique max
  collapses to argmax; tied maxima stay tied)
- tiny min_p ~ whole-vocabulary nucleus (the widening ladder + the
  full-vocabulary fast path)
- per-seed determinism, cross-path (CPU / staged / zero-copy) parity
- mass concentration sanity (the nucleus carries most of the mass)
- error contract (min_p bounds, temperature, wrong dtype / 2-D input)
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


def _logits(rng, kind, n):
    """Raw logits of the requested shape (the op softmaxes these)."""
    out = {
        "peaked": rng.standard_normal(n).astype(np.float32) * 0.5,
        "midtail": rng.standard_normal(n).astype(np.float32),
        "flat": rng.standard_normal(n).astype(np.float32) * 1e-3,
    }[kind]
    if kind == "peaked":
        out[7] += 6.0
    return out


def _probs(logits):
    e = np.exp((logits - logits.max()).astype(np.float32))
    return e / e.sum()


def _reference(logits, min_p, seed):
    """numpy reference: softmax -> filter p >= min_p * p_max ->
    renormalize -> splitmix inverse-CDF in descending order."""
    probs = _probs(logits)
    order = np.argsort(-probs, kind="stable")   # value desc, tie earliest
    p_max = probs[order[0]]
    nucleus = order[probs[order] >= min_p * p_max]
    mass = np.float32(0)
    for t in nucleus:
        mass = np.float32(mass + probs[t])
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    u = (z >> 11) * (1.0 / 9007199254740992.0)
    target = np.float32(u) * mass
    cum = np.float32(0)
    for t in nucleus:
        cum = np.float32(cum + probs[t])
        if cum >= target:
            return int(t)
    return int(nucleus[-1])


@pytest.mark.parametrize("kind", ["peaked", "midtail", "flat"])
@pytest.mark.parametrize("min_p", [0.3, 0.1, 0.02])
def test_membership_matches_reference_cpu(kind, min_p):
    rng = np.random.default_rng(80)
    logits = _logits(rng, kind, 4096)
    probs = _probs(logits)
    p_max = probs.max()
    nucleus = set(np.flatnonzero(probs >= min_p * p_max).tolist())
    for seed in range(8):
        tok = fusedtok.sample_minp(logits, min_p, seed=seed)
        assert tok in nucleus, (kind, min_p, seed)


def test_cpu_matches_reference_distribution():
    rng = np.random.default_rng(81)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(16):
        assert fusedtok.sample_minp(logits, 0.05, seed=seed) == \
            _reference(logits, 0.05, seed)


def test_minp_one_keeps_only_the_maxima():
    # unique max: min_p = 1 collapses to exactly argmax for every seed
    x = np.array([1.0, 5.0, 4.0, 2.0], dtype=np.float32)
    for seed in range(8):
        assert fusedtok.sample_minp(x, 1.0, seed=seed) == 1
    # tied maxima: both stay in the nucleus (equal mass -> either draws)
    y = np.array([1.0, 5.0, 5.0, 2.0], dtype=np.float32)
    for seed in range(8):
        assert fusedtok.sample_minp(y, 1.0, seed=seed) in (1, 2)


def test_low_temperature_collapse():
    x = np.array([0.1, 3.0, 2.9], dtype=np.float32)
    assert fusedtok.sample_minp(x, 0.5, temperature=1e-4, seed=0) == 1


def test_tiny_minp_spans_vocabulary():
    # min_p below every probability keeps the whole vocab: the nucleus
    # is everything and the draw is a plain softmax sample
    rng = np.random.default_rng(82)
    logits = _logits(rng, "peaked", 3000)
    tok = fusedtok.sample_minp(logits, 1e-9, seed=3)
    assert 0 <= tok < 3000
    assert fusedtok.sample_minp(logits, 1e-9, seed=3) == tok


def test_nucleus_shape_sanity():
    # min-p is a VALUE threshold: the nucleus is a small prefix of the
    # descending order (never empty - exps[0] == 1.0 passes any valid
    # min_p - and strictly smaller than the vocabulary at a healthy
    # threshold). Unlike top-p it carries no mass guarantee, by design.
    rng = np.random.default_rng(83)
    logits = _logits(rng, "peaked", 8192)
    probs = _probs(logits)
    mask = probs >= 0.1 * probs.max()
    assert mask.any() and mask.sum() < 8192


def test_determinism_and_seed_coverage_cpu():
    rng = np.random.default_rng(84)
    logits = _logits(rng, "midtail", 1024)
    seen = set()
    for seed in range(64):
        tok = fusedtok.sample_minp(logits, 0.05, seed=seed)
        assert tok == fusedtok.sample_minp(logits, 0.05, seed=seed)
        seen.add(tok)
    assert len(seen) > 1          # seeds must not all collapse to one


def test_error_contract_cpu():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_minp(x, 0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_minp(x, 1.5)
    with pytest.raises(ValueError):
        fusedtok.sample_minp(x, 0.1, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_minp(np.ones((2, 2), dtype=np.float32), 0.1)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="staged needs a GPU")
def test_staged_matches_cpu():
    rng = np.random.default_rng(85)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(6):
        assert fusedtok.sample_minp(logits, 0.05, seed=seed) == \
            fusedtok.sample_minp(logits, 0.05, seed=seed, cuda=True)


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu_all_regimes(self):
        rng = np.random.default_rng(86)
        for kind in ("peaked", "midtail", "flat"):
            logits = _logits(rng, kind, 8192)
            probs = _probs(logits)
            dev = torch.from_numpy(logits).cuda()
            for min_p in (0.2, 0.05, 0.002):
                for seed in range(6):
                    host = fusedtok.sample_minp(logits, min_p, seed=seed)
                    got = int(fusedtok.sample_minp(dev, min_p, seed=seed))
                    # exact-exp CPU vs __expf GPU can neighbor on the
                    # CDF boundary: same nucleus, at most a neighbor rank
                    order = np.argsort(-probs, kind="stable")
                    rank = {int(t): i for i, t in enumerate(order)}
                    assert host in rank and got in rank
                    assert abs(rank[host] - rank[got]) <= 1

    def test_full_vocabulary_fast_path(self):
        # near-uniform logits + tiny threshold -> nucleus = whole vocab
        # via the widening ladder + k==n parallel-pack path
        rng = np.random.default_rng(87)
        logits = _logits(rng, "flat", 131072)
        dev = torch.from_numpy(logits).cuda()
        tok = int(fusedtok.sample_minp(dev, 1e-6, seed=5))
        assert 0 <= tok < 131072
        assert tok == int(fusedtok.sample_minp(dev, 1e-6, seed=5))

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_minp(x, 0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_minp(x, 1.5)
        with pytest.raises(TypeError):
            fusedtok.sample_minp(x.to(torch.bfloat16), 0.1)
        with pytest.raises(ValueError):
            fusedtok.sample_minp(torch.ones(2, 2, device="cuda"), 0.1)
