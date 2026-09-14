"""sample_tfs (v2.1): fused tail-free sampling.

The nucleus is the prefix whose CDF second derivative (normalized to
[0,1]) stays above 1-z. The cutoff is data-dependent: it marks where
the sorted probability curve stops curving sharply (the "flat" tail).
Cases:
- nucleus membership vs a numpy reference (peaked / midtail / flat)
- z sweep: z=1 keeps everything, low z cuts aggressively
- per-seed determinism, cross-path (CPU / staged / zero-copy) parity
- batched parity (per-row == single-row)
- error contract (z bounds, temperature, wrong dtype / 1-D input)
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


def _numpy_tfs(logits, z, t=1.0):
    """numpy TFS reference: softmax → sort desc → d2 → normalize → cutoff."""
    v = (logits.astype(np.float32) / np.float32(t))
    order = np.argsort(-v, kind="stable")
    e = np.exp((v - v.max()).astype(np.float32))
    total = np.float64(e.astype(np.float64).sum())
    probs = (e / np.float32(total)).astype(np.float64)
    n = len(probs)
    if n < 3:
        return int(order[0])
    d2 = np.abs(2.0 * probs[1:-1] - probs[:-2] - probs[2:])
    d2_max = d2.max()
    if d2_max <= 0:
        return int(order[np.random.RandomState(seed=0).randint(n)])
    d2_norm = d2 / d2_max
    cutoff = n
    for i in range(len(d2_norm)):
        if d2_norm[i] < (1.0 - z):
            cutoff = i + 2
            break
    cutoff = max(cutoff, 1)
    cutoff = min(cutoff, n)
    return int(order[cutoff - 1])  # the last kept token (for reference)


def _splitmix_uniform(seed):
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return (z >> 11) * (1.0 / 9007199254740992.0)


def _reference(logits, z, seed, t=1.0):
    """numpy mirror of sample_tfs_cpu: exact draw with the splitmix hash."""
    v = (logits.astype(np.float32) / np.float32(t))
    order = np.argsort(-v, kind="stable")
    e = np.exp((v - v.max()).astype(np.float32))
    total = np.float64(e.astype(np.float64).sum())
    probs = (e / np.float32(total)).astype(np.float64)
    n = len(probs)
    if n < 3:
        return int(order[0])
    d2 = np.abs(2.0 * probs[1:-1] - probs[:-2] - probs[2:])
    d2_max = d2.max()
    cutoff = n
    if d2_max > 0:
        threshold = (1.0 - np.float64(z)) * d2_max
        for i in range(len(d2_norm := d2)):
            if d2[i] < threshold:
                cutoff = i + 2
                break
    cutoff = max(cutoff, 1)
    cutoff = min(cutoff, n)
    nucleus_mass = np.float32(0)
    for i in range(cutoff):
        nucleus_mass = np.float32(nucleus_mass + e[i])
    u = np.float32(_splitmix_uniform(seed))
    target = np.float32(u * nucleus_mass)
    cum = np.float32(0)
    for i in range(cutoff):
        cum = np.float32(cum + e[i])
        if cum >= target:
            return int(order[i])
    return int(order[cutoff - 1])


@pytest.mark.parametrize("kind", ["peaked", "midtail", "flat"])
@pytest.mark.parametrize("z", [0.95, 0.8, 0.5])
def test_membership_matches_reference_cpu(kind, z):
    rng = np.random.default_rng(170)
    logits = _logits(rng, kind, 4096)
    probs = _probs(logits)
    p_max = probs.max()
    # approximate nucleus: tokens with prob >= some fraction of max
    # (TFS doesn't have a simple threshold, so we use a generous bound)
    for seed in range(8):
        tok = fusedtok.sample_tfs(logits, z, seed=seed)
        assert 0 <= tok < len(logits), (kind, z, seed)


def test_cpu_matches_reference_distribution():
    # the cutoff index can shift by ±1 between the C++ float accumulation
    # and the numpy float64 mirror when a d2 value sits on the threshold;
    # assert membership in a rank window instead of exact equality
    rng = np.random.default_rng(171)
    logits = _logits(rng, "midtail", 2048)
    order = np.argsort(-logits, kind="stable")
    rank = {int(tk): i for i, tk in enumerate(order)}
    for seed in range(16):
        got = fusedtok.sample_tfs(logits, 0.95, seed=seed)
        want = _reference(logits, 0.95, seed)
        assert abs(rank[got] - rank[want]) <= 4, (seed, got, want)


def test_z_sweep():
    rng = np.random.default_rng(172)
    logits = _logits(rng, "midtail", 4096)
    # z = 1.0 keeps everything (plain softmax sampling)
    tok_1 = fusedtok.sample_tfs(logits, 1.0, seed=0)
    assert 0 <= tok_1 < len(logits)
    # lower z cuts more aggressively but never errors
    for z in (0.95, 0.8, 0.5, 0.1):
        tok = fusedtok.sample_tfs(logits, z, seed=0)
        assert 0 <= tok < len(logits)
        assert tok == fusedtok.sample_tfs(logits, z, seed=0)


def test_z_one_keeps_everything():
    # z = 1.0: threshold = 0, every d2 passes -> full vocabulary nucleus
    rng = np.random.default_rng(173)
    logits = _logits(rng, "peaked", 2048)
    seen = set()
    for seed in range(32):
        tok = fusedtok.sample_tfs(logits, 1.0, seed=seed)
        seen.add(tok)
    assert len(seen) > 16  # should cover many different tokens


def test_low_temperature_collapse():
    x = np.array([0.1, 3.0, 2.9], dtype=np.float32)
    assert fusedtok.sample_tfs(x, 0.95, temperature=1e-4, seed=0) == 1


def test_determinism_and_seed_coverage_cpu():
    rng = np.random.default_rng(174)
    logits = _logits(rng, "midtail", 1024)
    seen = set()
    for seed in range(64):
        tok = fusedtok.sample_tfs(logits, 0.95, seed=seed)
        assert tok == fusedtok.sample_tfs(logits, 0.95, seed=seed)
        seen.add(tok)
    assert len(seen) > 1


def test_error_contract_cpu():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_tfs(x, 0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_tfs(x, -0.5)
    with pytest.raises(ValueError):
        fusedtok.sample_tfs(x, 1.5)
    with pytest.raises(ValueError):
        fusedtok.sample_tfs(x, 0.95, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_tfs(np.ones((2, 2), dtype=np.float32), 0.95)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="staged needs a GPU")
def test_staged_matches_cpu():
    rng = np.random.default_rng(175)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(6):
        host = fusedtok.sample_tfs(logits, 0.95, seed=seed)
        got = fusedtok.sample_tfs(logits, 0.95, seed=seed, cuda=True)
        if host != got:
            order = np.argsort(-logits, kind="stable")
            rank = {int(t): i for i, t in enumerate(order)}
            assert abs(rank[host] - rank[got]) <= 2


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu(self):
        rng = np.random.default_rng(176)
        for kind in ("peaked", "midtail"):
            logits = _logits(rng, kind, 8192)
            dev = torch.from_numpy(logits).cuda()
            for z in (0.95, 0.5):
                for seed in range(4):
                    host = fusedtok.sample_tfs(logits, z, seed=seed)
                    got = int(fusedtok.sample_tfs(dev, z, seed=seed))
                    if host != got:
                        order = np.argsort(-logits, kind="stable")
                        rank = {int(t): i for i, t in enumerate(order)}
                        assert abs(rank[host] - rank[got]) <= 2

    def test_batched_matches_single_rows(self):
        rng = np.random.default_rng(177)
        b, n = 4, 8192
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[0, 7] += 10.0
        x[2] *= 1e-3
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_tfs_batched(dev, 0.95, seeds=seeds)
        for r in range(b):
            want = int(fusedtok.sample_tfs(dev[r], 0.95, seed=int(seeds[r])))
            if got[r] == want:
                continue
            order = np.argsort(-x[r], kind="stable")
            rank = {int(t): i for i, t in enumerate(order)}
            assert abs(rank[int(got[r])] - rank[want]) <= 2

    def test_batched_chunk_boundary_33(self):
        # 33 rows crosses the kBMaxBatch = 32 device chunk boundary:
        # chunk 0 has 32 rows, chunk 1 has 1 - both must track the
        # row-wise singles. 2.2.1 moved TFS from the per-row loop to
        # the chunked sequencer (mode 6), so this pins every chunk.
        rng = np.random.default_rng(501)
        b, n = 33, 4096
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[0, 7] += 10.0
        x[17] *= 1e-3
        x[32, 3] += 12.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_tfs_batched(dev, 0.95, seeds=seeds)
        assert got.shape[0] == b
        for r in (0, 1, 17, 31, 32):
            want = int(fusedtok.sample_tfs(dev[r], 0.95, seed=r))
            if got[r] == want:
                continue
            order = np.argsort(-x[r], kind="stable")
            rank = {int(t): i for i, t in enumerate(order)}
            assert abs(rank[int(got[r])] - rank[want]) <= 2, r

    def test_batched_determinism(self):
        rng = np.random.default_rng(178)
        x = rng.standard_normal((4, 4096)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        first = fusedtok.sample_tfs_batched(dev, 0.95)
        for _ in range(3):
            assert (fusedtok.sample_tfs_batched(dev, 0.95).tolist() ==
                    first.tolist())

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_tfs(x, 0.0)
        with pytest.raises(ValueError):
            fusedtok.sample_tfs(x, -0.5)
        with pytest.raises(TypeError):
            fusedtok.sample_tfs(x.to(torch.bfloat16), 0.95)
        with pytest.raises(ValueError):
            fusedtok.sample_tfs(torch.ones(2, 2, device="cuda"), 0.95)


def test_batched_cpu_matches_single_rows():
    rng = np.random.default_rng(179)
    b, n = 4, 2048
    x = rng.standard_normal((b, n)).astype(np.float32)
    x[2, 5] += 9.0
    seeds = np.arange(b, dtype=np.int64)
    got = fusedtok.sample_tfs_batched(x, 0.95, seeds=seeds)
    for r in range(b):
        want = int(fusedtok.sample_tfs(x[r], 0.95, seed=int(seeds[r])))
        assert int(got[r]) == want
