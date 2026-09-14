"""sample_xtc (v2.2): fused XTC (Exclude Top Choices) sampling.

With probability p, the top top_n tokens by probability are removed
from the sampling pool. Breaks LLM "template" outputs.
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
    }[kind]
    if kind == "peaked":
        out[7] += 6.0
    return out


def test_xtc_basic():
    rng = np.random.default_rng(190)
    logits = _logits(rng, "midtail", 4096)
    for seed in range(8):
        tok = fusedtok.sample_xtc(logits, 3, 0.8, seed=seed)
        assert 0 <= tok < len(logits)


def test_xtc_probability_zero_is_plain_softmax():
    # probability = 0 means suppression never triggers
    rng = np.random.default_rng(191)
    logits = _logits(rng, "midtail", 2048)
    for seed in range(8):
        tok_xtc = fusedtok.sample_xtc(logits, 3, 0.0, seed=seed)
        assert 0 <= tok_xtc < len(logits)


def test_xtc_determinism():
    rng = np.random.default_rng(192)
    logits = _logits(rng, "midtail", 2048)
    for z in (0.5, 0.8):
        for seed in range(16):
            a = fusedtok.sample_xtc(logits, 3, z, seed=seed)
            b = fusedtok.sample_xtc(logits, 3, z, seed=seed)
            assert a == b, (z, seed)


def test_xtc_top_n_zero_noop():
    # top_n = 0 means nothing to suppress
    rng = np.random.default_rng(193)
    logits = _logits(rng, "midtail", 1024)
    for seed in range(8):
        tok = fusedtok.sample_xtc(logits, 0, 0.9, seed=seed)
        assert 0 <= tok < len(logits)


def test_error_contract_cpu():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, -1, 0.5)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, 3, -0.1)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, 3, 1.5)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(x, 3, 0.5, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.sample_xtc(np.ones((2, 2), dtype=np.float32), 3, 0.5)


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu(self):
        rng = np.random.default_rng(194)
        logits = _logits(rng, "midtail", 8192)
        dev = torch.from_numpy(logits).cuda()
        for top_n in (2, 5):
            for prob in (0.5, 0.9):
                for seed in range(4):
                    host = fusedtok.sample_xtc(logits, top_n, prob, seed=seed)
                    got = int(fusedtok.sample_xtc(dev, top_n, prob, seed=seed))
                    _assert_neighbor_rank(logits, got, int(host),
                                          (top_n, prob, seed))

    def test_batched_matches_single_rows(self):
        rng = np.random.default_rng(195)
        b, n = 4, 4096
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[0, 7] += 10.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_xtc_batched(dev, 3, 0.8, seeds=seeds)
        assert len(got) == b
        for r in range(b):
            want = int(fusedtok.sample_xtc(x[r], 3, 0.8, seed=int(seeds[r])))
            _assert_neighbor_rank(x[r], int(got[r]), want, r)

    def test_batched_chunk_boundary_33(self):
        # 33 rows crosses the kBMaxBatch = 32 device chunk boundary.
        # 2.2.1 moved XTC from the per-row loop to the chunked
        # sequencer (mode 7); every row runs the full-vocabulary
        # window, so both chunks must track the row-wise singles.
        rng = np.random.default_rng(502)
        b, n = 33, 4096
        x = rng.standard_normal((b, n)).astype(np.float32)
        x[0, 7] += 10.0
        x[32, 3] += 12.0
        dev = torch.from_numpy(x).cuda()
        seeds = np.arange(b, dtype=np.int64)
        got = fusedtok.sample_xtc_batched(dev, 3, 0.8, seeds=seeds)
        assert got.shape[0] == b
        for r in (0, 1, 17, 31, 32):
            want = int(fusedtok.sample_xtc(dev[r], 3, 0.8, seed=r))
            _assert_neighbor_rank(x[r], int(got[r]), want, r)

    def test_batched_determinism(self):
        rng = np.random.default_rng(196)
        x = rng.standard_normal((4, 4096)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        first = fusedtok.sample_xtc_batched(dev, 3, 0.8)
        for _ in range(3):
            assert (fusedtok.sample_xtc_batched(dev, 3, 0.8).tolist() ==
                    first.tolist())

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.sample_xtc(x, -1, 0.5)
        with pytest.raises(ValueError):
            fusedtok.sample_xtc(x, 3, -0.1)
        with pytest.raises(TypeError):
            fusedtok.sample_xtc(x.to(torch.bfloat16), 3, 0.5)
        with pytest.raises(ValueError):
            fusedtok.sample_xtc(torch.ones(2, 2, device="cuda"), 3, 0.5)


def _assert_neighbor_rank(logits_row, got, want, what):
    """Exact parity with the documented ulp fallback: the GPU total is
    accumulated with per-block float atomics and __expf, so a draw
    landing exactly on a CDF boundary may pick a neighbor rank."""
    if got == want:
        return
    order = np.argsort(-logits_row, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="staged needs a GPU")
def test_staged_matches_cpu_full_vocab_tail():
    # 2.2.1 regression pin: the GPU path used to start the nucleus
    # window at kSelEarlyOut (1024) and the XTC serial walk always
    # resolved inside it, so on a > 1024 vocabulary every draw came
    # from the top ~1024 ranks and the widening ladder never fired
    # (the walk's target is u * the window's own mass). The window must
    # be the full vocabulary - on flat-ish logits the CPU reference
    # draws deep-tail tokens, and every staged draw must follow it.
    rng = np.random.default_rng(197)
    logits = _logits(rng, "midtail", 4096)
    deep_tail_seen = False
    order = np.argsort(-logits, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    for seed in range(40):
        host = int(fusedtok.sample_xtc(logits, 3, 0.8, seed=seed))
        got = int(fusedtok.sample_xtc(logits, 3, 0.8, seed=seed,
                                      cuda=True))
        assert abs(rank[host] - rank[got]) <= 2, seed
        if rank[host] > 2048:
            deep_tail_seen = True
    assert deep_tail_seen, "reference draws never reached the tail"

    # the staged batched path must track the row-wise CPU singles too
    x = np.stack([logits] * 3).astype(np.float32)
    x[1, 7] += 8.0
    seeds = np.arange(3, dtype=np.int64)
    got = fusedtok.sample_xtc_batched(x, 3, 0.8, seeds=seeds, cuda=True)
    for r in range(3):
        want = int(fusedtok.sample_xtc(x[r], 3, 0.8, seed=int(seeds[r])))
        _assert_neighbor_rank(x[r], int(got[r]), want, r)
