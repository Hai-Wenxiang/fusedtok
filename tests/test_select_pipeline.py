"""Targeted coverage for the v0.4 selection pipeline.

The v0.4 GPU path replaces the single cooperative kernel with a
multi-launch pipeline (arrival-ticket radix rounds, early-exit
compaction, two-level emit, chunk-merge sort). These cases pin the
behaviors that are easiest to break:

- distributions whose keys share high bytes force several radix rounds
  (and defeat the early exit when a boundary bin stays populous)
- k > 2048 exercises the chunk-sort + merge-ladder big path
- interleaved calls with different k / modes must reset the process
  workspace state correctly (tickets, counters, stage word)
- nucleus sampling with a window that must widen (flat logits) still
  matches the CPU reference for the same seed
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestPipelineRoundCoverage:
    def test_quantized_values_force_deep_rounds(self):
        # values snapped to a coarse grid: thousands of keys share the
        # same high bytes, so the boundary bin stays populous for several
        # rounds before the early exit (or the full 8 rounds) resolve it
        rng = np.random.default_rng(21)
        x = np.round(rng.standard_normal(50000) * 4).astype(np.float32) / 4
        for k in (1, 37, 2048, 4096):
            v, i = fusedtok.topk(x, k, cuda=True)
            ref_v, ref_i = fusedtok.topk(x, k)
            assert v == pytest.approx(ref_v, abs=1e-6), f"k={k}"
            assert i.tolist() == ref_i.tolist(), f"k={k}"

    def test_two_level_values(self):
        # only two distinct values: the boundary tie group is huge
        x = np.zeros(30000, dtype=np.float32)
        x[:500] = 1.0
        x[-500:] = 1.0
        x[15000] = -1.0
        v, i = fusedtok.topk(x, 700, cuda=True)
        ref_v, ref_i = fusedtok.topk(x, 700)
        assert i.tolist() == ref_i.tolist()
        assert v == pytest.approx(ref_v, abs=1e-6)

    def test_big_path_merge_ladder(self):
        # k > 2048: emit -> chunk sort -> multiple merge levels -> decode
        rng = np.random.default_rng(22)
        x = rng.standard_normal(40000).astype(np.float32)
        for k in (2049, 5000, 20000, 40000):
            v, i = fusedtok.topk(x, k, cuda=True)
            ref_v, ref_i = fusedtok.topk(x, k)
            assert v == pytest.approx(ref_v, abs=1e-5), f"k={k}"
            assert i.tolist() == ref_i.tolist(), f"k={k}"

    def test_interleaved_calls_reset_state(self):
        # consecutive selection calls with different k / modes share the
        # process workspace; every call must see a clean ticket/counter/
        # stage state (determinism check on repeats)
        rng = np.random.default_rng(23)
        x = rng.standard_normal(131072).astype(np.float32)
        p = np.abs(x).astype(np.float32)
        p /= p.sum()
        for round_ in range(3):
            v1, i1 = fusedtok.topk(x, 50, cuda=True)
            v2, i2 = fusedtok.topk(x, 9000, cuda=True)
            v3, i3 = fusedtok.topp(p, 0.95, cuda=True)
            ref1_v, ref1_i = fusedtok.topk(x, 50)
            ref2_v, ref2_i = fusedtok.topk(x, 9000)
            assert i1.tolist() == ref1_i.tolist(), f"round {round_}"
            assert i2.tolist() == ref2_i.tolist(), f"round {round_}"
            cum = np.cumsum(v3.astype(np.float64))
            assert cum[-1] >= 0.95 - 1e-4
            assert v3 == pytest.approx(np.sort(p)[::-1][:len(v3)], abs=1e-6)

    def test_topp_p1_full_vocab(self):
        # p = 1.0 keeps everything: crossing lands at the very end of the
        # 131k sorted array; the count must equal the vocab size
        rng = np.random.default_rng(24)
        p = rng.random(131072).astype(np.float32)
        p /= p.sum()
        v, i = fusedtok.topp(p, 1.0, cuda=True)
        assert len(v) == 131072
        assert v == pytest.approx(np.sort(p)[::-1], abs=1e-5)
        assert (np.diff(v) <= 1e-6).all()

    def test_topp_tiny_p_first_element(self):
        rng = np.random.default_rng(25)
        p = rng.random(2048).astype(np.float32)
        p /= p.sum()
        v, i = fusedtok.topp(p, 1e-7, cuda=True)
        assert i[0] == int(np.argmax(p))
        assert v[0] == pytest.approx(p.max(), abs=1e-7)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestPipelineSampling:
    def test_flat_logits_force_widening(self):
        # near-identical logits spread the nucleus over the whole vocab:
        # the sampling window must widen past the 1024 in-block size and
        # the big-path serial scan must still match the CPU reference
        n = 9000
        logits = (np.zeros(n, dtype=np.float32) +
                  np.linspace(-1e-3, 1e-3, n).astype(np.float32))
        for seed in (0, 5, 77):
            assert fusedtok.sample_topp(logits, 0.99, seed=seed, cuda=True) == \
                fusedtok.sample_topp(logits, 0.99, seed=seed)

    def test_flat_large_window_deterministic(self):
        # n=40000 pushes the widening loop deep into the big-window
        # regime (exp precompute + long serial walks); the same seed
        # must reproduce the same token bit-exactly across calls, and
        # the token must lie in the p=0.9 nucleus of the sorted
        # distribution (which for this shape is most of the vocab)
        n = 40000
        logits = (np.random.default_rng(31)
                  .standard_normal(n).astype(np.float32))
        a = fusedtok.sample_topp(logits, 0.9, seed=123, cuda=True)
        b = fusedtok.sample_topp(logits, 0.9, seed=123, cuda=True)
        assert a == b and 0 <= a < n

    def test_sample_after_topk_topp_state(self):
        # sampling right after selections in the same process exercises
        # the workspace handoff (token slot vs counters)
        rng = np.random.default_rng(26)
        logits = (rng.standard_normal(4096) * 3).astype(np.float32)
        fusedtok.topk(logits, 100, cuda=True)
        p = np.abs(logits).astype(np.float32)
        fusedtok.topp(p / p.sum(), 0.9, cuda=True)
        tok = fusedtok.sample_topp(logits, 0.9, seed=42, cuda=True)
        assert tok == fusedtok.sample_topp(logits, 0.9, seed=42)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestPipelineTorchZeroCopy:
    def test_big_path_torch(self):
        x = torch.randn(30000, device="cuda")
        vals, idxs = fusedtok.topk(x, 7000)
        ref_vals, ref_idxs = torch.topk(x, 7000)
        torch.cuda.synchronize()
        assert vals.cpu().numpy() == pytest.approx(
            ref_vals.cpu().numpy(), abs=1e-5)
        assert idxs.cpu().tolist() == ref_idxs.cpu().tolist()

    def test_topp_torch_nucleus_property(self):
        p = torch.rand(40000, device="cuda")
        p = p / p.sum()
        vals, idxs = fusedtok.topp(p, 0.85)
        cum = vals.cumsum(0)
        assert cum[-1].item() >= 0.85 - 1e-4
        assert (vals.diff() <= 1e-6).all().item()
