"""Sampling-side ops: top-k, top-p, argmax, temperature, repetition penalty."""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# top-k
# ---------------------------------------------------------------------------


def test_topk_hand_checkable():
    x = np.array([1.0, 3.0, 2.0, 3.0], dtype=np.float32)
    vals, idxs = fusedtok.topk(x, 3)
    # tie between index 1 and 3 -> earliest wins
    assert vals == pytest.approx([3.0, 3.0, 2.0], abs=1e-6)
    assert idxs.tolist() == [1, 3, 2]


def test_topk_full_sort():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100).astype(np.float32)
    vals, idxs = fusedtok.topk(x, 100)
    assert vals == pytest.approx(np.sort(x)[::-1], abs=1e-5)
    assert (np.diff(vals) <= 1e-6).all()          # descending


def test_topk_k_zero_and_errors():
    x = np.ones(5, dtype=np.float32)
    vals, idxs = fusedtok.topk(x, 0)
    assert vals.size == 0 and idxs.size == 0
    with pytest.raises(ValueError):
        fusedtok.topk(x, 6)
    with pytest.raises(ValueError):
        fusedtok.topk(np.ones((2, 2), dtype=np.float32), 1)


# ---------------------------------------------------------------------------
# top-p
# ---------------------------------------------------------------------------


def test_topp_includes_crossing_element():
    # probs 0.5, 0.3, 0.2: cumulative 0.5 < 0.7, 0.8 >= 0.7 -> keep two
    p = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    vals, idxs = fusedtok.topp(p, 0.7)
    assert vals == pytest.approx([0.5, 0.3], abs=1e-6)
    assert idxs.tolist() == [0, 1]


def test_topp_p_one_keeps_everything():
    rng = np.random.default_rng(1)
    p = rng.random(50).astype(np.float32)
    p /= p.sum()
    vals, idxs = fusedtok.topp(p, 1.0)
    assert len(vals) == 50


def test_topp_errors():
    with pytest.raises(ValueError):
        fusedtok.topp(np.ones(3, dtype=np.float32) / 3, 0.0)
    with pytest.raises(ValueError):
        fusedtok.topp(np.ones(3, dtype=np.float32) / 3, 1.5)


# ---------------------------------------------------------------------------
# argmax / temperature
# ---------------------------------------------------------------------------


def test_argmax_ties_earliest():
    x = np.array([1.0, 5.0, 5.0, 2.0], dtype=np.float32)
    assert fusedtok.argmax(x) == 1


def test_argmax_errors():
    with pytest.raises(ValueError):
        fusedtok.argmax(np.array([], dtype=np.float32))
    with pytest.raises(ValueError):
        fusedtok.argmax(np.ones((2, 2), dtype=np.float32))


def test_temperature_scales():
    x = np.array([2.0, -4.0], dtype=np.float32)
    assert fusedtok.temperature(x, 2.0) == pytest.approx([1.0, -2.0], abs=1e-6)
    assert fusedtok.temperature(x, 0.5) == pytest.approx([4.0, -8.0], abs=1e-5)
    with pytest.raises(ValueError):
        fusedtok.temperature(x, 0.0)


# ---------------------------------------------------------------------------
# repetition penalty
# ---------------------------------------------------------------------------


def test_rep_penalty_positive_divides():
    lg = np.ones(10, dtype=np.float32) * 4.0
    y = fusedtok.repetition_penalty(lg, [2], 2.0)
    assert y[2] == pytest.approx(2.0, abs=1e-6)
    assert y[0] == pytest.approx(4.0, abs=1e-6)


def test_rep_penalty_negative_multiplies():
    lg = np.full(10, -4.0, dtype=np.float32)
    y = fusedtok.repetition_penalty(lg, [0, 9], 2.0)
    assert y[0] == pytest.approx(-8.0, abs=1e-5)
    assert y[9] == pytest.approx(-8.0, abs=1e-5)
    assert y[5] == pytest.approx(-4.0, abs=1e-6)


def test_rep_penalty_zero_and_empty():
    lg = np.zeros(4, dtype=np.float32)
    y = fusedtok.repetition_penalty(lg, [1], 3.0)
    assert y[1] == 0.0
    y2 = fusedtok.repetition_penalty(lg, [], 3.0)
    assert (y2 == lg).all()
    with pytest.raises(ValueError):
        fusedtok.repetition_penalty(lg, [0], 0.0)
    with pytest.raises(ValueError):
        fusedtok.repetition_penalty(lg, [99], 1.1)


# ---------------------------------------------------------------------------
# GPU paths
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestRadixSelect:
    """Targeted coverage for the radix-select GPU path (v0.2).

    The radix kernel replaces the per-round selection loop; these cases pin
    the semantics that are easiest to break: heavy ties, boundary sizes
    around the shared/global bitonic cutover (m <= 2048 vs above), k = 1,
    k = n, and workspace reuse across calls with different k.
    """

    def test_heavy_ties_earliest_indices(self):
        # thousands of identical values: boundary falls inside one big tie
        x = np.full(5000, 0.5, dtype=np.float32)
        x[::7] = 1.0          # a few strictly larger
        x[3::11] = -0.5
        v, i = fusedtok.topk(x, 64, cuda=True)
        ref_v, ref_i = fusedtok.topk(x, 64)
        assert v == pytest.approx(ref_v, abs=1e-6)
        assert i.tolist() == ref_i.tolist()   # ties -> earliest index, exact

    def test_ties_span_boundary(self):
        # tie group straddles the k-th position exactly
        x = np.zeros(1000, dtype=np.float32)
        x[:100] = 2.0                        # top group
        x[100:300] = 1.0                     # tie group around the boundary
        v, i = fusedtok.topk(x, 150, cuda=True)
        ref_v, ref_i = fusedtok.topk(x, 150)
        assert i.tolist() == ref_i.tolist()

    def test_k1_equals_argmax(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(4096).astype(np.float32)
        v, i = fusedtok.topk(x, 1, cuda=True)
        assert i[0] == fusedtok.argmax(x, cuda=True)
        assert v[0] == pytest.approx(x.max(), abs=1e-6)

    def test_full_sort_non_power_of_two(self):
        # k = n exercises emit + sort with padding; 4097 is not a power of 2
        rng = np.random.default_rng(11)
        x = rng.standard_normal(4097).astype(np.float32)
        v, i = fusedtok.topk(x, 4097, cuda=True)
        assert v == pytest.approx(np.sort(x)[::-1], abs=1e-5)
        assert (np.diff(v) <= 1e-6).all()

    def test_across_bitonic_cutover(self):
        # m pads to 2048 (shared path) and 4096 (global path): both must agree
        rng = np.random.default_rng(12)
        x = rng.standard_normal(5000).astype(np.float32)
        for k in (2047, 2048, 2049, 3000):
            v, i = fusedtok.topk(x, k, cuda=True)
            ref_v, ref_i = fusedtok.topk(x, k)
            assert i.tolist() == ref_i.tolist(), f"k={k}"

    def test_workspace_growth_across_calls(self):
        # same process, increasing then decreasing k: the process-cached
        # key buffer must serve every call correctly
        rng = np.random.default_rng(13)
        x = rng.standard_normal(9000).astype(np.float32)
        for k in (5, 700, 8000, 33, 8000):
            v, i = fusedtok.topk(x, k, cuda=True)
            ref_v, ref_i = fusedtok.topk(x, k)
            assert v == pytest.approx(ref_v, abs=1e-5), f"k={k}"
            assert i.tolist() == ref_i.tolist(), f"k={k}"

    def test_large_vocab(self):
        # realistic LLM vocabulary size (global bitonic path)
        rng = np.random.default_rng(14)
        x = rng.standard_normal(131072).astype(np.float32)
        v, i = fusedtok.topk(x, 50, cuda=True)
        ref_v, ref_i = fusedtok.topk(x, 50)
        assert v == pytest.approx(ref_v, abs=1e-5)
        assert i.tolist() == ref_i.tolist()

    def test_all_negative_values(self):
        # negative-only range exercises the flipped mantissa ordering
        rng = np.random.default_rng(15)
        x = -rng.uniform(0.1, 5.0, 7000).astype(np.float32)
        v, i = fusedtok.topk(x, 100, cuda=True)
        ref_v, ref_i = fusedtok.topk(x, 100)
        assert v == pytest.approx(ref_v, abs=1e-5)
        assert i.tolist() == ref_i.tolist()

    def test_topp_nucleus_matches_reference(self):
        # The GPU count uses a parallel prefix scan; its float accumulation
        # order differs from the serial CPU loop, so at draws that land
        # exactly on the boundary the cut index may differ by one element
        # while both satisfy the nucleus definition. Assert the definition.
        rng = np.random.default_rng(16)
        p = rng.random(20000).astype(np.float32)
        p /= p.sum()
        v, i = fusedtok.topp(p, 0.9, cuda=True)
        ref_v, _ = fusedtok.topp(p, 0.9)
        # descending, indices consistent with values
        assert (np.diff(v) <= 1e-6).all()
        assert np.allclose(p[i], v, atol=1e-6)
        # nucleus property: prefix mass >= p, prefix-without-last < p
        cum = np.cumsum(v.astype(np.float64))
        assert cum[-1] >= 0.9 - 1e-4
        assert len(v) == 1 or cum[-2] < 0.9 + 1e-4
        # count within float-ordering drift of the CPU reference: the GPU
        # prefix scan accumulates in a different order, boundary shifts by
        # a couple of elements at 131k-scale vocabularies
        assert abs(len(v) - len(ref_v)) <= max(2, int(1e-4 * len(ref_v)))
        assert v == pytest.approx(ref_v[:len(v)], abs=1e-5)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_topk_matches_cpu(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(1000).astype(np.float32)
        v_cpu, i_cpu = fusedtok.topk(x, 10)
        v_gpu, i_gpu = fusedtok.topk(x, 10, cuda=True)
        assert v_gpu == pytest.approx(v_cpu, abs=1e-5)
        assert i_gpu.tolist() == i_cpu.tolist()

    def test_topp_matches_cpu(self):
        rng = np.random.default_rng(3)
        p = rng.random(500).astype(np.float32)
        p /= p.sum()
        v_cpu, i_cpu = fusedtok.topp(p, 0.9)
        v_gpu, i_gpu = fusedtok.topp(p, 0.9, cuda=True)
        assert v_gpu == pytest.approx(v_cpu, abs=1e-5)
        assert i_gpu.tolist() == i_cpu.tolist()

    def test_argmax_temperature(self):
        x = np.array([1.0, 9.0, 3.0], dtype=np.float32)
        assert fusedtok.argmax(x, cuda=True) == 1
        assert fusedtok.temperature(x, 3.0, cuda=True) == pytest.approx(
            [1 / 3, 3.0, 1.0], abs=1e-5)

    def test_rep_penalty(self):
        rng = np.random.default_rng(4)
        lg = rng.standard_normal(1000).astype(np.float32)
        ids = [3, 7, 250, 999]
        cpu = fusedtok.repetition_penalty(lg, ids, 1.15)
        gpu = fusedtok.repetition_penalty(lg, ids, 1.15, cuda=True)
        assert gpu == pytest.approx(cpu, abs=1e-5)


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_topk_gpu(self):
        x = torch.randn(1000, device="cuda", dtype=torch.float32)
        vals, idxs = fusedtok.topk(x, 10)
        ref_vals, ref_idxs = torch.topk(x, 10)
        torch.cuda.synchronize()
        assert vals.cpu().numpy() == pytest.approx(ref_vals.cpu().numpy(), abs=1e-5)
        assert set(idxs.cpu().tolist()) == set(ref_idxs.cpu().tolist())

    def test_topp_gpu_slices(self):
        p = torch.rand(500, device="cuda", dtype=torch.float32)
        p = p / p.sum()
        vals, idxs = fusedtok.topp(p, 0.9)
        cum = vals.cumsum(0)
        assert cum[-1].item() >= 0.9
        assert cum[-2].item() < 0.9 if len(vals) > 1 else True
        assert (vals.diff() <= 1e-6).all().item() or len(vals) == 1

    def test_argmax_rep_penalty_gpu(self):
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        assert fusedtok.argmax(x) == x.argmax().item()
        y = fusedtok.repetition_penalty(x, [0, 5, 99], 1.2)
        torch.cuda.synchronize()
        ref = x.clone()
        for i in (0, 5, 99):
            ref[i] = ref[i] / 1.2 if ref[i] > 0 else ref[i] * 1.2
        assert y.cpu().numpy() == pytest.approx(ref.cpu().numpy(), abs=1e-5)
