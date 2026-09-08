"""argmax_batched (v1.8): row-wise greedy argmax for a whole batch.

The single-row argmax rule applied per row of a 2-D batch in one
launch: packed-key max, ties toward the EARLIEST index of the row.
Cases:

- per-row equality with numpy argmax / the single-row op
- explicit tie rows (ties must resolve to the earliest index)
- large batches and full-decode-scale vocabularies
- return-type contract (CUDA in -> CUDA int64 tensor, zero-copy;
  CPU torch in -> CPU torch tensor; numpy in -> numpy array)
- determinism across repeats
- CUDA graph capture + replay with a mutated input (the launcher is
  capturable by design - no readback)
- error contract (1-D input, empty vocab, wrong dtype, bad rows)
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


def test_cpu_matches_numpy_per_row():
    rng = np.random.default_rng(150)
    x = rng.standard_normal((16, 1024)).astype(np.float32)
    got = fusedtok.argmax_batched(x)
    assert got.dtype == np.int64
    assert np.array_equal(got, np.argmax(x, axis=1))


def test_ties_resolve_to_earliest_index():
    # constant rows: every position ties, argmax must return 0
    x = np.zeros((4, 64), dtype=np.float32)
    got = fusedtok.argmax_batched(x)
    assert got.tolist() == [0, 0, 0, 0]
    # duplicated maxima: the first occurrence wins
    y = np.full((2, 8), -1.0, dtype=np.float32)
    y[0, 5] = 3.0
    y[0, 2] = 3.0
    y[1, 7] = -0.5
    y[1, 1] = -0.5
    assert fusedtok.argmax_batched(y).tolist() == [2, 1]


def test_cpu_matches_single_row_op():
    rng = np.random.default_rng(151)
    x = rng.standard_normal((7, 333)).astype(np.float32)
    got = fusedtok.argmax_batched(x)
    for r in range(7):
        assert int(got[r]) == fusedtok.argmax(x[r])


def test_batched_cpu_direct_surface_contract():
    from fusedtok import _fusedtok
    # a short buffer must be rejected instead of trusting rows * n
    with pytest.raises(ValueError):
        _fusedtok.argmax_batched_cpu(
            np.zeros(8, dtype=np.float32), 2, 8)
    with pytest.raises(ValueError):
        _fusedtok.argmax_batched_cpu(
            np.zeros((2, 8), dtype=np.float32), 2, 0)   # empty vocab
    with pytest.raises(ValueError):
        _fusedtok.argmax_batched_cpu(
            np.zeros((2, 8), dtype=np.float32), -1, 8)  # negative rows


def test_error_contract_cpu():
    with pytest.raises(ValueError):
        fusedtok.argmax_batched(np.ones(8, dtype=np.float32))  # 1-D
    with pytest.raises(ValueError):
        fusedtok.argmax_batched(
            np.zeros((2, 0), dtype=np.float32))                # empty row
    # host-side non-f32 dtypes are CONVERTED, not rejected (the
    # documented package-wide convention); the result matches
    got = fusedtok.argmax_batched(np.ones((2, 8), dtype=np.float16))
    assert got.tolist() == [0, 0]


@needs_gpu
class TestCuda:
    def test_zero_copy_matches_cpu(self):
        rng = np.random.default_rng(152)
        for shape in ((4, 1024), (16, 4096), (8, 131072)):
            x = rng.standard_normal(shape).astype(np.float32)
            dev = torch.from_numpy(x).cuda()
            got = fusedtok.argmax_batched(dev)
            want = fusedtok.argmax_batched(x)
            assert np.array_equal(got.cpu().numpy(), want), shape

    def test_large_batch(self):
        # 512 rows x 4096: one launch, one index per row
        rng = np.random.default_rng(153)
        x = rng.standard_normal((512, 4096)).astype(np.float32)
        x[np.arange(512), (rng.random(512) * 4096).astype(np.int64)] += 10.0
        dev = torch.from_numpy(x).cuda()
        got = fusedtok.argmax_batched(dev).cpu().numpy()
        assert np.array_equal(got, np.argmax(x, axis=1))

    def test_matches_single_row_zero_copy(self):
        rng = np.random.default_rng(154)
        x = rng.standard_normal((6, 2048)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        got = fusedtok.argmax_batched(dev)
        for r in range(6):
            assert int(got[r]) == int(fusedtok.argmax(dev[r]))

    def test_return_type_is_device_int64(self):
        x = torch.randn(3, 128, device="cuda")
        out = fusedtok.argmax_batched(x)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.int64 and out.is_cuda
        # cpu torch input: staged path returns a CPU torch tensor
        out_cpu = fusedtok.argmax_batched(x.cpu(), cuda=True)
        assert isinstance(out_cpu, torch.Tensor)
        assert out_cpu.dtype == torch.int64 and out_cpu.is_cpu

    def test_determinism_across_repeats(self):
        rng = np.random.default_rng(155)
        x = rng.standard_normal((8, 4096)).astype(np.float32)
        x[3, 17] += 9.0
        dev = torch.from_numpy(x).cuda()
        first = fusedtok.argmax_batched(dev).cpu().tolist()
        for _ in range(3):
            assert fusedtok.argmax_batched(dev).cpu().tolist() == first

    def test_mixed_with_selection_keeps_slots_clean(self):
        # the per-row arrival slots live where selection calls keep
        # their scratch: interleave the two families and make sure the
        # per-call clearing holds (the historical foot-gun this design
        # guards against)
        rng = np.random.default_rng(156)
        x = rng.standard_normal((4, 2048)).astype(np.float32)
        dev = torch.from_numpy(x).cuda()
        for _ in range(3):
            vals, idxs = fusedtok.topk(dev[0], 32)     # selection call
            assert idxs.shape[0] == 32
            got = fusedtok.argmax_batched(dev).cpu().numpy()
            assert np.array_equal(got, np.argmax(x, axis=1))

    def test_error_contract_cuda(self):
        x = torch.ones(8, device="cuda")
        with pytest.raises(ValueError):
            fusedtok.argmax_batched(x)                 # 1-D
        with pytest.raises(ValueError):
            fusedtok.argmax_batched(torch.ones(2, 0, device="cuda"))
        with pytest.raises(TypeError):
            fusedtok.argmax_batched(
                torch.ones(2, 8, device="cuda", dtype=torch.bfloat16))


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()
                         and hasattr(torch.cuda, "CUDAGraph")),
                    reason="no torch.cuda.CUDAGraph")
def test_graph_capture_replay():
    # the launcher is capturable by design: no readback, stream-ordered;
    # replay must recompute after the input mutates (a vacuously
    # captured empty graph would keep the warm-up answer)
    x = torch.randn(4, 1024, device="cuda")
    out = {}

    def fn():
        out["idx"] = fusedtok.argmax_batched(x)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    x[1, 500] = 100.0                  # move row 1's max
    out["idx"].fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    assert int(out["idx"][1]) == 500
