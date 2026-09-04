"""Cross-cutting validation contracts hardened in 1.2.1.

Every test here pins a fix from the 1.2.1 audit:

- empty-input guards that used to return garbage on the zero-copy path
  (argmax read an uninitialized output slot when the launcher skipped
  the empty input),
- GQA divisibility / paged group-size checks that only ran on the CUDA
  branch (the CPU reference silently accepted misshapen head counts),
- host-side lens / block-table / token-id validation: values arriving
  from the HOST are checked before the upload; device-resident tensors
  are trusted (reading them back would sync the stream and break
  CUDA-graph capture - the same trust boundary as raw pointers),
- dtype / contiguity rejection on the zero-copy INT8 ops (a wrong dtype
  or a strided view would otherwise be read as raw bytes),
- consistent return types: quantize_int8 / qadd_int8 return a Python
  float scale on EVERY path,
- honest operand names in binary-op validation errors,
- staged-upload value checks (1.3.1): the staged bindings receive host
  lens/block-table arrays directly - their VALUES are now validated in
  the Python wrapper before the upload, so a bad entry can no longer
  become a silent GPU out-of-bounds access (the in-place kv_append ops
  would have corrupted the cache permanently),
- integral host ints only (1.3.1): float lens/ids used to be silently
  truncated by the int32/int64 casts; 2-D id arrays used to be silently
  ravelled flat - both are now rejected.
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


# ---------------------------------------------------------------------------
# empty-input guard (argmax used to read uninitialized device memory)
# ---------------------------------------------------------------------------


def test_argmax_empty_raises_on_cpu():
    with pytest.raises(ValueError):
        fusedtok.argmax(np.zeros(0, dtype=np.float32))


@needs_gpu
def test_argmax_empty_raises_on_cuda():
    # the GPU launcher early-returns without writing the output slot, so
    # the wrapper must raise instead of reading uninitialized memory
    with pytest.raises(ValueError):
        fusedtok.argmax(torch.zeros(0, dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError):
        fusedtok.argmax(torch.zeros(0, dtype=torch.float32))  # CPU tensor


@needs_gpu
def test_argmax_nonempty_cuda_still_works():
    x = torch.tensor([1.0, 5.0, 5.0, 2.0], device="cuda")
    assert fusedtok.argmax(x) == 1  # earliest index on ties


# ---------------------------------------------------------------------------
# GQA validation on every execution path (was CUDA-branch-only)
# ---------------------------------------------------------------------------


def test_attention_decode_gqa_checked_on_cpu():
    # hq=3, hkv=2: not a multiple - the CPU reference must reject it
    q = np.zeros((1, 3, 8), dtype=np.float32)
    k = np.zeros((1, 2, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, k.copy())


def test_attention_prefill_gqa_checked_on_cpu():
    q = np.zeros((1, 3, 4, 8), dtype=np.float32)
    k = np.zeros((1, 2, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.attention_prefill(q, k, k.copy())


def test_paged_group_size_checked_on_cpu():
    # hq=6, hkv=2 -> group 3: not one of the supported 1/2/4/8/16
    q = np.zeros((1, 6, 8), dtype=np.float32)
    pool = np.zeros((4, 2, 2, 8), dtype=np.float32)
    table = np.zeros((1, 2), dtype=np.int32)
    with pytest.raises(ValueError):
        fusedtok.attention_decode_paged(q, pool, pool.copy(), table)


@needs_gpu
def test_attention_decode_gqa_checked_on_cuda():
    q = torch.zeros((1, 3, 8), device="cuda")
    k = torch.zeros((1, 2, 4, 8), device="cuda")
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, k.clone())


# ---------------------------------------------------------------------------
# host-side lens validation / device-lens trust (no stream sync)
# ---------------------------------------------------------------------------


def _decode_fixture(device):
    rng = np.random.default_rng(3)
    b, hq, hkv, t, d = 2, 4, 2, 16, 8
    q = torch.tensor(rng.standard_normal((b, hq, d)),
                     dtype=torch.float32, device=device)
    k = torch.tensor(rng.standard_normal((b, hkv, t, d)),
                     dtype=torch.float32, device=device)
    return q, k, b, t


@needs_gpu
def test_lens_host_values_validated_before_upload():
    q, k, b, t = _decode_fixture("cuda")
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, k.clone(), [t + 1, 0])
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, k.clone(), [0, -1])
    with pytest.raises(ValueError):
        # wrong element count is a shape error
        fusedtok.attention_decode(q, k, k.clone(), [0])
    # CPU-tensor lens get the same treatment (validated, then uploaded)
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, k.clone(),
                                  torch.tensor([0, t + 1]))


@needs_gpu
def test_lens_equal_capacity_is_valid():
    # the decode bound is inclusive: lens == T reads the whole cache
    q, k, b, t = _decode_fixture("cuda")
    out = fusedtok.attention_decode(q, k, k.clone(), [t, t])
    assert out.shape == q.shape
    ref = fusedtok.attention_decode(q, k, k.clone())
    assert torch.equal(out, ref)


@needs_gpu
def test_lens_host_validation_matches_device_result():
    # a validated host list and the equivalent device tensor must agree
    q, k, b, t = _decode_fixture("cuda")
    lens = [t, 7]
    out_list = fusedtok.attention_decode(q, k, k.clone(), lens)
    out_dev = fusedtok.attention_decode(q, k, k.clone(),
                                        torch.tensor(lens, device="cuda"))
    assert torch.equal(out_list, out_dev)


@needs_gpu
def test_lens_device_tensor_is_graph_capturable():
    # THE fix: value validation used to read the device lens back (two
    # stream syncs per call), which made CUDA-graph capture impossible.
    # Device lens are now trusted and capture works.
    q, k, b, t = _decode_fixture("cuda")
    lens = torch.tensor([t, 7], dtype=torch.int32, device="cuda")
    fusedtok.attention_decode(q, k, k.clone(), lens)  # warm-up (workspace)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fusedtok.attention_decode(q, k, k.clone(), lens)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        out = fusedtok.attention_decode(q, k, k.clone(), lens)
    ref = fusedtok.attention_decode(q, k, k.clone(), lens).clone()
    g.replay()
    assert torch.equal(out, ref)
    # replay after mutating the inputs recomputes (fixed-address reads)
    q.copy_(q * 0.5)
    g.replay()
    ref2 = fusedtok.attention_decode(q, k, k.clone(), lens)
    assert torch.equal(out, ref2)


@needs_gpu
def test_paged_block_table_host_values_validated():
    rng = np.random.default_rng(4)
    b, hq, hkv, nb, p, d = 1, 4, 2, 4, 4, 8
    q = torch.tensor(rng.standard_normal((b, hq, d)), dtype=torch.float32,
                     device="cuda")
    pool = torch.tensor(rng.standard_normal((nb, hkv, p, d)),
                        dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        # nb is out of range for a 4-block pool
        fusedtok.attention_decode_paged(q, pool, pool.clone(),
                                        [[0, 4]])
    with pytest.raises(ValueError):
        fusedtok.attention_decode_paged(q, pool, pool.clone(),
                                        [[0, -1]])
    # a device-resident table is trusted (documented boundary)
    out = fusedtok.attention_decode_paged(
        q, pool, pool.clone(),
        torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"))
    assert out.shape == q.shape


@needs_gpu
def test_kv_append_lens_bound_is_exclusive():
    # kv_append WRITES at position lens[b]: the table must map that
    # block, so lens == S*P is invalid (decode's bound is inclusive,
    # append's is not)
    rng = np.random.default_rng(5)
    b, hkv, nb, p, d = 1, 2, 4, 4, 8
    pool = torch.zeros((nb, hkv, p, d), dtype=torch.float32, device="cuda")
    k_new = torch.tensor(rng.standard_normal((b, hkv, d)),
                         dtype=torch.float32, device="cuda")
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    fusedtok.kv_append_paged(pool, pool.clone(), table, k_new, k_new.clone(),
                             [p - 1])  # last slot of block 1: valid
    with pytest.raises(ValueError):
        fusedtok.kv_append_paged(pool, pool.clone(), table,
                                 k_new, k_new.clone(), [2 * p])


# ---------------------------------------------------------------------------
# token-id validation for the penalty kernels (host-checked, device-trusted)
# ---------------------------------------------------------------------------


@needs_gpu
def test_repetition_penalty_host_ids_validated():
    logits = torch.randn(32, device="cuda")
    with pytest.raises(ValueError):
        fusedtok.repetition_penalty(logits, [0, 32], 1.1)
    with pytest.raises(ValueError):
        fusedtok.repetition_penalty(logits, [-1], 1.1)
    out = fusedtok.repetition_penalty(logits, [3, 3], 1.1)
    assert out.shape == logits.shape
    # a CUDA ids tensor is trusted - and still applies the penalty
    ids = torch.tensor([3, 3], dtype=torch.int64, device="cuda")
    out_dev = fusedtok.repetition_penalty(logits, ids, 1.1)
    assert torch.equal(out, out_dev)


@needs_gpu
def test_decode_step_host_ids_validated():
    logits = torch.randn(32, device="cuda")
    with pytest.raises(ValueError):
        fusedtok.decode_step(logits, [32], 1.1, seed=0)
    tok = fusedtok.decode_step(logits, [0, 1], 1.1, seed=0)
    assert isinstance(tok, int) and 0 <= tok < 32


# ---------------------------------------------------------------------------
# zero-copy INT8 ops: dtype / contiguity rejection
# ---------------------------------------------------------------------------


@needs_gpu
def test_dequantize_rejects_wrong_dtype_and_layout():
    x = torch.randn(64, device="cuda")
    q, s = fusedtok.quantize_int8(x)
    with pytest.raises(TypeError):
        fusedtok.dequantize_int8(x, s)  # float32, not int8
    strided = torch.tile(q, (2,))[::2]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError):
        fusedtok.dequantize_int8(strided, s)


@needs_gpu
def test_qgemm_rejects_non_contiguous_operands():
    rng = np.random.default_rng(6)
    a = torch.tensor(rng.standard_normal((8, 16)),
                     device="cuda").to(torch.int8)
    b = torch.tensor(rng.standard_normal((8, 16)),
                     device="cuda").to(torch.int8)
    strided_b = torch.tile(b, (1, 2))[:, ::2]
    assert not strided_b.is_contiguous()
    with pytest.raises(ValueError):
        fusedtok.qgemm(a, 1.0, strided_b, 1.0)
    with pytest.raises(ValueError):
        fusedtok.qgemm_perchannel(a, 1.0, strided_b,
                                  torch.ones(8, device="cuda"))
    # the contiguous originals still work
    y = fusedtok.qgemm(a, 1.0, b, 1.0)
    assert y.shape == (8, 8)


@needs_gpu
def test_qadd_rejects_non_contiguous():
    a = torch.randn(64, device="cuda")
    qa, sa = fusedtok.quantize_int8(a)
    strided = torch.tile(qa, (2,))[::2]
    with pytest.raises(ValueError):
        fusedtok.qadd_int8(qa, sa, strided, sa)


# ---------------------------------------------------------------------------
# consistent return types: Python float scale on every path
# ---------------------------------------------------------------------------


def test_quantize_scale_is_python_float_on_cpu():
    q, s = fusedtok.quantize_int8(np.float32([1.0, -2.0]))
    assert isinstance(s, float)


@needs_gpu
def test_quantize_scale_is_python_float_on_cuda():
    x = torch.randn(100, device="cuda")
    q, s = fusedtok.quantize_int8(x)
    assert isinstance(s, float)
    assert q.dtype is torch.int8
    # the natural pipeline works without tensor-scale acrobatics
    back = fusedtok.dequantize_int8(*fusedtok.quantize_int8(x))
    assert (back - x).abs().max().item() <= s * 0.51


@needs_gpu
def test_qadd_scale_is_python_float_on_cuda():
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    qa, sa = fusedtok.quantize_int8(a)
    qb, sb = fusedtok.quantize_int8(b)
    qy, sy = fusedtok.qadd_int8(qa, sa, qb, sb)
    assert isinstance(sy, float)


# ---------------------------------------------------------------------------
# honest operand names in binary-op errors
# ---------------------------------------------------------------------------


@needs_gpu
def test_binary_errors_name_the_failing_operand():
    a = torch.randn(8, device="cuda")
    with pytest.raises(TypeError, match="b must be"):
        fusedtok.add(a, a.cpu())  # b on the wrong device
    bad_b = torch.randn(8, dtype=torch.float64, device="cuda")
    with pytest.raises(TypeError, match="b must be"):
        fusedtok.add(a, bad_b)  # b with the wrong dtype
    good_gate = torch.randn(8, device="cuda")
    bad_up = torch.randn(8, dtype=torch.float64, device="cuda")
    with pytest.raises(TypeError, match="up must be"):
        fusedtok.swiglu(good_gate, bad_up)


@needs_gpu
def test_mixed_device_inputs_rejected_with_guidance():
    x = torch.randn((4, 16))  # CPU
    w = torch.rand(16, device="cuda")  # CUDA weight: device mismatch
    with pytest.raises(TypeError):
        fusedtok.rmsnorm(x, w)


@needs_gpu
def test_cpu_operand_on_zero_copy_path_rejected_everywhere():
    # a host pointer handed to a kernel is an asynchronous illegal
    # access that POISONS the CUDA context for every later call - every
    # zero-copy helper must reject CPU tensors up front (found by the
    # v1.3 kv_append tests: one poisoned launch failed 100+ later tests)
    q = torch.randn(1, 4, 8, device="cuda")
    k = torch.randn(1, 2, 8, 8, device="cuda")
    with pytest.raises(TypeError):
        fusedtok.attention_decode(q, k.cpu(), k.clone())
    with pytest.raises(TypeError):
        fusedtok.attention_decode(q, k, k.clone().cpu())
    kn = torch.randn(1, 2, 8)
    kc = torch.zeros(1, 2, 8, 8, device="cuda")
    with pytest.raises(TypeError):
        fusedtok.kv_append(kc, kc.clone(), kn, kn.clone(), [0])
    pool = torch.zeros(4, 2, 4, 8, device="cuda")
    with pytest.raises(TypeError):
        fusedtok.attention_decode_paged(q, pool.cpu(), pool.clone(),
                                        [[0, 1]])
    # the CUDA context must still be healthy after all the rejections
    out = fusedtok.attention_decode(q, k, k.clone())
    assert out.shape == q.shape


# ---------------------------------------------------------------------------
# staged-upload value validation (1.3.1: the staged bindings got host
# lens/table arrays whose VALUES were unchecked - a bad entry became a
# silent GPU out-of-bounds access, and the in-place kv_append ops would
# have corrupted the cache permanently; the docstrings claimed the
# validation existed)
# ---------------------------------------------------------------------------


def _host_cache(t_rows=8):
    k = np.zeros((2, 2, t_rows, 4), dtype=np.float32)
    v = np.zeros_like(k)
    kn = np.ones((2, 2, 4), dtype=np.float32)
    vn = np.zeros_like(kn)
    return k, v, kn, vn


@needs_gpu
def test_kv_append_staged_lens_out_of_range_rejected():
    # lens[b] == T writes one row past sequence b's cache rows
    k, v, kn, vn = _host_cache(t_rows=8)
    with pytest.raises(ValueError):
        fusedtok.kv_append(k, v, kn, vn, [8, 0], cuda=True)
    with pytest.raises(ValueError):
        fusedtok.kv_append(k, v, kn, vn, [0, -1], cuda=True)
    # the CPU reference path validates in C++ - same contract
    with pytest.raises(ValueError):
        fusedtok.kv_append(k, v, kn, vn, [8, 0])


def test_kv_append_staged_valid_lens_still_work():
    # regression guard: valid staged lens must keep round-tripping
    k, v, kn, vn = _host_cache(t_rows=8)
    fusedtok.kv_append(k, v, kn, vn, [3, 5])
    assert np.allclose(k[0, :, 3, :], 1.0)
    assert np.allclose(k[1, :, 5, :], 1.0)
    assert np.allclose(k[0, :, 0, :], 0.0)  # untouched rows stay zero


def test_kv_append_empty_batch_lens_still_accepted():
    # empty batch passes lens=[] - and np.asarray([]) is FLOAT64, so the
    # new integral-dtype rejection must not fire on empty input
    k = np.zeros((0, 2, 8, 4), dtype=np.float32)
    kn = np.zeros((0, 2, 4), dtype=np.float32)
    assert fusedtok.kv_append(k, k.copy(), kn, kn.copy(), []) is None


@needs_gpu
def test_kv_append_paged_staged_bad_table_rejected():
    pool = np.zeros((4, 2, 4, 4), dtype=np.float32)
    kn = np.zeros((2, 2, 4), dtype=np.float32)
    table = np.zeros((2, 3), dtype=np.int32)
    table[1, 0] = 4  # only blocks [0, 4) exist
    with pytest.raises(ValueError):
        fusedtok.kv_append_paged(pool, pool.copy(), table, kn, kn.copy(),
                                 [0, 0], cuda=True)
    table[1, 0] = -1
    with pytest.raises(ValueError):
        fusedtok.kv_append_paged(pool, pool.copy(), table, kn, kn.copy(),
                                 [0, 0], cuda=True)


@needs_gpu
def test_kv_append_paged_staged_lens_out_of_range_rejected():
    pool = np.zeros((4, 2, 4, 4), dtype=np.float32)
    kn = np.zeros((2, 2, 4), dtype=np.float32)
    table = np.zeros((2, 3), dtype=np.int32)
    # span = 3 pages * 4 rows = 12; lens 12 is one past the last row
    with pytest.raises(ValueError):
        fusedtok.kv_append_paged(pool, pool.copy(), table, kn, kn.copy(),
                                 [12, 0], cuda=True)


@needs_gpu
def test_attention_decode_staged_lens_out_of_range_rejected():
    q = np.zeros((1, 4, 8), dtype=np.float32)
    k = np.zeros((1, 2, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, k.copy(), [9], cuda=True)
    with pytest.raises(ValueError):
        fusedtok.attention_decode(q, k, k.copy(), [-1], cuda=True)


@needs_gpu
def test_attention_decode_paged_staged_bad_table_rejected():
    # the staged paged-decode binding also validates in C++; the Python
    # check fires first with the same error family - pinned either way
    q = np.zeros((1, 4, 8), dtype=np.float32)
    pool = np.zeros((4, 2, 4, 8), dtype=np.float32)
    table = np.zeros((1, 3), dtype=np.int32)
    table[0, 2] = 99
    with pytest.raises(ValueError):
        fusedtok.attention_decode_paged(q, pool, pool.copy(), table,
                                        [4], cuda=True)


def test_kv_append_host_v_shape_mismatch_rejected():
    # same TOTAL size, different shape: without the shape check the
    # binding's size-only guard would alias the write into v
    k = np.zeros((1, 2, 8, 4), dtype=np.float32)
    v = np.zeros((1, 2, 4, 8), dtype=np.float32)
    kn = np.zeros((1, 2, 4), dtype=np.float32)
    vn = np.zeros((1, 2, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.kv_append(k, v, kn, vn, [0])
    with pytest.raises(ValueError):
        fusedtok.kv_append(k, k.copy(), kn, vn, [0])


def test_kv_append_paged_host_v_shape_mismatch_rejected():
    pool = np.zeros((4, 2, 4, 4), dtype=np.float32)
    vpool = np.zeros((2, 2, 4, 8), dtype=np.float32)  # same total size
    kn = np.zeros((2, 2, 4), dtype=np.float32)
    table = np.zeros((2, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        fusedtok.kv_append_paged(pool, vpool, table, kn, kn.copy(),
                                 [0, 0])


# ---------------------------------------------------------------------------
# integral host ints + 1-D ids (1.3.1: float lens/ids were silently
# truncated by the int casts; 2-D id arrays were silently ravelled flat)
# ---------------------------------------------------------------------------


@needs_gpu
def test_attention_decode_cuda_float_lens_rejected():
    q = torch.zeros((1, 4, 8), device="cuda")
    k = torch.zeros((1, 2, 8, 8), device="cuda")
    with pytest.raises(TypeError):
        fusedtok.attention_decode(q, k, k.clone(), [1.5])


def test_kv_append_host_float_lens_rejected():
    k, v, kn, vn = _host_cache(t_rows=8)
    with pytest.raises(TypeError):
        fusedtok.kv_append(k, v, kn, vn, [1.5, 0])


def test_repetition_penalty_host_2d_ids_rejected():
    logits = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.repetition_penalty(logits, [[1, 2], [3, 4]], 1.2)


def test_decode_step_host_2d_ids_rejected():
    logits = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.decode_step(logits, [[1, 2], [3, 4]])


def test_decode_step_host_float_ids_rejected():
    logits = np.zeros(16, dtype=np.float32)
    with pytest.raises(TypeError):
        fusedtok.decode_step(logits, [1.5, 2])


def test_decode_step_host_ids_range_message_names_parameter():
    logits = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError, match="sampled_ids"):
        fusedtok.decode_step(logits, [16])


# ---------------------------------------------------------------------------
# 1.4.1: batched-binding hardening. The Python layer derives rows/n
# from the array shape, but _fusedtok is a supported direct surface and
# the 1.4.0 staged trio copied rows*n floats without checking the
# buffer (an oversized pair read past the numpy array) while the _cpu
# trio also lacked the n <= 0 guard (an inverted pointer range is UB).
# ---------------------------------------------------------------------------

def _ft():
    return fusedtok._fusedtok


def test_batched_staged_shape_mismatch_rejected():
    if not fusedtok.cuda_available():
        pytest.skip("staged needs a GPU")
    x = np.ones((2, 8), dtype=np.float32)
    seeds = np.zeros(4, dtype=np.int64)
    # rows*n = 32 would read past the 16-float buffer
    with pytest.raises(ValueError, match="shape"):
        _ft().sample_topp_batched(x, 4, 8, 0.9, 1.0, seeds)
    # swapped rows/n passes the total but not the shape
    with pytest.raises(ValueError, match="shape"):
        _ft().sample_topk_batched(x, 8, 2, 4, 1.0, seeds)
    with pytest.raises(ValueError, match="shape"):
        _ft().sample_minp_batched(x, 2, 9, 0.05, 1.0,
                                  np.zeros(2, dtype=np.int64))


def test_batched_cpu_n_nonpositive_rejected():
    x = np.ones((2, 8), dtype=np.float32)
    seeds = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="empty logits"):
        _ft().sample_topp_batched_cpu(x, 2, 0, 0.9, 1.0, seeds)
    with pytest.raises(ValueError, match="empty logits"):
        _ft().sample_topk_batched_cpu(x, 2, -8, 4, 1.0, seeds)
    with pytest.raises(ValueError, match="shape"):
        _ft().sample_minp_batched_cpu(x, 3, 8, 0.05, 1.0, seeds)


def test_batched_staged_empty_vocab_rejected():
    # (0, 0): zero rows do not bypass the empty-vocab guard (same
    # contract as the single-row samplers)
    if not fusedtok.cuda_available():
        pytest.skip("staged needs a GPU")
    x = np.empty((0, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="empty logits"):
        _ft().sample_topp_batched(x, 0, 0, 0.9, 1.0,
                                  np.empty(0, dtype=np.int64))


def test_batched_launcher_safe_invalids_rejected():
    if not fusedtok.cuda_available():
        pytest.skip("launcher needs a GPU")
    x = np.ones((2, 8), dtype=np.float32)
    seeds = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="empty logits"):
        _ft().sample_topp_batched_launch(0, 2, 0, 0.9, 1.0, seeds)
    with pytest.raises(ValueError, match="rows"):
        _ft().sample_topk_batched_launch(0, -1, 8, 4, 1.0, seeds)
    with pytest.raises(ValueError, match="seeds"):
        _ft().sample_minp_batched_launch(0, 2, 8, 0.05, 1.0,
                                         np.zeros(3, dtype=np.int64))


def test_batch_seeds_uint64_wrap_rejected():
    # uint64 values above 2**63 - 1 wrap negative in the int64 staging
    # array; the check runs AFTER the cast so they cannot sneak through
    x = np.ones((2, 8), dtype=np.float32)
    bad = np.array([0, 2 ** 63], dtype=np.uint64)
    with pytest.raises(ValueError, match=r"2\*\*63"):
        fusedtok.sample_topp_batched(x, 0.9, seeds=bad)
    ok = np.array([0, 2 ** 63 - 1], dtype=np.uint64)
    out = fusedtok.sample_topp_batched(x, 0.9, seeds=ok)
    assert out.shape == (2,)
