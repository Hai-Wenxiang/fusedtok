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
- honest operand names in binary-op validation errors.
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
