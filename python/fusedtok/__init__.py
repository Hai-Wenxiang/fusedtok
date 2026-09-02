"""fusedtok: small fused CUDA kernels for LLM inference.

The Python layer dispatches between three execution paths:

- torch CUDA tensors  -> zero-copy: kernels read/write torch device buffers
                         directly via data_ptr(); no staging, no host sync
                         (results are stream-ordered with other torch ops).
- numpy / torch CPU + ``cuda=True`` -> staged: data is copied to the GPU,
                         the kernel runs, results are copied back.
- numpy / torch CPU (default) -> the C++ CPU reference implementation
                         (ground truth; runs on machines without a GPU).

All functions accept float32 numpy arrays or torch tensors (other dtypes
are converted to float32 with a copy when needed; outputs are float32).
CUDA torch tensors keep their native storage dtype where kernels support
it (bfloat16 everywhere data moves, float16 on attention). Row-wise ops
accept 1-D (one row) or 2-D contiguous arrays.
"""

import numpy as np

try:
    from . import _fusedtok
except ImportError:  # development build tree: _fusedtok lives in build/
    import _fusedtok

_HAS_TORCH = False
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # torch is an optional dependency
    torch = None

__version__ = "1.3.0"

__all__ = [
    "cuda_available",
    "axpy",
    "rmsnorm",
    "layernorm",
    "softmax",
    "rope",
    "swiglu",
    "silu",
    "gelu",
    "gelu_tanh",
    "relu",
    "tanh",
    "sigmoid",
    "add",
    "mul",
    "temperature",
    "repetition_penalty",
    "argmax",
    "topk",
    "topp",
    "sample_topp",
    "sample_topk",
    "sample_minp",
    "quantize_int8",
    "dequantize_int8",
    "qadd_int8",
    "qgemm",
    "qgemm_perchannel",
    "decode_step",
    "attention_decode",
    "kv_append",
    "attention_decode_paged",
    "attention_prefill",
    "kv_append_paged",
]


def cuda_available():
    """Return True if a usable CUDA device is present."""
    return _fusedtok.cuda_available()


# ---------------------------------------------------------------------------
# internal dispatch helpers
# ---------------------------------------------------------------------------


def _is_torch(x):
    return _HAS_TORCH and torch.is_tensor(x)


def _torch_to_numpy(t, name):
    """CPU torch tensor -> float32 C-contiguous numpy array (zero-copy when
    the tensor already is float32/contiguous)."""
    if t.is_cuda:
        raise TypeError(
            f"{name} is a CUDA tensor while another input is host-side; "
            "move all inputs to the same device (CUDA tensors select the "
            "zero-copy path automatically, host inputs the CPU reference)"
        )
    if t.dtype is not torch.float32:
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t.detach().numpy()


def _numpy_to_torch_like(arr):
    """Fresh numpy array -> torch tensor sharing its memory."""
    return torch.from_numpy(arr)


def _require_cuda(t, name):
    """Device-family guard for secondary operands: once the primary input
    picked the zero-copy path, every tensor operand must live on the same
    GPU (a host copy would silently diverge from the kernel's view)."""
    if not (_is_torch(t) and t.is_cuda):
        raise TypeError(f"{name} must be a CUDA tensor when the primary "
                        "input is on CUDA")


def _check_contiguous(t, name):
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_torch_f32(t, name):
    # the zero-copy helpers only ever run on the torch-cuda path, where
    # EVERY tensor operand must live on the GPU - a host pointer handed
    # to a kernel is an asynchronous illegal access that poisons the
    # CUDA context for every later call, so reject it here
    if not t.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor (got a CPU tensor "
                        "on the zero-copy path)")
    if t.dtype is not torch.float32:
        raise TypeError(f"{name} must be float32, got {t.dtype} "
                        "(convert with .to(torch.float32))")
    _check_contiguous(t, name)


def _check_torch_float(t, name):
    """Accept float32 or bfloat16 (the two dtypes with CUDA kernels)."""
    if not t.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor (got a CPU tensor "
                        "on the zero-copy path)")
    if t.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"{name} must be float32 or bfloat16, got {t.dtype} "
                        "(convert with .to(torch.float32))")
    _check_contiguous(t, name)


def _check_torch_att(t, name):
    """Accept the three attention storage dtypes: float32, bfloat16,
    float16 (attention kernels are templated on the storage dtype and
    compute in float32)."""
    if not t.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor (got a CPU tensor "
                        "on the zero-copy path)")
    if t.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"{name} must be float32, bfloat16 or float16, "
                        f"got {t.dtype} (convert with .to(torch.float32))")
    _check_contiguous(t, name)


def _launch_for_dtype(dtype, base, fp16=False):
    """Bindings entry point for a storage dtype. Suffix convention:
    ``base`` = float32 kernel, ``base + '_bf16'`` / ``base + '_fp16'``
    = the half-precision instantiations (fp16 exists for attention
    only)."""
    if dtype is torch.bfloat16:
        return getattr(_fusedtok, base + "_bf16")
    if fp16 and dtype is torch.float16:
        return getattr(_fusedtok, base + "_fp16")
    return getattr(_fusedtok, base)


def _att_launch_fn(t, base):
    """The attention launcher for t's storage dtype."""
    return _launch_for_dtype(t.dtype, base, fp16=True)


def _norm_weight_f32(weight):
    """Norm weights must reach the kernel as float32 (they commonly are in
    checkpoints, and the bf16 kernels read them as float). Small [cols]
    upcast copy when the caller hands us bf16."""
    if weight.dtype is torch.float32:
        return weight
    return weight.to(torch.float32)



def _cuda_stream():
    """Current torch CUDA stream handle (0 = legacy default stream).

    Passing the live stream keeps the zero-copy launchers ordered with
    surrounding torch work and makes them CUDA-graph capturable.
    """
    return torch.cuda.current_stream().cuda_stream


def _host_int_array(t):
    """Host-origin value (list / numpy / CPU tensor) -> int numpy array,
    or None when t is a device-resident tensor."""
    if _is_torch(t):
        if t.is_cuda:
            return None
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _i32_cuda_arg(t, device):
    """Coerce a host-side value / torch tensor to a contiguous int32 CUDA
    tensor (lens, block tables). Device-resident tensors pass through
    with at most a device-side dtype/contiguity fixup - their VALUES are
    never read back: a readback would sync the stream (breaking the
    zero-copy contract and CUDA-graph capture), the same trust boundary
    as a raw pointer. Host-origin values ARE validated by the callers
    (via _host_int_array) before the upload."""
    host = _host_int_array(t)
    if host is not None:
        return torch.from_numpy(
            np.ascontiguousarray(host, dtype=np.int32)).to(device)
    tt = t
    if tt.dtype is not torch.int32:
        tt = tt.to(torch.int32)
    if not tt.is_contiguous():
        tt = tt.contiguous()
    return tt


def _lens_arg(lens, batch, limit, device, upper_inclusive=True):
    """Validated lens tensor for the attention kernels: 1-D int32 with
    batch entries, values in [0, limit] (or [0, limit) when the caller
    marks the upper bound exclusive - kv_append writes AT lens[b], so
    lens[b] itself must be a mapped position). Device-resident lens
    tensors are trusted; host-origin values are checked before upload."""
    host = _host_int_array(lens)
    if host is not None:
        host = np.ascontiguousarray(host, dtype=np.int32)
        if host.ndim != 1 or host.shape[0] != batch:
            raise ValueError("lens must be 1-D with batch entries")
        bracket = "]" if upper_inclusive else ")"
        if host.size and (host.min() < 0 or host.max() > limit
                          or (not upper_inclusive and host.max() == limit)):
            raise ValueError(f"lens entries must be in [0, {limit}{bracket}")
        return torch.from_numpy(host).to(device)
    lt = _i32_cuda_arg(lens, device)
    if lt.ndim != 1 or lt.numel() != batch:
        raise ValueError("lens must be 1-D with batch entries")
    return lt


def _table_arg(table, batch, nblocks, device):
    """Validated block_table tensor: [B, S] int32, host-origin values in
    [0, nblocks). Device-resident tables are trusted (see
    _i32_cuda_arg for the trust boundary)."""
    host = _host_int_array(table)
    if host is not None:
        host = np.ascontiguousarray(host, dtype=np.int32)
        if host.ndim != 2 or host.shape[0] != batch:
            raise ValueError("block_table must be [B, S] with batch rows")
        if host.size and (host.min() < 0 or host.max() >= nblocks):
            raise ValueError(f"block_table entries must be in [0, {nblocks})")
        return torch.from_numpy(host).to(device)
    bt = _i32_cuda_arg(table, device)
    if bt.ndim != 2 or bt.shape[0] != batch:
        raise ValueError("block_table must be [B, S] with batch rows")
    return bt


def _ids_arg(ids, vocab, device, name):
    """Coerced token-id tensor for the penalty kernels: 1-D int64 on
    `device`. Host-origin values are validated against [0, vocab) before
    the upload; a device-resident tensor is trusted (no stream sync -
    same boundary as _i32_cuda_arg)."""
    if _is_torch(ids) and ids.is_cuda:
        it = ids
        if it.dtype is not torch.int64:
            it = it.to(torch.int64)
        if not it.is_contiguous():
            it = it.contiguous()
        if it.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
        return it
    if _is_torch(ids):
        host = np.ascontiguousarray(ids.detach().cpu().numpy(),
                                    dtype=np.int64)
        if host.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
    else:
        host = np.ascontiguousarray(np.asarray(ids),
                                    dtype=np.int64).ravel()
    if host.size and (host.min() < 0 or host.max() >= vocab):
        raise ValueError("token id out of range")
    return torch.from_numpy(host).to(device)


def _as_numpy(x, name):
    """Anything -> float32 C-contiguous numpy array (copy only if needed)."""
    if isinstance(x, np.ndarray):
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x
    if _is_torch(x):
        return _torch_to_numpy(x, name)
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def _device_path(x, cuda):
    """Decide the execution path for input x.

    Returns one of: 'torch-cuda' (zero-copy GPU), 'staged' (GPU with
    host staging), 'cpu'.
    """
    if _is_torch(x) and x.is_cuda:
        return "torch-cuda"
    return "staged" if cuda else "cpu"


# ---------------------------------------------------------------------------
# elementwise unary
# ---------------------------------------------------------------------------


def _unary(x, cuda, name):
    """Unary elementwise op dispatch: `name` selects the staged/cpu
    bindings and the launcher family (`name`/`name + '_cpu'`/
    `name + '_launch'[_bf16]`)."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_float(x, "x")
        out = torch.empty_like(x)
        _launch_for_dtype(x.dtype, name + "_launch")(
            x.data_ptr(), out.data_ptr(), x.numel(), _cuda_stream())
        return out
    arr = _as_numpy(x, "x")
    if path == "staged":
        res = getattr(_fusedtok, name)(arr)
    else:
        res = getattr(_fusedtok, name + "_cpu")(arr)
    return _numpy_to_torch_like(res) if _is_torch(x) else res


def silu(x, *, cuda=False):
    """SiLU / Swish activation: ``v * sigmoid(v)``."""
    return _unary(x, cuda, "silu")


def gelu(x, *, cuda=False):
    """GeLU activation, exact erf form: ``0.5 v (1 + erf(v / sqrt(2)))``."""
    return _unary(x, cuda, "gelu")


def gelu_tanh(x, *, cuda=False):
    """GeLU activation, tanh approximation (BERT/GPT checkpoint variant)."""
    return _unary(x, cuda, "gelu_tanh")


def relu(x, *, cuda=False):
    """ReLU activation: ``max(v, 0)``."""
    return _unary(x, cuda, "relu")


def tanh(x, *, cuda=False):
    """Hyperbolic tangent activation."""
    return _unary(x, cuda, "tanh")


def sigmoid(x, *, cuda=False):
    """Logistic sigmoid: ``1 / (1 + exp(-v))``."""
    return _unary(x, cuda, "sigmoid")


def temperature(x, t, *, cuda=False):
    """Logit temperature scaling: ``x / t`` with ``t > 0``.

    ``t < 1`` sharpens the distribution, ``t > 1`` flattens it. The CUDA
    path is float32-only (logits are float32 by convention).
    """
    if not t > 0.0:
        raise ValueError("temperature must be > 0")
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        out = torch.empty_like(x)
        _fusedtok.temperature_launch(x.data_ptr(), out.data_ptr(),
                                     x.numel(), t, _cuda_stream())
        return out
    arr = _as_numpy(x, "x")
    if path == "staged":
        res = _fusedtok.temperature(arr, t)
    else:
        res = _fusedtok.temperature_cpu(arr, t)
    return _numpy_to_torch_like(res) if _is_torch(x) else res


def axpy(x, a=1.0, b=0.0, *, cuda=False):
    """Skeleton demo op: ``y = a * x + b`` elementwise (float32 CUDA
    path)."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        out = torch.empty_like(x)
        _fusedtok.axpy_launch(x.data_ptr(), out.data_ptr(),
                              x.numel(), a, b, _cuda_stream())
        return out
    arr = _as_numpy(x, "x")
    if path == "staged":
        res = _fusedtok.axpy(arr, a, b)
    else:
        res = _fusedtok.axpy_cpu(arr, a, b)
    return _numpy_to_torch_like(res) if _is_torch(x) else res


# ---------------------------------------------------------------------------
# elementwise binary
# ---------------------------------------------------------------------------


def _binary(a, b, cuda, name):
    """Binary elementwise op dispatch (same naming convention as _unary).
    The operand names keep validation errors honest about WHICH input
    failed."""
    name_a, name_b = ("gate", "up") if name == "swiglu" else ("a", "b")
    if _is_torch(a) and a.is_cuda:
        _require_cuda(b, name_b)
        _check_torch_float(a, name_a)
        _check_torch_float(b, name_b)
        if a.dtype is not b.dtype:
            raise TypeError("inputs must have the same dtype")
        if a.shape != b.shape:
            raise ValueError("inputs must have the same shape")
        out = torch.empty_like(a)
        _launch_for_dtype(a.dtype, name + "_launch")(
            a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(),
            _cuda_stream())
        return out
    arr_a = _as_numpy(a, name_a)
    arr_b = _as_numpy(b, name_b)
    if arr_a.shape != arr_b.shape:
        raise ValueError("inputs must have the same shape")
    if cuda:
        res = getattr(_fusedtok, name)(arr_a, arr_b)
    else:
        res = getattr(_fusedtok, name + "_cpu")(arr_a, arr_b)
    return _numpy_to_torch_like(res) if _is_torch(a) else res


def add(a, b, *, cuda=False):
    """Elementwise ``a + b`` (the fused add + residual pattern)."""
    return _binary(a, b, cuda, "add")


def mul(a, b, *, cuda=False):
    """Elementwise ``a * b``."""
    return _binary(a, b, cuda, "mul")


def swiglu(gate, up, *, cuda=False):
    """SwiGLU activation: ``silu(gate) * up``."""
    return _binary(gate, up, cuda, "swiglu")


# ---------------------------------------------------------------------------
# row-wise normalization / softmax
# ---------------------------------------------------------------------------


def rmsnorm(x, weight, *, residual=None, eps=1e-6, cuda=False):
    """RMSNorm (LLaMA/Qwen style), optionally fused with a residual add:

    ``y = (x + r) * rsqrt(mean((x + r)^2) + eps) * weight``

    x: [rows, cols] (or 1-D [cols]), weight: [cols], residual: same shape as x.
    """
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_float(x, "x")
        _require_cuda(weight, "weight")
        weight = _norm_weight_f32(weight)
        r_ptr = None
        if residual is not None:
            _require_cuda(residual, "residual")
            _check_torch_float(residual, "residual")
            if residual.dtype is not x.dtype:
                raise TypeError("residual must have the same dtype as x")
            if residual.shape != x.shape:
                raise ValueError("residual must have the same shape as x")
            r_ptr = residual.data_ptr()
        rows, cols = _shape_rows_cols(x)
        out = torch.empty_like(x)
        _launch_for_dtype(x.dtype, "rmsnorm_launch")(
            x.data_ptr(), weight.data_ptr(), r_ptr, out.data_ptr(),
            rows, cols, eps, _cuda_stream())
        return out
    arr_x = _as_numpy(x, "x")
    args = (arr_x, _as_numpy(weight, "weight"),
            None if residual is None else _as_numpy(residual, "residual"),
            eps)
    if path == "staged":
        res = _fusedtok.rmsnorm(*args)
    else:
        res = _fusedtok.rmsnorm_cpu(*args)
    return _numpy_to_torch_like(res) if _is_torch(x) else res


def _shape_rows_cols(t):
    if t.ndim == 1:
        return 1, t.shape[0]
    if t.ndim == 2:
        return t.shape[0], t.shape[1]
    raise ValueError("expected a 1-D or 2-D tensor")


def layernorm(x, weight, bias, *, eps=1e-6, cuda=False):
    """LayerNorm with affine transform:

    ``y = (x - mean) / sqrt(var + eps) * weight + bias``

    x: [rows, cols] (or 1-D [cols]), weight/bias: [cols].
    """
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_float(x, "x")
        for name, tv in (("weight", weight), ("bias", bias)):
            _require_cuda(tv, name)
        weight = _norm_weight_f32(weight)
        bias = _norm_weight_f32(bias)
        rows, cols = _shape_rows_cols(x)
        out = torch.empty_like(x)
        _launch_for_dtype(x.dtype, "layernorm_launch")(
            x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
            out.data_ptr(), rows, cols, eps, _cuda_stream())
        return out
    arr_x = _as_numpy(x, "x")
    args = (arr_x, _as_numpy(weight, "weight"), _as_numpy(bias, "bias"), eps)
    if path == "staged":
        res = _fusedtok.layernorm(*args)
    else:
        res = _fusedtok.layernorm_cpu(*args)
    return _numpy_to_torch_like(res) if _is_torch(x) else res


def softmax(x, *, cuda=False):
    """Row-wise numerically stable softmax over the last dimension."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_float(x, "x")
        rows, cols = _shape_rows_cols(x)
        out = torch.empty_like(x)
        _launch_for_dtype(x.dtype, "softmax_launch")(
            x.data_ptr(), out.data_ptr(), rows, cols, _cuda_stream())
        return out
    arr_x = _as_numpy(x, "x")
    if path == "staged":
        res = _fusedtok.softmax(arr_x)
    else:
        res = _fusedtok.softmax_cpu(arr_x)
    return _numpy_to_torch_like(res) if _is_torch(x) else res


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rope(q, k=None, *, theta=10000.0, pos_offset=0, neox=False, cuda=False):
    """Rotary position embedding on (q, k).

    q, k: [seq, dim] with even dim. Two pair layouts:

    - ``neox=False``: interleaved pairs (2j, 2j+1) - original RoFormer form
    - ``neox=True``: rotate_half layout across the row halves
      (GPT-NeoX / LLaMA-HF checkpoints)

    ``pos_offset`` gives the absolute position of row 0, for kv-cache
    decoding where new tokens continue an existing sequence.
    Returns ``(q_rotated, k_rotated)``; the second element is None if k is.
    """
    path = _device_path(q, cuda)
    if path == "torch-cuda":
        _check_torch_float(q, "q")
        if q.ndim != 2:
            raise ValueError("q must be 2-D [seq, dim]")
        seq, dim = q.shape
        if dim % 2:
            raise ValueError("dim must be even (RoPE pairs elements)")
        if pos_offset < 0:
            raise ValueError("pos_offset must be >= 0")
        q_out = torch.empty_like(q)
        k_out = None
        if k is not None:
            _require_cuda(k, "k")
            _check_torch_float(k, "k")
            if k.dtype is not q.dtype:
                raise TypeError("k must have the same dtype as q")
            if k.shape != q.shape:
                raise ValueError("k must have the same shape as q")
            k_out = torch.empty_like(k)
        # float32 ships one bool-dispatch binding; bf16 has per-layout
        # kernels (the suffix convention cannot express the bool flag)
        if q.dtype is torch.bfloat16:
            def _rot(src, dst):
                launch = (_fusedtok.rope_neox_launch_bf16 if neox
                          else _fusedtok.rope_launch_bf16)
                launch(src, dst, seq, dim, theta, pos_offset,
                       _cuda_stream())
        else:
            def _rot(src, dst):
                _fusedtok.rope_launch(neox, src, dst, seq, dim, theta,
                                      pos_offset, _cuda_stream())
        _rot(q.data_ptr(), q_out.data_ptr())
        if k_out is not None:
            _rot(k.data_ptr(), k_out.data_ptr())
        return q_out, k_out
    arr_q = _as_numpy(q, "q")
    arr_k = None if k is None else _as_numpy(k, "k")
    call = _fusedtok.rope if path == "staged" else _fusedtok.rope_cpu
    q_res, k_res = call(neox, arr_q, arr_k, theta, pos_offset)
    if _is_torch(q):
        q_res = _numpy_to_torch_like(q_res)
        if k_res is not None:
            k_res = _numpy_to_torch_like(k_res)
    return q_res, k_res


# ---------------------------------------------------------------------------
# attention (decode step)
# ---------------------------------------------------------------------------


def attention_decode(q, k_cache, v_cache, lens=None, *, cuda=False):
    """Single-token (decode step) causal attention with GQA:

    ``out[b, h] = softmax(q[b,h] . K[b,kv(h)]^T / sqrt(D)) . V[b,kv(h)]``

    q: [B, Hq, D] (the new token's queries); k_cache / v_cache:
    [B, Hkv, T, D] contiguous kv-cache; optional lens: [B] ints giving
    each sequence's valid cache length (<= T; None = all rows valid, so
    rows past lens[b] are treated as padding and variable-length batches
    share one cache tensor). q head h attends with kv head
    ``h // (Hq // Hkv)`` - contiguous GQA groups; ``Hq == Hkv`` is plain
    MHA. Sequences with length 0 produce zero output rows.

    CUDA tensors may be float32, bfloat16 or float16 - output matches
    the input dtype and every accumulator stays float32, so the
    half-precision paths halve the kv-cache bytes (the decode-step
    bottleneck) without changing the softmax numerics. Non-CUDA inputs
    run the float32 CPU reference (numpy has no bf16/fp16).

    ``lens`` values are validated for host-side inputs (lists, numpy,
    CPU tensors) before the upload; a CUDA lens tensor is trusted as-is
    - reading it back would sync the stream and break CUDA-graph
    capture (the same trust boundary as raw device pointers).
    """
    path = _device_path(q, cuda)
    if path == "torch-cuda":
        for name, t in (("q", q), ("k_cache", k_cache), ("v_cache", v_cache)):
            _check_torch_att(t, name)
        for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
            if t.dtype is not q.dtype:
                raise TypeError(f"{name} must have the same dtype as q")
        if q.ndim != 3 or k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("q must be [B, Hq, D]; k/v caches [B, Hkv, T, D]")
        b, hq, d = q.shape
        b2, hkv, t_rows, d2 = k_cache.shape
        if (b2, d2) != (b, d) or v_cache.shape != k_cache.shape:
            raise ValueError("k_cache/v_cache must match q's batch and dim")
        if hq % hkv:
            raise ValueError("q heads must be a multiple of kv heads")
        lens_ptr = None
        if lens is not None:
            lt = _lens_arg(lens, b, t_rows, q.device)
            lens_ptr = lt.data_ptr()
        out = torch.empty((b, hq, d), dtype=q.dtype, device=q.device)
        if b * hq > 0:
            _att_launch_fn(q, "attention_decode_launch")(
                q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
                lens_ptr, out.data_ptr(), b, hq, hkv, t_rows, d,
                _cuda_stream())
        return out
    arr_q = _as_numpy(q, "q")
    arr_k = _as_numpy(k_cache, "k_cache")
    arr_v = _as_numpy(v_cache, "v_cache")
    if arr_q.ndim != 3 or arr_k.ndim != 4 or arr_v.ndim != 4:
        raise ValueError("q must be [B, Hq, D]; k/v caches [B, Hkv, T, D]")
    b, hq, d = arr_q.shape
    b2, hkv, t_rows, d2 = arr_k.shape
    if (b2, d2) != (b, d) or arr_v.shape != arr_k.shape:
        raise ValueError("k_cache/v_cache must match q's batch and dim")
    if hq % hkv:
        raise ValueError("q heads must be a multiple of kv heads")
    arr_lens = None if lens is None else np.ascontiguousarray(
        np.asarray(lens, dtype=np.int32))
    call = (_fusedtok.attention_decode if path == "staged"
            else _fusedtok.attention_decode_cpu)
    res = call(arr_q, arr_k, arr_v, arr_lens, b, hq, hkv, t_rows, d)
    return _numpy_to_torch_like(res) if _is_torch(q) else res


def attention_prefill(q, k, v, causal=True, *, cuda=False):
    """Prefill (fresh-sequence) attention over S query rows:

    ``out[b, h, i] = softmax(q . K^T / sqrt(D)) . V`` where query row i
    attends to key rows ``[0, i]`` (``causal=True``, the prefill
    diagonal) or all S rows (``causal=False``, bidirectional / encoder
    style).

    q: [B, Hq, S, D]; k / v: [B, Hkv, S, D]; out: [B, Hq, S, D]. Same
    contiguous-group GQA mapping as :func:`attention_decode` (Hq must be
    a multiple of Hkv). dim: multiple of 4, at most 512.

    CUDA tensors may be float32, bfloat16 or float16 - output matches
    the input dtype and every accumulator stays float32 (this is the
    convenience path: half precision halves the IO bytes, the heavy
    prefill still belongs to SDPA/FlashAttention). Non-CUDA inputs run
    the float32 CPU reference.
    """
    path = _device_path(q, cuda)
    if path == "torch-cuda":
        for name, t in (("q", q), ("k", k), ("v", v)):
            _check_torch_att(t, name)
        for name, t in (("k", k), ("v", v)):
            if t.dtype is not q.dtype:
                raise TypeError(f"{name} must have the same dtype as q")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be [B, heads, S, D]")
        b, hq, s, d = q.shape
        b2, hkv, s2, d2 = k.shape
        if (b2, s2, d2) != (b, s, d) or v.shape != k.shape:
            raise ValueError("k/v must match q's batch, seq and dim")
        if hq % hkv:
            raise ValueError("q heads must be a multiple of kv heads")
        out = torch.empty((b, hq, s, d), dtype=q.dtype, device=q.device)
        if b * hq * s > 0:
            _att_launch_fn(q, "attention_prefill_launch")(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                b, hq, hkv, s, d, bool(causal), _cuda_stream())
        return out
    arr_q = _as_numpy(q, "q")
    arr_k = _as_numpy(k, "k")
    arr_v = _as_numpy(v, "v")
    if arr_q.ndim != 4 or arr_k.ndim != 4 or arr_v.ndim != 4:
        raise ValueError("q/k/v must be [B, heads, S, D]")
    b, hq, s, d = arr_q.shape
    b2, hkv, s2, d2 = arr_k.shape
    if (b2, s2, d2) != (b, s, d) or arr_v.shape != arr_k.shape:
        raise ValueError("k/v must match q's batch, seq and dim")
    if hq % hkv:
        raise ValueError("q heads must be a multiple of kv heads")
    call = (_fusedtok.attention_prefill if path == "staged"
            else _fusedtok.attention_prefill_cpu)
    res = call(arr_q, arr_k, arr_v, b, hq, hkv, s, d, bool(causal))
    return _numpy_to_torch_like(res) if _is_torch(q) else res


def attention_decode_paged(q, k_pool, v_pool, block_table, lens=None,
                           *, cuda=False):
    """Single-token causal attention with GQA over a PAGED kv-cache:

    ``out[b, h] = softmax(q[b,h] . K[b,kv(h)]^T / sqrt(D)) . V[b,kv(h)]``

    The cache is a pool of fixed-size token blocks (the vLLM-style layout
    that keeps fragmentation out of the cache): k_pool / v_pool are
    [Nb, Hkv, P, D] (P = tokens per block, derived from the pool shape),
    and block_table [B, S] maps sequence b's token t to pool block
    ``block_table[b, t // P]``, offset ``t % P``. Optional lens: [B] ints
    with each sequence's valid length (None = every sequence uses its
    full table width, S * P rows).

    Same GQA mapping, zero-row-for-length-0 convention and dim limits as
    :func:`attention_decode`. The paged path supports GQA group sizes
    1/2/4/8/16 (other divisors: use the contiguous op). CUDA tensors may
    be float32, bfloat16 or float16 (output matches the input dtype,
    accumulators stay float32). Block-table VALUES are validated on the
    CPU/staged paths (ValueError) and trusted on the zero-copy path - a
    device table is not host-readable without a sync, the same trust
    boundary as raw pointers. CUDA-graph capture requires warming the
    shape up once outside the capture (the split workspace must
    pre-exist).
    """
    path = _device_path(q, cuda)
    if path == "torch-cuda":
        for name, t in (("q", q), ("k_pool", k_pool), ("v_pool", v_pool)):
            _check_torch_att(t, name)
        for name, t in (("k_pool", k_pool), ("v_pool", v_pool)):
            if t.dtype is not q.dtype:
                raise TypeError(f"{name} must have the same dtype as q")
        if q.ndim != 3 or k_pool.ndim != 4 or v_pool.ndim != 4:
            raise ValueError("q must be [B, Hq, D]; pools [Nb, Hkv, P, D]")
        b, hq, d = q.shape
        nb, hkv, page, d2 = k_pool.shape
        if d2 != d or v_pool.shape != k_pool.shape:
            raise ValueError("pools must match q's dim and each other")
        if hq % hkv:
            raise ValueError("q heads must be a multiple of kv heads")
        if hq // hkv not in (1, 2, 4, 8, 16):
            raise ValueError("paged decode supports GQA group sizes "
                             "1/2/4/8/16 (use attention_decode otherwise)")
        bt = _table_arg(block_table, b, nb, q.device)
        s_width = bt.shape[1]
        lens_ptr = None
        if lens is not None:
            lt = _lens_arg(lens, b, s_width * page, q.device)
            lens_ptr = lt.data_ptr()
        out = torch.empty((b, hq, d), dtype=q.dtype, device=q.device)
        if b * hq > 0:
            _att_launch_fn(q, "attention_decode_paged_launch")(
                q.data_ptr(), k_pool.data_ptr(), v_pool.data_ptr(),
                bt.data_ptr(), lens_ptr, out.data_ptr(),
                b, hq, hkv, page, s_width, d, _cuda_stream())
        return out
    arr_q = _as_numpy(q, "q")
    arr_k = _as_numpy(k_pool, "k_pool")
    arr_v = _as_numpy(v_pool, "v_pool")
    arr_t = np.ascontiguousarray(np.asarray(block_table, dtype=np.int32))
    if arr_q.ndim != 3 or arr_k.ndim != 4 or arr_v.ndim != 4 \
            or arr_t.ndim != 2:
        raise ValueError("q must be [B, Hq, D]; pools [Nb, Hkv, P, D]; "
                         "block_table [B, S]")
    b, hq, d = arr_q.shape
    nb, hkv, page, d2 = arr_k.shape
    if d2 != d or arr_v.shape != arr_k.shape:
        raise ValueError("pools must match q's dim and each other")
    if hq % hkv:
        raise ValueError("q heads must be a multiple of kv heads")
    if hq // hkv not in (1, 2, 4, 8, 16):
        raise ValueError("paged decode supports GQA group sizes "
                         "1/2/4/8/16 (use attention_decode otherwise)")
    if arr_t.shape[0] != b:
        raise ValueError("block_table must have one row per sequence")
    arr_lens = None if lens is None else np.ascontiguousarray(
        np.asarray(lens, dtype=np.int32))
    call = (_fusedtok.attention_decode_paged
            if path == "staged"
            else _fusedtok.attention_decode_paged_cpu)
    res = call(arr_q, arr_k, arr_v, arr_t, arr_lens, b, hq, hkv, page,
               arr_t.shape[1], nb, d)
    return _numpy_to_torch_like(res) if _is_torch(q) else res


def kv_append_paged(k_pool, v_pool, block_table, k_new, v_new, lens,
                    *, cuda=False):
    """Append ONE fresh token's k/v rows per sequence into the paged
    kv-cache (in place - the write side of the paged decode loop):

    sequence b's new rows ``k_new[b]``/``v_new[b]`` (each [Hkv, D]) land
    at pool block ``block_table[b, lens[b] // P]``, offset ``lens[b] % P``.

    k_pool / v_pool: [Nb, Hkv, P, D]; k_new / v_new: [B, Hkv, D]; lens
    [B] (REQUIRED - the write position is each sequence's current
    length). The block table itself is owned by the scheduler: this
    writes data into already-mapped blocks and never touches table
    entries (a position in an unmapped block is invalid input).

    CUDA pools may be float32, bfloat16 or float16 (rows copied in the
    storage dtype; k_new must match the pool dtype). Host-origin
    table/lens values are validated before the upload; device-resident
    tensors are trusted (same trust boundary as
    attention_decode_paged). Returns None; the pools are mutated in
    place.
    """
    path = _device_path(k_pool, cuda)
    if path == "torch-cuda":
        for name, t in (("k_pool", k_pool), ("v_pool", v_pool),
                        ("k_new", k_new), ("v_new", v_new)):
            _check_torch_att(t, name)
        for name, t in (("v_pool", v_pool), ("k_new", k_new),
                        ("v_new", v_new)):
            if t.dtype is not k_pool.dtype:
                raise TypeError(f"{name} must have the pool dtype")
        if k_pool.ndim != 4 or v_pool.ndim != 4:
            raise ValueError("pools must be [Nb, Hkv, P, D]")
        if k_new.ndim != 3 or v_new.ndim != 3:
            raise ValueError("k_new/v_new must be [B, Hkv, D]")
        nb, hkv, page, d = k_pool.shape
        b, hkv2, d2 = k_new.shape
        if (hkv2, d2) != (hkv, d) or v_pool.shape != k_pool.shape \
                or v_new.shape != k_new.shape:
            raise ValueError("k/v operands must match the pool layout")
        if b == 0:
            return None
        bt = _table_arg(block_table, b, nb, k_pool.device)
        lt = _lens_arg(lens, b, bt.shape[1] * page, k_pool.device,
                       upper_inclusive=False)
        _att_launch_fn(k_pool, "kv_append_paged_launch")(
            k_new.data_ptr(), v_new.data_ptr(), bt.data_ptr(),
            lt.data_ptr(), k_pool.data_ptr(), v_pool.data_ptr(),
            b, hkv, d, page, bt.shape[1], _cuda_stream())
        return None
    # in-place op: a dtype/layout conversion would silently drop the
    # mutation (numpy would view a converted copy), so pools must
    # already be float32 C-contiguous on the host paths
    for name, arr in (("k_pool", k_pool), ("v_pool", v_pool)):
        raw = arr.numpy() if _is_torch(arr) else arr
        if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 \
                or not raw.flags["C_CONTIGUOUS"]:
            raise TypeError(f"{name} must be a float32 C-contiguous array "
                            "(in-place op; conversion would drop writes)")
    arr_kp = _as_numpy(k_pool, "k_pool")
    arr_vp = _as_numpy(v_pool, "v_pool")
    arr_kn = _as_numpy(k_new, "k_new")
    arr_vn = _as_numpy(v_new, "v_new")
    arr_t = np.ascontiguousarray(np.asarray(block_table, dtype=np.int32))
    arr_l = np.ascontiguousarray(np.asarray(lens, dtype=np.int32))
    if arr_kp.ndim != 4 or arr_kn.ndim != 3 or arr_t.ndim != 2:
        raise ValueError("pools [Nb,Hkv,P,D]; k_new [B,Hkv,D]; table [B,S]")
    nb, hkv, page, d = arr_kp.shape
    b = arr_kn.shape[0]
    if arr_t.shape[0] != b or arr_l.shape != (b,):
        raise ValueError("block_table rows / lens must match the batch")
    if path == "staged":
        _fusedtok.kv_append_paged(arr_kn, arr_vn, arr_t, arr_l, arr_kp,
                                  arr_vp, b, hkv, d, page, arr_t.shape[1],
                                  nb)
    else:
        _fusedtok.kv_append_paged_cpu(arr_kn, arr_vn, arr_t, arr_l,
                                      arr_kp, arr_vp, b, hkv, d, page,
                                      arr_t.shape[1], nb)
    return None


def kv_append(k_cache, v_cache, k_new, v_new, lens, *, cuda=False):
    """Append ONE fresh token's k/v rows per sequence into the
    contiguous kv-cache (in place - the write side of the decode loop):

    sequence b's new rows ``k_new[b]``/``v_new[b]`` (each [Hkv, D]) land
    at cache row ``lens[b]`` of ``k_cache[b]`` / ``v_cache[b]``.

    k_cache / v_cache: [B, Hkv, T, D]; k_new / v_new: [B, Hkv, D]; lens
    [B] (REQUIRED - the write position is each sequence's current
    length). The contiguous twin of :func:`kv_append_paged`: the typical
    loop appends at ``lens[b]`` and then decodes with ``lens + 1``.

    CUDA caches may be float32, bfloat16 or float16 (rows copied in the
    storage dtype; k_new must match the cache dtype). Host-origin lens
    values are validated in ``[0, T)`` before the upload;
    device-resident tensors are trusted (same trust boundary as
    :func:`attention_decode`). Host paths require float32 C-contiguous
    caches (in-place op; a conversion would drop the writes). Returns
    None; the caches are mutated in place.
    """
    path = _device_path(k_cache, cuda)
    if path == "torch-cuda":
        for name, t in (("k_cache", k_cache), ("v_cache", v_cache),
                        ("k_new", k_new), ("v_new", v_new)):
            _check_torch_att(t, name)
        for name, t in (("v_cache", v_cache), ("k_new", k_new),
                        ("v_new", v_new)):
            if t.dtype is not k_cache.dtype:
                raise TypeError(f"{name} must have the cache dtype")
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("caches must be [B, Hkv, T, D]")
        if k_new.ndim != 3 or v_new.ndim != 3:
            raise ValueError("k_new/v_new must be [B, Hkv, D]")
        b, hkv, t_rows, d = k_cache.shape
        b2, hkv2, d2 = k_new.shape
        if (b2, hkv2, d2) != (b, hkv, d) \
                or v_cache.shape != k_cache.shape \
                or v_new.shape != k_new.shape:
            raise ValueError("k/v operands must match the cache layout")
        if b == 0:
            return None
        lt = _lens_arg(lens, b, t_rows, k_cache.device,
                       upper_inclusive=False)
        _att_launch_fn(k_cache, "kv_append_launch")(
            k_new.data_ptr(), v_new.data_ptr(), lt.data_ptr(),
            k_cache.data_ptr(), v_cache.data_ptr(),
            b, hkv, d, t_rows, _cuda_stream())
        return None
    # in-place op: a dtype/layout conversion would silently drop the
    # mutation (numpy would view a converted copy), so caches must
    # already be float32 C-contiguous on the host paths
    for name, arr in (("k_cache", k_cache), ("v_cache", v_cache)):
        raw = arr.numpy() if _is_torch(arr) else arr
        if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 \
                or not raw.flags["C_CONTIGUOUS"]:
            raise TypeError(f"{name} must be a float32 C-contiguous array "
                            "(in-place op; conversion would drop writes)")
    arr_kc = _as_numpy(k_cache, "k_cache")
    arr_vc = _as_numpy(v_cache, "v_cache")
    arr_kn = _as_numpy(k_new, "k_new")
    arr_vn = _as_numpy(v_new, "v_new")
    arr_l = np.ascontiguousarray(np.asarray(lens, dtype=np.int32))
    if arr_kc.ndim != 4 or arr_kn.ndim != 3:
        raise ValueError("caches [B,Hkv,T,D]; k_new [B,Hkv,D]")
    b, hkv, t_rows, d = arr_kc.shape
    if arr_l.shape != (arr_kn.shape[0],):
        raise ValueError("lens must have one entry per sequence")
    if path == "staged":
        _fusedtok.kv_append(arr_kn, arr_vn, arr_l, arr_kc, arr_vc,
                            b, hkv, d, t_rows)
    else:
        _fusedtok.kv_append_cpu(arr_kn, arr_vn, arr_l, arr_kc, arr_vc,
                                b, hkv, d, t_rows)
    return None


# ---------------------------------------------------------------------------
# sampling / logits post-processing
# ---------------------------------------------------------------------------


def argmax(x, *, cuda=False):
    """Index of the largest element (earliest index on ties)."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        if x.ndim != 1:
            raise ValueError("x must be 1-D for argmax")
        if x.numel() == 0:
            # the GPU launcher skips empty inputs without writing the
            # output slot - raise here to match the CPU reference
            raise ValueError("argmax of empty input")
        out = torch.empty(1, dtype=torch.int32, device=x.device)
        _fusedtok.argmax_launch(x.data_ptr(), out.data_ptr(),
                                x.numel(), _cuda_stream())
        return int(out.item())
    arr = _as_numpy(x, "x")
    if arr.ndim != 1:
        raise ValueError("x must be 1-D for argmax")
    call = _fusedtok.argmax if path == "staged" else _fusedtok.argmax_cpu
    return int(call(arr))


def topk(x, k, *, cuda=False):
    """Top-k selection: the k largest elements and their indices,
    descending, earliest index on ties. Returns ``(values, indices)``."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        if x.ndim != 1:
            raise ValueError("x must be 1-D for topk")
        n = x.numel()
        if not 0 <= k <= n:
            raise ValueError("k must be in [0, n]")
        vals = torch.empty(k, dtype=torch.float32, device=x.device)
        idxs = torch.empty(k, dtype=torch.int64, device=x.device)
        if k > 0:
            _fusedtok.topk_launch(x.data_ptr(), vals.data_ptr(),
                                  idxs.data_ptr(), n, k, _cuda_stream())
        return vals, idxs
    arr = _as_numpy(x, "x")
    if arr.ndim != 1:
        raise ValueError("x must be 1-D for topk")
    call = _fusedtok.topk if path == "staged" else _fusedtok.topk_cpu
    vals, idxs = call(arr, k)
    if _is_torch(x):
        vals = _numpy_to_torch_like(vals)
        idxs = torch.from_numpy(idxs)
    return vals, idxs


def topp(probs, p, *, cuda=False):
    """Top-p (nucleus) selection: the smallest set of highest-probability
    elements whose cumulative mass reaches p (crossing element included).
    Returns ``(values, indices)`` descending."""
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    path = _device_path(probs, cuda)
    if path == "torch-cuda":
        _check_torch_f32(probs, "probs")
        if probs.ndim != 1:
            raise ValueError("probs must be 1-D for topp")
        n = probs.numel()
        if n == 0:
            return (torch.empty(0, dtype=torch.float32, device=probs.device),
                    torch.empty(0, dtype=torch.int64, device=probs.device))
        vals = torch.empty(n, dtype=torch.float32, device=probs.device)
        idxs = torch.empty(n, dtype=torch.int64, device=probs.device)
        cnt = torch.empty(1, dtype=torch.int32, device=probs.device)
        _fusedtok.topp_select_launch(probs.data_ptr(), vals.data_ptr(),
                                     idxs.data_ptr(), n, p, cnt.data_ptr(),
                                     _cuda_stream())
        c = int(cnt.item())
        return vals[:c], idxs[:c]
    arr = _as_numpy(probs, "probs")
    if arr.ndim != 1:
        raise ValueError("probs must be 1-D for topp")
    call = _fusedtok.topp if path == "staged" else _fusedtok.topp_cpu
    vals, idxs = call(arr, p)
    if _is_torch(probs):
        vals = _numpy_to_torch_like(vals)
        idxs = torch.from_numpy(idxs)
    return vals, idxs


def repetition_penalty(logits, token_ids, penalty, *, cuda=False):
    """CTRL-style repetition penalty on logits before sampling:

    for every id in token_ids (previously generated tokens):
    ``logit /= penalty`` if positive, ``logit *= penalty`` if negative.

    logits: 1-D [vocab]; token_ids: 1-D ints; penalty > 0 (1.0 = disabled).
    Host-side token id values are validated against the vocab; a CUDA
    ids tensor is trusted (no stream sync - see the lens note in
    :func:`attention_decode`).
    """
    if not penalty > 0.0:
        raise ValueError("penalty must be > 0")
    path = _device_path(logits, cuda)
    if path == "torch-cuda":
        _check_torch_f32(logits, "logits")
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        n = logits.numel()
        ids = _ids_arg(token_ids, n, logits.device, "token_ids")
        out = torch.empty_like(logits)
        _fusedtok.repetition_penalty_launch(
            logits.data_ptr(), ids.data_ptr(), out.data_ptr(),
            n, ids.numel(), penalty, _cuda_stream())
        return out
    arr = _as_numpy(logits, "logits")
    ids = np.asarray(token_ids, dtype=np.int64).ravel()
    call = (_fusedtok.repetition_penalty if path == "staged"
            else _fusedtok.repetition_penalty_cpu)
    res = call(arr, ids, penalty)
    return _numpy_to_torch_like(res) if _is_torch(logits) else res


def quantize_int8(x):
    """Symmetric per-tensor INT8 quantization (storage path).

    ``scale = max(|x|) / 127``; ``q = clamp(round(x / scale), -127, 127)``.
    Returns ``(q, scale)`` where q matches the input family and scale is
    a Python float on EVERY path (reading the scale back is inherent to
    returning a host value; every consumer - dequantize, qgemm - takes a
    host float, so a device-resident scale would just defer the same
    readback with an inconsistent return type).
    """
    path = _device_path(x, cuda=False)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        scale = torch.empty(1, dtype=torch.float32, device=x.device)
        _fusedtok.quantize_launch(x.data_ptr(), q.data_ptr(),
                                  scale.data_ptr(), x.numel(),
                                  _cuda_stream())
        return q, float(scale)
    arr = _as_numpy(x, "x")
    q, s = _fusedtok.quantize_int8_cpu(arr)
    if _is_torch(x):
        return torch.from_numpy(q), s
    return q, s


def dequantize_int8(q, scale):
    """Dequantize int8 to float32: ``x = q * scale``.

    Accepts the UNPACKED pair from :func:`quantize_int8`:
    ``dequantize_int8(*quantize_int8(x))`` - passing the tuple itself is
    not valid. The CUDA path requires an int8 C-contiguous tensor (a
    wrong dtype or layout would be read as raw bytes); numpy and CPU
    tensors take the C++ CPU reference."""
    if _device_path(q, cuda=False) == "torch-cuda":
        if q.dtype is not torch.int8:
            raise TypeError(f"q must be int8, got {q.dtype}")
        _check_contiguous(q, "q")
        x = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        _fusedtok.dequantize_launch(q.data_ptr(), x.data_ptr(),
                                    float(scale), q.numel(),
                                    _cuda_stream())
        return x
    arr = np.asarray(q)
    out = _fusedtok.dequantize_int8_cpu(arr, float(scale))
    return _numpy_to_torch_like(out) if _is_torch(q) else out


def qadd_int8(qa, sa, qb, sb):
    """Fused dequant-add-requant for int8 tensors: computes
    ``qa*sa + qb*sb`` in float32 and requantizes with the output's own
    per-tensor scale. Returns ``(qy, out_scale)`` with out_scale a
    Python float (same readback note as :func:`quantize_int8`). One
    device pass instead of dequant -> add -> quant round trips."""
    _require_cuda(qa, "qa")
    _require_cuda(qb, "qb")
    if qa.dtype is not torch.int8 or qb.dtype is not torch.int8:
        raise TypeError("inputs must be int8")
    _check_contiguous(qa, "qa")
    _check_contiguous(qb, "qb")
    if qa.shape != qb.shape:
        raise ValueError("inputs must have the same shape")
    qy = torch.empty(qa.shape, dtype=torch.int8, device=qa.device)
    out_scale = torch.empty(1, dtype=torch.float32, device=qa.device)
    _fusedtok.qadd_launch(qa.data_ptr(), qb.data_ptr(), float(sa), float(sb),
                          qy.data_ptr(), out_scale.data_ptr(), qa.numel(),
                          _cuda_stream())
    return qy, float(out_scale)


def _qgemm_operands(a_q, b_q):
    """Shared CUDA-path validation for qgemm / qgemm_perchannel: both
    int8, both 2-D row-major-along-K, inner dims matching. Returns
    (m, n, k)."""
    if not (_is_torch(b_q) and b_q.is_cuda):
        raise TypeError("both operands must be CUDA int8 tensors")
    if a_q.dtype is not torch.int8 or b_q.dtype is not torch.int8:
        raise TypeError("operands must be int8")
    _check_contiguous(a_q, "a_q")
    _check_contiguous(b_q, "b_q")
    if a_q.ndim != 2 or b_q.ndim != 2:
        raise ValueError("operands must be 2-D [rows, K]")
    m, k = a_q.shape
    n, k2 = b_q.shape
    if k != k2:
        raise ValueError("inner dimensions must match")
    return m, n, k


def qgemm(a_q, a_scale, b_q, b_scale, *, cuda=False):
    """INT8 matmul with int32-exact accumulation:

    ``y[M, N] = (A_q[M, K] int8  @  B_q[N, K] int8 ^T) * (a_scale*b_scale)``

    Both operands are row-major along K - the LLM-friendly layout
    (``activations @ linear_weight.T``). ``M == 1`` dispatches to a
    warp-per-row GEMV kernel (the decode step). Results are bit-identical
    across the CPU / staged / zero-copy paths: integer accumulation is
    exact and the combined scale applies once at the store.
    """
    if _device_path(a_q, cuda=False) == "torch-cuda":
        m, n, k = _qgemm_operands(a_q, b_q)
        y = torch.empty((m, n), dtype=torch.float32, device=a_q.device)
        # the launcher no-ops empty operands and zero-fills K == 0
        if m > 0 and n > 0:
            _fusedtok.qgemm_launch(a_q.data_ptr(), b_q.data_ptr(),
                                   y.data_ptr(), m, n, k,
                                   float(a_scale), float(b_scale),
                                   _cuda_stream())
        return y
    a = np.ascontiguousarray(a_q, dtype=np.int8)
    b = np.ascontiguousarray(b_q, dtype=np.int8)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("operands must be 2-D [rows, K]")
    m, k = a.shape
    n, k2 = b.shape
    if k != k2:
        raise ValueError("inner dimensions must match")
    call = (_fusedtok.qgemm if cuda else _fusedtok.qgemm_cpu)
    res = call(a, b, m, n, k, float(a_scale), float(b_scale))
    return _numpy_to_torch_like(res) if _is_torch(a_q) else res


def qgemm_perchannel(a_q, a_scale, b_q, b_scales, *, cuda=False):
    """INT8 matmul with per-output-channel weight scales (W8A8):

    ``y[M, N] = (A_q[M, K] int8 @ B_q[N, K] int8^T) * (a_scale * b_scales[j])``

    ``b_scales`` is a float32 vector of length N - one scale per output
    row of B_q (per output channel of the layer), the SmoothQuant /
    TensorRT-LLM INT8 inference layout. Per-channel weight scales absorb
    the outlier structure of real weights that a single per-tensor scale
    cannot, at the same INT8 storage cost.

    Exactness contract (identical to :func:`qgemm`): the integer
    accumulation is exact, the output scale is composed as
    ``float32(a_scale * b_scales[j])`` with a single rounding, and the
    product applies once - CPU, staged and zero-copy results are
    BIT-IDENTICAL. ``M == 1`` dispatches to the warp-per-row GEMV.
    """
    if _device_path(a_q, cuda=False) == "torch-cuda":
        m, n, k = _qgemm_operands(a_q, b_q)
        if not _is_torch(b_scales):
            b_scales = torch.as_tensor(b_scales, dtype=torch.float32,
                                       device=a_q.device)
        if b_scales.ndim != 1 or b_scales.numel() != n:
            raise ValueError("b_scales must be 1-D with n entries")
        if b_scales.dtype is not torch.float32:
            b_scales = b_scales.to(torch.float32)
        if not b_scales.is_contiguous():
            b_scales = b_scales.contiguous()
        if not b_scales.is_cuda:
            b_scales = b_scales.to(a_q.device)
        y = torch.empty((m, n), dtype=torch.float32, device=a_q.device)
        # the launcher no-ops empty operands and zero-fills K == 0
        if m > 0 and n > 0:
            _fusedtok.qgemm_perchannel_launch(
                a_q.data_ptr(), b_q.data_ptr(), b_scales.data_ptr(),
                y.data_ptr(), m, n, k, float(a_scale), _cuda_stream())
        return y
    a = np.ascontiguousarray(a_q, dtype=np.int8)
    b = np.ascontiguousarray(b_q, dtype=np.int8)
    sb = np.ascontiguousarray(b_scales, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("operands must be 2-D [rows, K]")
    m, k = a.shape
    n, k2 = b.shape
    if k != k2:
        raise ValueError("inner dimensions must match")
    if sb.ndim != 1 or sb.shape[0] != n:
        raise ValueError("b_scales must be 1-D with n entries")
    call = (_fusedtok.qgemm_perchannel if cuda
            else _fusedtok.qgemm_perchannel_cpu)
    res = call(a, b, sb, m, n, k, float(a_scale))
    return _numpy_to_torch_like(res) if _is_torch(a_q) else res


def sample_topp(logits, p, *, temperature=1.0, seed=0, cuda=False):
    """Fused nucleus sampling: one GPU round trip from raw logits to a token.

    Pipeline (selection pipeline with a global-mass threshold): softmax of
    ``logits / temperature`` -> truncate to the smallest top-p nucleus ->
    inverse-CDF draw using a hash-uniform of ``seed``. Deterministic per
    seed; the RNG is a splitmix-style hash (reproducible, NOT
    cryptographically secure).

    Returns the sampled token id (int). ``p`` in (0, 1], temperature > 0.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    if not temperature > 0.0:
        raise ValueError("temperature must be > 0")
    path = _device_path(logits, cuda)
    if path == "torch-cuda":
        _check_torch_f32(logits, "logits")
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        return int(_fusedtok.sample_topp_launch(logits.data_ptr(),
                                                logits.numel(), p,
                                                temperature, seed,
                                                _cuda_stream()))
    arr = _as_numpy(logits, "logits")
    if arr.ndim != 1:
        raise ValueError("logits must be 1-D")
    call = (_fusedtok.sample_topp if path == "staged"
            else _fusedtok.sample_topp_cpu)
    return int(call(arr, p, temperature, seed))


def sample_topk(logits, k, *, temperature=1.0, seed=0, cuda=False):
    """Fused top-k sampling: one GPU round trip from raw logits to a token.

    Pipeline: softmax of ``logits / temperature`` -> keep the k
    highest-probability tokens -> renormalize WITHIN the k survivors ->
    inverse-CDF draw using a hash-uniform of ``seed``. Deterministic per
    seed; the RNG is a splitmix-style hash (reproducible, NOT
    cryptographically secure).

    Returns the sampled token id (int). ``k`` in ``[1, vocab]``
    (``k = 1`` is greedy; ``k >= vocab`` samples the whole distribution).
    temperature > 0.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if not temperature > 0.0:
        raise ValueError("temperature must be > 0")
    path = _device_path(logits, cuda)
    if path == "torch-cuda":
        _check_torch_f32(logits, "logits")
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        return int(_fusedtok.sample_topk_launch(logits.data_ptr(),
                                                logits.numel(), k,
                                                temperature, seed,
                                                _cuda_stream()))
    arr = _as_numpy(logits, "logits")
    if arr.ndim != 1:
        raise ValueError("logits must be 1-D")
    call = (_fusedtok.sample_topk if path == "staged"
            else _fusedtok.sample_topk_cpu)
    return int(call(arr, k, temperature, seed))


def sample_minp(logits, min_p, *, temperature=1.0, seed=0, cuda=False):
    """Fused min-p sampling: one GPU round trip from raw logits to a token.

    Pipeline: softmax of ``logits / temperature`` -> keep every token
    whose probability is at least ``min_p`` times the MAXIMUM
    probability -> renormalize within that nucleus -> inverse-CDF draw
    using a hash-uniform of ``seed``. Deterministic per seed; the RNG
    is a splitmix-style hash (reproducible, NOT cryptographically
    secure).

    Returns the sampled token id (int). ``min_p`` in (0, 1]
    (1.0 collapses to greedy among the max-probability tokens),
    temperature > 0.
    """
    if not 0.0 < min_p <= 1.0:
        raise ValueError("min_p must be in (0, 1]")
    if not temperature > 0.0:
        raise ValueError("temperature must be > 0")
    path = _device_path(logits, cuda)
    if path == "torch-cuda":
        _check_torch_f32(logits, "logits")
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        return int(_fusedtok.sample_minp_launch(logits.data_ptr(),
                                                logits.numel(), min_p,
                                                temperature, seed,
                                                _cuda_stream()))
    arr = _as_numpy(logits, "logits")
    if arr.ndim != 1:
        raise ValueError("logits must be 1-D")
    call = (_fusedtok.sample_minp if path == "staged"
            else _fusedtok.sample_minp_cpu)
    return int(call(arr, min_p, temperature, seed))


def decode_step(logits, sampled_ids, penalty=1.0, *, p=0.9, temperature=1.0,
                seed=0, cuda=False):
    """Fused decode step: one call from raw logits to the next token.

    Applies the CTRL-style repetition penalty over ``sampled_ids``
    (previously generated tokens), then the temperature scale, then
    nucleus-samples - the whole chain runs inside the selection pipeline
    (a vocab bitmap marks penalized ids; every logit read applies
    penalty then temperature, matching the composed reference order).

    Returns the sampled token id (int). Deterministic per seed.
    ``penalty`` > 0 (1.0 disables), ``p`` in (0, 1], temperature > 0.
    """
    if not penalty > 0.0:
        raise ValueError("penalty must be > 0")
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    if not temperature > 0.0:
        raise ValueError("temperature must be > 0")
    path = _device_path(logits, cuda)
    if path == "torch-cuda":
        _check_torch_f32(logits, "logits")
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        n = logits.numel()
        ids = _ids_arg(sampled_ids, n, logits.device, "sampled_ids")
        return int(_fusedtok.decode_step_launch(
            logits.data_ptr(), ids.data_ptr(), n, ids.numel(),
            penalty, p, temperature, seed, _cuda_stream()))
    arr = _as_numpy(logits, "logits")
    if arr.ndim != 1:
        raise ValueError("logits must be 1-D")
    ids = np.asarray(sampled_ids, dtype=np.int64).ravel()
    if ids.size and (ids.min() < 0 or ids.max() >= arr.size):
        raise ValueError("token id out of range")
    if path == "staged":
        return int(_fusedtok.decode_step(arr, ids, penalty, p, temperature,
                                         seed))
    # CPU reference: the composed three calls, same operation order
    penalized = _fusedtok.repetition_penalty_cpu(arr, ids, penalty)
    return int(_fusedtok.sample_topp_cpu(penalized, p, temperature, seed))
