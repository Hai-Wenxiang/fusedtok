"""fusedtok: small fused CUDA kernels for LLM inference.

The Python layer dispatches between three execution paths:

- torch CUDA tensors  -> zero-copy: kernels read/write torch device buffers
                         directly via data_ptr(); no staging, no host sync
                         (results are stream-ordered with other torch ops).
- numpy / torch CPU + ``cuda=True`` -> staged: data is copied to the GPU,
                         the kernel runs, results are copied back.
- numpy / torch CPU (default) -> the C++ CPU reference implementation
                         (ground truth; runs on machines without a GPU).

All functions accept float32 numpy arrays or torch tensors (any dtype is
accepted and converted to float32 with a copy; outputs are float32).
Row-wise ops accept 1-D (one row) or 2-D contiguous arrays.
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

__version__ = "0.2.0"

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
            f"{name} is a CUDA tensor; call with cuda=True or keep it on GPU "
            "(the zero-copy path is selected automatically)"
        )
    if t.dtype is not torch.float32:
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t.detach().numpy()


def _numpy_to_torch_like(arr, ref):
    """Fresh numpy array -> torch tensor sharing its memory."""
    return torch.from_numpy(arr)


def _check_torch_f32(t, name):
    if t.dtype is not torch.float32:
        raise TypeError(f"{name} must be float32, got {t.dtype} "
                        "(convert with .to(torch.float32))")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_torch_float(t, name):
    """Accept float32 or bfloat16 (the two dtypes with CUDA kernels)."""
    if t.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"{name} must be float32 or bfloat16, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _norm_weight_f32(weight, ref):
    """Norm weights must reach the kernel as float32 (they commonly are in
    checkpoints, and the bf16 kernels read them as float). Small [cols]
    upcast copy when the caller hands us bf16."""
    if weight.dtype is torch.float32:
        return weight
    return weight.to(torch.float32)


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


def _unary(x, cuda, staged, cpu, launch, launch_bf16=None):
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_float(x, "x")
        out = torch.empty_like(x)
        if x.dtype is torch.bfloat16 and launch_bf16 is not None:
            launch_bf16(x.data_ptr(), out.data_ptr(), x.numel())
        else:
            launch(x.data_ptr(), out.data_ptr(), x.numel())
        return out
    arr = _as_numpy(x, "x")
    res = staged(arr) if path == "staged" else cpu(arr)
    return _numpy_to_torch_like(res, x) if _is_torch(x) else res


def silu(x, *, cuda=False):
    """SiLU / Swish activation: ``v * sigmoid(v)``."""
    return _unary(x, cuda, _fusedtok.silu, _fusedtok.silu_cpu,
                  _fusedtok.silu_launch, _fusedtok.silu_launch_bf16)


def gelu(x, *, cuda=False):
    """GeLU activation, exact erf form: ``0.5 v (1 + erf(v / sqrt(2)))``."""
    return _unary(x, cuda, _fusedtok.gelu, _fusedtok.gelu_cpu,
                  _fusedtok.gelu_launch, _fusedtok.gelu_launch_bf16)


def gelu_tanh(x, *, cuda=False):
    """GeLU activation, tanh approximation (BERT/GPT checkpoint variant)."""
    return _unary(x, cuda, _fusedtok.gelu_tanh, _fusedtok.gelu_tanh_cpu,
                  _fusedtok.gelu_tanh_launch, _fusedtok.gelu_tanh_launch_bf16)


def relu(x, *, cuda=False):
    """ReLU activation: ``max(v, 0)``."""
    return _unary(x, cuda, _fusedtok.relu, _fusedtok.relu_cpu,
                  _fusedtok.relu_launch, _fusedtok.relu_launch_bf16)


def tanh(x, *, cuda=False):
    """Hyperbolic tangent activation."""
    return _unary(x, cuda, _fusedtok.tanh, _fusedtok.tanh_cpu,
                  _fusedtok.tanh_launch, _fusedtok.tanh_launch_bf16)


def sigmoid(x, *, cuda=False):
    """Logistic sigmoid: ``1 / (1 + exp(-v))``."""
    return _unary(x, cuda, _fusedtok.sigmoid, _fusedtok.sigmoid_cpu,
                  _fusedtok.sigmoid_launch, _fusedtok.sigmoid_launch_bf16)


def temperature(x, t, *, cuda=False):
    """Logit temperature scaling: ``x / t`` with ``t > 0``.

    ``t < 1`` sharpens the distribution, ``t > 1`` flattens it.
    """
    if not t > 0.0:
        raise ValueError("temperature must be > 0")
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        out = torch.empty_like(x)
        _fusedtok.temperature_launch(x.data_ptr(), out.data_ptr(), x.numel(), t)
        return out
    arr = _as_numpy(x, "x")
    res = _fusedtok.temperature(arr, t) if path == "staged" else _fusedtok.temperature_cpu(arr, t)
    return _numpy_to_torch_like(res, x) if _is_torch(x) else res


def axpy(x, a=1.0, b=0.0, *, cuda=False):
    """Skeleton demo op: ``y = a * x + b`` elementwise."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        out = torch.empty_like(x)
        _fusedtok.axpy_launch(x.data_ptr(), out.data_ptr(), x.numel(), a, b)
        return out
    arr = _as_numpy(x, "x")
    res = _fusedtok.axpy(arr, a, b) if path == "staged" else _fusedtok.axpy_cpu(arr, a, b)
    return _numpy_to_torch_like(res, x) if _is_torch(x) else res


# ---------------------------------------------------------------------------
# elementwise binary
# ---------------------------------------------------------------------------


def _binary(a, b, cuda, staged, cpu, launch, name, launch_bf16=None):
    if _is_torch(a) and a.is_cuda:
        if not (_is_torch(b) and b.is_cuda):
            raise TypeError("both inputs must be CUDA tensors")
        _check_torch_float(a, name)
        _check_torch_float(b, name)
        if a.dtype is not b.dtype:
            raise TypeError("inputs must have the same dtype")
        if a.shape != b.shape:
            raise ValueError("inputs must have the same shape")
        out = torch.empty_like(a)
        if a.dtype is torch.bfloat16 and launch_bf16 is not None:
            launch_bf16(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel())
        else:
            launch(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel())
        return out
    arr_a = _as_numpy(a, name)
    arr_b = _as_numpy(b, name)
    if arr_a.shape != arr_b.shape:
        raise ValueError("inputs must have the same shape")
    res = staged(arr_a, arr_b) if cuda else cpu(arr_a, arr_b)
    return _numpy_to_torch_like(res, a) if _is_torch(a) else res


def add(a, b, *, cuda=False):
    """Elementwise ``a + b`` (the fused add + residual pattern)."""
    return _binary(a, b, cuda, _fusedtok.add, _fusedtok.add_cpu,
                   _fusedtok.add_launch, "a", _fusedtok.add_launch_bf16)


def mul(a, b, *, cuda=False):
    """Elementwise ``a * b``."""
    return _binary(a, b, cuda, _fusedtok.mul, _fusedtok.mul_cpu,
                   _fusedtok.mul_launch, "a", _fusedtok.mul_launch_bf16)


def swiglu(gate, up, *, cuda=False):
    """SwiGLU activation: ``silu(gate) * up``."""
    return _binary(gate, up, cuda, _fusedtok.swiglu, _fusedtok.swiglu_cpu,
                   _fusedtok.swiglu_launch, "gate", _fusedtok.swiglu_launch_bf16)


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
        if not (_is_torch(weight) and weight.is_cuda):
            raise TypeError("weight must be a CUDA tensor when x is on CUDA")
        weight = _norm_weight_f32(weight, x)
        r_ptr = None
        if residual is not None:
            if not (_is_torch(residual) and residual.is_cuda):
                raise TypeError("residual must be a CUDA tensor when x is on CUDA")
            _check_torch_float(residual, "residual")
            if residual.dtype is not x.dtype:
                raise TypeError("residual must have the same dtype as x")
            if residual.shape != x.shape:
                raise ValueError("residual must have the same shape as x")
            r_ptr = residual.data_ptr()
        rows, cols = _shape_rows_cols(x)
        out = torch.empty_like(x)
        if x.dtype is torch.bfloat16:
            _fusedtok.rmsnorm_launch_bf16(x.data_ptr(), weight.data_ptr(),
                                          r_ptr, out.data_ptr(), rows, cols, eps)
        else:
            _fusedtok.rmsnorm_launch(x.data_ptr(), weight.data_ptr(), r_ptr,
                                     out.data_ptr(), rows, cols, eps)
        return out
    arr_x = _as_numpy(x, "x")
    res = (
        _fusedtok.rmsnorm(arr_x, _as_numpy(weight, "weight"),
                          None if residual is None else _as_numpy(residual, "residual"),
                          eps)
        if path == "staged"
        else _fusedtok.rmsnorm_cpu(arr_x, _as_numpy(weight, "weight"),
                                   None if residual is None else _as_numpy(residual, "residual"),
                                   eps)
    )
    return _numpy_to_torch_like(res, x) if _is_torch(x) else res


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
            if not (_is_torch(tv) and tv.is_cuda):
                raise TypeError(f"{name} must be a CUDA tensor when x is on CUDA")
        weight = _norm_weight_f32(weight, x)
        bias = _norm_weight_f32(bias, x)
        rows, cols = _shape_rows_cols(x)
        out = torch.empty_like(x)
        if x.dtype is torch.bfloat16:
            _fusedtok.layernorm_launch_bf16(x.data_ptr(), weight.data_ptr(),
                                            bias.data_ptr(), out.data_ptr(),
                                            rows, cols, eps)
        else:
            _fusedtok.layernorm_launch(x.data_ptr(), weight.data_ptr(),
                                       bias.data_ptr(), out.data_ptr(),
                                       rows, cols, eps)
        return out
    arr_x = _as_numpy(x, "x")
    args = (arr_x, _as_numpy(weight, "weight"), _as_numpy(bias, "bias"), eps)
    res = _fusedtok.layernorm(*args) if path == "staged" else _fusedtok.layernorm_cpu(*args)
    return _numpy_to_torch_like(res, x) if _is_torch(x) else res


def softmax(x, *, cuda=False):
    """Row-wise numerically stable softmax over the last dimension."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_float(x, "x")
        rows, cols = _shape_rows_cols(x)
        out = torch.empty_like(x)
        if x.dtype is torch.bfloat16:
            _fusedtok.softmax_launch_bf16(x.data_ptr(), out.data_ptr(), rows, cols)
        else:
            _fusedtok.softmax_launch(x.data_ptr(), out.data_ptr(), rows, cols)
        return out
    arr_x = _as_numpy(x, "x")
    res = _fusedtok.softmax(arr_x) if path == "staged" else _fusedtok.softmax_cpu(arr_x)
    return _numpy_to_torch_like(res, x) if _is_torch(x) else res


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
            if not (_is_torch(k) and k.is_cuda):
                raise TypeError("k must be a CUDA tensor when q is on CUDA")
            _check_torch_float(k, "k")
            if k.dtype is not q.dtype:
                raise TypeError("k must have the same dtype as q")
            if k.shape != q.shape:
                raise ValueError("k must have the same shape as q")
            k_out = torch.empty_like(k)
        if q.dtype is torch.bfloat16:
            def _launch_rope(neox_flag, src, dst, s, d, th, off):
                if neox_flag:
                    _fusedtok.rope_neox_launch_bf16(src, dst, s, d, th, off)
                else:
                    _fusedtok.rope_launch_bf16(src, dst, s, d, th, off)
        else:
            def _launch_rope(neox_flag, src, dst, s, d, th, off):
                _fusedtok.rope_launch(neox_flag, src, dst, s, d, th, off)
        _launch_rope(neox, q.data_ptr(), q_out.data_ptr(), seq, dim,
                     theta, pos_offset)
        if k_out is not None:
            _launch_rope(neox, k.data_ptr(), k_out.data_ptr(), seq, dim,
                         theta, pos_offset)
        return q_out, k_out
    arr_q = _as_numpy(q, "q")
    arr_k = None if k is None else _as_numpy(k, "k")
    call = _fusedtok.rope if path == "staged" else _fusedtok.rope_cpu
    q_res, k_res = call(neox, arr_q, arr_k, theta, pos_offset)
    if _is_torch(q):
        q_res = _numpy_to_torch_like(q_res, q)
        if k_res is not None:
            k_res = _numpy_to_torch_like(k_res, k)
    return q_res, k_res


# ---------------------------------------------------------------------------
# sampling / logits post-processing
# ---------------------------------------------------------------------------


def argmax(x, *, cuda=False):
    """Index of the largest element (earliest index on ties)."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        if x.ndim != 1:
            raise ValueError("argmax expects 1-D input")
        out = torch.empty(1, dtype=torch.int32, device=x.device)
        _fusedtok.argmax_launch(x.data_ptr(), out.data_ptr(), x.numel())
        return int(out.item())
    arr = _as_numpy(x, "x")
    if arr.ndim != 1:
        raise ValueError("argmax expects 1-D input")
    return int(_fusedtok.argmax(arr) if path == "staged" else _fusedtok.argmax_cpu(arr))


def topk(x, k, *, cuda=False):
    """Top-k selection: the k largest elements and their indices,
    descending, earliest index on ties. Returns ``(values, indices)``."""
    path = _device_path(x, cuda)
    if path == "torch-cuda":
        _check_torch_f32(x, "x")
        if x.ndim != 1:
            raise ValueError("topk expects 1-D input")
        n = x.numel()
        if not 0 <= k <= n:
            raise ValueError("k must be in [0, n]")
        vals = torch.empty(k, dtype=torch.float32, device=x.device)
        idxs = torch.empty(k, dtype=torch.int64, device=x.device)
        if k > 0:
            _fusedtok.topk_launch(x.data_ptr(), vals.data_ptr(),
                                  idxs.data_ptr(), n, k)
        return vals, idxs
    arr = _as_numpy(x, "x")
    if arr.ndim != 1:
        raise ValueError("topk expects 1-D input")
    call = _fusedtok.topk if path == "staged" else _fusedtok.topk_cpu
    vals, idxs = call(arr, k)
    if _is_torch(x):
        vals = _numpy_to_torch_like(vals, x)
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
            raise ValueError("topp expects 1-D input")
        n = probs.numel()
        if n == 0:
            return (torch.empty(0, dtype=torch.float32, device=probs.device),
                    torch.empty(0, dtype=torch.int64, device=probs.device))
        vals = torch.empty(n, dtype=torch.float32, device=probs.device)
        idxs = torch.empty(n, dtype=torch.int64, device=probs.device)
        cnt = torch.empty(1, dtype=torch.int32, device=probs.device)
        _fusedtok.topp_select_launch(probs.data_ptr(), vals.data_ptr(),
                                     idxs.data_ptr(), n, p, cnt.data_ptr())
        c = int(cnt.item())
        return vals[:c], idxs[:c]
    arr = _as_numpy(probs, "probs")
    if arr.ndim != 1:
        raise ValueError("topp expects 1-D input")
    call = _fusedtok.topp if path == "staged" else _fusedtok.topp_cpu
    vals, idxs = call(arr, p)
    if _is_torch(probs):
        vals = _numpy_to_torch_like(vals, probs)
        idxs = torch.from_numpy(idxs)
    return vals, idxs


def repetition_penalty(logits, token_ids, penalty, *, cuda=False):
    """CTRL-style repetition penalty on logits before sampling:

    for every id in token_ids (previously generated tokens):
    ``logit /= penalty`` if positive, ``logit *= penalty`` if negative.

    logits: 1-D [vocab]; token_ids: 1-D ints; penalty > 0 (1.0 = disabled).
    """
    if not penalty > 0.0:
        raise ValueError("penalty must be > 0")
    path = _device_path(logits, cuda)
    if path == "torch-cuda":
        _check_torch_f32(logits, "logits")
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        n = logits.numel()
        if _is_torch(token_ids):
            ids = token_ids
        else:
            ids = torch.as_tensor(token_ids, dtype=torch.int64)
        if ids.ndim != 1:
            raise ValueError("token_ids must be 1-D")
        if ids.numel() and (ids.min() < 0 or ids.max() >= n):
            raise ValueError("token id out of range")
        if not ids.is_cuda:
            ids = ids.to(logits.device)
        ids = ids.to(torch.int64).contiguous()
        out = torch.empty_like(logits)
        _fusedtok.repetition_penalty_launch(
            logits.data_ptr(), ids.data_ptr(), out.data_ptr(),
            n, ids.numel(), penalty)
        return out
    arr = _as_numpy(logits, "logits")
    ids = np.asarray(token_ids, dtype=np.int64).ravel()
    call = (_fusedtok.repetition_penalty if path == "staged"
            else _fusedtok.repetition_penalty_cpu)
    res = call(arr, ids, penalty)
    return _numpy_to_torch_like(res, logits) if _is_torch(logits) else res


def sample_topp(logits, p, *, temperature=1.0, seed=0, cuda=False):
    """Fused nucleus sampling: one GPU round trip from raw logits to a token.

    Pipeline (single cooperative kernel): softmax(logits / temperature) ->
    truncate to the smallest top-p nucleus -> inverse-CDF draw using a
    hash-uniform of ``seed``. Deterministic per seed; the RNG is a
    splitmix-style hash (reproducible, NOT cryptographically secure).

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
                                                temperature, seed))
    arr = _as_numpy(logits, "logits")
    if arr.ndim != 1:
        raise ValueError("logits must be 1-D")
    call = _fusedtok.sample_topp if path == "staged" else _fusedtok.sample_topp_cpu
    return int(call(arr, p, temperature, seed))
