# The INT8 path

fusedtok's INT8 path covers quantized storage and compute: symmetric
per-tensor quantization utilities, the integer-exact `qgemm` matmul,
and the per-channel `qgemm_perchannel` (W8A8) variant that real INT8
inference deploys.

**Other languages:** [中文：INT8 路径](../zh/int8.md)

- [Quantization utilities](#quantization-utilities)
- [The matmuls](#the-matmuls)
- [The exactness contract](#the-exactness-contract)
- [Performance, honestly](#performance-honestly)

## Quantization utilities

```python
q, scale = fusedtok.quantize_int8(x)     # scale = max|x|/127, a Python
                                         # float on EVERY path
x_back = fusedtok.dequantize_int8(q, scale)
qy, s_out = fusedtok.qadd_int8(qa, sa, qb, sb)   # fused dequant-add-requant
```

- `quantize_int8`: symmetric per-tensor - `scale = max|x| / 127`,
  `q = clamp(round(x / scale), -127, 127)`. The scale is read back to
  the host once at the source (every consumer takes a host float), so
  the return type is consistent across all execution paths.
- `dequantize_int8(q, scale)`: `x ~= q * scale`. Accepts the unpacked
  pair: `dequantize_int8(*quantize_int8(x))`. The zero-copy path
  requires int8 C-contiguous tensors (a wrong dtype would be read as
  raw bytes).
- `qadd_int8(qa, sa, qb, sb)`: computes `qa*sa + qb*sb` in float32 and
  requantizes with the output's own per-tensor scale, in one device
  pass - instead of a dequant -> add -> quant round trip.

`quantize_int8` / `qadd_int8` are the one documented exception to the
async-launcher contract: composing pass 2 needs the reduced absmax on
the host, so they sync the caller's stream once mid-call (the copies
ride the caller's stream and are error-checked).

## The matmuls

Both operands are row-major along K - the LLM-friendly layout, so
`activations @ linear_weight.T` needs no transpose:

```python
y = fusedtok.qgemm(a_q, a_scale, b_q, b_scale)
# y[M, N] = (A_q[M, K] int8 @ B_q[N, K] int8 ^T) * (a_scale * b_scale)

y = fusedtok.qgemm_perchannel(a_q, a_scale, b_q, b_scales)
# y[M, N] = (A_q @ B_q^T) * (a_scale * b_scales[j])    # W8A8
```

- `M == 1` dispatches to a bandwidth-bound warp-per-row GEMV kernel
  (the decode-step projection); larger `M` runs the tensor-core IMMA
  pipeline with runtime tile tuning (64x64 or 128x128 tiles,
  cp.async double-buffered).
- `qgemm_perchannel` is the layout INT8 inference actually uses
  (SmoothQuant / TensorRT-LLM style W8A8): activations carry one
  per-tensor scale, weights one scale **per output channel**
  (`b_scales[j]`, a float32 vector of length N). Per-channel scales
  absorb weight outliers that a single per-tensor scale cannot -
  the end-to-end tests quantify a 5x+ error reduction on spiky
  weights.

The zero-copy path requires int8 C-contiguous operands (and float32
for the scale vector); non-contiguous tensors are rejected rather
than read wrong.

## The exactness contract

Integer accumulation is exact int32, and the combined scale applies
exactly once at the store: CPU, staged and zero-copy results are
**bit-identical**. `qgemm_perchannel` composes its per-element scale
as `float32(a_scale * b_scales[j])` with a single rounding, matching
the CPU reference's order; with a constant `b_scales` vector its
output is bit-equal to per-tensor `qgemm`. Both facts are pinned by
tests.

No tolerance games: if an INT8 result differs across paths, that is a
bug.

## Performance, honestly

- The **decode GEMV** (`M == 1`) moves half the bytes of an fp16
  projection and runs at full memory bandwidth - roughly 2x the fp16
  projection. This is the per-token hot path and the point of INT8
  weights.
- The **pipelined IMMA GEMM** reaches ~39 TOPS on an RTX 3060 and ~67
  TOPS on an RTX 5060 Ti (2x-4x the original v0.4 kernel), but
  cuBLASLt (`torch._int_mm`) retains roughly a 2.1-2.6x lead with
  deeper-pipelined tiles and per-arch epilogues. fusedtok's INT8 path
  is the **exact / graph-capturable / zero-copy** one, not the
  fastest one; a CUTLASS-class schedule stays on the roadmap.
- `qgemm_perchannel`'s scale multiply is fused into the same epilogue
  at zero kernel cost - it is the composite torch reference that pays
  for the scale broadcast separately, which is where its ratio comes
  from.
