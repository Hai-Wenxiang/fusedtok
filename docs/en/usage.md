# fusedtok usage guide (English)

Topic-structured guide to using fusedtok. For the operator index and
benchmark tables see the [README](../../README.md); for a runnable tour of
every operator see [`examples/demo.py`](../../examples/demo.py). This
guide explains the execution model first (it is shared by every operator)
and then walks each operator family.

**Other languages:** [中文使用指南](../zh/usage.md)

- [Install and first steps](#install-and-first-steps)
- [The three execution paths](#the-three-execution-paths)
- [dtype support](#dtype-support)
- [Streams and CUDA graphs](#streams-and-cuda-graphs)
- [Attention](#attention)
- [Norms and RoPE](#norms-and-rope)
- [Elementwise and activations](#elementwise-and-activations)
- [Sampling and selection](#sampling-and-selection)
- [The INT8 path](#the-int8-path)
- [Error contract](#error-contract)
- [Running the benchmarks](#running-the-benchmarks)

## Install and first steps

```bash
pip install fusedtok
```

Prebuilt wheels cover Linux x86_64 (cp310-cp313) and Windows x86_64
(cp311-cp313), built with CUDA 12.4. Ampere (RTX 30, sm_80/86) runs the
shipped cubins natively; newer GPUs (RTX 40/50, H100, ...) JIT the PTX
fallback through the driver. Building from source needs CUDA Toolkit
>= 12.0 and a C++17 compiler.

Check that a usable GPU is visible to the extension:

```python
import fusedtok
print(fusedtok.cuda_available())   # True when a CUDA device is usable
```

Every operator also runs on CPU (a float32 reference implementation in
C++), so the same code works on machines without a GPU - handy for tests
and debugging.

## The three execution paths

Each operator accepts the same logical inputs in three flavors, and the
flavor picks the execution path automatically:

| You pass | Path | What happens |
|---|---|---|
| numpy arrays (default) | **CPU reference** | C++ float32 reference, no GPU involved |
| numpy arrays + `cuda=True` | **staged CUDA** | copies inputs to GPU, runs the kernel, copies back |
| CUDA torch tensors | **zero-copy CUDA** | kernels read/write torch's buffers via `data_ptr()`, no staging copies, no host sync |

```python
import numpy as np
import torch
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y1 = fusedtok.rmsnorm(x, w)                    # CPU reference
y2 = fusedtok.rmsnorm(x, w, cuda=True)         # staged CUDA (numpy in/out)

xt, wt = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)                  # zero-copy: CUDA torch in/out
```

Outputs follow the input family: numpy in gives numpy out, CUDA torch in
gives CUDA torch out. The zero-copy path is what an inference loop wants
- kernels launch on torch's current stream and interoperate with other
GPU work without hidden transfers.

## dtype support

| Operator family | numpy | CUDA torch |
|---|---|---|
| elementwise, activations, norms, RoPE | float32 | float32, bfloat16 |
| `attention_decode`, `attention_prefill` | float32 | float32, bfloat16, float16 |
| selection and sampling (`topk`, `sample_*`, ...) | float32 | float32 |
| INT8 ops (`qgemm`, ...) | int8 operands, float32 scale/out | same |

Rules worth knowing:

- Half-precision inputs keep float32 **compute**: loads widen at the
  boundary, stores narrow back round-to-nearest. The attention softmax
  and accumulators stay float32 on every dtype, so numerics change only
  through the input rounding.
- Attention output **matches the input dtype**; on other families the
  output matches the input dtype as well (bf16 in, bf16 out).
- Norm weights (`weight`, `bias`, `residual`) are upcast to float32
  automatically when the activations are half precision - checkpoints
  commonly store them in fp32.
- CPU/staged paths are always float32 (numpy has no bf16/fp16).

## Streams and CUDA graphs

Zero-copy launches ride torch's **current stream** (`torch.cuda.current_stream()`),
so ordinary stream ordering applies. The whole library is
CUDA-graph-capturable: capture with `torch.cuda.graph` as usual.

Two practical notes:

- Do a **warm-up call before capturing**. First calls may allocate a
  per-shape workspace (attention split path, selection pipeline) or tune
  a config (row-kernel block size, qgemm tiles); both happen outside
  captures by design - the warm-up gets them out of the way.
- Kernels cached in a graph read their per-call parameters from device
  memory, so replays observe new tensor contents written between
  replays. Mutate-in-place + replay re-computes, as the tests assert.

```python
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = fusedtok.rmsnorm(xt, wt)         # warm-up (tuning/workspace)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = fusedtok.rmsnorm(xt, wt)
g.replay()                                      # replayed as one launch batch
```

## Attention

### attention_decode - the decode step

```python
out = fusedtok.attention_decode(q, k_cache, v_cache, lens)
```

- `q`: `[B, Hq, D]` - the new token's query heads.
- `k_cache`, `v_cache`: `[B, Hkv, T, D]` - a **contiguous** kv-cache
  (`T` = allocated rows, not necessarily used rows).
- `lens`: optional `[B]` int32 (torch tensor or numpy) - per-sequence
  valid cache length. `None` means all `T` rows are valid. Sequences
  with length 0 produce zero output rows.

GQA mapping is **contiguous groups**: q head `h` uses kv head
`h // (Hq // Hkv)`. `Hq == Hkv` degenerates to plain MHA. This is the
layout used by LLaMA-style checkpoints where q heads of a group sit
next to each other.

Constraints: `Hq % Hkv == 0`, `D` a multiple of 4 and at most 512.
Long caches split automatically (flash-decoding style): the sequence is
cut into slices, partials computed in parallel, then reduced - one call
regardless of `T`.

Performance frame: decode attention is bandwidth-bound (it streams the
whole kv-cache once per token). f32 roughly saturates effective
bandwidth vs SDPA; bf16/fp16 caches halve the bytes. At batch 1 the
kernel is latency-bound, so half precision buys modest absolute time
there - the byte savings scale with batch.

### attention_decode_paged - the vLLM-style block-pool cache (v1.2)

```python
out = fusedtok.attention_decode_paged(q, k_pool, v_pool, block_table, lens)
```

- `k_pool`, `v_pool`: `[Nb, Hkv, P, D]` - a **pool of fixed-size token
  blocks** (`P` = tokens per block, read from the pool shape) instead of
  per-sequence contiguous spans: the layout that keeps memory
  fragmentation out of the cache as sequences grow, shrink and get
  evicted.
- `block_table`: `[B, S]` int32 - sequence `b`'s token `t` lives in
  pool block `block_table[b, t // P]` at offset `t % P`. Any valid table
  is allowed (non-monotonic, sharing, holes) - the kernel walks the
  indirection.
- `lens`: optional `[B]` - valid length per sequence (`None` = every
  sequence uses its full table width `S * P`).

Same math, GQA mapping, zero-row convention and dtype matrix as the
contiguous op; the split pipeline and its workspace are shared. GQA
group sizes 1/2/4/8/16 (other divisors: use the contiguous op).
Block-table **values** are validated on the CPU/staged paths
(`ValueError`) and trusted on the zero-copy path - a device table is
not host-readable without a sync, the same trust boundary as raw
pointers. CUDA-graph capture requires warming the shape up once outside
the capture (the split workspace must pre-exist).

Measured cost of the indirection (v1.2, 3060, b=1 GQA 32/8 D=128
T=16384 P=16): **1.06x** the contiguous op with bit-identical output on
matching slice schedules.

### kv_append_paged - writing tokens into the pool (v1.2)

```python
fusedtok.kv_append_paged(k_pool, v_pool, block_table, k_new, v_new, lens)
```

The cache-write side of the paged loop: sequence `b`'s fresh rows
`k_new[b]` / `v_new[b]` (each `[Hkv, D]`) land at pool block
`block_table[b, lens[b] // P]`, offset `lens[b] % P`. **In place**
(returns `None`); the block table itself belongs to the scheduler -
this writes data into already-mapped blocks and never touches table
entries. Host paths require float32 C-contiguous pools (a conversion
would silently drop the writes - rejected with `TypeError`); the torch
path supports the full f32/bf16/fp16 storage matrix. One tiny kernel,
stream-ordered and graph-capturable. Typical loop: append at
`lens[b]`, then decode with `lens + 1`.

### attention_prefill - fresh sequences

```python
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)
```

- `q`: `[B, Hq, S, D]`, `k`/`v`: `[B, Hkv, S, D]`; query row `i`
  attends to key rows `[0, i]` when `causal=True`, to all rows when
  `causal=False`.
- Same GQA/dtype/dim rules as decode.

Honest scope: this is the **convenience path** - a single tiled kernel
without tensor cores. It exists so small prefills and mixed workloads
stay in fusedtok; heavyweight prefill belongs to SDPA /
FlashAttention (see the benchmark tables for the honest ratios).

## Norms and RoPE

```python
h = fusedtok.rmsnorm(x, w, residual=r, eps=1e-6)
# y = (x + r) * rsqrt(mean((x + r)^2) + eps) * w
# residual=None drops the add; x may be [rows, cols] or [cols]

y = fusedtok.layernorm(x, w, b, eps=1e-6)
# y = (x - mean) / sqrt(var + eps) * w + b

q_rot, k_rot = fusedtok.rope(q, k, theta=10000.0, pos_offset=0, neox=True)
```

`rmsnorm`'s fused residual is the decode-loop workhorse: one kernel
reads `x` and `r`, writes the normed sum - no intermediate tensor.

RoPE applies to `[seq, dim]` rows with even `dim`:

- `neox=False`: interleaved pairs `(2j, 2j+1)` (original RoFormer form)
- `neox=True`: rotate-half across the row halves (GPT-NeoX /
  LLaMA-HuggingFace checkpoints)
- `pos_offset` is the absolute position of row 0 - pass the cache length
  when decoding into an existing sequence (`k=None` for query-only).
- `k` is optional (`q`-only call returns `(q_rot, None)`).

## Elementwise and activations

`silu`, `gelu` (erf form), `gelu_tanh`, `relu`, `tanh`, `sigmoid`,
`softmax` (row-wise, numerically stable), `add`, `mul`, and the fused
MLP gate `swiglu(gate, up)` (= `silu(gate) * up`). All follow the three
execution paths and the dtype rules above. `temperature(x, t)` scales
logits; `axpy(x, a, b)` computes `a * x + b` in one pass.

## Sampling and selection

All selection ops resolve ties to the **earliest index**; sampling is
**deterministic per seed** (a splitmix-style hash RNG - reproducible,
not cryptographically secure).

```python
i = fusedtok.argmax(logits)                      # earliest index on ties
vals, idxs = fusedtok.topk(logits, 50)           # descending, (vals, idxs)
vals, idxs = fusedtok.topp(probs, 0.9)           # input = probabilities
```

- `topk` takes raw scores, `k` in `[0, n]`.
- `topp` takes **probabilities** (already softmaxed), `p` in `(0, 1]`;
  returns the smallest top-p set, crossing element included.

The fused samplers run logits -> token in one GPU round trip:

```python
tok = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
tok = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
tok = fusedtok.decode_step(logits, history, penalty=1.1,
                           p=0.9, temperature=0.8, seed=step)
```

- `sample_topp`: softmax(logits / T) -> nucleus cut with a
  **global-mass** threshold -> inverse-CDF draw.
- `sample_topk`: softmax(logits / T) -> keep k -> renormalize within
  the survivors -> draw. `k=1` is exactly greedy; `k >= vocab` samples
  the whole distribution.
- `decode_step`: CTRL-style `repetition_penalty` over `history`, then
  temperature, then nucleus sampling - one call, one readback. Same
  result as composing the three ops (identical order, identical seed).
- `repetition_penalty(logits, token_ids, penalty)` is also exposed
  standalone: positive logits are divided by `penalty`, negative
  multiplied (`penalty=1.0` disables).

Same-token guarantee: for a fixed seed, CPU / staged / zero-copy paths
draw the same token (documented rounding boundary: CPU uses precise
`exp`, GPU uses `__expf`; the parity tests pin where this can matter -
on the CDF boundary the draw may differ by one element). At very large
vocab with near-uniform logits, every draw sits on that boundary at
scale, so the CPU and GPU references may pick neighboring (equally
valid) tokens for the same seed; the GPU result itself is always
bit-identical per seed, and the window-widening schedule (adaptive
since v1.2) never changes the sampled token.

Flat-distribution caveat: when the nucleus spans most of the vocab
(uniform-ish logits), `sample_topp` must effectively order the whole
vocabulary and remains slower than torch's fully parallel sort -
documented honestly in the benchmarks. v1.2 cut this worst case ~8x on
a 3060 (18.2ms -> 2.2ms at n=131072) with three contract-preserving
changes: the widening jump goes straight to the `p*total` mass bound
instead of stepping the ladder, the full-vocabulary attempt skips the
radix selection entirely, and the serial sampling walk batches its
loads (the strictly-sequential float adds - the CPU-parity contract -
are untouched, so every token is bit-identical). Real decode logits
are peaked; the flat case is the worst case, not the typical one.

## The INT8 path

```python
q, scale = fusedtok.quantize_int8(x)        # symmetric per-tensor:
                                            # scale = max|x|/127
x_back = fusedtok.dequantize_int8(q, scale)
qy, s_out = fusedtok.qadd_int8(qa, sa, qb, sb)   # fused dequant-add-requant
```

Matmul - the LLM-friendly layout (both operands row-major along K, so
`activations @ linear_weight.T` needs no transpose):

```python
y = fusedtok.qgemm(a_q, a_scale, b_q, b_scale)
# y[M, N] = (A_q[M, K] @ B_q[N, K]^T) * (a_scale * b_scale)

y = fusedtok.qgemm_perchannel(a_q, a_scale, b_q, b_scales)
# y[M, N] = (A_q @ B_q^T) * (a_scale * b_scales[j])   # W8A8
```

- `M == 1` dispatches to a bandwidth-bound GEMV kernel (the decode
  step); larger `M` runs the tensor-core IMMA pipeline with runtime
  tile tuning.
- **Exactness contract**: integer accumulation is exact int32 and the
  combined scale applies once at the store - CPU, staged and zero-copy
  results are **bit-identical**. `qgemm_perchannel` with a constant
  `b_scales` vector equals per-tensor `qgemm` bit-for-bit.
- `qgemm_perchannel` is the layout real INT8 inference uses
  (SmoothQuant / TensorRT-LLM style W8A8): one scale per output channel
  absorbs weight outliers that a single per-tensor scale cannot.
- Honest performance: cuBLASLt (`torch._int_mm`) remains ~2.2-2.6x
  faster on big GEMMs; fusedtok's INT8 path is the exact /
  graph-capturable / zero-copy one. The decode GEMV, the common case
  per token, moves half the bytes of an fp16 projection at full
  bandwidth.

## Error contract

Stable since 1.0 (enforced by tests):

- `ValueError` - shape mismatches and out-of-range values
  (`k` outside `[0, n]`, `p` outside `(0, 1]`, negative `pos_offset`,
  `lens` entries beyond `T`, ...)
- `TypeError` - wrong dtype family or device family (float64 input,
  CPU tensor where a CUDA one is required, mixed dtypes between q/k/v,
  ...)
- `RuntimeError` - CUDA execution failures (kernel launch or driver
  errors)

## Running the benchmarks

```bash
python benchmarks/bench.py            # full suite, ~a few minutes
```

Protocol: CUDA-event timing, 3 independent timed rounds per
configuration (each with its own warmup), means reported and per-round
values kept in the JSON. Output lands in `docs/benchmarks/`: one JSON +
one single-panel speedup chart per GPU (file names carry the device). See
the README benchmark section for the current numbers on an RTX 3060
and an RTX 5060 Ti.
