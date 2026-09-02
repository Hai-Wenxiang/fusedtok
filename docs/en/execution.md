# The execution model

Every fusedtok operator shares one execution model. This page explains
it once so the topic pages can focus on the operators themselves:
the three dispatch paths, the dtype rules, stream and CUDA-graph
behavior, and the error contract.

**Other languages:** [中文：执行模型](../zh/execution.md)

## The three execution paths

Each operator accepts the same logical inputs in three flavors, and the
flavor picks the path automatically:

| You pass | Path | What happens |
|---|---|---|
| numpy arrays (default) | **CPU reference** | a C++ float32 reference implementation; no GPU involved |
| numpy arrays + `cuda=True` | **staged CUDA** | inputs copied to the GPU, kernel runs, results copied back |
| CUDA torch tensors | **zero-copy CUDA** | kernels read/write torch's buffers via `data_ptr()` - no staging copies, no host sync |

```python
import numpy as np, torch, fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y1 = fusedtok.rmsnorm(x, w)                # CPU reference
y2 = fusedtok.rmsnorm(x, w, cuda=True)     # staged CUDA (numpy in/out)

xt, wt = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)              # zero-copy: CUDA torch in/out
```

Outputs follow the input family: numpy in gives numpy out, CUDA torch
in gives CUDA torch out (a CPU torch input gives a CPU torch output via
the reference path).

The zero-copy path is what an inference loop wants. Kernels launch on
torch's **current stream** (`torch.cuda.current_stream()`), so they
interoperate with surrounding GPU work with ordinary stream ordering
and no hidden transfers.

The CPU reference is the ground truth: it implements the identical
algorithm (same accumulation order where that matters - see the
[sampling contract](sampling.md#the-same-token-guarantee)), runs on
machines without a GPU, and powers the parity tests.

## dtype rules

| Operator family | numpy | CUDA torch |
|---|---|---|
| elementwise, activations, norms, RoPE | float32 | float32, bfloat16 |
| `attention_decode`, `attention_decode_paged`, `attention_prefill`, `kv_append_paged` | float32 | float32, bfloat16, float16 |
| selection and sampling (`topk`, `sample_*`, `decode_step`, ...) | float32 | float32 |
| INT8 ops (`qgemm`, ...) | int8 operands, float32 scales/outputs | same |

Rules worth knowing:

- **Half-precision inputs keep float32 compute.** Loads widen to
  float32 at the memory boundary, stores narrow back (round-to-nearest
  even). The attention softmax and every accumulator stay float32 on
  all dtypes, so numerics change only through the input rounding.
- **Output dtype matches input dtype** on every family (bf16 in, bf16
  out; attention additionally round-trips fp16).
- Norm weights (`weight`, `bias`, `residual`) are upcast to float32
  automatically when the activations are half precision - checkpoints
  commonly store these in fp32.
- CPU/staged paths are always float32 (numpy has no bf16/fp16).
- Other input dtypes (float64, ...) are converted to float32 with a
  copy on the CPU/staged paths and rejected with `TypeError` on the
  zero-copy path.

## Streams and CUDA graphs

Zero-copy launches ride the caller's current stream, so the ordinary
torch stream semantics apply. The whole library is
CUDA-graph-capturable; capture with `torch.cuda.graph` as usual. Two
practical rules:

1. **Warm up before capturing.** First calls may allocate a per-shape
   workspace (attention split path, selection pipeline) or tune a
   launch config (row-kernel block size, qgemm tiles). Both happen
   outside captures by design - a warm-up call gets them out of the
   way.
2. **Captured kernels read their per-call parameters from device
   memory.** Replays observe new tensor contents written between
   replays; mutate-in-place + replay recomputes (tests pin this).
   Per-call *values* that travel as kernel parameters (e.g. the
   sampling `seed`) are baked at capture time like any kernel argument.

```python
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = fusedtok.rmsnorm(xt, wt)     # warm-up (tuning/workspace)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = fusedtok.rmsnorm(xt, wt)
g.replay()                                  # replayed as one launch batch
```

Notable exceptions (by design, documented per operator):

- The fused samplers return a host `int`, so each call ends in one
  small device-to-host readback - they are not meant to be captured.
- `quantize_int8` / `qadd_int8` must read the reduced scale back to
  compose pass 2, so they sync the caller's stream once mid-call.
- On the zero-copy path, integer inputs (attention `lens`, paged
  `block_table`, penalty token ids) are **trusted** when they arrive
  as CUDA tensors - validating their values would require a stream
  sync, which would break capture. Host-side values (lists, numpy, CPU
  tensors) ARE validated before the upload.

## The error contract

Stable since 1.0 and pinned by `tests/test_api.py`:

| Exception | Raised for | Examples |
|---|---|---|
| `ValueError` | shapes and value ranges | `k` outside `[0, n]`, `p` outside `(0, 1]`, negative `pos_offset`, `lens` entries beyond the cache, mismatched shapes, host-origin `lens`/`block_table` values out of range |
| `TypeError` | dtype / device-family problems | float64 input on the zero-copy path, CPU tensor where the primary input is CUDA, mixed dtypes between q/k/v, non-contiguous tensors where the kernel reads raw pointers |
| `RuntimeError` | CUDA execution failures | kernel launch faults, driver errors, failed copies |

Note the contiguity rule: zero-copy kernels address device memory
directly, so a non-contiguous tensor is rejected (`ValueError`) rather
than silently read wrong - pass `.contiguous()` views explicitly.

## Where to go next

- [Attention operators](attention.md)
- [Sampling and selection](sampling.md)
- [The INT8 path](int8.md)
