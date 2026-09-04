# fusedtok

[![CI](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusedtok.svg)](https://pypi.org/project/fusedtok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/pyproject.toml)

**Fused CUDA kernels for LLM inference** — RMSNorm / RoPE / SwiGLU / attention
decode and friends, with **zero-copy torch tensor support**: up to
**8.7x faster than PyTorch SDPA** (attention decode, RTX 3060, see
[Benchmarks](#benchmarks)).

**中文文档请看 [README_zh.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/README_zh.md)** | English below.

## Why

LLM inference frameworks launch many small, memory-bound operators per token. Each launch
round-trips through global memory. `fusedtok` fuses them into single kernels to cut memory
traffic and launch overhead.

## Operators

37 operators + helpers (the frozen `fusedtok.__all__` of the 1.x series;
see "API stability" below). `axpy` is the hello-world demo op kept from
the v0.x skeleton - functional, but not a performance feature.

| Status | Kernel | Notes |
|---|---|---|
| ✅ | RMSNorm (+residual) | LLaMA/Qwen style, fused residual add |
| ✅ | LayerNorm | with affine |
| ✅ | RoPE | interleaved **and** NeoX layouts, kv-cache `pos_offset` |
| ✅ | SwiGLU | fused MLP activation |
| ✅ | Softmax (row-wise) | numerically stable |
| ✅ | SiLU / GeLU / GeLU-tanh / ReLU / Tanh / Sigmoid | elementwise |
| ✅ | add / mul | elementwise binary (fused add+residual pattern) |
| ✅ | top-k / top-p (nucleus) | arrival-ticket radix + early-exit compaction, replayed from a cached CUDA graph; deterministic ties (1.57x / 1.51x vs torch/CUB @131k k=50, parity-to-winning across the whole k range on both test GPUs) |
| ✅ | argmax / temperature | greedy decoding helpers |
| ✅ | sample_topp | fused nucleus sampling: softmax -> top-p -> seeded draw, global-mass threshold |
| ✅ | sample_topk | fused top-k sampling: softmax -> top-k -> renormalize within the window -> seeded draw (2.04x / 2.03x vs the topk+multinomial composite @131k) |
| ✅ | sample_minp | fused min-p sampling (v1.3): keep every token with p >= min_p * p_max -> renormalize -> seeded draw - a value threshold, so no global-mass reduction; adaptive nucleus by construction |
| ✅ | sample_topp/minp/topk_batched | batched sampling (v1.4): one call, `[rows, vocab]` logits in, one seeded token per row out - every row runs the single-row pipeline verbatim; vs the per-row loop 4-6x wall on submission-bound hosts, at native batched-multinomial level on peaked decode logits |
| ✅ | repetition penalty | CTRL-style, applied to sampled token ids |
| ✅ | decode_step | the whole decode step fused: penalty -> temperature -> nucleus sample, one call, one readback |
| ✅ | quantize_int8 / dequantize_int8 / qadd_int8 | symmetric per-tensor INT8, fused dequant-add-requant |
| ✅ | qgemm | INT8 matmul, int32-exact: cp.async double-buffered pipelined IMMA GEMM with runtime tile tuning (64x64 / 128x128) + warp-per-row GEMV (M=1 decode; 2x vs fp16 projection) |
| ✅ | qgemm_perchannel | the W8A8 layout real INT8 inference uses: per-output-channel weight scales fused into the same kernel's epilogue at zero cost |
| ✅ | attention_decode | single-token causal attention with GQA over a contiguous kv-cache: online softmax, flash-decoding split over long caches, per-sequence lengths; **float32 / bfloat16 / float16 storage** (half-precision cache = half the decode bytes, softmax stays float32) |
| ✅ | kv_append | the cache-write side of the contiguous decode loop (v1.3): one fresh token's k/v rows per sequence scattered in place into the cache at row `lens[b]` (one tiny kernel, f32/bf16/fp16) |
| ✅ | attention_decode_paged | the v1.2 headline: the same decode attention over a **vLLM-style block-pool kv-cache** `[Nb, Hkv, P, D]` walked through a per-sequence block table - fragmentation-free cache memory; any valid table honored, f32/bf16/fp16 storage, 1.09-1.13x the contiguous op (7.96x / 4.28x vs SDPA on the pre-expanded-heads reference) |
| ✅ | kv_append_paged | the cache-write side of the paged loop: one fresh token's k/v rows per sequence scattered in place into the pool at position `lens[b]` (one tiny kernel, f32/bf16/fp16) |
| ✅ | attention_prefill | fresh-sequence attention over S query rows (causal / bidirectional), float32 / bf16 / fp16 storage; convenience path - heavyweight prefill stays SDPA/flash territory (honest ~0.45x f32) |
| ✅ | axpy | `a*x + b` - the v0.x hello-world demo op, kept for API compatibility |

## Install

```bash
pip install fusedtok
```

Prebuilt wheels on PyPI (built with CUDA 12.4): **Linux x86_64**
(manylinux, cp310-cp313) and **Windows x86_64** (cp311-cp313). On other
platforms or Python versions pip builds from source automatically:

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

**Requirements:**

- NVIDIA GPU of **RTX 30 series (Ampere) or newer** — e.g. RTX 3060/3090, RTX 4080, RTX 5090, A100, H100
- CUDA Toolkit >= 12.0
- A C++17 compiler (MSVC on Windows, GCC/Clang on Linux); Python 3.10+

<details>
<summary>What is "compute capability"? (click to expand)</summary>

Compute capability is NVIDIA's version number for a GPU architecture generation — not a
performance score. CUDA code must be compiled for a specific architecture to run on it.
The wheel builds native cubins for compute capability 8.0 (A100) and 8.6 (RTX 30) plus a
compute_86 PTX fallback, so Ampere runs natively and newer architectures (RTX 40/50, ...)
JIT the PTX with their driver.

| Compute capability | Architecture | Example GPUs |
|---|---|---|
| 7.5 | Turing | GTX 16xx, RTX 20xx (not supported) |
| 8.0 / 8.6 | Ampere | A100, RTX 30xx |
| 8.9 | Ada | RTX 40xx (via PTX) |
| 9.0 | Hopper | H100 (via PTX) |
| 12.0 | Blackwell | RTX 50xx (via PTX) |

Check yours: run `nvidia-smi` to see your GPU model, then look it up at
https://developer.nvidia.com/cuda-gpus

</details>

## Usage

numpy in / numpy out, or torch in / torch out — including **zero-copy CUDA**:
kernels read and write torch device buffers directly via `data_ptr()`, with
no staging copies and no host synchronization.

```python
import numpy as np
import torch
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32)

# CPU reference implementation (ground truth, runs anywhere)
y = fusedtok.rmsnorm(x, w, eps=1e-6)

# staged CUDA: copies to GPU, runs kernel, copies back
y = fusedtok.rmsnorm(x, w, cuda=True)

# zero-copy CUDA with torch tensors: kernels run in torch's own buffers,
# stream-ordered with other torch operations
xt, wt = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)          # -> CUDA torch tensor

# RoPE with kv-cache position offset, NeoX (LLaMA-HF) layout
q = torch.randn(1, 4096, device="cuda")          # new token only
q_rot, k_rot = fusedtok.rope(q, k=None, pos_offset=1023, neox=True)

# attention over a GQA kv-cache: one call per decode step, no score
# materialization, variable-length batches share one cache tensor
out = fusedtok.attention_decode(
    q_heads,                                    # [B, Hq, D] new token
    k_cache, v_cache,                           # [B, Hkv, T, D]
    lens=torch.tensor([1023, 512], dtype=torch.int32, device="cuda"))
# grow a contiguous cache: append each new token's rows, then decode
fusedtok.kv_append(k_cache, v_cache, k_new, v_new, lens)
out = fusedtok.attention_decode(q_heads, k_cache, v_cache, lens + 1)
# ...or over a paged (vLLM-style block-pool) cache: pools [Nb, Hkv, P, D]
# + a per-sequence block table; append each new token, then decode
fusedtok.kv_append_paged(k_pool, v_pool, block_table, k_new, v_new, lens)
out = fusedtok.attention_decode_paged(q_heads, k_pool, v_pool,
                                      block_table, lens + 1)
# fresh-sequence prefill (causal by default; convenience path)
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)

# sampling side: the whole decode step in one fused call
token = fusedtok.decode_step(logits, sampled_ids, penalty=1.1,
                             p=0.9, temperature=0.8, seed=step)
# or step by step:
logits = fusedtok.repetition_penalty(logits, sampled_ids, penalty=1.1)
token = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
# top-k sampling variant (renormalizes within the k survivors)
token = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
# concurrent decode: [B, vocab] in, one token per row out (default
# seeds are 0..B-1, so identical rows still draw independently)
tokens = fusedtok.sample_topp_batched(batch_logits, p=0.9,
                                      temperature=0.8)
```

A minimal per-token sampling loop:

```python
import torch, fusedtok as ft

h = torch.zeros(1, 4096, device="cuda")            # decoder state
w = torch.load("rms_weight.pt").cuda()             # float32 weights
wq, wscale = ft.quantize_int8(weight_f32.ravel())  # int8 weights
generated = []
for step in range(256):
    h = ft.rmsnorm(h, w, residual=h)               # fused add + norm
    q = ft.rope(q, k=None, pos_offset=step, neox=True)
    logits = model_output(h)                       # your model
    tok = ft.decode_step(logits, generated, penalty=1.1,
                         p=0.9, temperature=0.8, seed=step)
    generated.append(int(tok))
```

Every function accepts float32 numpy arrays or torch tensors (other dtypes
are converted with a copy) and returns float32 outputs of the same family.
CUDA torch tensors may be **bfloat16** on every operator that moves tensor
data (elementwise / norms / RoPE / attention), and the attention operators
additionally take **float16** — kernels compute in float32 and convert at
the load/store boundary (norm weights are upcast to float32 automatically;
sampling/selection ops stay float32).
CUDA torch tensors select the zero-copy path automatically.

See `examples/demo.py` for a runnable tour of every operator, and the
[usage guide](https://github.com/Hai-Wenxiang/fusedtok/blob/main/docs/en/usage.md) for the topic-structured manual - one page per theme:
quickstart, the execution model (paths / dtypes / streams / CUDA
graphs), attention, the sampling contract, the INT8 workflow, how to
read the benchmarks, and an FAQ with a glossary. Also available in
[中文](https://github.com/Hai-Wenxiang/fusedtok/blob/main/docs/zh/usage.md).

## Correctness

Every kernel ships with a CPU reference implementation and element-wise parity tests
(pytest). Tests run on machines without a GPU (CUDA cases skip automatically).

## API stability

1.0 freezes the public surface: the names in `fusedtok.__all__` (37
operators + helpers) keep their signatures across the 1.x series.
Type stubs (`__init__.pyi`, PEP 561 `py.typed`) ship with the package.
New operators arrive in minor releases; breaking changes require a new
major version and a deprecation window. Determinism promises: selection
ties resolve to the earliest index; sampling is deterministic per seed.

## Benchmarks

RTX 3060 (sm_86), float32, zero-copy torch tensors, CUDA-event timing over
3 independent rounds (means below; per-round values in the JSON), vs
the equivalent PyTorch reference (composite eager expressions; attention
references use **pre-expanded** heads - `repeat_interleave` outside the
timed region). Largest shape per op; full data:
`docs/benchmarks/benchmark_rtx3060.json`, reproduce with `python benchmarks/bench.py`:

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| attention_decode (GQA) | T=16384, D=128 | 884 µs | 7715 µs (SDPA) | **8.73x** |
| attention_decode_paged (GQA) | T=16384, D=128, P=16 | 982 µs | 7724 µs (SDPA) | **7.87x** |
| RoPE NeoX (q+k) | [8192×4096] | 1664 µs | 10203 µs | **6.13x** |
| kv_append (contiguous cache write) | B=8, T=4096 | 16 µs | 110 µs (advanced indexing) | **6.97x** |
| RMSNorm (+residual) | [4096×4096] | 616 µs | 2101 µs | **3.41x** |
| sample_topp p=0.9 (peaked) | [131072] | 139 µs | 437 µs (sort+mask+multinomial) | **3.15x** |
| attention_decode bf16 | T=16384, D=128 | 861 µs | 1817 µs (SDPA bf16) | **2.11x** |
| sample_topk k=50 | [131072] | 148 µs | 318 µs (topk+multinomial) | **2.15x** |
| SwiGLU | [4096×4096] | 644 µs | 1045 µs | **1.62x** |
| top-k (k=50) | [131072] | 88 µs | 134 µs | **1.52x** |
| LayerNorm | [4096×4096] | 458 µs | 627 µs | **1.37x** |
| sample_minp p=0.05 (peaked) | [131072] | 156 µs | 212 µs (mask+multinomial) | **1.36x** |
| top-k (k=4096, mid-k) | [131072] | 111 µs | 136 µs | **1.22x** |
| Softmax | [4096×4096] | 423 µs | 431 µs | **1.02x** |
| SiLU / GeLU / add | [4096×4096] | ~416-625 µs | ~417-622 µs | ~1.0x |
| argmax | [131072] | 48 µs | 42 µs | 0.87x (event-timed, noisy on WDDM; wall probe 1.12x - see below) |
| int8 qgemm pc (W8A8) | [4096×4096×4096] | 3568 µs (38.5 TOPS) | 2066 µs (cuBLASLt + broadcast) | 0.58x (honest) |
| int8 qgemm (IMMA) | [4096×11008×4096] | 9580 µs (38.6 TOPS) | 4528 µs (cuBLASLt) | 0.47x (honest) |
| attention_prefill (causal) | S=1024, D=128 | 5797 µs | 2627 µs (SDPA flash) | 0.45x (honest) |
| sample_topp p=0.9 (flat worst case) | [131072] | 1394 µs | 349 µs | 0.25x (honest, see below) |
| sample_minp p=0.05 (wide nucleus) | [131072] | 597 µs | 230 µs | 0.39x (honest: one adaptive-jump retry + a 64k sort; the torch boolean-mask composite never sorts - see the peaked min-p row above for min-p's win scenario) |

Batched samplers (v1.4) - one call for the whole `[8, 131072]` batch,
referenced against torch's native batched draw (softmax + multinomial
over the 2-D tensor; per-round values in the JSON):

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| sample_topk_batched k=50 | [8×131072] | 210 µs | 489 µs (topk+multinomial) | **2.33x** |
| sample_minp_batched p=0.05 | [8×131072] | 264 µs | 311 µs (mask+multinomial) | **1.18x** |
| sample_topp_batched p=0.9 | [8×131072] | 280 µs | 234 µs (multinomial) | 0.84x (within WDDM variance of the reference) |
| sample_topp_batched (flat worst case) | [8×131072] | 3608 µs | 214 µs | 0.06x (honest, same caveat as the single row) |

Row-wise kernels (norms, softmax) autotune their thread-block size per
shape at first call (v0.4.1); the table reflects the tuned choices.

![fusedtok vs PyTorch reference](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmarks/benchmark_rtx3060.png)

**RTX 5060 Ti (Blackwell, sm_120)** — same suite, largest shape per op
(full data: `docs/benchmarks/benchmark_rtx5060ti.json`):

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [8192×4096] | 1384 µs | 8373 µs | **6.05x** |
| attention_decode (GQA) | T=16384, D=128 | 574 µs | 2681 µs (SDPA) | **4.67x** |
| attention_decode_paged (GQA) | T=16384, D=128, P=16 | 627 µs | 2681 µs (SDPA) | **4.28x** |
| attention_decode bf16 | T=16384, D=128 | 548 µs | 642 µs (SDPA bf16) | **1.17x** |
| RMSNorm (+residual) | [4096×4096] | 505 µs | 1657 µs | **3.28x** |
| sample_topp p=0.9 (peaked) | [131072] | 62 µs | 158 µs (sort+mask+multinomial) | **2.53x** |
| sample_topk k=50 | [131072] | 45 µs | 94 µs (topk+multinomial) | **2.08x** |
| SwiGLU | [4096×4096] | 504 µs | 859 µs | **1.70x** |
| top-k (k=50) | [131072] | 27 µs | 41 µs (CUB) | **1.51x** |
| sample_minp p=0.05 (peaked) | [131072] | 62 µs | 74 µs (mask+multinomial) | **1.20x** |
| top-k (k=4096, mid-k) | [131072] | 50 µs | 54 µs (CUB) | 1.09x |
| LayerNorm / Softmax | [4096×4096] | ~346 µs | ~344-351 µs | ~1.0x |
| argmax | [131072] | 17 µs | 14 µs | 0.82x (event-timed, noisy; wall probe 0.96x) |
| int8 qgemm pc (W8A8) | [4096×4096×4096] | 2077 µs (66.2 TOPS) | 1141 µs (cuBLASLt + broadcast) | 0.55x (honest) |
| attention_prefill (causal) | S=1024, D=128 | 3298 µs | 1421 µs (SDPA flash) | 0.43x (honest) |
| int8 qgemm (IMMA) | [4096×11008×4096] | 5477 µs (67.4 TOPS) | 2183 µs (cuBLASLt) | 0.40x (honest) |
| kv_append (contiguous cache write) | B=8, T=4096 | 10 µs | 21 µs (advanced indexing) | 2.21x |
| sample_topp p=0.9 (flat worst case) | [131072] | 985 µs | 167 µs | 0.17x (honest, see below) |
| sample_minp p=0.05 (wide nucleus) | [131072] | 262 µs | 74 µs | 0.28x (honest: same wide-nucleus caveat as the 3060 row) |

Batched samplers (v1.4) on the same `[8, 131072]` shapes:

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| sample_topk_batched k=50 | [8×131072] | 94 µs | 118 µs (topk+multinomial) | **1.25x** |
| sample_minp_batched p=0.05 | [8×131072] | 118 µs | 112 µs (mask+multinomial) | 0.95x (parity) |
| sample_topp_batched p=0.9 | [8×131072] | 131 µs | 83 µs (multinomial) | 0.63x |
| sample_topp_batched (flat worst case) | [8×131072] | 1755 µs | 83 µs | 0.05x (honest) |

On smaller shapes the Blackwell card shows bigger wins (softmax 1.7x,
RMSNorm 3.1x at 256 rows, attention decode 3.81x at T=4096 running
188 GB/s) - the launch-overhead share shrinks as shapes grow; full
sweep in the JSON.

![fusedtok vs PyTorch reference (RTX 5060 Ti)](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmarks/benchmark_rtx5060ti.png)

The PyPI wheel ships sm_80/sm_86 cubins plus a compute_86 PTX fallback —
verified to JIT and run correctly on Blackwell (sm_120) drivers.

Fusions win big (RoPE / RMSNorm / SwiGLU) because eager mode round-trips
intermediate tensors through global memory. The v0.4 selection pipeline
(arrival-ticket radix rounds + early-exit compaction, replayed from a
cached CUDA graph) beats torch's CUB radix select at small k on both
GPUs; the v1.0 retune (in-block-sort threshold and sort chunk both
dropped 2048 -> 1024 - a single block bitonic-sorting 2048 keys was the
whole mid-k regression) brings the mid-k window to parity-or-winning as
well (k=4096 @131k: 1.22x / 1.09x). The fused samplers win against the
eager composites when the logits look like real decode output
(sample_topp peaked: 3.15x / 2.53x; sample_topk: 2.15x / 2.08x; the
peaked row's composite reference itself swings ~15% run-to-run on
WDDM - per-round values in the JSON); on a
FLAT distribution sample_topp is honestly 0.25x / 0.17x — the nucleus then
spans ~90% of the vocabulary and the pipeline must effectively order
the whole thing. v1.2 cut that worst case ~8.5x (18.2ms -> 2.2ms at
n=131072 on a 3060) with three token-preserving changes - an adaptive
window jump driven by a p*total mass bound, a full-vocabulary fast
path that skips the selection stages, and a batched-load serial sampling
walk (the strictly sequential float adds are the CPU-parity determinism
contract; only the loads got pipelined); v1.3 cut it another ~1.6x
(2.2ms -> 1.4ms on a 3060, 1.6ms -> 1.0ms on a 5060 Ti) by having the
first walk record prefix-sum checkpoints that the inverse-CDF walk
binary-searches - still bit-identical tokens - but torch's fully parallel
sort still owns that regime. sample_minp (v1.3) wins on peaked logits
(1.36x / 1.20x vs the softmax+mask+multinomial composite) and honestly
loses the wide-nucleus row (0.28-0.39x: one widening retry plus a 32-64k
sort; the torch boolean-mask composite never sorts) - v1.4 gave min-p
the same adaptive jump top-p has had since v1.2 (a sufficient bound
from the one-time global total: wide-nucleus rows skip the ladder's
intermediate stops, -28-30% on that row, tokens bit-identical).
The v1.4 batched samplers put the whole `[rows, vocab]` batch through
one call: every row runs the single-row pipeline verbatim (per-row
parity, including the widening loop with per-row finish tracking), so
the win over the per-row loop is pure batching - 4-6x wall time at
B=8 on submission-bound hosts (3060: topp 1340 -> 274 µs, minp
1399 -> 237 µs; 5060 Ti: 3.9x / 4.3x), landing at torch's native
batched-multinomial level on peaked logits (topk batched outright wins:
2.33x / 1.25x) and keeping the same honest flat-worst caveat as the
singles (0.05-0.06x). attention_decode
wins big at decode (one launch streams the GQA cache once at up to
~154 GB/s effective on a 3060 / ~234 GB/s on a 5060 Ti, while SDPA pays
head expansion or small-query inefficiency);
attention_decode_paged (v1.2) pays only 1.09-1.13x for the block-table
indirection of the fragmentation-free vLLM-style cache layout, with
bit-identical output on matching slice schedules;
attention_prefill is the honest convenience path at ~0.45x of SDPA's
flash backend — no tensor cores by design, so heavyweight prefill stays
with SDPA/FlashAttention. kv_append (v1.3) writes one token's k/v rows
into the contiguous cache in a single launch (6.97x / 2.21x over the
advanced-indexing scatter; a tiny launch-bound op - parity-level cost
per decode step). The INT8 decode GEMV moves half the bytes of
an fp16 projection and runs at full memory bandwidth (2x); the pipelined
IMMA GEMM (v1.0 rework: cp.async double-buffered slabs, runtime-tuned
64x64 / 128x128 tiles) reaches ~38 TOPS on a 3060 and ~67 TOPS on a
5060 Ti — 2x-4x the v0.4 kernel — but cuBLASLt (`torch._int_mm`) still
holds a ~2.1-2.5x lead on the per-tensor rows (its tiles pipeline
deeper and its epilogue is tuned per-arch); the W8A8 rows' gap is only
~1.7-1.8x. For now qgemm is the exact / graph-capturable /
zero-copy INT8 path, not the fastest one; honest numbers, a
CUTLASS-class schedule stays future work. The per-channel variant
(`qgemm_perchannel`, the W8A8 layout INT8 inference actually uses)
fuses the per-output-channel scale multiply into the same epilogue at
zero kernel cost — the composite torch reference pays for that
broadcast separately, which is where its 0.55-0.58x comes from.

All sampling rows above measure FIXED logits since 1.1.1 (the bench
seeds torch's RNG); since v1.2 the peaked row spikes +20 and the flat
row uses near-uniform logits — both regimes are now deterministically
what their labels say (the v1.1 peaked row sat on the coverage boundary
at n=131072 and flipped regimes per seed). The argmax rows are
event-timed over a host-synchronized call and swing on WDDM
(0.73-1.24x across runs); wall-clock probes with the sync excluded
from the timed loop measure 1.12x (3060) / 0.96x (5060 Ti) — v1.2
removed one CUDA submission and one allocation per call.

## Development

See [CONTRIBUTING.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) for the full guide (test rules,
error contract, determinism invariants). Quick start:

```bash
# Windows: run inside a VS developer prompt (vcvars64)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
# from repo root: PYTHONPATH picks up the built module, conftest.py adds python/
$env:PYTHONPATH = "$PWD/build"        # Windows
PYTHONPATH=$PWD/build                 # Linux
python -m pytest tests -q
python benchmarks/bench.py            # GPU benchmark + chart
```

Windows / Linux. Windows uses MSVC via nvcc; CI builds and runs the CPU test
suite on every push.

## Roadmap

- v0.2 (done): bf16 zero-copy, radix-select top-k/top-p, fused nucleus
  sampling, single-read softmax, CUDA-graph verified
- v0.3 (done): chunk-merge selection sort + parallel nucleus count,
  bf16x4/x8 vectorized elementwise, INT8 quantize/dequantize utilities
- v0.4 (done): arrival-ticket selection pipeline (no cooperative
  launch, early-exit compaction, cached CUDA graphs), stream-aware
  launchers everywhere (real CUDA-graph capture), INT8 compute path
  (IMMA qgemm + decode GEMV), fused decode_step sampling
- v0.4.1 (done): runtime block-size autotuning for the row-wise kernels
  (norms/softmax pick 128..1024 threads per shape at first call)
- v0.5 (done): attention - GQA decode attention over a contiguous
  kv-cache (flash-decoding split over long caches, per-sequence lengths)
  and a tiled prefill path (honest ~0.45x of SDPA flash - the
  convenience path); single-chart-per-GPU benchmarks; Windows wheels in
  the PyPI publish pipeline
- 1.0 (released): pipelined tensor-core INT8 GEMM (cp.async
  double-buffering, runtime tile tuning; 17 -> 39 TOPS on a 3060) with
  per-channel weight scales (W8A8), fused top-k sampling (2.1x vs the
  topk+multinomial composite), top-k mid-range-k parity, text hygiene
  gate, wheel matrix expansion (Linux cp310-313, Windows cp311-313),
  API freeze
- 1.1 (released): half-precision attention — `attention_decode` /
  `attention_prefill` accept bfloat16 and float16 caches (float32
  compute, half the decode bytes); parallel exp precompute halves the
  flat-distribution sampling worst case with bit-identical tokens
- 1.2 (released): paged kv-cache attention — `attention_decode_paged`
  over a vLLM-style block pool `[Nb, Hkv, P, D]` + per-sequence block
  tables (1.06-1.07x the contiguous op, any valid table honored) and
  `kv_append_paged` (the in-place cache-write side); flat-distribution
  sampling worst case cut ~8.5x (adaptive widening jump + full-vocab
  fast path + batched-load serial walk, tokens bit-identical); argmax
  launch diet (one submission and one allocation less per call)
- 1.2.1 (released): audit-driven hardening - selection workspace
  overflow past 131072-token vocabularies fixed (Qwen-scale vocabs),
  host-side lens/table/id validation with a documented device-tensor
  trust boundary (CUDA-graph capture works with `lens` now), empty-
  input and dtype/contiguity guards on the zero-copy paths, honest
  benchmark bandwidth (four rows were over-billed 1.5x), warning-clean
  builds (MSVC /W3 + GCC -Wall -Wextra), and the documentation
  restructured into topic pages with a natural-sounding Chinese
  rewrite
- 1.3 (released): `sample_minp` (min-p sampling - value-threshold
  nucleus relative to p_max, no global-mass reduction, adaptive by
  construction) and `kv_append` (the contiguous cache-write side); the
  sampling serial walk gains checkpoint bisection (walk 1 records
  prefix sums, walk 2 binary-searches them - flat worst case cut
  another ~1.6x, tokens bit-identical); zero-copy helpers reject CPU
  operands (a host pointer in a kernel poisons the CUDA context)
- 1.3.1 (released): audit-driven hardening - staged-path lens/block-table
  value validation closed (bad values used to become silent GPU
  out-of-bounds writes) plus integer-input hardening; two latent
  sampling-walk bugs fixed (stride>=2 checkpoint double-count, adaptive
  widening reading the wrong workspace word - mid-tail ~28% faster with
  bit-identical tokens); softmax tuner capped to keep sanitizer gates
  clean; kernel/launcher cleanup; benchmark tables fully regenerated
- 1.4 (released): batched sampling - `sample_topp/minp/topk_batched`
  sample a whole `[rows, vocab]` batch in one call (per-row parity with
  the singles, one seed per row, rows finish at their own window sizes);
  min-p gains the adaptive widening jump (a sufficient bound from the
  one-time global total - wide-nucleus rows skip the ladder's
  intermediate stops, tokens bit-identical)
  (new min-p peaked and kv_append rows) and a documentation overhaul
  (stale numbers, stiff Chinese phrasing, glossary additions)

## Community

- [Contributing guide](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) — setup, rules of the road, PR process
- [Code of conduct](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [Security policy](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [Changelog](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## License

MIT — see [LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE). Third-party notices: [NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md).
