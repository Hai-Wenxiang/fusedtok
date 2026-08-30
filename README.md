# fusedtok

[![CI](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusedtok.svg)](https://pypi.org/project/fusedtok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/pyproject.toml)

**Fused CUDA kernels for LLM inference** — RMSNorm / RoPE / SwiGLU / attention
decode and friends, with **zero-copy torch tensor support**: up to
**9.3x faster than PyTorch SDPA** (attention decode, RTX 3060, see
[Benchmarks](#benchmarks)).

**中文文档请看 [README_zh.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/README_zh.md)** | English below.

## Why

LLM inference frameworks launch many small, memory-bound operators per token. Each launch
round-trips through global memory. `fusedtok` fuses them into single kernels to cut memory
traffic and launch overhead.

## Operators

| Status | Kernel | Notes |
|---|---|---|
| ✅ | RMSNorm (+residual) | LLaMA/Qwen style, fused residual add |
| ✅ | LayerNorm | with affine |
| ✅ | RoPE | interleaved **and** NeoX layouts, kv-cache `pos_offset` |
| ✅ | SwiGLU | fused MLP activation |
| ✅ | Softmax (row-wise) | numerically stable |
| ✅ | SiLU / GeLU / GeLU-tanh / ReLU / Tanh / Sigmoid | elementwise |
| ✅ | add / mul | elementwise binary (fused add+residual pattern) |
| ✅ | top-k / top-p (nucleus) | arrival-ticket radix + early-exit compaction, replayed from a cached CUDA graph; deterministic ties (1.5x vs torch/CUB @131k on a 36-SM RTX 5060 Ti) |
| ✅ | argmax / temperature | greedy decoding helpers |
| ✅ | sample_topp | fused nucleus sampling: softmax -> top-p -> seeded draw, global-mass threshold |
| ✅ | repetition penalty | CTRL-style, applied to sampled token ids |
| ✅ | decode_step | the whole decode step fused: penalty -> temperature -> nucleus sample, one call, one readback |
| ✅ | quantize_int8 / dequantize_int8 / qadd_int8 | symmetric per-tensor INT8, fused dequant-add-requant |
| ✅ | qgemm | INT8 matmul, int32-exact: cp.async double-buffered pipelined IMMA GEMM with runtime tile tuning (64x64 / 128x128) + warp-per-row GEMV (M=1 decode; 2x vs fp16 projection) |
| ✅ | attention_decode | single-token causal attention with GQA over a contiguous kv-cache: online softmax, flash-decoding split over long caches, per-sequence lengths |
| ✅ | attention_prefill | fresh-sequence attention over S query rows (causal / bidirectional); convenience path - heavyweight prefill stays SDPA/flash territory (honest ~0.45x) |

## Install

```bash
pip install fusedtok
```

Prebuilt wheels on PyPI (built with CUDA 12.4): **Linux x86_64** (manylinux,
cp310) and **Windows x86_64** (cp312). On other platforms or Python versions
pip builds from source automatically:

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
# fresh-sequence prefill (causal by default; convenience path)
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)

# sampling side: the whole decode step in one fused call
token = fusedtok.decode_step(logits, sampled_ids, penalty=1.1,
                             p=0.9, temperature=0.8, seed=step)
# or step by step:
logits = fusedtok.repetition_penalty(logits, sampled_ids, penalty=1.1)
token = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
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
CUDA torch tensors may also be **bfloat16** - the kernels compute in float32
and convert at the load/store boundary (norm weights are upcast to float32
automatically; sampling/selection ops stay float32).
CUDA torch tensors select the zero-copy path automatically.

See `examples/demo.py` for a runnable tour of every operator.

## Correctness

Every kernel ships with a CPU reference implementation and element-wise parity tests
(pytest). Tests run on machines without a GPU (CUDA cases skip automatically).

## Benchmarks

RTX 3060 (sm_86), float32, zero-copy torch tensors, CUDA-event timing over
3 independent rounds (means below; per-round values in the JSON), vs
the equivalent PyTorch reference (composite eager expressions; attention
references use **pre-expanded** heads - `repeat_interleave` outside the
timed region). Largest shape per op; full data:
`docs/benchmark_rtx3060.json`, reproduce with `python benchmarks/bench.py`:

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| attention_decode (GQA) | T=16384, D=128 | 853 µs | 7614 µs (SDPA) | **8.92x** |
| RoPE NeoX (q+k) | [8192×4096] | 1641 µs | 10061 µs | **6.13x** |
| RMSNorm (+residual) | [4096×4096] | 614 µs | 2061 µs | **3.36x** |
| SwiGLU | [4096×4096] | 614 µs | 1025 µs | **1.67x** |
| top-k (k=50) | [131072] | 80 µs | 127 µs | **1.59x** |
| LayerNorm | [4096×4096] | 446 µs | 616 µs | **1.38x** |
| Softmax | [4096×4096] | 414 µs | 432 µs | 1.04x |
| SiLU / GeLU / add | [4096×4096] | ~412 µs | ~411 µs | ~1.0x |
| argmax | [131072] | 65 µs | 45 µs | 0.69x (incl. host readback) |
| int8 qgemm (IMMA) | [4096×4096×4096] | 3554 µs (38.7 TOPS) | 1634 µs (cuBLASLt) | 0.46x (honest) |
| attention_prefill (causal) | S=1024, D=128 | 5732 µs | 2560 µs (SDPA flash) | 0.45x (honest) |

Row-wise kernels (norms, softmax) autotune their thread-block size per
shape at first call (v0.4.1); the table reflects the tuned choices.

![fusedtok vs PyTorch reference](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rtx3060.png)

**RTX 5060 Ti (Blackwell, sm_120)** — same suite, largest shape per op
(full data: `docs/benchmark_rtx5060ti.json`):

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [8192×4096] | 1384 µs | 8368 µs | **6.04x** |
| attention_decode (GQA) | T=16384, D=128 | 575 µs | 2682 µs (SDPA) | **4.67x** |
| RMSNorm (+residual) | [4096×4096] | 504 µs | 1657 µs | **3.29x** |
| SwiGLU | [4096×4096] | 504 µs | 858 µs | **1.70x** |
| top-k (k=50) | [131072] | 27 µs | 41 µs (CUB) | **1.50x** |
| LayerNorm / Softmax | [4096×4096] | ~345 µs | ~348 µs | 1.0x |
| argmax | [131072] | 17 µs | 14 µs | 0.83x (incl. host readback) |
| int8 qgemm (IMMA) | [4096×4096×4096] | 2063 µs (66.6 TOPS) | 800 µs (cuBLASLt) | 0.39x (honest) |
| attention_prefill (causal) | S=1024, D=128 | 3291 µs | 1421 µs (SDPA flash) | 0.43x (honest) |

On smaller shapes the Blackwell card shows bigger wins (softmax 2.5x,
RMSNorm 3.2x at 256 rows, attention decode 3.8x at T=4096 running
235 GB/s) - the launch-overhead share shrinks as shapes grow; full
sweep in the JSON.

![fusedtok vs PyTorch reference (RTX 5060 Ti)](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rtx5060ti.png)

The PyPI wheel ships sm_80/sm_86 cubins plus a compute_86 PTX fallback —
verified to JIT and run correctly on Blackwell (sm_120) drivers.

Fusions win big (RoPE / RMSNorm / SwiGLU) because eager mode round-trips
intermediate tensors through global memory. The v0.4 selection pipeline
(arrival-ticket radix rounds + early-exit compaction, replayed from a
cached CUDA graph) beats torch's CUB radix select at small k on both
GPUs; mid-range k (2048..n) stays at or below parity — honest numbers,
a pipelined tensor-core sort stays future work. attention_decode wins
big at decode (one launch streams the GQA cache once at up to ~157 GB/s
effective while SDPA pays head expansion or small-query inefficiency);
attention_prefill is the honest convenience path at ~0.45x of SDPA's
flash backend — no tensor cores by design, so heavyweight prefill stays
with SDPA/FlashAttention. The INT8 decode GEMV moves half the bytes of
an fp16 projection and runs at full memory bandwidth (2x); the pipelined
IMMA GEMM (v1.0 rework: cp.async double-buffered slabs, runtime-tuned
64x64 / 128x128 tiles) reaches ~39 TOPS on a 3060 and ~67 TOPS on a
5060 Ti — 2x-4x the v0.4 kernel — but cuBLASLt (`torch._int_mm`) still
holds a ~2.2-2.6x lead: its tiles pipeline deeper and its epilogue is
tuned per-arch. For now qgemm is the exact / graph-capturable /
zero-copy INT8 path, not the fastest one; honest numbers, a
CUTLASS-class schedule stays future work.

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
- 1.0 (in development): pipelined tensor-core INT8 GEMM (cp.async
  double-buffering, runtime tile tuning; 17 -> 39 TOPS on a 3060),
  per-channel weight scales for qgemm (SmoothQuant-style W8A8), fused
  top-k sampling, top-k mid-range-k parity, text hygiene gate, wheel
  matrix expansion, API freeze

## Community

- [Contributing guide](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) — setup, rules of the road, PR process
- [Code of conduct](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [Security policy](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [Changelog](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## License

MIT — see [LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE). Third-party notices: [NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md).
