# fusedtok

[![CI](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusedtok.svg)](https://pypi.org/project/fusedtok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/pyproject.toml)

**Fused CUDA kernels for LLM inference** — RMSNorm / RoPE / SwiGLU and friends,
with **zero-copy torch tensor support**: up to **6.2x faster than PyTorch eager**
(RoPE, RTX 3060, see [Benchmarks](#benchmarks)).

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
| ✅ | qgemm | INT8 matmul, int32-exact: tensor-core IMMA GEMM + warp-per-row GEMV (M=1 decode; 2x vs fp16 projection) |
| ✅ | attention_decode | single-token causal attention with GQA over a contiguous kv-cache: online softmax, flash-decoding split over long caches, per-sequence lengths |
| ✅ | attention_prefill | fresh-sequence attention over S query rows (causal / bidirectional); convenience path - heavyweight prefill stays SDPA/flash territory (honest ~0.45x) |

## Install

```bash
pip install fusedtok
```

Prebuilt Linux x86_64 wheels (manylinux, built with CUDA 12.4) are on PyPI.
On Windows (or any platform without a matching wheel) pip builds from
source automatically:

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

RTX 3060 (sm_86), float32, zero-copy torch tensors, CUDA-event timing, vs
the equivalent PyTorch reference (composite eager expressions; attention
references use **pre-expanded** heads - `repeat_interleave` outside the
timed region). Largest shape per op; full data:
`docs/benchmark_rtx3060.json`, reproduce with `python benchmarks/bench.py`:

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| attention_decode (GQA) | T=16384, D=128 | 857 µs | 7667 µs (SDPA) | **8.9x** |
| RoPE NeoX (q+k) | [8192×4096] | 1654 µs | 10092 µs | **6.1x** |
| RMSNorm (+residual) | [4096×4096] | 613 µs | 2058 µs | **3.4x** |
| SwiGLU | [4096×4096] | 610 µs | 1031 µs | **1.7x** |
| top-k (k=50) | [131072] | 78 µs | 125 µs | **1.6x** |
| LayerNorm | [4096×4096] | 441 µs | 615 µs | **1.4x** |
| Softmax | [4096×4096] | 415 µs | 427 µs | 1.0x |
| SiLU / GeLU / add | [4096×4096] | ~414 µs | ~411 µs | ~1.0x |
| argmax | [131072] | 39 µs | 35 µs | 0.9x (incl. host readback) |
| attention_prefill (causal) | S=1024, D=128 | 5764 µs | 2607 µs (SDPA flash) | 0.45x (honest) |

Row-wise kernels (norms, softmax) autotune their thread-block size per
shape at first call (v0.4.1); the table reflects the tuned choices.

![fusedtok vs PyTorch reference](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rtx3060.png)

**RTX 5060 Ti (Blackwell, sm_120)** — same suite, highlights (full data:
`docs/benchmark_rtx5060ti.json`):

| Op | Shape | fusedtok | PyTorch reference | Speedup |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [512×4096] | 29 µs | 239 µs | **8.3x** |
| RMSNorm (+residual) | [4096×4096] | 512 µs | 1662 µs | **3.3x** |
| Softmax | [1024×4096] | 20 µs | 51 µs | **2.6x** |
| top-k (k=50) | [131072] | 27 µs | 41 µs (CUB) | **1.5x** |
| SwiGLU | [4096×4096] | 504 µs | 859 µs | **1.7x** |
| argmax | [32000] | 15 µs | 22 µs | **1.5x** |
| LayerNorm | [1024×4096] | 27 µs | 28 µs | ~1.0x |

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
an fp16 projection and runs at full memory bandwidth (2x); the IMMA
GEMM path (~17 TOPS) is correctness-first — cuBLASLt (`torch._int_mm`)
remains faster for large prefill matmuls.

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
- v0.4+: lightweight fused attention; prebuilt wheels on PyPI;
  pipelined tensor-core INT8 GEMM

## Community

- [Contributing guide](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) — setup, rules of the road, PR process
- [Code of conduct](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [Security policy](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [Changelog](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## License

MIT — see [LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE). Third-party notices: [NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md).
