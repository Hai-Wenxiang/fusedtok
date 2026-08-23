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
| ✅ | top-k / top-p (nucleus) | radix-select, deterministic ties (1.4x vs torch @131k) |
| ✅ | argmax / temperature | greedy decoding helpers |
| ✅ | sample_topp | fused nucleus sampling: softmax -> top-p -> seeded draw, one kernel |
| ✅ | repetition penalty | CTRL-style, applied to sampled token ids |
| ⏳ | INT8/FP8 quantized path | planned v0.3 |

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

# sampling side: penalty + fused nucleus draw (one kernel, seeded)
logits = fusedtok.repetition_penalty(logits, sampled_ids, penalty=1.1)
token = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
```

A minimal per-token sampling loop:

```python
import torch, fusedtok as ft

h = torch.zeros(1, 4096, device="cuda")            # decoder state
w = torch.load("rms_weight.pt").cuda()             # float32 weights
generated = []
for step in range(256):
    h = ft.rmsnorm(h, w, residual=h)               # fused add + norm
    q = ft.rope(q, k=None, pos_offset=step, neox=True)
    logits = model_output(h)                       # your model
    logits = ft.repetition_penalty(logits, generated, 1.1)
    tok = ft.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
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

RTX 3060 (sm_86), float32, zero-copy torch tensors, CUDA-event timing, vs the
equivalent PyTorch eager expressions (full data: `docs/benchmark_rt3060.json`,
reproduce with `python benchmarks/bench.py`):

| Op | Shape | fusedtok | PyTorch eager | Speedup |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [2048×4096] | 416 µs | 2570 µs | **6.2x** |
| RMSNorm (+residual) | [1024×4096] | 260 µs | 538 µs | **2.1x** |
| SwiGLU | [1024×4096] | 153 µs | 257 µs | **1.7x** |
| LayerNorm | [1024×4096] | 168 µs | 162 µs | ~1.0x |
| SiLU | [1024×4096] | 105 µs | 112 µs | ~1.0x |
| Softmax | [1024×4096] | 159 µs | 115 µs | 0.7x |
| argmax | [131072] | 36 µs | 46 µs | **1.3x** |
| top-k (k=50) | [131072] | 168 µs | 129 µs | 0.8x |

![fusedtok vs PyTorch eager](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rt3060.png)

**RTX 5060 Ti (Blackwell, sm_120)** — same suite, torch 2.11/cu128, highlights:

| Op | Shape | fusedtok | PyTorch eager | Speedup |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [512×4096] | 29 µs | 240 µs | **8.3x** |
| RMSNorm (+residual) | [4096×4096] | 512 µs | 1662 µs | **3.3x** |
| Softmax | [1024×4096] | 20 µs | 51 µs | **2.6x** |
| SwiGLU | [4096×4096] | 504 µs | 858 µs | **1.7x** |
| argmax | [32000] | 11 µs | 22 µs | **1.9x** |
| LayerNorm | [1024×4096] | 27 µs | 28 µs | ~1.0x |

![fusedtok vs PyTorch eager (RTX 5060 Ti)](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rt5060ti.png)

The PyPI wheel ships sm_80/sm_86 cubins plus a compute_86 PTX fallback —
verified to JIT and run correctly on Blackwell (sm_120) drivers.

Fusions win big (RoPE / RMSNorm / SwiGLU) because eager mode round-trips
intermediate tensors through global memory. Pure memory-bound elementwise ops
run at the same ~330-500 GB/s as PyTorch's tuned kernels (silu, gelu, add ≈
parity). Softmax and top-k remain behind PyTorch's CUB-based kernels —
honest numbers, on the v0.2 roadmap.

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
- v0.4: decoupled-lookback selection (CUB-class on many-SM GPUs),
  INT8 GEMM path, block-size autotuning
- v0.4+: lightweight fused attention; prebuilt wheels on PyPI

## Community

- [Contributing guide](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) — setup, rules of the road, PR process
- [Code of conduct](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [Security policy](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [Changelog](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## License

MIT — see [LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE). Third-party notices: [NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md).
