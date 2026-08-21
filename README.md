# fusedtok

**Fused CUDA kernels for LLM inference** — RMSNorm / RoPE / SwiGLU, one `pip install` away.

**中文文档请看 [README_zh.md](README_zh.md)** | English below.

## Why

LLM inference frameworks launch many small, memory-bound operators per token. Each launch
round-trips through global memory. `fusedtok` fuses them into single kernels to cut memory
traffic and launch overhead.

## Operators

| Status | Kernel | Notes |
|---|---|---|
| ✅ | RMSNorm (+residual) | naive version, v0.1 |
| ✅ | LayerNorm | with affine, naive |
| ✅ | RoPE | interleaved **and** NeoX layouts, naive |
| ✅ | SwiGLU | naive version, v0.1 |
| ✅ | Softmax (row-wise) | numerically stable, naive |
| ✅ | SiLU / GeLU / ReLU / Tanh | elementwise, naive |
| ✅ | top-k / top-p (nucleus) | deterministic ties, naive |
| ✅ | argmax / temperature | sampling helpers, naive |
| ⏳ | INT8/FP8 quantized path | planned v0.3 |

## Install

```bash
pip install fusedtok   # not yet published — building from source until v0.1
```

Build from source:

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

**Requirements:**

- NVIDIA GPU of **RTX 30 series (Ampere) or newer** — e.g. RTX 3060/3090, RTX 4080, RTX 5090, A100, H100
- CUDA Toolkit >= 12.0

<details>
<summary>What is "compute capability"? (click to expand)</summary>

Compute capability is NVIDIA's version number for a GPU architecture generation — not a
performance score. CUDA code must be compiled for a specific architecture to run on it.
Kernels in this library target compute capability 8.0+ because features they rely on
(e.g. `__nv_bfloat16`) only exist from that generation on.

| Compute capability | Architecture | Example GPUs |
|---|---|---|
| 7.5 | Turing | GTX 16xx, RTX 20xx |
| 8.0 / 8.6 | Ampere | A100, RTX 30xx |
| 8.9 | Ada | RTX 40xx |
| 9.0 | Hopper | H100 |
| 12.0 | Blackwell | RTX 50xx, B200 |

Check yours: run `nvidia-smi` to see your GPU model, then look it up at
https://developer.nvidia.com/cuda-gpus

</details>

## Usage (preview)

```python
import fusedtok

x = [1.0, 2.0, 3.0]
y = fusedtok.axpy(x, 2.0, 1.0, cuda=True)   # current skeleton op

# RMSNorm over a flattened [rows, cols] tensor, optional fused residual
h = fusedtok.rmsnorm(x, w=[1.0, 1.0, 1.0], rows=1, cols=3, eps=1e-6,
                     residual=skip, cuda=True)
```

> The final API targets zero-copy torch tensors; the current skeleton uses plain lists
> to stay dependency-free while the framework is under construction.

## Correctness

Every kernel ships with a CPU reference implementation and element-wise parity tests
(pytest). Tests run on machines without a GPU (CUDA cases skip automatically).

## Benchmarks

Coming with v0.1 — comparisons vs PyTorch eager on RTX 3060 (sm_86).

## Development

```bash
# Windows: run inside a VS developer prompt (vcvars64)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
py -3.12 -m pytest tests -q      # with build dir on PYTHONPATH
```

Windows / Linux. Windows uses MSVC via nvcc; CI builds on both.

## License

MIT
