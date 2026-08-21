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
| 🚧 | RMSNorm (+residual) | planned v0.1 |
| 🚧 | RoPE | planned v0.1 |
| 🚧 | SwiGLU | planned v0.1 |
| ⏳ | top-p / top-k sampling | planned v0.2 |
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

Requirements: CUDA Toolkit >= 12.0, compute capability >= 8.0 (Ampere+).

## Usage (preview)

```python
import fusedtok

x = [1.0, 2.0, 3.0]
y = fusedtok.axpy(x, 2.0, 1.0, cuda=True)   # current skeleton op
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
