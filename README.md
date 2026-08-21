# fusedtok

**Fused CUDA kernels for LLM inference** — RMSNorm / RoPE / SwiGLU, one `pip install` away.

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
import torch
import fusedtok

x = torch.randn(8, 4096, dtype=torch.float16, device="cuda")
w = torch.randn(4096, dtype=torch.float16, device="cuda")

y = fusedtok.rms_norm(x, w, eps=1e-5)          # fused residual optional
q, k = fusedtok.rope(q, k, theta=10000.0)      # in-place rotation
```

## Correctness

Every kernel ships with a CPU reference implementation and element-wise parity tests
(GoogleTest + pytest). Tests run on machines without a GPU.

## Benchmarks

Coming with v0.1 — comparisons vs PyTorch eager on RTX 3060 (sm_86).

## Development

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Windows / Linux. Windows uses MSVC via nvcc `-ccbin`; CI builds on both.

## License

MIT
