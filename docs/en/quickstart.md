# Quickstart

This page gets you from `pip install` to running kernels on your GPU in
a few minutes. If you are evaluating the library, start here; the
[execution model](execution.md) page explains what happens under the
hood, and the topic pages ([attention](attention.md),
[sampling](sampling.md), [INT8](int8.md)) go deep on each operator
family.

**Other languages:** [中文快速上手](../zh/quickstart.md)

## Install

```bash
pip install fusedtok
```

Prebuilt wheels (built with CUDA 12.4) cover **Linux x86_64**
(manylinux, CPython 3.10-3.13) and **Windows x86_64** (3.11-3.13). On
other platforms or Python versions, pip builds from source
automatically:

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

**Requirements** (source builds): an NVIDIA GPU of the RTX 30 series
(Ampere) or newer, CUDA Toolkit >= 12.0, and a C++17 compiler. Prebuilt
wheels need only a matching driver. See the
[FAQ](faq.md#which-gpus-are-supported) for the full architecture table
and how JIT fallback works on RTX 40/50 cards.

## Thirty-second tour

```python
import numpy as np
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y = fusedtok.rmsnorm(x, w)          # CPU reference - runs anywhere
y = fusedtok.rmsnorm(x, w, cuda=True)   # staged: copy to GPU and back
```

Every operator accepts numpy arrays and (when torch is installed) torch
tensors. A CUDA torch tensor switches to the **zero-copy path**: the
kernels read and write torch's own device buffers directly, launch on
torch's current stream, and never stage through host memory.

```python
import torch

xt = torch.from_numpy(x).cuda()
wt = torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)       # zero-copy CUDA, output on GPU
```

## A taste of the fast paths

The two headline operators, both one call on the zero-copy path:

```python
# attention over a GQA kv-cache: one launch streams the whole cache
q = torch.randn(1, 32, 128, device="cuda")       # [B, Hq, D]
k_cache = torch.randn(1, 8, 16384, 128, device="cuda")   # [B, Hkv, T, D]
v_cache = torch.randn(1, 8, 16384, 128, device="cuda")
lens = torch.tensor([16384], dtype=torch.int32, device="cuda")
out = fusedtok.attention_decode(q, k_cache, v_cache, lens)

# the whole decode-step sampling chain in one call, one readback
logits = torch.randn(131072, device="cuda")
token = fusedtok.decode_step(logits, [], penalty=1.1,
                             p=0.9, temperature=0.8, seed=0)
```

`examples/demo.py` in the repository tours every operator with
closed-form checks - it doubles as executable documentation.

## What to read next

- [The execution model](execution.md) - the three paths, dtype rules,
  streams, CUDA graphs, and the error contract
- [Attention operators](attention.md) - decode, paged caches, prefill
- [Sampling and selection](sampling.md) - top-k/top-p, fused samplers,
  and the determinism contract
- [The INT8 path](int8.md) - quantization and integer-exact GEMM
- [Benchmarks](benchmarks.md) - how the numbers are measured and how to
  read them
- [FAQ / troubleshooting](faq.md)
