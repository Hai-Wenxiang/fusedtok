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
(manylinux, CPython 3.11-3.13) and **Windows x86_64** (3.11-3.13). On
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

The headline operators, each one call on the zero-copy path:

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

# serving a batch: one call, one seeded token per row (spike the
# logits like real decode output - on flat random logits the batched
# win shrinks, see the benchmarks)
batch_logits = torch.randn(8, 131072, device="cuda")
batch_logits[torch.arange(8, device="cuda"),
             batch_logits.argmax(dim=1)] += 20.0
tokens = fusedtok.sample_topp_batched(batch_logits, p=0.9)

# ...with per-row repetition penalties included: ragged histories
# (a list of per-row id lists), one call, one token per row
histories = [[5, 9], [], [1, 2, 2], [7] * 16,
             [3], [], [0], [4, 4]]
tokens = fusedtok.decode_step_batched(batch_logits, histories,
                                      penalty=1.3, p=0.9)

# entropy-adaptive cutoffs (v1.6): the truncation threshold is
# derived from the distribution's own entropy, not fixed
token = fusedtok.sample_eta(logits, eta=1e-3, temperature=0.8, seed=0)
token = fusedtok.sample_typical(logits, typical=0.9,
                                temperature=0.8, seed=0)

# the combined HF penalties (v1.6.1): one call, once per distinct id
penalized = fusedtok.logit_penalties(logits, [3, 3, 7], repetition=1.2,
                                     presence=0.1, frequency=0.05)

# value-threshold pair (v1.8): both cutoffs ride min-p's prefix
# machinery - top-a keys off the squared peak, nsigma off the row's
# own spread
token = fusedtok.sample_topa(logits, top_a=0.2, temperature=0.8, seed=0)
token = fusedtok.sample_nsigma(logits, nsigma=1.5, temperature=0.8,
                               seed=0)

# batched greedy argmax (v1.8): one launch for the whole batch, ties
# to the earliest index - the batched decode shortcut with no seeding
idx = fusedtok.argmax_batched(batch_logits)
```

`examples/demo.py` in the repository tours every operator with
closed-form checks - it doubles as executable documentation.

## What to read next

- [The execution model](execution.md) - the three paths, dtype rules,
  streams, CUDA graphs, and the error contract
- [Attention operators](attention.md) - decode, paged caches, the
  contiguous and paged append write sides, prefill
- [Sampling and selection](sampling.md) - top-k/top-p/min-p, the
  value-threshold pair (top-a, top-n-sigma), the entropy-adaptive
  samplers (eta, typical), the combined logit penalties, and the
  determinism contract
- [The INT8 path](int8.md) - quantization and integer-exact GEMM
- [Benchmarks](benchmarks.md) - how the numbers are measured and how to
  read them
- [FAQ / troubleshooting](faq.md)
