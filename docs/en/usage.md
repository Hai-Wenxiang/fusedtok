# fusedtok usage guide (English)

Topic-structured documentation for fusedtok. This page is the index:
each link opens one focused topic. For the operator index and
benchmark tables see the [README](../../README.md); for a runnable
tour of every operator see [`examples/demo.py`](../../examples/demo.py).

**Other languages:** [中文使用指南](../zh/usage.md)

## The map

| Page | What it covers |
|---|---|
| [Quickstart](quickstart.md) | install, first kernels, a taste of the fast paths |
| [The execution model](execution.md) | the three dispatch paths, dtype rules, streams, CUDA graphs, the error contract |
| [Attention operators](attention.md) | decode over contiguous and paged kv-caches, the append write sides (contiguous + paged), prefill |
| [Sampling and selection](sampling.md) | top-k / top-p / min-p / argmax, the fused samplers, the same-token determinism guarantee |
| [The INT8 path](int8.md) | quantization utilities, integer-exact qgemm, W8A8 per-channel scales |
| [Benchmarks](benchmarks.md) | the measurement protocol, how to read the tables (including the honest losses), reproduction |
| [FAQ / troubleshooting](faq.md) | GPU support matrix, common rejections, timing caveats, glossary |

## The one-minute version

```python
import numpy as np
import torch
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y = fusedtok.rmsnorm(x, w)                    # CPU reference
y = fusedtok.rmsnorm(x, w, cuda=True)         # staged CUDA
yt = fusedtok.rmsnorm(torch.from_numpy(x).cuda(),
                      torch.from_numpy(w).cuda())   # zero-copy CUDA
```

numpy in / numpy out, or torch in / torch out. CUDA torch tensors
select the zero-copy path automatically: kernels run in torch's own
device buffers, stream-ordered with other torch work, with no staging
copies and no host synchronization. Start with the
[quickstart](quickstart.md) for the full orientation.
