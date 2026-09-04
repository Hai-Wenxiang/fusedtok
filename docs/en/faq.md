# FAQ / troubleshooting

**Other languages:** [中文：常见问题](../zh/faq.md)

## Which GPUs are supported?

Ampere (RTX 30, compute capability 8.0/8.6) or newer. The wheels ship
sm_80/sm_86 cubins plus a compute_86 PTX fallback, so newer
architectures (RTX 40/50, H100) JIT through their driver - verified
on Blackwell (sm_120). Turing (GTX 16xx / RTX 20xx, cc 7.5) and older
are not supported.

| Compute capability | Architecture | Example GPUs | How fusedtok runs |
|---|---|---|---|
| 7.5 | Turing | GTX 16xx, RTX 20xx | not supported |
| 8.0 / 8.6 | Ampere | A100, RTX 30xx | native cubins |
| 8.9 | Ada | RTX 40xx | PTX JIT |
| 9.0 | Hopper | H100 | PTX JIT |
| 12.0 | Blackwell | RTX 50xx | PTX JIT |

Check yours with `nvidia-smi`, then look the model up at
https://developer.nvidia.com/cuda-gpus.

## It says "no CUDA device" but I have a GPU

- `fusedtok.cuda_available()` returns False when a CUDA context cannot
  be created: check that `nvidia-smi` works, that the driver is
  current for your CUDA runtime (>= 12.0 class), and that a device is
  actually visible to the machine.
- All CPU-reference functionality works without a GPU - only the
  CUDA paths need one.

## Why is my first call slower?

First calls may do one-time work per shape: allocate a workspace
(attention split path, selection pipeline) or micro-benchmark launch
configs (row-kernel block size, qgemm tiles). The choices are cached
for the process. Warm up once before capturing a CUDA graph or timing
(see [execution model](execution.md#streams-and-cuda-graphs)).

## Why was my tensor rejected?

| Error | Typical cause | Fix |
|---|---|---|
| `TypeError: ... must be float32, bfloat16 or float16` | unsupported dtype on the zero-copy path | convert with `.to(torch.float32)` (or bf16 where supported) |
| `ValueError: ... must be contiguous` | strided view on the zero-copy path (kernels address raw pointers) | pass `.contiguous()` |
| `TypeError: ... must be a CUDA tensor when the primary input is on CUDA` | mixed devices | move every operand to the same device |
| `ValueError: lens entries must be in [0, T]` | out-of-range length in a host-side `lens` | fix the values (device-resident `lens` tensors are trusted - validate them yourself) |
| `RuntimeError: ... launch ...` | CUDA-side failure | check `nvidia-smi` for context/memory; open an issue with the reproducer |

Full contract: [execution model - the error contract](execution.md#the-error-contract).

## Why do CPU and GPU sample different tokens (rarely)?

The samplers are deterministic per seed, and the CPU / staged /
zero-copy paths agree - except when a draw lands exactly on an
exp-rounding boundary of the CDF (CPU uses exact `exp`, GPU uses the
~2-ulp `__expf`); the two draws are then neighboring, equally valid
samples. At very large vocabularies with near-uniform logits this
happens at scale (a small rank window, measured ~14 ranks at
n=152064). There is also a rarer GPU-side boundary: the global
softmax total is accumulated with per-block float atomics whose
scheduling order differs between processes, so one draw in a boundary
case may pick a neighboring token after a process restart. Details:
[sampling - the same-token guarantee](sampling.md#the-same-token-guarantee).

## Why is my batched sampler slower than torch on flat logits?

`torch.multinomial` never sorts - it draws against the full
distribution with a boolean-mask pass. The fusedtok samplers must
ORDER the nucleus (the selection pipeline), and on near-uniform
logits the nucleus is ~90% of the vocabulary, so they honestly lose
that regime (0.05-0.06x batched, 0.17-0.25x single-row). Real decode
logits are peaked, where the samplers sit at native-multinomial
level or better. Details:
[sampling - flat distributions](sampling.md#flat-distributions---the-honest-worst-case).

## Are the samplers cryptographically secure?

No. The RNG is a splitmix-style hash of the seed - reproducible and
evenly distributed, but not a CSPRNG. Use external randomness if your
application needs one.

## Can I capture CUDA graphs?

Yes, library-wide, with warm-up first. Exceptions: the fused samplers
return host ints (each call ends in a readback) - the `_batched`
variants additionally re-launch kernels from a readback inside their
widening loop, so int64 arrays or not, they stay outside graphs - and
`quantize_int8`/`qadd_int8` sync once mid-call to compose their
scales. The selection pipeline captures its own internal graph
automatically - you do not manage it.

## Windows-specific timing notes

Windows runs GeForce drivers in WDDM mode: kernel submissions cost
tens of microseconds and host-side timing is noisy. Benchmarks use
CUDA events (see [benchmarks](benchmarks.md#the-protocol)); if you
micro-benchmark yourself, prefer events over wall clock and expect
`argmax`-class tiny ops to swing.

## Glossary

- **GQA** - grouped-query attention: `Hq` query heads share `Hkv < Hq`
  key/value heads in contiguous groups (`q head h -> kv head
  h // (Hq/Hkv)`).
- **kv-cache** - stored keys/values of previous tokens; decode
  attention reads the whole cache each step.
- **Paged cache / block table** - cache layout where tokens live in
  fixed-size blocks anywhere in a pool, mapped per sequence by a block
  table (the vLLM design). Avoids fragmentation from growing,
  shrinking and evicted sequences.
- **Flash-decoding** - long-sequence decode strategy: split the cache
  into slices, compute partial softmaxes in parallel, then reduce.
- **Nucleus** - the truncated candidate set a sampler draws from:
  top-p takes the prefix whose cumulative probability mass reaches p;
  min-p takes the prefix of probabilities at least min_p times the
  maximum.
- **Batched sampling** - sampling a whole `[rows, vocab]` batch in one
  call (the `_batched` samplers): every row runs the single-row
  pipeline verbatim with its own seed; the speedup over a per-row loop
  comes from collapsing launch/submission overhead, not from different
  math.
- **multinomial** - torch's probability-weighted draw
  (`torch.multinomial`); the reference implementation this library's
  samplers are benchmarked against (composed with softmax, and with
  top-k / a boolean mask where the row label says so).
- **Radix** - the selection pipeline's ordering technique: candidate
  keys are histogrammed round by round on their high bytes (the radix
  sort idea).
- **SDPA** - PyTorch's `scaled_dot_product_attention`, the official
  reference implementation this library benchmarks against.
- **Cooperative launch** - a CUDA launch primitive that starts every
  block at once and provides grid-wide synchronization; the selection
  pipeline replaces it with plain launches plus arrival tickets.
- **arrival-ticket** - the selection pipeline's cross-block ordering
  trick: the last block to arrive at a counter decides the round, so
  plain kernel launches replace grid-wide barriers.
- **Materialize** - write an intermediate tensor to global memory
  (fusedtok's selling point is not materializing scores/intermediates).
- **W8A8** - INT8 inference layout: 8-bit weights with per-output-
  channel scales, 8-bit activations with per-tensor scales.
- **WDDM** - the Windows display driver model; adds kernel submission
  latency that Linux's driver model does not have.
- **Zero-copy** - kernels operate directly on torch's device buffers
  via `data_ptr()`; no staging copies, no host sync.
