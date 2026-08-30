# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- attention_decode / attention_prefill accept **bfloat16 and float16**
  CUDA tensors (v1.1 headline): the kernels are templated on the
  storage dtype and compute entirely in float32, so half-precision
  caches halve the global bytes on the bandwidth-bound decode path
  while the softmax numerics stay float32. Output matches the input
  dtype; CPU paths remain float32 (numpy has no bf16/fp16). 12 new
  tests pin half-precision parity (vs float64 attention on the
  half-rounded inputs), the GQA mapping through a constant-V probe,
  poisoned-padding and zero-length handling, single-vs-split path
  agreement, graph capture, and mixed-dtype rejection. Benchmarked
  against SDPA in the SAME dtype: bf16 decode @T=16384 = 2.11x on a
  3060 / 1.17x on a 5060 Ti (vs our own f32 path the absolute gain at
  batch 1 is modest - the kernel is latency-bound there; the byte
  savings grow with batch).
- benchmarks/bench.py: attention decode bf16 rows; sample_topp rows in
  two labeled regimes (peaked / flat worst case, see the 1.0.1 notes).
- examples/demo.py: qgemm_perchannel and sample_topk tours.
- test_select_pipeline: large-window (n=40000) sampling determinism
  pin - the same seed reproduces the same token across calls in the
  exp-precompute regime.

### Changed
- sampling serial scans precompute their exp column in PARALLEL
  (exp_window_kernel writes exp(v - row_max) into the idle ping-pong
  key buffer; zero extra memory) and the single-threaded walkers then
  only add. The accumulation order of every walk is UNCHANGED, so
  per-seed tokens are bit-identical to the 1.0 serial scans - only the
  expf compute left the serial path. Flat-distribution worst case:
  25.4ms -> 13.9ms at n=131072 on a 3060 (1.8x), 6.5ms -> 3.1ms at
  n=32000.
- the dim<32 prefill fallback kernel is GONE: under dtype templating it
  miscompiled (hq >= 2 with dim 16 produced wrong rows even in
  float32), and the tiled kernel's dim<=128 band handles tiny heads
  correctly through zero-padded staging and per-lane bounds guards -
  one code path fewer, correct in all three storage dtypes.

## [1.0.1] - 2026-08-30

Maintenance release: honest sampling benchmarks, an x8 sampling-window retry jump, documentation and metadata corrections. No API changes; all 338 tests green on RTX 3060 (Windows, CUDA 13.3) and RTX 5060 Ti (Linux, CUDA 13.2).

### Added
- benchmarks/bench.py: sample_topp rows in TWO honestly labeled
  regimes - "peaked" (a dominant token, the decode-time case that the
  first sampling window covers) and "flat (worst case)" (randn logits,
  where the nucleus spans most of the vocab and the widening loop
  reruns the pipeline on ever-larger windows). The peaked case is a
  real win (3.11x on a 3060 / 2.49x on a 5060 Ti vs the softmax+sort+
  mask+multinomial composite); the flat case is honestly 0.01-0.02x and
  now both the number and the reason sit in the README tables and
  prose instead of being invisible.
- examples/demo.py: tours for the two 1.0 operators the demo never
  covered - qgemm_perchannel (bit-exact W8A8 parity vs numpy) and
  sample_topk (top-k set membership, k=1-is-greedy, per-seed
  CUDA-vs-CPU parity).

### Changed
- sampling window widening jumps x8 instead of x4 (sample_topp and
  decode_step): flat distributions need windows 50-100x the vocab tail
  and x4 needed up to five full pipeline attempts on n=131072, each
  re-histogramming every key. The sampled token is unaffected by the
  jump size (a covered nucleus samples identically - the threshold is
  the global mass, the renormalization the nucleus mass), only the
  retry count drops (5 -> 4 attempts worst case at n=131072).
- README roadmap: 1.0 marked as released; install-facing metadata moved
  to the stable classifier (Development Status 5 - Production/Stable).
- quantize.cu: the file header still claimed "INT8 GEMM is out of scope
  (v0.4+)" - the compute half shipped in v0.4 and gained per-channel
  scales in 1.0; the header now points at qgemm.cu.
- topk.cu: merged a duplicated (and partly contradictory) doc block over
  select_round_kernel into one accurate description - the rounds only
  ever run stage 0; compaction belongs to select_finalize_kernel.

## [1.0.0] - 2026-08-30

First stable release. The public surface (30 operators and helpers in
`fusedtok.__all__`) is now frozen: additions land in minor releases,
signature changes require a major version and a deprecation window.
Typed (PEP 561 `py.typed` + stubs), bilingual docs, a text-hygiene CI
gate, and a seven-way release wheel matrix (Linux cp310-cp313, Windows
cp311-cp313).

### Added
- API freeze infrastructure for 1.0: `python/fusedtok/__init__.pyi`
  type stubs (full signatures for the whole frozen surface) plus the
  PEP 561 `py.typed` marker - IDEs and type checkers now see every
  operator signature. `tests/test_api.py` pins the frozen surface
  (an accidental addition to or removal from `__all__` fails), keeps
  every public callable documented, pins `__version__` as semver,
  guards against undocumented public leaks, and asserts the stub
  tracks `__all__`. The error contract is now written down completely:
  ValueError for shape/value problems, TypeError for dtype/device/
  mixed-input problems, RuntimeError for CUDA execution failures, and
  the 1.0 stability policy (additions in minor releases; signature
  changes need a major + a deprecation window) is documented in
  CONTRIBUTING and both READMEs.
- scripts/check_utf8.py: a repository-wide text hygiene gate - every
  tracked text file must be strict UTF-8, BOM-free, without U+FFFD
  replacement characters or the double-encoded (UTF-8-as-CP1252)
  mojibake shape that bit the 0.1.1/0.1.2 PyPI pages. CI runs it on
  every push before anything compiles.
- qgemm_perchannel(a_q, a_scale, b_q, b_scales): per-output-channel
  weight scales - the W8A8 layout real INT8 inference uses (activations
  per-tensor, weights one scale per output row). The per-channel scale
  multiply is fused into the pipelined kernel's epilogue (and the M==1
  decode GEMV) at zero kernel cost: measured dead even with per-tensor
  qgemm (38.7 vs 38.7 TOPS on a 3060; 66.5 vs 66.1 on a 5060 Ti) while
  the composite torch reference (cuBLASLt + separate scale broadcast)
  pays for the epilogue separately. Exactness contract unchanged and
  pinned: f32(sa * sb[j]) composes with ONE rounding, the product
  applies once, CPU / staged / zero-copy are bit-identical; with all
  b_scales equal the output is bit-equal to per-tensor qgemm. New
  end-to-end test quantizes spiky weight rows per channel and asserts
  the 5x+ error reduction over per-tensor scales on non-outlier rows.
- sample_topk(logits, k, temperature=1.0, seed=0): fused top-k sampling
  - softmax of the temperature-scaled logits, top-k truncation,
  renormalization WITHIN the k survivors, and a seeded inverse-CDF
  draw, one call with one readback. The k window is covered by
  construction, so unlike the nucleus sampler there is no global-mass
  threshold and no widening loop; the serial draw reuses the identical
  splitmix RNG and accumulation order as sample_topp (deterministic per
  seed; same caveat: exact-exp on CPU vs __expf on GPU can move draws
  sitting exactly on an exp-rounding boundary). Measured vs the
  composite an inference loop would write (torch.topk + softmax +
  multinomial): 2.13x on a 3060 and 1.91x on a 5060 Ti at
  k=50 @131072 - the timing is the fair part, the composite's draw is
  not seed-reproducible while fusedtok's is. 16 new tests pin set
  membership, k=1-is-greedy, full-vocab clamping, cold-temperature
  collapse to argmax, mass concentration, per-seed determinism and
  cross-path parity.

### Changed
- qgemm (INT8 matmul) rewritten as a cp.async double-buffered pipelined
  IMMA kernel: K streams through two shared-memory slabs - the DMA for
  slab s+1 overlaps the tensor-core work of slab s (the v0.4 kernel
  round-tripped every byte through registers and stalled the whole block
  on a barrier per 64-element slab); boundary tiles are padded by
  pre-zeroed shared memory instead of per-load predicates; and the tile
  size is a runtime-tuned choice between 64x64 (256 threads; slab 64 or
  128) and a 128x128 DRAM-intensity tile (512 threads, 96 KB opt-in
  dynamic shared memory). Integer accumulation is untouched - CPU /
  staged / zero-copy stay bit-identical. Measured on RTX 3060 (3-round
  averages, 4096^3): 17 TOPS (v0.4) -> 39 TOPS, 0.15x -> 0.46x vs
  cuBLASLt (torch._int_mm); the honest gap to cuBLASLt (~83 TOPS, about
  82% of the sm_86 dense-INT8 peak) is analyzed in the README
  performance section.
- top-k selection retuned for the mid-k range (v0.4's documented weak
  spot): the early-exit compaction threshold and the per-block sort
  chunk BOTH drop from 2048 to 1024. The entire regression window was a
  single block bitonic-sorting 2048 keys - every SM but one idle - and
  for k > 1024 the parallel chunk+merge tail does the same work spread
  across the device. Measured on the canonical 3-round protocol at
  n=131072, k=4096: RTX 3060 143 -> 113 us (0.87-0.93x -> 1.12x vs
  torch.topk), RTX 5060 Ti 0.79-0.84x -> 1.09x; large k improves too
  (k=16384 on the 3060: 1.59x -> 2.12x), small k unchanged. The first
  fused-kernel attempt (chunk sort + merge ladder + decode in ONE
  launch behind arrival-ticket barriers) measured SLOWER - inside the
  cached graph the per-kernel savings are ~1-2us while the 16-block
  co-residency cap cuts the ladder's natural parallelism - and was
  reverted; the one measurement that mattered was the sort's serial
  span.

### Fixed
- qgemm with K == 0 zero-fills the output on every path (the v0.4
  launcher skipped the GPU write and returned torch.empty garbage; the
  CPU reference always produced zeros). Pinned by a new cross-path test.
- quantize.cu: removed a dead index variable (nvcc 13.x warning #177-D
  on every build; leftover from an old 4x-unroll iteration).

## [0.5.1] - 2026-08-30

### Fixed
- docs: README headline speedup refreshed to the 0.5.0 attention numbers
  (9.3x vs SDPA at decode; the old 6.2x RoPE line predated attention);
  install notes now mention the PyPI Windows wheel (cp312); the usage
  tour covers attention_decode / attention_prefill; both languages.
- CONTRIBUTING new-kernel checklist: two v0.5 lessons added (runtime
  loop bounds demote register arrays to local memory - a measured 7x;
  full-warp-mask shuffles deadlock under warp divergence).

### Changed
- benchmarks/bench.py: every configuration is now timed over THREE
  independent rounds (each with its own warmup) and reported as the
  mean; the per-round values ride along in the JSON and the console
  shows the round spread, so jitter stays auditable. The chart's x-axis
  ticks always include 1 - matplotlib's default step-2 ticks at x_max
  ~ 9.5 (the RTX 3060 chart) read 0/2/4/6/8 and lost the parity anchor
  (the 5060 Ti chart happened to stay under 10 and kept it).
- attention kernels: the per-lane chunk index had two names (nc at
  load, j at use) across the three online-softmax kernels - unified so
  the load-to-store mapping reads as one invariant (no behavior
  change; 287 tests green, perf unchanged, re-verified on both GPUs
  under the 3-round protocol).
- README benchmark tables regenerated from the 3-round JSONs on both
  GPUs (headlines: attn decode 8.92x @ 3060 / 4.67x @ 5060 Ti; argmax
  honestly at 0.69x including the host readback).

## [0.5.0] - 2026-08-30

### Added
- attention_decode(q, k_cache, v_cache, lens=None): single-token
  (decode step) causal attention with GQA over a contiguous
  [B, Hkv, T, D] kv-cache - `out = softmax(q . K^T / sqrt(D)) . V` with
  q head h attending via kv head h // (Hq/Hkv) (contiguous groups; Hq ==
  Hkv is plain MHA). Two strategies behind one launcher: short caches /
  saturated grids take a single kernel (one block per (batch, q head),
  8 warps striding the rows with a running ONLINE softmax each, merged
  in shared memory); long caches (>= ~512 rows) take a flash-decoding
  split path - the sequence is sliced, stage-1 blocks (one per
  (batch, kv head, slice)) compute the partials of ALL q heads of the
  GQA group over their slice in one pass (k/v rows read once, reused
  across the group) into a process-cached workspace, and stage-2 blocks
  max-rescale-merge the partials. Scores never materialize; q/K/V are
  read exactly once. Stream-ordered and CUDA-graph capturable on both
  paths (workspace allocation happens outside captures; a first call
  racing an active capture falls back to the single-kernel path).
  Optional per-sequence `lens` mark valid cache rows so
  variable-length batches share one cache tensor (padding rows are
  ignored; a zero-length sequence produces a zero output row). float32;
  dim multiple of 4, at most 512. RTX 3060 vs pre-expanded
  torch SDPA (Lq=1): 4.2x @ T=512, 6.5x @ 4k, 9.3x @ 32k (163 GB/s
  effective); 13.3x against the expand+SDPA composite most code
  actually writes.
- attention_prefill(q, k, v, causal=True): fresh-sequence attention
  over S query rows - q [B,Hq,S,D] x k/v [B,Hkv,S,D], causal row i
  attending to keys [0, i] (or all rows when causal=False). Tiled
  single kernel: a 64/32/16-row query tile (by head size) lives in
  shared memory while K/V stream through staged chunks, and lanes split
  into per-row groups (4/8/16 lanes) so a dot product needs only
  log2(lanes) xor-shuffles instead of a 10-step warp reduction - the
  register arrays stay register-resident via compile-time trip counts
  (a runtime bound silently demotes them to local memory, measured 7x).
  Same GQA grouping, zero-row and dtype conventions as the decode op;
  CUDA-graph capturable. HONEST numbers (RTX 3060, D=128): ~0.45x of
  torch SDPA's flash backend at S=256..4096 - this is the convenience
  path (GQA + causal + fusedtok pipeline integration without
  materializing scores); heavyweight prefill belongs to SDPA /
  FlashAttention (tensor cores); the library's competitive attention
  surface is the decode step.
- tests/test_attention.py: GQA mapping (constant-V probe), float64
  reference parity across shapes (single key, odd T, empty cache, long
  4096-row cache, D=4..256), padding-poisoning, error contract, staged
  and zero-copy paths, an independent torch repeat_interleave
  crosscheck, graph capture-replay with mutation on both kernel paths,
  and split-path pins for every templated group width (1/2/4/8/16),
  lens crossing slice boundaries, batched long caches and 16k rows.

## [0.4.1] - 2026-08-29

### Added
- runtime block-size autotuning for the row-wise kernels (rmsnorm,
  layernorm, softmax, both dtypes): the first call for a (op, dtype,
  rows, cols) shape micro-benchmarks the candidate thread blocks (128 /
  256 / 512 / 1024) with the REAL kernel on the caller's own buffers at
  full size and caches the winner for the process. Tuning a truncated
  scratch problem misleads the choice (small grids favor big blocks,
  full grids do not) - measured before shipping. Stream captures skip
  tuning and use the default block; structurally unlaunchable
  candidates (register-resident softmax at 1024 threads) score as slow
  instead of failing. Measured on RTX 3060 vs the fixed 256-thread
  baseline: layernorm +17..53% across shapes (e.g. [4096x4096] 671 ->
  460 us, [512x8192] 204 -> 108 us), rmsnorm+residual [4096x4096] +39%
  (1010 -> 616 us), softmax ~+1% (kept for the wide-row online variant
  where big blocks win). tests/test_autotune.py pins correctness across
  tuned blocks, bit-identical cached repeats, dtype-specific choices,
  and capture paths.
- benchmarks/bench.py: the chart now uses two LINEAR panels (split at
  the largest gap of the sorted per-op maxima) with direct microsecond
  labels and color-coded speedup badges, replacing the log-scale axis.

## [0.4.0] - 2026-08-29

### Added
- qgemm(a_q, a_scale, b_q, b_scale): INT8 matmul closing the v0.4
  quantized-compute loop - y = (A_q[M,K] int8 @ B_q[N,K] int8^T) *
  (sa*sb) with int32-exact accumulation (CPU / staged / zero-copy
  results are bit-identical; the integer math has no tolerance games and
  the single float scale differs from a numpy float64 reference by at
  most one rounding). Both operands row-major along K (the LLM
  activations @ weight.T layout); M == 1 dispatches to a warp-per-row
  GEMV kernel. The GEMM path uses tensor-core IMMA (wmma s8xs8->s32,
  64x64 tile, 32x16 per warp). Honest numbers (RTX 3060): decode GEMV
  [1x4096 @ 131072x4096] runs at 337 GB/s effective - exactly 2x faster
  than the same projection in fp16, which is the point of INT8 weights;
  mid-size GEMM reaches ~17 TOPS vs cuBLASLt torch._int_mm's 83 TOPS (a
  pipelined/CUTLASS-class kernel stays future work; correctness and
  stream/graph integration are complete). All launchers are
  stream-aware and CUDA-graph capturable.
- tests/test_qgemm.py: exact integer parity across shapes (tile
  boundaries, K tails, k=1), int8 extremes, end-to-end quantized error
  bound, decode-shaped GEMV, and graph capture-replay with mutation.

- decode_step(logits, sampled_ids, penalty, p=0.9, temperature=1.0,
  seed=0): the fused decode loop - repetition penalty (vocab bitmap,
  applied to the raw logit before the temperature scale, matching the
  composed reference order), temperature, and nucleus sampling, all
  inside the selection pipeline with a single host readback. Same seed
  gives the same token as the composed repetition_penalty ->
  temperature -> sample_topp calls on CPU and every GPU path. 131k-vocab
  decode: 309us/token vs 354us for the composed three calls (1.15x, and
  one API call instead of three); the launch is raw (no internal graph -
  the penalty rides as a kernel parameter, which cached graphs would
  bake stale).
- tests/test_decode_step.py: composed-reference parity across penalties
  and seeds, distribution shift under heavy penalty, empty/disabled
  penalty paths, a 40-token generation loop, torch zero-copy parity.

### Changed
- selection kernels (top-k / top-p / sampling) rewritten as a multi-launch
  pipeline of plain kernels: per-round arrival-ticket radix refinement (the
  last block to arrive decides the round - no grid-wide barriers, no
  cooperative launch), early-exit compaction when a boundary bin holds at
  most 2048 candidates (one block sorts the survivors in shared memory),
  two-level emit counting (one global atomic per block; k = n skips
  counting entirely), and a merge-path sort ladder with one launch per
  level for k > 2048. The whole sequence is captured into a process-cached
  CUDA graph per (n, k, mode), so a call submits as ONE graph launch;
  per-call pointers and p travel through a pinned argument ring copied
  into a device-side argument block that the kernels dereference at
  runtime (graph nodes stay pointer-stable). Devices without cooperative
  launch no longer need the v0.1 host-rounds fallback (removed).
- every `_launch` entry point (and the staged drivers' signatures) gained
  a trailing `stream` argument; the Python layer passes the live torch
  stream on zero-copy paths. This FIXES CUDA-graph capture library-wide:
  the previous launchers used the legacy default stream, so
  `torch.cuda.graph` captures came back EMPTY and the graph tests passed
  vacuously against stale warm-up results. The capture tests now mutate
  inputs between replays and assert recomputation.
- `sample_topp` nucleus threshold now compares against the GLOBAL softmax
  mass (computed by two new reduction kernels) instead of the window-local
  total. The old window-local renormalization silently shrank the nucleus
  whenever the distribution was flat enough that the widening window did
  not yet cover it (semantic bug present since v0.2; exposed by a new
  flat-logits widening test). Flat distributions now widen correctly up to
  the full vocabulary; realistic peaked logits typically sample from the
  first 2048-token window.

### Performance (dual-GPU, CUDA events, honest)
- top-k k=50 @131k: RTX 5060 Ti 26.7us vs torch.topk (CUB radix) 40.7us =
  1.53x (v0.3: 132us, 0.31x - the many-SM grid.sync barrier pathology is
  gone); RTX 3060 85.6us wall = 1.56x (v0.3 absolute time roughly halved;
  the remaining Windows/WDDM gap is CUDA API submission cost, ~44us/call
  for graph launch + argument copy, not GPU work).
- top-p @131k: 5060 Ti 158us (v0.3: 672us, 4.3x faster), 0.44x vs a
  torch sort+cumsum composite; 3060 351us (v0.3: 1146us).
- sample_topp (realistic peaked logits, p=0.9): 3060 299us including the
  host readback (v0.3: 976us with the old, incorrect window semantics);
  flat worst-case distributions honestly cost milliseconds (full-vocab
  serial scan is the price of the corrected global-mass semantics).
- top-k mid-range k (2048..5000) remains at or below parity on both GPUs
  (honest numbers; the CUB radix select is hard to beat when k is large
  relative to n).

### Added
- tests/test_select_pipeline.py: deep-prefix distributions that force
  several radix rounds, tie groups spanning the k boundary, the k > 2048
  merge ladder, interleaved calls sharing the process workspace, top-p at
  p = 1.0 over a full 131k vocab, sampling that forces window widening,
  and torch zero-copy variants of the big paths.

## [0.3.1] - 2026-08-24

### Fixed
- README operator table synced with the shipped 0.3.0 (INT8 row was still
  marked "planned"; top-k speedup updated to the measured 1.6x; stale
  "v0.2 roadmap" pointer corrected) - bilingual

### Changed
- demo tours the INT8 utilities (roundtrip error bound, zero-copy scale
  parity, fused qadd vs an explicit float reference)
- dequantize_int8 dispatches through the shared device-path helper
  (consistency only; behavior unchanged)

## [0.3.0] - 2026-08-24

### Added
- INT8 symmetric per-tensor quantization utilities: `quantize_int8` /
  `dequantize_int8` (scale = absmax/127, clamp to [-127, 127]) and the
  fused `qadd_int8` (dequant -> add -> requant in one device pass, the
  output using its own absmax scale). Storage/dtype path; INT8 GEMM stays
  on the v0.4+ roadmap.
- bf16x8 (uint4, 16-byte) vectorization tier for elementwise kernels:
  8-byte accesses saturate Ampere GDDR6 but leave ~2.8x on the table on
  GDDR7 parts - measured bf16 silu on RTX 5060 Ti: 28.3 -> 10.2 us
  (1.05x vs torch). Dispatch tiers x8 / x4 / scalar by alignment.

### Performance interpretation (dual-GPU, honest)
- Ampere (RTX 3060, 28 SM): topk(131k, k=50) 1.62x vs torch; topp
  nucleus count now parallel (2178 -> 1146 us at 131k); bf16 elementwise
  at DRAM parity (325 GB/s effective).
- Blackwell (RTX 5060 Ti, 36 SM): bare topk remains 0.31x vs torch's CUB
  radix-select (40 us) - the chunk-merge sort removed the barrier
  pathology but CUB's decoupled-lookback passes stay ahead at high SM
  counts; sample_topp 903 -> 672 us. Documented as the v0.4 selection
  work item; argmax 1.07x, bf16 silu 1.05x.

### Changed
- selection sort phase rewritten: per-block chunk sorts + merge-path levels
  (one co-rank search per 256-output tile) replace the global bitonic;
  barriers drop from O(log^2 m) to O(log nb). topk(131k) 1.62x vs torch on
  Ampere; the topp nucleus count is now a grid-wide parallel scan (was a
  ~0.9 ms single-thread loop at 131k vocab)
- bf16 elementwise kernels vectorize 4 elements per thread (ushort4,
  8B accesses) with scalar tail and misalignment fallback; bf16 element
  throughput doubles vs the v0.2 scalar kernels at the same DRAM bandwidth

## [0.2.1] - 2026-08-23

### Fixed
- `sample_topp` was missing from `__all__` (shadowed by star-imports)
- demo.py: stale import fallback only probed `build/`; now probes both
  common dev build dirs, declares `ALL_OK` global correctly, and tours the
  v0.2 features (fused sampling determinism + zero-copy parity, bf16
  rmsnorm/softmax)

### Changed
- removed a dead functor in activations.cu that raised nvcc warning
  #177-D on every Linux build
- benchmark chart filenames derive from the queried device; docs/
  benchmark JSONs renamed per device (rt3060 / rt5060ti)

### Added
- READMEs: minimal per-token sampling loop example (English + Chinese)

## [0.2.0] - 2026-08-23

### Added
- `sample_topp(logits, p, *, temperature=1.0, seed=0)`: fused nucleus
  sampling - softmax over raw logits with temperature, top-p truncation and
  an inverse-CDF draw driven by a splitmix-hash uniform of `seed`, all in a
  single cooperative kernel (widening-window candidate sort; ~0.9 ms per
  call at 131k vocab on RTX 3060 incl. the token readback). Deterministic
  per seed; the RNG is reproducible but NOT cryptographically secure.
  CPU reference implements the identical algorithm; boundary draws may pick
  a neighbor token due to exact-exp vs fast-exp rounding (both valid).
  Same seed gives the same token across the CPU / staged / zero-copy paths.
- bfloat16 support on the zero-copy torch path for the inference core:
  SiLU / GeLU (both forms) / ReLU / Tanh / Sigmoid, add / mul / SwiGLU,
  RMSNorm / LayerNorm (norm weights upcast to float32 automatically),
  row-wise softmax, and both RoPE layouts. Kernels are templated on the
  storage dtype and compute in float32, converting at the load/store
  boundary (round-to-nearest-even). Sampling/selection ops remain float32
  (logits are f32). numpy paths are unaffected (numpy has no native bf16).
### Changed
- softmax rewritten for the LLM-relevant row widths (cols <= 8192):
  register-resident single-read kernel - each thread keeps its slice of the
  row in registers, so x is read once, exp is computed once, and y is
  written from registers; __expf (2-ulp fast approximation) replaces expf.
  Wide rows fall back to an online (max, sum) streaming kernel with the
  same tolerance. Benchmarks: 0.70-0.86x -> 0.98-1.13x vs PyTorch.
- top-k / top-p GPU path rewritten: single cooperative kernel doing 8-round
  256-bin radix refinement over order-preserving packed keys, one emit scan,
  and a bitonic sort (shared-memory fast path for k <= 2048, global-memory
  grid-participating path above). Deterministic ties (earliest index) are
  preserved exactly; process-cached workspace avoids per-call device syncs
  and keeps the hot path CUDA-graph-capturable. top-k @131072: 0.77x ->
  1.42x vs PyTorch; top-p: 26x faster than v0.1 (still behind torch.sort at
  small vocab sizes - honest numbers, merge-sort upgrade planned).
  Host-driven per-round fallback retained for non-cooperative devices.


### Verified
- Linux/Blackwell validation: full suite (134 tests) green on RTX 5060 Ti
  (sm_120, CUDA 13.2, torch 2.11/cu128), including torch zero-copy, bf16
  and CUDA-graph cases.
- Benchmarks on Blackwell: RMSNorm+res 3.3x, RoPE 8.3x, softmax 2.6x,
  argmax 1.9x vs PyTorch eager (chart in docs/).
- The released PyPI build (sm_86 cubins + compute_86 PTX) JIT-runs
  correctly on sm_120 drivers.

## [0.1.2] - 2026-08-23

### Fixed
- Chinese text and typographic dashes/micro signs in both READMEs were
  mojibake on the PyPI 0.1.1 page (encoding mishap while editing). Both
  READMEs are restored and verified UTF-8 clean; the PyPI description now
  renders correctly.
- Cross-links between the English and Chinese READMEs pointed at the wrong
  file; both now reference each other correctly.

## [0.1.1] - 2026-08-23

### Fixed
- README links on PyPI: relative links (README_zh.md, docs image, community
  files) resolved against pypi.org and 404'd. All links are now absolute
  GitHub URLs, so the Chinese README, the benchmark chart, and the
  contributing/security/coc/changelog links work from the PyPI page.

## [0.1.0] - 2026-08-23

First public release.

### Added
- Operators: RMSNorm (+fused residual), LayerNorm (affine), RoPE
  (interleaved and NeoX layouts, kv-cache `pos_offset`), SwiGLU, row-wise
  softmax, SiLU / GeLU (erf + tanh) / ReLU / Tanh / Sigmoid, elementwise
  add / mul, temperature scaling, repetition penalty, top-k, top-p
  (nucleus), argmax.
- Three execution paths per op: CPU reference (ground truth, no GPU
  needed), staged CUDA (numpy in / numpy out), and **zero-copy CUDA**
  (kernels run directly in torch device buffers via `data_ptr()`).
- `pip install .` via scikit-build-core; sm_80/sm_86 cubins plus
  compute_86 PTX for JIT on newer architectures.
- Optimized kernels: block-reduced norms/softmax, float4-vectorized
  elementwise, exp2f/sincosf RoPE frequencies (6.2x vs eager), parallel
  packed-key selection for top-k/top-p/argmax with deterministic ties.
- Benchmark suite (`benchmarks/bench.py`) with CUDA-event timing and
  chart; bilingual README; CI on GitHub Actions (build + CPU tests).
