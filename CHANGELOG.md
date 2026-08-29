# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- attention_decode(q, k_cache, v_cache, lens=None): single-token
  (decode step) causal attention with GQA over a contiguous
  [B, Hkv, T, D] kv-cache - `out = softmax(q . K^T / sqrt(D)) . V` with
  q head h attending via kv head h // (Hq/Hkv) (contiguous groups; Hq ==
  Hkv is plain MHA). One block per (batch, q head); the block's 8 warps
  stride the key rows, each keeping a running ONLINE softmax (max,
  denominator, [D] accumulator) in registers, and a shared-memory merge
  folds the warp partials - scores are never materialized, q/K/V are
  read exactly once, and the whole step is one stream-ordered,
  CUDA-graph-capturable kernel launch. Optional per-sequence `lens`
  mark valid cache rows so variable-length batches share one cache
  tensor (padding rows are ignored; a zero-length sequence produces a
  zero output row). float32; dim multiple of 4, at most 512.
- tests/test_attention.py: GQA mapping (constant-V probe), float64
  reference parity across shapes (single key, odd T, empty cache, long
  4096-row cache, D=4..256), padding-poisoning, error contract, staged
  and zero-copy paths, an independent torch repeat_interleave
  crosscheck, and graph capture-replay with mutation.

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
