# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

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
### Added
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
### Changed
- top-k / top-p GPU path rewritten: single cooperative kernel doing 8-round
  256-bin radix refinement over order-preserving packed keys, one emit scan,
  and a bitonic sort (shared-memory fast path for k <= 2048, global-memory
  grid-participating path above). Deterministic ties (earliest index) are
  preserved exactly; process-cached workspace avoids per-call device syncs
  and keeps the hot path CUDA-graph-capturable. top-k @131072: 0.77x ->
  1.42x vs PyTorch; top-p: 26x faster than v0.1 (still behind torch.sort at
  small vocab sizes - honest numbers, merge-sort upgrade planned).
  Host-driven per-round fallback retained for non-cooperative devices.

## [0.2.0] - Unreleased

### Changed
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
- CUDA graph capture+replay verified for elementwise, norms, softmax,
  top-k and RoPE launchers (sample_topp documented as not capturable).
- Windows wheel on CI: skipped - windows runners lack a CUDA toolkit
  (3GB/30min install per run); local wheel builds are proven, GitHub
  Releases ship a cp312 Windows wheel, and pip source-build fallback is
  verified on Windows.

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
