# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

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
