# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

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
