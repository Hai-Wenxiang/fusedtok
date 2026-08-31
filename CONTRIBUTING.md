# Contributing to fusedtok

Thanks for your interest in improving fusedtok!

## Ways to help

- **Bug reports**: open an issue with the op name, shapes, dtypes, GPU model,
  driver/CUDA version, and a minimal reproducer. Include the CPU vs GPU
  mismatch if it is a numerical issue.
- **Performance ideas**: profiles, Nsight Compute reports, or concrete kernel
  suggestions are extremely welcome. The bar for merging a faster kernel is
  that the full test suite stays green.
- **New operators**: check the roadmap in the README first; avoid operators
  that duplicate PyTorch one-to-one without a fusion or sampling benefit.

## Development setup

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install -e .            # or: cmake -S . -B build -G Ninja && cmake --build build
pip install pytest torch    # torch optional but recommended for the CUDA tests
pytest tests -q
```

Requirements: CUDA Toolkit >= 12.0, a CUDA GPU of compute capability 8.0+
(Ampere or newer). CPU-only machines can still run the CPU reference tests —
CUDA cases skip automatically.

## Rules of the road

1. **Every kernel ships with a CPU reference implementation** and parity
   tests (multiple shapes, edge cases, error paths, GPU-vs-CPU comparison).
2. **Determinism where promised**: selection ops (top-k / top-p / argmax)
   resolve ties toward the earliest index; keep that invariant.
3. **Error contract**: shape and value problems (bad k / p / temperature,
   mismatched shapes, out-of-range ids) raise `ValueError`; dtype,
   device-family and mixed-input problems raise `TypeError`; CUDA
   execution failures raise `RuntimeError` (mapped from
   `std::runtime_error` in C++). The full surface is pinned by
   `tests/test_api.py`.
4. **API freeze (1.0)**: the names in `fusedtok.__all__` are frozen.
   Additions may land in minor releases; changing a signature, renaming
   or removing a public name requires a major release and a deprecation
   window of at least one minor release (DeprecationWarning) before it.
   `python/fusedtok/__init__.pyi` + `py.typed` ship the typed surface -
   keep them in sync with `__all__` in the same change (the API test
   fails otherwise).
5. **Comments in English**, explaining *why* (design constraints, GPU
   micro-arch reasons), not *what*.
6. Benchmarks use CUDA events, never wall clock (WDDM makes host timing
   on Windows meaningless).
7. Keep the CI green: `ubuntu-latest` + CUDA container build and CPU tests
   run on every push.

## New-kernel checklist

Lessons baked into the v0.3/v0.4/v0.5/v1.0/v1.1 sprints - run through this
before pushing any new GPU kernel:

- [ ] **Sanitizer trio before review**: `compute-sanitizer --tool memcheck`
      and `--tool racecheck` on a smoke that exercises every code path
      (both kernels, fallbacks, boundary shapes). Cross-block
      synchronization bugs only show up here.
- [ ] **Dual-GPU evidence**: numbers (or at least test runs) from one
      Ampere and one newer part (Blackwell exposes different occupancy /
      atomic throughput; the v0.3 selection regression was invisible on
      the dev GPU).
- [ ] **Float-order tolerance**: if accumulation order can differ between
      CPU and GPU paths, state the tolerance in the test and the docstring
      (see the top-p count drift notes).
- [ ] **Stream discipline**: launch on the caller's `stream` argument
      (never the legacy default stream) and keep the launcher free of
      allocations, syncs, and event queries so it stays CUDA-graph
      capturable. If the kernel sequence is cached in a graph, remember
      kernel *parameters* get baked - per-call values must travel through
      device memory.
- [ ] **Compile-time loop bounds for register arrays**: a per-thread
      array indexed by a loop whose trip count is a RUNTIME value is
      demoted to local memory (spill traffic every iteration - the
      v0.5 prefill kernel lost 7x to this before the bound became a
      `#pragma unroll`-able constant with an early `break`).
- [ ] **Masked shuffles under warp divergence**: `__shfl_*_sync` with a
      full-warp mask deadlocks when only part of the warp arrives (the
      v0.5 prefill lanes split into per-row groups with different
      `live` flags). Compute the mask for the lane group that actually
      executes the shuffle.
- [ ] **Platform traps**: `1ULL << 64` is UB in device code; `char4`
      fields are plain `char` (unsigned on MSVC - cast through
      `(signed char)`); single-thread serial volatile loads are latency
      poison (stage cooperatively through shared memory).
- [ ] **cp.async beats register staging for GEMM-style slabs** (v1.0
      qgemm): staging global tiles in per-thread register arrays keeps
      them live across the compute phase - the first pipelined qgemm
      measured 156 registers (1 block/SM, slab128: 255 + spills) and ran
      SLOWER than the v0.4 single-buffered kernel. `__pipeline_memcpy_async`
      moves the same bytes global->shared with no register detour
      (111-128 registers, 2 blocks/SM). Check `nvcc --ptxas-options=-v`
      before believing a pipeline.
- [ ] **Tuned kernel configs need a capture-safe default** (v1.0 qgemm):
      if first-call micro-benchmarks pick a tile/slab config, captures
      must skip the tuning (events + syncs are illegal mid-capture) and
      launch a default config instead - and a config needing
      `cudaFuncSetAttribute` opt-in (> 48 KB dynamic smem) must raise the
      attribute inside the tuner, never during a capture. Pin both with a
      capture-after-tuning test.
- [ ] **One code path, even under dtype templates** (v1.1 attention):
      templating a kernel on the storage dtype recompiles EVERYTHING -
      including the float32 instantiation you did not touch - and nvcc
      does not guarantee textually-equivalent specializations produce
      equivalent code (the dim<32 prefill fallback miscompiled for f32
      after templating; deleting it and routing small heads through the
      tiled kernel's zero-padded, bounds-guarded band was the fix). Prefer
      deleting a fallback over keeping a second path alive through a
      template rework, and re-run the FULL parity suite on every dtype
      after any template change.
- [ ] **Address math is codegen input** (v1.1.1 prefill): `ld(p + i * 4)`
      and `float4*[i]` compute the same address, but the sm_120 backend
      scheduled the element-scaled form 71% slower in the prefill staging
      loop (Ampere identical, register counts identical - only the
      instruction mix changed). Keep vector loads/stores in chunk-unit
      subscripts, and when a refactor is "textually equivalent", A/B the
      perf on every GPU generation you ship to before believing it.

## Pull requests

- One logical change per PR, with tests and README updates in the same PR.
- PRs are merged only with passing CI and maintainer review.
- By contributing you agree your contributions are licensed under the MIT
  license (see [LICENSE](LICENSE)).
