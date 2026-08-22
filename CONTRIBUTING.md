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
3. **Error contract**: shape problems raise `ValueError`, CUDA problems
   raise `RuntimeError` (mapped from `std::invalid_argument` /
   `std::runtime_error` in C++).
4. **Comments in English**, explaining *why* (design constraints, GPU
   micro-arch reasons), not *what*.
5. Benchmarks use CUDA events, never wall clock (WDDM makes host timing
   on Windows meaningless).
6. Keep the CI green: `ubuntu-latest` + CUDA container build and CPU tests
   run on every push.

## Pull requests

- One logical change per PR, with tests and README updates in the same PR.
- PRs are merged only with passing CI and maintainer review.
- By contributing you agree your contributions are licensed under the MIT
  license (see [LICENSE](LICENSE)).
