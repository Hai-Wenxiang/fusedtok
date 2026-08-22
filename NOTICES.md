# Third-party notices

fusedtok is MIT licensed (see LICENSE). This product builds on:

## pybind11

Licensed under the BSD-style license found in
https://github.com/pybind/pybind11/blob/master/LICENSE.
Used at build time to generate the Python bindings.

## NVIDIA CUDA Toolkit

The compiled extension links the CUDA Runtime (cudart) and is built with the
NVIDIA CUDA Compiler. Use of the CUDA Toolkit is governed by NVIDIA's
software license: https://developer.nvidia.com/cuda-toolkit-software-license-agreement

## Optional runtime dependencies

- NumPy (BSD-3-Clause) - array interface for the staged path.
- PyTorch (BSD-3-Clause) - optional; used for the zero-copy CUDA path and in
  the benchmark suite.
