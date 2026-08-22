#pragma once

// Shared helpers for the kernel launcher implementations.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace fusedtok {

// Surface a kernel launch failure as std::runtime_error (Python RuntimeError).
// Called right after every <<<>>> launch; catches bad launch configs etc.
inline void check_launch(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Default threads-per-block for the simple elementwise kernels.
constexpr int kBlock = 256;

// Grid size covering n items with kBlock threads, rounded up.
inline long long grid_for(long long n) { return (n + kBlock - 1) / kBlock; }

} // namespace fusedtok
