#include "fusedtok/fusedtok.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace fusedtok {

namespace {

// Naive kernel: one thread computes one output element.
// No memory coalescing tricks, no vectorized loads - correctness first.
__global__ void axpy_kernel(const float* x, float* y, float a, float b,
                            long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}

} // namespace

std::vector<float> axpy_cpu(const std::vector<float>& x, float a, float b) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = a * x[i] + b;
    return y;
}

void axpy_launch(const float* x, float* y, long long n, float a, float b, std::uintptr_t stream) {
    if (n <= 0) return;
    axpy_kernel<<<(unsigned)grid_for(n), kBlock, 0, (cudaStream_t)stream>>>(x, y, a, b, n);
    check_launch("axpy kernel launch");
}

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace fusedtok
