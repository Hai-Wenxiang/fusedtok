#include "fusedtok/fusedtok.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace fusedtok {

// Naive kernel: one thread computes one output element.
// No memory coalescing tricks, no vectorized loads - correctness first.
__global__ void axpy_kernel(const float* x, float* y, float a, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}

std::vector<float> axpy_cpu(const std::vector<float>& x, float a, float b) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = a * x[i] + b;
    return y;
}

std::vector<float> axpy_cuda(const std::vector<float>& x, float a, float b) {
    const int n = static_cast<int>(x.size());
    if (n == 0) return {};
    std::vector<float> y(n);

    // Allocate device buffers; release them on every early-exit path
    float *dx = nullptr, *dy = nullptr;
    if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc dx failed");
    if (cudaMalloc(&dy, n * sizeof(float)) != cudaSuccess) {
        cudaFree(dx);
        throw std::runtime_error("cudaMalloc dy failed");
    }

    if (cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy);
        throw std::runtime_error("H2D copy failed");
    }

    // 256 threads per block is a safe default; grid covers n with one guard
    axpy_kernel<<<(n + 255) / 256, 256>>>(dx, dy, a, b, n);

    // Synchronize before reading back: also surfaces kernel launch errors
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(dx); cudaFree(dy);
        throw std::runtime_error("kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));
    }

    if (cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(dx); cudaFree(dy);
        throw std::runtime_error("D2H copy failed");
    }

    cudaFree(dx);
    cudaFree(dy);
    return y;
}

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}
