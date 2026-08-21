#include "fusedtok/activations.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// erff is a CUDA math-library intrinsic; accuracy matches CPU std::erf
// closely enough for the 1e-5 parity tolerance used in tests.
__global__ void silu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v * sigmoidf_(v);
    }
}

__global__ void gelu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + erff(v / 1.4142135623730951f));
    }
}

__global__ void relu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void tanh_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

// Shared host-side driver for the elementwise kernels above: allocates,
// copies in, launches, syncs, copies back, frees. The naive version repeats
// this boilerplate per operator on purpose - it is easy to read and diff.
template <typename Kernel>
std::vector<float> elementwise_cuda(const std::vector<float>& x, Kernel kernel) {
    const int n = static_cast<int>(x.size());
    if (n == 0) return {};
    std::vector<float> y(n);

    float *dx = nullptr, *dy = nullptr;
    if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc x failed");
    if (cudaMalloc(&dy, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc y failed");
    if (cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D x failed");

    kernel<<<(n + 255) / 256, 256>>>(dx, dy, n);

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("elementwise kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));
    if (cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H y failed");

    cudaFree(dx); cudaFree(dy);
    return y;
}

} // namespace

std::vector<float> silu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
    return y;
}

std::vector<float> silu_cuda(const std::vector<float>& x) {
    return elementwise_cuda(x, silu_kernel);
}

std::vector<float> gelu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::erf(v / 1.4142135623730951f));
    }
    return y;
}

std::vector<float> gelu_cuda(const std::vector<float>& x) {
    return elementwise_cuda(x, gelu_kernel);
}

std::vector<float> relu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return y;
}

std::vector<float> relu_cuda(const std::vector<float>& x) {
    return elementwise_cuda(x, relu_kernel);
}

std::vector<float> tanh_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = std::tanh(x[i]);
    return y;
}

std::vector<float> tanh_cuda(const std::vector<float>& x) {
    return elementwise_cuda(x, tanh_kernel);
}

}
