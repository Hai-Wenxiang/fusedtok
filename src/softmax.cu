#include "fusedtok/softmax.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fusedtok {

namespace {

void softmax_check(const std::vector<float>& x, int rows, int cols) {
    if (rows < 0 || cols <= 0)
        throw std::invalid_argument("rows must be >= 0 and cols must be > 0");
    if (static_cast<long long>(rows) * cols != static_cast<long long>(x.size()))
        throw std::invalid_argument("x.size() must equal rows * cols");
}

} // namespace

// One thread per row: three serial passes over the row (max, exp-sum, write).
// Every thread recomputes expf per element twice - deliberately wasteful,
// kept for clarity and 1:1 correspondence with the CPU reference.
__global__ void softmax_kernel(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    float m = -INFINITY;
    for (int i = 0; i < cols; ++i)
        if (xr[i] > m) m = xr[i];

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i)
        sum += expf(xr[i] - m);

    float inv = 1.0f / sum;
    for (int i = 0; i < cols; ++i)
        yr[i] = expf(xr[i] - m) * inv;
}

std::vector<float> softmax_cpu(const std::vector<float>& x, int rows, int cols) {
    softmax_check(x, rows, cols);
    std::vector<float> y(x.size());
    for (int row = 0; row < rows; ++row) {
        const size_t base = (size_t)row * cols;
        float m = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < cols; ++i)
            if (x[base + i] > m) m = x[base + i];

        float sum = 0.0f;
        for (int i = 0; i < cols; ++i)
            sum += std::exp(x[base + i] - m);

        float inv = 1.0f / sum;
        for (int i = 0; i < cols; ++i)
            y[base + i] = std::exp(x[base + i] - m) * inv;
    }
    return y;
}

std::vector<float> softmax_cuda(const std::vector<float>& x, int rows, int cols) {
    softmax_check(x, rows, cols);
    if (x.empty()) return {};
    std::vector<float> y(x.size());

    float *dx = nullptr, *dy = nullptr;
    if (cudaMalloc(&dx, x.size() * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc x failed");
    if (cudaMalloc(&dy, x.size() * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc y failed");
    if (cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D x failed");

    softmax_kernel<<<(rows + 255) / 256, 256>>>(dx, dy, rows, cols);

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("softmax kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));
    if (cudaMemcpy(y.data(), dy, x.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H y failed");

    cudaFree(dx); cudaFree(dy);
    return y;
}

}
