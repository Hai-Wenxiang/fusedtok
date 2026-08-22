// Row-wise softmax, max-subtracted for numerical stability.
//
// Single-kernel block-per-row implementation: three strided passes over the
// row (block-reduced max, block-reduced exp-sum, write). exp is recomputed
// in the write pass - the row data lives in L2 between passes, and softmax
// is bandwidth-bound anyway.

#include "fusedtok/softmax.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

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

constexpr int kSmBlock = 256;

__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int cols) {
    __shared__ float shared[kSmBlock / 32];
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* yr = y + (size_t)blockIdx.x * cols;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += kSmBlock)
        m = fmaxf(m, xr[i]);
    m = block_reduce_max<kSmBlock>(m, shared);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += kSmBlock)
        sum += expf(xr[i] - m);
    const float inv = 1.0f / block_reduce_sum<kSmBlock>(sum, shared);

    for (int i = threadIdx.x; i < cols; i += kSmBlock)
        yr[i] = expf(xr[i] - m) * inv;
}

} // namespace

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

void softmax_launch(const float* x, float* y, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    softmax_kernel<<<rows, kSmBlock>>>(x, y, cols);
    check_launch("softmax kernel launch");
}

} // namespace fusedtok
