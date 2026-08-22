// LayerNorm with affine weight/bias.
//
// Naive single kernel: one thread per row, serial passes for mean, biased
// variance, and the normalized write. Mirrors the CPU reference 1:1.

#include "fusedtok/layernorm.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

void layernorm_check(const std::vector<float>& x, const std::vector<float>& w,
                     const std::vector<float>& b, int rows, int cols) {
    if (rows < 0 || cols <= 0)
        throw std::invalid_argument("rows must be >= 0 and cols must be > 0");
    if (static_cast<long long>(rows) * cols != static_cast<long long>(x.size()))
        throw std::invalid_argument("x.size() must equal rows * cols");
    if (w.size() != static_cast<size_t>(cols) || b.size() != static_cast<size_t>(cols))
        throw std::invalid_argument("weight and bias must have length cols");
}

__global__ void layernorm_kernel(const float* x, const float* w, const float* b,
                                 float* y, int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    float mean = 0.0f;
    for (int i = 0; i < cols; ++i) mean += xr[i];
    mean /= cols;

    float var = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float d = xr[i] - mean;
        var += d * d;
    }
    var /= cols;

    float inv_std = rsqrtf(var + eps);
    for (int i = 0; i < cols; ++i)
        yr[i] = (xr[i] - mean) * inv_std * w[i] + b[i];
}

} // namespace

std::vector<float> layernorm_cpu(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 const std::vector<float>& b,
                                 int rows, int cols, float eps) {
    layernorm_check(x, w, b, rows, cols);
    std::vector<float> y(x.size());
    for (int row = 0; row < rows; ++row) {
        const size_t base = (size_t)row * cols;
        float mean = 0.0f;
        for (int i = 0; i < cols; ++i) mean += x[base + i];
        mean /= cols;

        float var = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float d = x[base + i] - mean;
            var += d * d;
        }
        var /= cols;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < cols; ++i)
            y[base + i] = (x[base + i] - mean) * inv_std * w[i] + b[i];
    }
    return y;
}

void layernorm_launch(const float* x, const float* w, const float* b,
                      float* y, int rows, int cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    layernorm_kernel<<<(rows + kBlock - 1) / kBlock, kBlock>>>(
        x, w, b, y, rows, cols, eps);
    check_launch("layernorm kernel launch");
}

} // namespace fusedtok
