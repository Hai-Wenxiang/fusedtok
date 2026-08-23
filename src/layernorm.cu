// LayerNorm with affine weight/bias.
//
// Single-kernel block-per-row implementation with two block reductions
// (mean, then biased variance computed against the mean for numerical
// stability), templated on storage dtype: float32 compute, bf16 converts
// at the load/store boundary. Weight/bias stay float32 in both cases.

#include "fusedtok/layernorm.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

constexpr int kLnBlock = 256;

template <typename T>
__global__ void layernorm_kernel(const T* __restrict__ x,
                                 const float* __restrict__ w,
                                 const float* __restrict__ b,
                                 T* __restrict__ y,
                                 int cols, float eps) {
    __shared__ float shared[kLnBlock / 32];
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += kLnBlock)
        sum += ld_f(xr, i);
    const float mean = block_reduce_sum<kLnBlock>(sum, shared) / cols;

    float var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += kLnBlock) {
        float d = ld_f(xr, i) - mean;
        var += d * d;
    }
    const float total_var = block_reduce_sum<kLnBlock>(var, shared) / cols;

    const float inv_std = rsqrtf(total_var + eps);
    for (int i = threadIdx.x; i < cols; i += kLnBlock)
        st_f(yr, i, (ld_f(xr, i) - mean) * inv_std * w[i] + b[i]);
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
    layernorm_kernel<float><<<rows, kLnBlock>>>(x, w, b, y, cols, eps);
    check_launch("layernorm kernel launch");
}

void layernorm_launch_bf16(const __nv_bfloat16* x, const float* w,
                           const float* b, __nv_bfloat16* y,
                           int rows, int cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    layernorm_kernel<__nv_bfloat16><<<rows, kLnBlock>>>(x, w, b, y, cols, eps);
    check_launch("layernorm bf16 kernel launch");
}

} // namespace fusedtok
