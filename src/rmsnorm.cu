// RMSNorm (with optional fused residual add).
//
// Single-kernel block-per-row implementation:
//   - one thread block owns one row
//   - pass 1: strided accumulation of sum of squares, warp-shuffle +
//     shared-memory block reduction (deterministic, no scratch buffer)
//   - pass 2: strided write of the normalized, weighted output
// Works for any cols (large rows are re-read from cache on pass 2).

#include "fusedtok/fusedtok.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

// Check shapes shared by CPU and CUDA paths.
// Convention: shape problems throw std::invalid_argument (-> Python ValueError),
// CUDA problems throw std::runtime_error (-> Python RuntimeError).
void rmsnorm_check(const std::vector<float>& x, const std::vector<float>& w,
                   int rows, int cols, const std::vector<float>* residual) {
    if (rows < 0 || cols <= 0)
        throw std::invalid_argument("rows must be >= 0 and cols must be > 0");
    if (static_cast<long long>(rows) * cols != static_cast<long long>(x.size()))
        throw std::invalid_argument("x.size() must equal rows * cols");
    if (w.size() != static_cast<size_t>(cols))
        throw std::invalid_argument("weight.size() must equal cols");
    if (residual && residual->size() != x.size())
        throw std::invalid_argument("residual.size() must equal x.size()");
}

constexpr int kRmsBlock = 256;

__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ r,
                               const float* __restrict__ w,
                               float* __restrict__ y,
                               int cols, float eps) {
    __shared__ float shared[kRmsBlock / 32];
    const float* xr = x + (size_t)blockIdx.x * cols;
    const float* rr = r ? r + (size_t)blockIdx.x * cols : nullptr;
    float* yr = y + (size_t)blockIdx.x * cols;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += kRmsBlock) {
        float v = xr[i] + (rr ? rr[i] : 0.0f);
        acc += v * v;
    }
    const float total = block_reduce_sum<kRmsBlock>(acc, shared);
    const float inv = rsqrtf(total / cols + eps);

    for (int i = threadIdx.x; i < cols; i += kRmsBlock) {
        float v = xr[i] + (rr ? rr[i] : 0.0f);
        yr[i] = v * inv * w[i];
    }
}

} // namespace

std::vector<float> rmsnorm_cpu(const std::vector<float>& x,
                               const std::vector<float>& w,
                               int rows, int cols, float eps,
                               const std::vector<float>* residual) {
    rmsnorm_check(x, w, rows, cols, residual);
    std::vector<float> y(x.size());
    for (int row = 0; row < rows; ++row) {
        const size_t base = (size_t)row * cols;
        float acc = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float v = x[base + i] + (residual ? (*residual)[base + i] : 0.0f);
            acc += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(acc / cols + eps);
        for (int i = 0; i < cols; ++i) {
            float v = x[base + i] + (residual ? (*residual)[base + i] : 0.0f);
            y[base + i] = v * inv_rms * w[i];
        }
    }
    return y;
}

void rmsnorm_launch(const float* x, const float* w, const float* r,
                    float* y, int rows, int cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    rmsnorm_kernel<<<rows, kRmsBlock>>>(x, r, w, y, cols, eps);
    check_launch("rmsnorm kernel launch");
}

} // namespace fusedtok
