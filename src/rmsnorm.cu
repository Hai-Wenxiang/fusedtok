// RMSNorm (with optional fused residual add).
//
// Two-kernel naive split:
//   kernel 1: one thread per row, serial loop accumulates sum of squares
//   kernel 2: one thread per element, applies the row scale and weight
// The launcher allocates a small [rows] scratch for inverse RMS values.

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

// Kernel 1: one thread per row. Serially accumulates sum of squares over the
// row (O(cols) per thread) and stores the inverse RMS scalar.
__global__ void rmsnorm_rms_kernel(const float* x, const float* r,
                                   float* inv_rms, int rows, int cols,
                                   float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * cols;
    const float* rr = r ? r + (size_t)row * cols : nullptr;

    float acc = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float v = xr[i] + (rr ? rr[i] : 0.0f);
        acc += v * v;
    }
    float rms = sqrtf(acc / cols + eps);
    inv_rms[row] = 1.0f / rms;
}

// Kernel 2: one thread per element. Reads the precomputed inverse RMS of its
// row and applies normalization + learned weight.
__global__ void rmsnorm_apply_kernel(const float* x, const float* r,
                                      const float* w, const float* inv_rms,
                                      float* y, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int row = i / cols;
    int col = i - row * cols;
    y[i] = (x[i] + (r ? r[i] : 0.0f)) * inv_rms[row] * w[col];
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
    // Scratch for per-row inverse RMS; cudaMalloc/free here keeps the launch
    // self-contained. Note cudaFree performs an implicit device sync.
    float* dinv = nullptr;
    if (cudaMalloc(&dinv, rows * sizeof(float)) != cudaSuccess)
        throw std::runtime_error(std::string("rmsnorm scratch alloc failed: ") +
                                 cudaGetErrorString(cudaGetLastError()));

    rmsnorm_rms_kernel<<<(rows + kBlock - 1) / kBlock, kBlock>>>(
        x, r, dinv, rows, cols, eps);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        const long long n = (long long)rows * cols;
        rmsnorm_apply_kernel<<<(unsigned)grid_for(n), kBlock>>>(
            x, r, w, dinv, y, rows, cols);
        err = cudaGetLastError();
    }
    cudaFree(dinv);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("rmsnorm kernel launch: ") +
                                 cudaGetErrorString(err));
}

} // namespace fusedtok
