#include "fusedtok/fusedtok.hpp"

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

} // namespace

// Kernel 1: one thread per row. Serially accumulates sum of squares over the
// row (O(cols) per thread) and stores the inverse RMS scalar.
// Deliberately naive - no warp reductions, no vectorized loads.
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

std::vector<float> rmsnorm_cuda(const std::vector<float>& x,
                                const std::vector<float>& w,
                                int rows, int cols, float eps,
                                const std::vector<float>* residual) {
    rmsnorm_check(x, w, rows, cols, residual);
    const size_t n = x.size();
    if (n == 0) return {};
    std::vector<float> y(n);

    const size_t bytes_x = n * sizeof(float);
    const size_t bytes_w = w.size() * sizeof(float);

    float *dx = nullptr, *dy = nullptr, *dw = nullptr, *dr = nullptr, *dinv = nullptr;
    if (cudaMalloc(&dx, bytes_x) != cudaSuccess) throw std::runtime_error("cudaMalloc x failed");
    if (cudaMalloc(&dy, bytes_x) != cudaSuccess) throw std::runtime_error("cudaMalloc y failed");
    if (cudaMalloc(&dw, bytes_w) != cudaSuccess) throw std::runtime_error("cudaMalloc w failed");
    if (cudaMalloc(&dinv, rows * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc inv_rms failed");
    bool r_on_device = residual != nullptr;
    if (r_on_device && cudaMalloc(&dr, bytes_x) != cudaSuccess) throw std::runtime_error("cudaMalloc r failed");

    // Simplified cleanup: rely on CUDA context teardown on throw for brevity
    // in the naive version; success path frees explicitly below.
    if (cudaMemcpy(dx, x.data(), bytes_x, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D x failed");
    if (cudaMemcpy(dw, w.data(), bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D w failed");
    if (r_on_device && cudaMemcpy(dr, residual->data(), bytes_x, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D r failed");

    rmsnorm_rms_kernel<<<(rows + 255) / 256, 256>>>(dx, r_on_device ? dr : nullptr, dinv, rows, cols, eps);
    rmsnorm_apply_kernel<<<((int)n + 255) / 256, 256>>>(dx, r_on_device ? dr : nullptr, dw, dinv, dy, rows, cols);

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("rmsnorm kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));

    if (cudaMemcpy(y.data(), dy, bytes_x, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H y failed");

    cudaFree(dx); cudaFree(dy); cudaFree(dw); cudaFree(dinv);
    if (r_on_device) cudaFree(dr);
    return y;
}

}
