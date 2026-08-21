#include "fusedtok/layernorm.hpp"

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

} // namespace

// One thread per row: serial passes for mean, biased variance, then the
// affine write. Mirrors the CPU reference 1:1.
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

std::vector<float> layernorm_cuda(const std::vector<float>& x,
                                  const std::vector<float>& w,
                                  const std::vector<float>& b,
                                  int rows, int cols, float eps) {
    layernorm_check(x, w, b, rows, cols);
    if (x.empty()) return {};
    std::vector<float> y(x.size());

    float *dx = nullptr, *dw = nullptr, *db = nullptr, *dy = nullptr;
    if (cudaMalloc(&dx, x.size() * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc x failed");
    if (cudaMalloc(&dw, w.size() * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc w failed");
    if (cudaMalloc(&db, b.size() * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc b failed");
    if (cudaMalloc(&dy, x.size() * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc y failed");

    if (cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D x failed");
    if (cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D w failed");
    if (cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D b failed");

    layernorm_kernel<<<(rows + 255) / 256, 256>>>(dx, dw, db, dy, rows, cols, eps);

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("layernorm kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));
    if (cudaMemcpy(y.data(), dy, x.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H y failed");

    cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dy);
    return y;
}

}
