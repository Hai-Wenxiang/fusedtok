#include "fusedtok/fusedtok.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fusedtok {

namespace {

void rope_check(const std::vector<float>& q, const std::vector<float>* k,
                int seq, int dim) {
    if (seq < 0 || dim <= 0)
        throw std::invalid_argument("seq must be >= 0 and dim must be > 0");
    if (dim % 2 != 0)
        throw std::invalid_argument("dim must be even (RoPE pairs elements)");
    if (static_cast<long long>(seq) * dim != static_cast<long long>(q.size()))
        throw std::invalid_argument("q.size() must equal seq * dim");
    if (k && k->size() != q.size())
        throw std::invalid_argument("k.size() must equal q.size()");
}

} // namespace

// One thread per (position, pair): computes the rotation angle from scratch
// with powf (naive - a real implementation precomputes frequencies once).
__global__ void rope_kernel(const float* x, float* y, int seq, int dim,
                            float theta) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;   // global pair index
    int pairs_per_row = dim / 2;
    if (p >= seq * pairs_per_row) return;

    int row = p / pairs_per_row;                     // position m
    int j = p - row * pairs_per_row;                 // pair index within row

    float angle = row * powf(theta, -2.0f * j / dim);
    float c = cosf(angle);
    float s = sinf(angle);

    int even = row * dim + 2 * j;
    int odd = even + 1;
    float xe = x[even];
    float xo = x[odd];
    y[even] = xe * c - xo * s;
    y[odd] = xe * s + xo * c;
}

std::pair<std::vector<float>, std::vector<float>>
rope_cpu(const std::vector<float>& q, const std::vector<float>* k,
         int seq, int dim, float theta) {
    rope_check(q, k, seq, dim);
    auto rotate = [&](const std::vector<float>& x) {
        std::vector<float> y(x.size());
        for (int m = 0; m < seq; ++m) {
            for (int j = 0; j < dim / 2; ++j) {
                double angle = m * std::pow((double)theta, -2.0 * j / dim);
                double c = std::cos(angle), s = std::sin(angle);
                size_t even = (size_t)m * dim + 2 * j, odd = even + 1;
                float xe = x[even], xo = x[odd];
                y[even] = (float)(xe * c - xo * s);
                y[odd] = (float)(xe * s + xo * c);
            }
        }
        return y;
    };

    std::vector<float> k_out;
    if (k) k_out = rotate(*k);
    return {rotate(q), std::move(k_out)};
}

std::pair<std::vector<float>, std::vector<float>>
rope_cuda(const std::vector<float>& q, const std::vector<float>* k,
          int seq, int dim, float theta) {
    rope_check(q, k, seq, dim);
    const size_t n = q.size();
    if (n == 0) return {{}, {}};

    auto rotate_cuda = [&](const std::vector<float>& x) {
        std::vector<float> y(n);
        float *dx = nullptr, *dy = nullptr;
        if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc x failed");
        if (cudaMalloc(&dy, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc y failed");
        if (cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D x failed");

        int pairs = seq * (dim / 2);
        rope_kernel<<<(pairs + 255) / 256, 256>>>(dx, dy, seq, dim, theta);

        if (cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("rope kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));
        if (cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H y failed");

        cudaFree(dx); cudaFree(dy);
        return y;
    };

    std::vector<float> k_out;
    if (k) k_out = rotate_cuda(*k);
    return {rotate_cuda(q), std::move(k_out)};
}

}
