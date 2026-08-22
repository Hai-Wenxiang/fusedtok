// RoPE (Rotary Position Embedding), interleaved and NeoX (rotate_half)
// layouts, with an optional position offset for kv-cache decoding.

#include "fusedtok/fusedtok.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fusedtok {

namespace {

void rope_check(const std::vector<float>& q, const std::vector<float>* k,
                int seq, int dim, int pos_offset) {
    if (seq < 0 || dim <= 0)
        throw std::invalid_argument("seq must be >= 0 and dim must be > 0");
    if (dim % 2 != 0)
        throw std::invalid_argument("dim must be even (RoPE pairs elements)");
    if (pos_offset < 0)
        throw std::invalid_argument("pos_offset must be >= 0");
    if (static_cast<long long>(seq) * dim != static_cast<long long>(q.size()))
        throw std::invalid_argument("q.size() must equal seq * dim");
    if (k && k->size() != q.size())
        throw std::invalid_argument("k.size() must equal q.size()");
}

// One thread per (position, pair): computes the rotation angle from scratch
// with powf (naive - a real implementation precomputes frequencies once).
__global__ void rope_kernel(const float* x, float* y, int seq, int dim,
                            float theta, int pos_offset) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;   // global pair index
    int pairs_per_row = dim / 2;
    if (p >= seq * pairs_per_row) return;

    int row = p / pairs_per_row;                     // row within this batch
    int j = p - row * pairs_per_row;                 // pair index within row
    int m = row + pos_offset;                        // absolute position

    float angle = m * powf(theta, -2.0f * j / dim);
    float c = cosf(angle);
    float s = sinf(angle);

    int even = row * dim + 2 * j;
    int odd = even + 1;
    float xe = x[even];
    float xo = x[odd];
    y[even] = xe * c - xo * s;
    y[odd] = xe * s + xo * c;
}

// One thread per (position, j): pairs row halves instead of adjacent
// elements. Same naive per-thread powf as the interleaved variant.
__global__ void rope_neox_kernel(const float* x, float* y, int seq, int dim,
                                 float theta, int pos_offset) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;   // global j index
    int half = dim / 2;
    if (p >= seq * half) return;

    int row = p / half;
    int j = p - row * half;
    int m = row + pos_offset;

    float angle = m * powf(theta, -2.0f * j / dim);
    float c = cosf(angle);
    float s = sinf(angle);

    int i1 = row * dim + j;          // first half element
    int i2 = row * dim + half + j;   // matching second half element
    float x1 = x[i1];
    float x2 = x[i2];
    y[i1] = x1 * c - x2 * s;
    y[i2] = x1 * s + x2 * c;
}

} // namespace

std::pair<std::vector<float>, std::vector<float>>
rope_cpu(const std::vector<float>& q, const std::vector<float>* k,
         int seq, int dim, float theta, int pos_offset) {
    rope_check(q, k, seq, dim, pos_offset);
    auto rotate = [&](const std::vector<float>& x) {
        std::vector<float> y(x.size());
        for (int r = 0; r < seq; ++r) {
            int m = r + pos_offset;
            for (int j = 0; j < dim / 2; ++j) {
                double angle = m * std::pow((double)theta, -2.0 * j / dim);
                double c = std::cos(angle), s = std::sin(angle);
                size_t even = (size_t)r * dim + 2 * j, odd = even + 1;
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
rope_neox_cpu(const std::vector<float>& q, const std::vector<float>* k,
              int seq, int dim, float theta, int pos_offset) {
    rope_check(q, k, seq, dim, pos_offset);
    auto rotate = [&](const std::vector<float>& x) {
        std::vector<float> y(x.size());
        for (int r = 0; r < seq; ++r) {
            int m = r + pos_offset;
            for (int j = 0; j < dim / 2; ++j) {
                double angle = m * std::pow((double)theta, -2.0 * j / dim);
                double c = std::cos(angle), s = std::sin(angle);
                size_t i1 = (size_t)r * dim + j, i2 = i1 + dim / 2;
                float x1 = x[i1], x2 = x[i2];
                y[i1] = (float)(x1 * c - x2 * s);
                y[i2] = (float)(x1 * s + x2 * c);
            }
        }
        return y;
    };

    std::vector<float> k_out;
    if (k) k_out = rotate(*k);
    return {rotate(q), std::move(k_out)};
}

void rope_launch(const float* x, float* y, int seq, int dim, float theta,
                 int pos_offset) {
    if (seq <= 0 || dim <= 0) return;
    int pairs = seq * (dim / 2);
    rope_kernel<<<(pairs + kBlock - 1) / kBlock, kBlock>>>(
        x, y, seq, dim, theta, pos_offset);
    check_launch("rope kernel launch");
}

void rope_neox_launch(const float* x, float* y, int seq, int dim, float theta,
                      int pos_offset) {
    if (seq <= 0 || dim <= 0) return;
    int threads_needed = seq * (dim / 2);
    rope_neox_kernel<<<(threads_needed + kBlock - 1) / kBlock, kBlock>>>(
        x, y, seq, dim, theta, pos_offset);
    check_launch("rope_neox kernel launch");
}

} // namespace fusedtok
