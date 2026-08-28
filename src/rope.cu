// RoPE (Rotary Position Embedding), interleaved and NeoX (rotate_half)
// layouts, with an optional position offset for kv-cache decoding.
// Templated on storage dtype (float32 compute, bf16 converts at the
// load/store boundary).

#include "fusedtok/fusedtok.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

// One thread per (position, pair). The pair frequency theta^(-2j/dim) is
// computed as exp2f(-(2j/dim) * log2(theta)) - a fast hardware-friendly
// rewrite of powf with the same value semantics - and the rotation uses a
// single sincosf call for both trig components.
template <typename T>
__global__ void rope_kernel(const T* x, T* y, int seq, int dim,
                            float theta, int pos_offset) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;   // global pair index
    int pairs_per_row = dim / 2;
    if (p >= seq * pairs_per_row) return;

    int row = p / pairs_per_row;                     // row within this batch
    int j = p - row * pairs_per_row;                 // pair index within row
    int m = row + pos_offset;                        // absolute position

    const float inv_log2_theta = log2f(theta);
    float freq = exp2f(-(2.0f * j / dim) * inv_log2_theta);
    float angle = m * freq;
    float c, s;
    sincosf(angle, &s, &c);

    const long long even = (long long)row * dim + 2 * j;
    const long long odd = even + 1;
    float xe = ld_f(x, even);
    float xo = ld_f(x, odd);
    st_f(y, even, xe * c - xo * s);
    st_f(y, odd, xe * s + xo * c);
}

// One thread per (position, j): pairs row halves instead of adjacent
// elements. Same frequency computation as the interleaved variant.
template <typename T>
__global__ void rope_neox_kernel(const T* x, T* y, int seq, int dim,
                                 float theta, int pos_offset) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;   // global j index
    int half = dim / 2;
    if (p >= seq * half) return;

    int row = p / half;
    int j = p - row * half;
    int m = row + pos_offset;

    const float inv_log2_theta = log2f(theta);
    float freq = exp2f(-(2.0f * j / dim) * inv_log2_theta);
    float angle = m * freq;
    float c, s;
    sincosf(angle, &s, &c);

    const long long i1 = (long long)row * dim + j;          // first half
    const long long i2 = i1 + half;                          // second half
    float x1 = ld_f(x, i1);
    float x2 = ld_f(x, i2);
    st_f(y, i1, x1 * c - x2 * s);
    st_f(y, i2, x1 * s + x2 * c);
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
                 int pos_offset, std::uintptr_t stream) {
    if (seq <= 0 || dim <= 0) return;
    int pairs = seq * (dim / 2);
    rope_kernel<float><<<(pairs + kBlock - 1) / kBlock, kBlock, 0, (cudaStream_t)stream>>>(
        x, y, seq, dim, theta, pos_offset);
    check_launch("rope kernel launch");
}

void rope_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int seq,
                      int dim, float theta, int pos_offset, std::uintptr_t stream) {
    if (seq <= 0 || dim <= 0) return;
    int pairs = seq * (dim / 2);
    rope_kernel<__nv_bfloat16><<<(pairs + kBlock - 1) / kBlock, kBlock, 0, (cudaStream_t)stream>>>(
        x, y, seq, dim, theta, pos_offset);
    check_launch("rope bf16 kernel launch");
}

void rope_neox_launch(const float* x, float* y, int seq, int dim, float theta,
                      int pos_offset, std::uintptr_t stream) {
    if (seq <= 0 || dim <= 0) return;
    int threads_needed = seq * (dim / 2);
    rope_neox_kernel<float><<<(threads_needed + kBlock - 1) / kBlock, kBlock, 0, (cudaStream_t)stream>>>(
        x, y, seq, dim, theta, pos_offset);
    check_launch("rope_neox kernel launch");
}

void rope_neox_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int seq,
                           int dim, float theta, int pos_offset, std::uintptr_t stream) {
    if (seq <= 0 || dim <= 0) return;
    int threads_needed = seq * (dim / 2);
    rope_neox_kernel<__nv_bfloat16><<<(threads_needed + kBlock - 1) / kBlock, kBlock, 0, (cudaStream_t)stream>>>(
        x, y, seq, dim, theta, pos_offset);
    check_launch("rope_neox bf16 kernel launch");
}

} // namespace fusedtok
