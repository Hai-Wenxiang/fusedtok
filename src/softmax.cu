// Row-wise softmax, max-subtracted for numerical stability.
//
// Two kernel variants behind one launcher:
//
// 1. shared-memoization path (cols <= kSmSharedMax, the LLM-relevant
//    regime): pass 1 reduces the row max (no exp), pass 2 computes
//    __expf(x - m) ONCE per element, stores it in dynamic shared memory,
//    and reduces the sum, pass 3 scales from shared. Only one exp per
//    element total - the exp/SFU unit, not bandwidth, is this kernel's
//    bottleneck on consumer GPUs.
//
// 2. online (flash-style) path for very wide rows: streaming (max, sum)
//    reduction with rescaling, then a plain write pass. Row data is read
//    once for the reduction.
//
// __expf (fast SFU approximation, ~2 ulp) is used throughout: softmax
// exactness is not contractual; parity tests use a matching tolerance.

#include "fusedtok/softmax.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
constexpr int kSmWarps = kSmBlock / 32;
// --- variant 1: register-resident path (cols <= kSmRegMax * kSmBlock) ------
// Each thread's slice of the row lives in registers, so x is read ONCE,
// exp is computed ONCE, and y is written from the register copy - minimal
// DRAM traffic, which is the true bottleneck once exp is single-pass.

constexpr int kSmPerThread = 32;                   // register budget per thread

template <typename T>
__global__ void softmax_reg_kernel(const T* __restrict__ x,
                                   T* __restrict__ y,
                                   int cols) {
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;
    __shared__ float sh_m[kSmWarps], sh_s[kSmWarps];

    // load this thread's slice (strided; coalesced across the warp)
    float v[kSmPerThread];
    int cnt = 0;
    for (int i = threadIdx.x; i < cols && cnt < kSmPerThread; i += kSmBlock)
        v[cnt++] = ld_f(xr, i);

    // row max over the register slice + block reduce
    float m = -INFINITY;
    for (int j = 0; j < cnt; ++j) m = fmaxf(m, v[j]);
    {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) sh_m[warp] = m;
        __syncthreads();
        if (warp == 0) {
            m = (threadIdx.x < kSmWarps) ? sh_m[lane] : -INFINITY;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
            if (lane == 0) sh_m[0] = m;
        }
        __syncthreads();
        m = sh_m[0];
    }

    // single exp per element; keep the exp values for the write pass
    float e[kSmPerThread];
    float s = 0.0f;
    for (int j = 0; j < cnt; ++j) {
        e[j] = __expf(v[j] - m);
        s += e[j];
    }
    {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, off);
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) sh_s[warp] = s;
        __syncthreads();
        if (warp == 0) {
            s = (threadIdx.x < kSmWarps) ? sh_s[lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_down_sync(0xffffffffu, s, off);
            if (lane == 0) sh_s[0] = s;
        }
        __syncthreads();
    }
    const float inv = 1.0f / sh_s[0];

    // write straight from registers - x is never re-read
    int j = 0;
    for (int i = threadIdx.x; i < cols && j < cnt; i += kSmBlock, ++j)
        st_f(yr, i, e[j] * inv);
}

// --- variant 2: online streaming reduction (wide rows) -----------------------

// Combine two (max, sum) partial states; math keeps s consistent with m.
__device__ __forceinline__ void merge_state(float& m, float& s,
                                            float m2, float s2) {
    const float m_new = fmaxf(m, m2);
    const float r1 = (m == -INFINITY && m_new == -INFINITY) ? 1.0f : __expf(m - m_new);
    const float r2 = (m2 == -INFINITY && m_new == -INFINITY) ? 1.0f : __expf(m2 - m_new);
    s = s * r1 + s2 * r2;
    m = m_new;
}

template <typename T>
__global__ void softmax_online_kernel(const T* __restrict__ x,
                                      T* __restrict__ y,
                                      int cols) {
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;

    float m = -INFINITY, s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += kSmBlock) {
        const float v = ld_f(xr, i);
        if (v > m) {
            s = (m == -INFINITY) ? 1.0f : s * __expf(m - v) + 1.0f;
            m = v;
        } else {
            s += __expf(v - m);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float mo = __shfl_down_sync(0xffffffffu, m, off);
        const float so = __shfl_down_sync(0xffffffffu, s, off);
        merge_state(m, s, mo, so);
    }
    __shared__ float sh_m[kSmWarps], sh_s[kSmWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { sh_m[warp] = m; sh_s[warp] = s; }
    __syncthreads();
    if (warp == 0) {
        m = (threadIdx.x < kSmWarps) ? sh_m[lane] : -INFINITY;
        s = (threadIdx.x < kSmWarps) ? sh_s[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float mo = __shfl_down_sync(0xffffffffu, m, off);
            const float so = __shfl_down_sync(0xffffffffu, s, off);
            merge_state(m, s, mo, so);
        }
        if (lane == 0) { sh_m[0] = m; sh_s[0] = s; }
    }
    __syncthreads();
    const float inv = 1.0f / sh_s[0];

    for (int i = threadIdx.x; i < cols; i += kSmBlock)
        st_f(yr, i, __expf(ld_f(xr, i) - sh_m[0]) * inv);
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

void softmax_launch(const float* x, float* y, int rows, int cols, std::uintptr_t stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols <= kSmPerThread * kSmBlock) {
        softmax_reg_kernel<float><<<rows, kSmBlock, 0, (cudaStream_t)stream>>>(x, y, cols);
        check_launch("softmax kernel launch");
    } else {
        softmax_online_kernel<float><<<rows, kSmBlock, 0, (cudaStream_t)stream>>>(x, y, cols);
        check_launch("softmax kernel launch");
    }
}

void softmax_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y,
                         int rows, int cols, std::uintptr_t stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols <= kSmPerThread * kSmBlock) {
        softmax_reg_kernel<__nv_bfloat16><<<rows, kSmBlock, 0, (cudaStream_t)stream>>>(x, y, cols);
        check_launch("softmax bf16 kernel launch");
    } else {
        softmax_online_kernel<__nv_bfloat16><<<rows, kSmBlock, 0, (cudaStream_t)stream>>>(x, y, cols);
        check_launch("softmax bf16 kernel launch");
    }
}

} // namespace fusedtok
