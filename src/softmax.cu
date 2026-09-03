// Row-wise softmax, max-subtracted for numerical stability.
//
// Two kernel variants behind one launcher:
//
// 1. register-resident path (cols <= kSmPerThread * kSmBlock, the
//    LLM-relevant regime): each thread keeps its slice of the row in
//    registers, so x is read ONCE, __expf(x - m) is computed ONCE per
//    element, and y is written straight from the register copy - one
//    DRAM round trip and one exp per element, which leaves bandwidth
//    (not the SFU) as the bottleneck.
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

constexpr int kSmBlock = 256;   // default block + variant-boundary basis
// (the register path's coverage bound derives from the runtime block:
// WARPS = BLOCK/32 inside the templated kernels)
// --- variant 1: register-resident path (cols <= kSmPerThread * block) ---
// Each thread's slice of the row lives in registers, so x is read ONCE,
// exp is computed ONCE, and y is written from the register copy - minimal
// DRAM traffic, which is the true bottleneck once exp is single-pass.

constexpr int kSmPerThread = 32;                   // register budget per thread

template <typename T, int BLOCK>
__global__ void softmax_reg_kernel(const T* __restrict__ x,
                                   T* __restrict__ y,
                                   int cols) {
    constexpr int WARPS = BLOCK / 32;
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;
    __shared__ float sh_m[WARPS], sh_s[WARPS];

    // load this thread's slice (strided; coalesced across the warp)
    float v[kSmPerThread];
    int cnt = 0;
    for (int i = threadIdx.x; i < cols && cnt < kSmPerThread; i += BLOCK)
        v[cnt++] = ld_f(xr, i);

    // row max over the register slice + shared block reduce
    float m = -INFINITY;
    for (int j = 0; j < cnt; ++j) m = fmaxf(m, v[j]);
    m = block_reduce_max<BLOCK>(m, sh_m);

    // single exp per element; keep the exp values for the write pass
    float e[kSmPerThread];
    float s = 0.0f;
    for (int j = 0; j < cnt; ++j) {
        e[j] = __expf(v[j] - m);
        s += e[j];
    }
    s = block_reduce_sum<BLOCK>(s, sh_s);
    const float inv = 1.0f / s;

    // write straight from registers - x is never re-read
    int j = 0;
    for (int i = threadIdx.x; i < cols && j < cnt; i += BLOCK, ++j)
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

template <typename T, int BLOCK>
__global__ void softmax_online_kernel(const T* __restrict__ x,
                                      T* __restrict__ y,
                                      int cols) {
    constexpr int WARPS = BLOCK / 32;
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;

    float m = -INFINITY, s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK) {
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
    __shared__ float sh_m[WARPS], sh_s[WARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { sh_m[warp] = m; sh_s[warp] = s; }
    __syncthreads();
    if (warp == 0) {
        m = (threadIdx.x < WARPS) ? sh_m[lane] : -INFINITY;
        s = (threadIdx.x < WARPS) ? sh_s[lane] : 0.0f;
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

    for (int i = threadIdx.x; i < cols; i += BLOCK)
        st_f(yr, i, __expf(ld_f(xr, i) - sh_m[0]) * inv);
}

// Block dispatch + one-time tuning. The reg/online variant boundary
// stays based on the default block (kSmBlock) so a wider tuned block
// cannot flip kernels underneath the numerics tests; only the chosen
// variant's thread count varies.
template <typename T>
void softmax_dispatch(const T* x, T* y, int rows, int cols, int block,
                      cudaStream_t cs) {
    if (cols <= kSmPerThread * kSmBlock) {
        switch (block) {
        case 128:
            softmax_reg_kernel<T, 128><<<rows, 128, 0, cs>>>(x, y, cols);
            break;
        case 512:
            softmax_reg_kernel<T, 512><<<rows, 512, 0, cs>>>(x, y, cols);
            break;
        case 1024:
            softmax_reg_kernel<T, 1024><<<rows, 1024, 0, cs>>>(x, y, cols);
            break;
        default:
            softmax_reg_kernel<T, 256><<<rows, 256, 0, cs>>>(x, y, cols);
            break;
        }
    } else {
        switch (block) {
        case 128:
            softmax_online_kernel<T, 128><<<rows, 128, 0, cs>>>(x, y, cols);
            break;
        case 512:
            softmax_online_kernel<T, 512><<<rows, 512, 0, cs>>>(x, y, cols);
            break;
        case 1024:
            softmax_online_kernel<T, 1024><<<rows, 1024, 0, cs>>>(x, y, cols);
            break;
        default:
            softmax_online_kernel<T, 256><<<rows, 256, 0, cs>>>(x, y, cols);
            break;
        }
    }
    check_launch("softmax kernel launch");
}

// Tuning on the caller's own buffers at full size (see the layernorm
// note for why a truncated scratch problem misleads the choice). The
// register-resident variant has a hard coverage ceiling of
// BLOCK * kSmPerThread elements per row, so candidates below
// ceil(cols / kSmPerThread) would silently drop the row tail - the
// tuner never sees them (caught on a 5060 Ti that liked 128-thread
// blocks; the RTX 3060 never picked one there). The register variant
// is also capped at 512 threads: its v[32]/e[32] register slices sit
// right at the per-SM register ceiling for 1024-thread blocks, where
// sanitizer/profiler instrumentation tips the launch over the limit -
// the tuner handles the failure by design, but the reported error
// pollutes sanitizer gates (observed as 4x cudaErrorLaunchOutOfResour-
// ces on this driver generation).
template <typename T>
int softmax_pick_block(const char* tag, const T* x, T* y, int rows,
                       int cols, cudaStream_t cs) {
    int min_block = 1;
    int max_block = 1024;
    if (cols <= kSmPerThread * kSmBlock) {
        min_block = (cols + kSmPerThread - 1) / kSmPerThread;
        max_block = 512;
    }
    return autotune_block(tag, ((long long)rows << 32) | (unsigned)cols,
                          [&](int b) {
                              softmax_dispatch<T>(x, y, rows, cols, b, cs);
                          },
                          cs, min_block, max_block);
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
    cudaStream_t cs = (cudaStream_t)stream;
    int block = kSmBlock;
    if (!stream_is_capturing(cs))
        block = softmax_pick_block<float>("softmax:f32", x, y, rows, cols,
                                          cs);
    softmax_dispatch<float>(x, y, rows, cols, block, cs);
}

void softmax_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y,
                         int rows, int cols, std::uintptr_t stream) {
    if (rows <= 0 || cols <= 0) return;
    cudaStream_t cs = (cudaStream_t)stream;
    int block = kSmBlock;
    if (!stream_is_capturing(cs))
        block = softmax_pick_block<__nv_bfloat16>("softmax:bf16", x, y,
                                                  rows, cols, cs);
    softmax_dispatch<__nv_bfloat16>(x, y, rows, cols, block, cs);
}

} // namespace fusedtok
