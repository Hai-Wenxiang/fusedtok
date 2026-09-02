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

constexpr int kLnBlock = 256;     // default when tuning is skipped

template <typename T, int BLOCK>
__global__ void layernorm_kernel(const T* __restrict__ x,
                                 const float* __restrict__ w,
                                 const float* __restrict__ b,
                                 T* __restrict__ y,
                                 int cols, float eps) {
    __shared__ float shared[BLOCK / 32];
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK)
        sum += ld_f(xr, i);
    const float mean = block_reduce_sum<BLOCK>(sum, shared) / cols;

    float var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK) {
        float d = ld_f(xr, i) - mean;
        var += d * d;
    }
    const float total_var = block_reduce_sum<BLOCK>(var, shared) / cols;

    const float inv_std = rsqrtf(total_var + eps);
    for (int i = threadIdx.x; i < cols; i += BLOCK)
        st_f(yr, i, (ld_f(xr, i) - mean) * inv_std * w[i] + b[i]);
}

template <typename T>
void layernorm_dispatch(const T* x, const float* w, const float* b, T* y,
                        int rows, int cols, float eps, int block,
                        cudaStream_t cs) {
    switch (block) {
    case 128:
        layernorm_kernel<T, 128><<<rows, 128, 0, cs>>>(x, w, b, y, cols, eps);
        break;
    case 512:
        layernorm_kernel<T, 512><<<rows, 512, 0, cs>>>(x, w, b, y, cols, eps);
        break;
    case 1024:
        layernorm_kernel<T, 1024><<<rows, 1024, 0, cs>>>(x, w, b, y, cols, eps);
        break;
    default:
        layernorm_kernel<T, 256><<<rows, 256, 0, cs>>>(x, w, b, y, cols, eps);
        break;
    }
    check_launch("layernorm kernel launch");
}

// One-time block tuning on the caller's own buffers at full size (a
// truncated scratch problem misleads the choice - small grids favor
// big blocks, full grids do not; the kernel is deterministic so the
// repeated tuning writes are byte-identical to the final launch). The
// layernorm sweep showed the largest autotune win of the three row-wise
// kernels: 512 threads beats 256 by ~30% on 4096-wide rows, 1024 wins
// by ~47% at 8192.
template <typename T>
int layernorm_pick_block(const char* tag, const T* x, const float* w,
                         const float* b, T* y, int rows, int cols,
                         float eps, cudaStream_t cs) {
    return autotune_block(tag, ((long long)rows << 32) | (unsigned)cols,
                          [&](int bl) {
                              layernorm_dispatch<T>(x, w, b, y, rows, cols,
                                                    eps, bl, cs);
                          },
                          cs);
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
                      float* y, int rows, int cols, float eps, std::uintptr_t stream) {
    if (rows <= 0 || cols <= 0) return;
    cudaStream_t cs = (cudaStream_t)stream;
    int block = kLnBlock;
    if (!stream_is_capturing(cs))
        block = layernorm_pick_block<float>("layernorm:f32", x, w, b, y,
                                            rows, cols, eps, cs);
    layernorm_dispatch<float>(x, w, b, y, rows, cols, eps, block, cs);
}

void layernorm_launch_bf16(const __nv_bfloat16* x, const float* w,
                           const float* b, __nv_bfloat16* y,
                           int rows, int cols, float eps, std::uintptr_t stream) {
    if (rows <= 0 || cols <= 0) return;
    cudaStream_t cs = (cudaStream_t)stream;
    int block = kLnBlock;
    if (!stream_is_capturing(cs))
        block = layernorm_pick_block<__nv_bfloat16>("layernorm:bf16", x, w,
                                                    b, y, rows, cols, eps,
                                                    cs);
    layernorm_dispatch<__nv_bfloat16>(x, w, b, y, rows, cols, eps, block, cs);
}

} // namespace fusedtok
