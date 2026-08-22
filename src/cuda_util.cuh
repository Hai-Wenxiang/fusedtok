#pragma once

// Shared helpers for the kernel launcher implementations: launch error
// surfacing, standard block size, and reusable warp/block reductions plus
// the order-preserving float key packing used by the selection kernels.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace fusedtok {

// Surface a kernel launch failure as std::runtime_error (Python RuntimeError).
// Called right after every <<<>>> launch; catches bad launch configs etc.
inline void check_launch(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Default threads-per-block for elementwise kernels.
constexpr int kBlock = 256;

// Grid size covering n items with kBlock threads, rounded up.
inline long long grid_for(long long n) { return (n + kBlock - 1) / kBlock; }

// ---------------------------------------------------------------------------
// Warp / block reductions (deterministic: fixed shuffle order per warp)
// ---------------------------------------------------------------------------

// Sum `val` across the warp; result valid on lane 0 only.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, off);
    return val;
}

// Max of `val` across the warp; result valid on lane 0 only.
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, off));
    return val;
}

// Block-wide sum via shared memory over warp partials. All threads return
// the reduced value. `shared` must hold BLOCK/32 floats.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (threadIdx.x < BLOCK / 32) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// Block-wide max via shared memory over warp partials. All threads return
// the reduced value. `shared` must hold BLOCK/32 floats.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (threadIdx.x < BLOCK / 32) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ---------------------------------------------------------------------------
// Order-preserving float32 -> uint32 packing for selection kernels.
//
// Maps floats monotonically onto unsigned ints so a single integer max
// computes both the largest value AND (via the low bits) the smallest index
// among ties. NaNs are not order-preserving - selection with NaN input is
// undefined, matching common library behavior.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int fkey(float v) {
    unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float unfkey(unsigned int u) {
    unsigned int bits = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return __uint_as_float(bits);
}

} // namespace fusedtok
