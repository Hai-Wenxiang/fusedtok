#pragma once

// Shared helpers for the kernel launcher implementations: launch error
// surfacing, standard block size, and reusable warp/block reductions plus
// the order-preserving float key packing used by the selection kernels.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

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

// ---------------------------------------------------------------------------
// dtype-generic load/store for kernel templates: compute always happens in
// float32; bf16 buffers convert at the memory boundary. This keeps one code
// path per kernel with full numerical parity tooling.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float ld_f(const float* p, long long i) {
    return p[i];
}
__device__ __forceinline__ float ld_f(const __nv_bfloat16* p, long long i) {
    return __bfloat162float(p[i]);
}
__device__ __forceinline__ void st_f(float* p, long long i, float v) {
    p[i] = v;
}
__device__ __forceinline__ void st_f(__nv_bfloat16* p, long long i, float v) {
    p[i] = __float2bfloat16_rn(v);   // round-to-nearest-even
}

// ---------------------------------------------------------------------------
// Runtime block-size autotuning (v0.4.1)
// ---------------------------------------------------------------------------

// One-time per (tag, shape key) micro-benchmark: every candidate block
// runs the REAL kernel on the caller's own buffers (full size - a
// truncated problem misleads the choice, see the op files) for a few
// warmup + timed iterations on the caller's stream, CUDA events pick
// the winner, and the choice is cached for the process. The per-TU
// static cache means each op file keeps its own table - no cross-op
// contention. Callers must skip tuning while a stream capture is active
// (events and syncs are illegal mid-capture) and fall back to the
// default block. Structurally unlaunchable candidates (register
// pressure at 1024 threads on register-heavy kernels) score as slow
// instead of failing the call.
inline int autotune_block(const char* tag, long long shape_key,
                          const std::function<void(int block)>& launch,
                          cudaStream_t cs) {
    static std::mutex mu;
    static std::map<std::pair<const char*, long long>, int> cache;
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_pair(tag, shape_key);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;
    static const int cands[4] = {128, 256, 512, 1024};
    cudaEvent_t s = nullptr, e = nullptr;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    float best_ms = 1e30f;
    int best_block = 256;
    for (int b : cands) {
        // A candidate can be structurally unlaunchable (e.g. register
        // pressure at 1024 threads); score those as infinitely slow
        // instead of failing the call, and clear the sticky error.
        bool ok = true;
        try {
            for (int i = 0; i < 3; ++i)
                launch(b);                  // warmup (occupancy settle)
        } catch (const std::runtime_error&) {
            cudaGetLastError();
            ok = false;
        }
        if (!ok)
            continue;
        cudaEventRecord(s, cs);
        for (int i = 0; i < 8; ++i)
            launch(b);
        cudaEventRecord(e, cs);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        if (ms < best_ms) {
            best_ms = ms;
            best_block = b;
        }
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cudaGetLastError();                   // clear benign residue
    cache.emplace(key, best_block);
    return best_block;
}

// True when the stream is being captured (tuning must be skipped).
inline bool stream_is_capturing(cudaStream_t cs) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cs, &cap) != cudaSuccess)
        cudaGetLastError();               // do not poison the caller
    return cap == cudaStreamCaptureStatusActive;
}

} // namespace fusedtok
