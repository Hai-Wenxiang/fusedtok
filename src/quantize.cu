// INT8 symmetric per-tensor quantization utilities.
//
//   scale = max(|x|) / 127
//   q[i]  = round(x[i] * (1/scale))  clamped to [-127, 127]
//           (both paths multiply by the same pre-computed f32
//            reciprocal, so codes are bit-identical CPU vs GPU)
//   dequant: x ~= q[i] * scale
//
// Fused variant (qadd): dequantize two int8 operands, add in float,
// requantize against the OUTPUT's own max (estimated in the same kernel
// via a two-pass trick: pass 1 reduces the elementwise |a*sa + b*sb| max,
// pass 2 writes the requantized int8) - one round trip instead of three.
//
// The storage/dtype half of the INT8 path; the compute half (IMMA
// qgemm / decode GEMV, per-tensor and per-channel scales) lives in
// qgemm.cu.
//
// Stream/sync note (the one documented exception to cuda_launch.hpp's
// "launchers are async" contract): quantize and qadd must read the
// reduced absmax back to the HOST to compose the scale before pass 2
// launches, so these launchers sync the caller's stream once mid-call.
// All copies still ride the caller's stream (stream-ordered with the
// kernels around them) and are error-checked; they are therefore NOT
// CUDA-graph capturable, by design.

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

// symmetric INT8 range bound: codes live in [-127, 127] (no -128, so
// dequant stays sign-symmetric around zero)
constexpr float kInt8Max = 127.0f;

// stream-ordered memcpy with error propagation - check_launch covers
// kernels only, and a silently-failed copy here would leave the scale
// stale and corrupt every downstream byte
void checked_copy_async(void* dst, const void* src, size_t bytes,
                        cudaMemcpyKind kind, cudaStream_t cs,
                        const char* what) {
    if (cudaMemcpyAsync(dst, src, bytes, kind, cs) != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(cudaGetLastError()));
}

// sync the caller's stream, surfacing both the sync error itself and any
// sticky kernel fault from the preceding launch
float checked_readback_float(const float* dp, cudaStream_t cs,
                             const char* what) {
    if (cudaStreamSynchronize(cs) != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(cudaGetLastError()));
    float v = 0.0f;
    if (cudaMemcpy(&v, dp, sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        throw std::runtime_error(std::string(what) +
                                 " (readback): " +
                                 cudaGetErrorString(cudaGetLastError()));
    return v;
}

// Pass 1 of quantize: block-reduce max(|x|), one block handles a tile;
// atomicMax the running global max. (Simpler than a full scan: quantize is
// bandwidth-bound and one extra pass over f32 is cheap relative to the
// 4x storage savings it unlocks.)
__global__ void absmax_kernel(const float* __restrict__ x,
                              float* __restrict__ out_max, long long n) {
    __shared__ float shared[kBlock / 32];
    float m = 0.0f;
    // grid-stride loop: consecutive threads read consecutive addresses
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        m = fmaxf(m, fabsf(x[i]));
    }
    m = block_reduce_max<kBlock>(m, shared);
    if (threadIdx.x == 0 && m > 0.0f)
        // bit-pattern compare is order-preserving for non-negative floats
        atomicMax(reinterpret_cast<int*>(out_max), __float_as_int(m));
}

__global__ void quantize_kernel(const float* __restrict__ x,
                                signed char* __restrict__ q,
                                float inv_scale, long long n) {
    // inv_scale is the PRE-COMPUTED 1/scale (the caller folds the
    // division once); the CPU reference multiplies by the same f32
    // reciprocal, so near-tie codes stay bit-identical across paths
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        float v = fmaxf(-kInt8Max, fminf(kInt8Max, rintf(x[i] * inv_scale)));
        q[i] = (signed char)v;
    }
}

__global__ void dequantize_kernel(const signed char* __restrict__ q,
                                  float* __restrict__ x,
                                  float scale, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        x[i] = (float)q[i] * scale;
    }
}

// Fused qadd pass 1: reduce max(|qa*sa + qb*sb|) in float
__global__ void qadd_absmax_kernel(const signed char* __restrict__ qa,
                                   const signed char* __restrict__ qb,
                                   float sa, float sb,
                                   float* __restrict__ out_max, long long n) {
    __shared__ float shared[kBlock / 32];
    float m = 0.0f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const float v = (float)qa[i] * sa + (float)qb[i] * sb;
        m = fmaxf(m, fabsf(v));
    }
    m = block_reduce_max<kBlock>(m, shared);
    if (threadIdx.x == 0 && m > 0.0f)
        atomicMax(reinterpret_cast<int*>(out_max), __float_as_int(m));
}

// Fused qadd pass 2: add in float, requantize with 1/out_max
__global__ void qadd_kernel(const signed char* __restrict__ qa,
                            const signed char* __restrict__ qb,
                            float sa, float sb, float inv_out_scale,
                            signed char* __restrict__ qy, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const float v = (float)qa[i] * sa + (float)qb[i] * sb;
        float qv = fmaxf(-kInt8Max, fminf(kInt8Max, rintf(v * inv_out_scale)));
        qy[i] = (signed char)qv;
    }
}

struct DevMax {
    float* p = nullptr;
    DevMax() {
        if (cudaMalloc(&p, sizeof(float)) != cudaSuccess)
            throw std::runtime_error(
                std::string("quantize scratch alloc failed: ") +
                cudaGetErrorString(cudaGetLastError()));
    }
    ~DevMax() { if (p) cudaFree(p); }
    DevMax(const DevMax&) = delete;
    DevMax& operator=(const DevMax&) = delete;
};

// shared scale composition: max(0, absmax)/127 with a 1.0 fallback for
// all-zero input (a 0 scale would make the dequant a black hole)
inline float make_scale(float absmax) {
    return absmax > 0.0f ? absmax / kInt8Max : 1.0f;
}

// zero the scratch on the caller's stream, launch the absmax reduction,
// sync once, read the max back (see the stream/sync note in the header)
float reduce_device_absmax(const std::function<void(cudaStream_t)>& zero_and_launch,
                           DevMax& dm, cudaStream_t cs, const char* what) {
    zero_and_launch(cs);
    return checked_readback_float(dm.p, cs, what);
}

} // namespace

std::pair<std::vector<signed char>, float>
quantize_int8_cpu(const std::vector<float>& x) {
    float m = 0.0f;
    for (float v : x) m = std::fmax(m, std::fabs(v));
    const float scale = make_scale(m);
    // multiply by the same pre-computed f32 reciprocal the GPU kernel
    // uses - dividing here would round near-tie values differently and
    // flip codes by one against the GPU path
    const float inv_scale = 1.0f / scale;
    std::vector<signed char> q(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = std::rint(x[i] * inv_scale);
        if (v > kInt8Max) v = kInt8Max;
        if (v < -kInt8Max) v = -kInt8Max;
        q[i] = (signed char)v;
    }
    return {std::move(q), scale};
}

std::vector<float> dequantize_int8_cpu(const std::vector<signed char>& q,
                                       float scale) {
    std::vector<float> x(q.size());
    for (size_t i = 0; i < q.size(); ++i) x[i] = (float)q[i] * scale;
    return x;
}

void quantize_int8_launch(const float* x, signed char* q,
                          float* scale_out, long long n, std::uintptr_t stream) {
    if (n <= 0) return;
    static DevMax dm;                      // process-cached scratch
    cudaStream_t cs = (cudaStream_t)stream;
    const float zero = 0.0f;
    const float m = reduce_device_absmax(
        [&](cudaStream_t s) {
            checked_copy_async(dm.p, &zero, sizeof(float),
                               cudaMemcpyHostToDevice, s, "quantize zero");
            absmax_kernel<<<(unsigned)grid_for(n), kBlock, 0, s>>>(x, dm.p, n);
            check_launch("absmax kernel launch");
        },
        dm, cs, "quantize absmax");
    const float scale = make_scale(m);
    checked_copy_async(scale_out, &scale, sizeof(float),
                       cudaMemcpyHostToDevice, cs, "quantize scale");
    quantize_kernel<<<(unsigned)grid_for(n), kBlock, 0, cs>>>(
        x, q, 1.0f / scale, n);
    check_launch("quantize kernel launch");
}

void dequantize_int8_launch(const signed char* q, float* x,
                            float scale, long long n, std::uintptr_t stream) {
    if (n <= 0) return;
    dequantize_kernel<<<(unsigned)grid_for(n), kBlock, 0, (cudaStream_t)stream>>>(q, x, scale, n);
    check_launch("dequantize kernel launch");
}

void qadd_int8_launch(const signed char* qa, const signed char* qb,
                      float sa, float sb,
                      signed char* qy, float* out_scale, long long n, std::uintptr_t stream) {
    if (n <= 0) return;
    static DevMax dm;
    cudaStream_t cs = (cudaStream_t)stream;
    const float zero = 0.0f;
    const float m = reduce_device_absmax(
        [&](cudaStream_t s) {
            checked_copy_async(dm.p, &zero, sizeof(float),
                               cudaMemcpyHostToDevice, s, "qadd zero");
            qadd_absmax_kernel<<<(unsigned)grid_for(n), kBlock, 0, s>>>(
                qa, qb, sa, sb, dm.p, n);
            check_launch("qadd absmax kernel launch");
        },
        dm, cs, "qadd absmax");
    const float scale = make_scale(m);
    checked_copy_async(out_scale, &scale, sizeof(float),
                       cudaMemcpyHostToDevice, cs, "qadd scale");
    qadd_kernel<<<(unsigned)grid_for(n), kBlock, 0, cs>>>(qa, qb, sa, sb,
                                                   1.0f / scale, qy, n);
    check_launch("qadd kernel launch");
}

} // namespace fusedtok
