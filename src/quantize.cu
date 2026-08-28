// INT8 symmetric per-tensor quantization utilities.
//
//   scale = max(|x|) / 127
//   q[i]  = round(x[i] / scale)  clamped to [-127, 127]
//   dequant: x ~= q[i] * scale
//
// Fused variant (qadd): dequantize two int8 operands, add in float,
// requantize against the OUTPUT's own max (estimated in the same kernel
// via a two-pass trick: pass 1 reduces the elementwise |a*sa + b*sb| max,
// pass 2 writes the requantized int8) - one round trip instead of three.
//
// This is the storage/dtype path only; INT8 GEMM is out of scope (v0.4+).

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

// Pass 1 of quantize: block-reduce max(|x|), one block handles a tile;
// atomicMax the running global max. (Simpler than a full scan: quantize is
// bandwidth-bound and one extra pass over f32 is cheap relative to the
// 4x storage savings it unlocks.)
__global__ void absmax_kernel(const float* __restrict__ x,
                              float* __restrict__ out_max, long long n) {
    __shared__ float shared[kBlock / 32];
    const long long i0 = (long long)blockIdx.x * blockDim.x * 4 + threadIdx.x;
    float m = 0.0f;
    // grid-stride with 4x unroll for coalescing bandwidth
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        m = fmaxf(m, fabsf(x[i]));
    }
    m = block_reduce_max<kBlock>(m, shared);
    if (threadIdx.x == 0 && m > 0.0f)
        atomicMax(reinterpret_cast<int*>(out_max), __float_as_int(m));
        // bit-pattern compare is order-preserving for non-negative floats
}

__global__ void quantize_kernel(const float* __restrict__ x,
                                signed char* __restrict__ q,
                                float scale, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        float v = fmaxf(-127.0f, fminf(127.0f, rintf(x[i] * scale)));
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
        float qv = fmaxf(-127.0f, fminf(127.0f, rintf(v * inv_out_scale)));
        qy[i] = (signed char)qv;
    }
}

struct DevMax {
    float* p = nullptr;
    DevMax() {
        if (cudaMalloc(&p, sizeof(float)) != cudaSuccess)
            throw std::runtime_error("quantize scratch alloc failed");
    }
    ~DevMax() { if (p) cudaFree(p); }
    DevMax(const DevMax&) = delete;
    DevMax& operator=(const DevMax&) = delete;
};

float device_absmax(const float* x, long long n, DevMax& dm, std::uintptr_t stream = 0) {
    const float zero = 0.0f;
    cudaMemcpy(dm.p, &zero, sizeof(float), cudaMemcpyHostToDevice);
    absmax_kernel<<<(unsigned)grid_for(n), kBlock, 0, (cudaStream_t)stream>>>(x, dm.p, n);
    check_launch("absmax kernel launch");
    float m = 0.0f;
    cudaMemcpy(&m, dm.p, sizeof(float), cudaMemcpyDeviceToHost);
    return m;
}

} // namespace

std::pair<std::vector<signed char>, float>
quantize_int8_cpu(const std::vector<float>& x) {
    float m = 0.0f;
    for (float v : x) m = std::fmax(m, std::fabs(v));
    const float scale = m > 0.0f ? m / 127.0f : 1.0f;
    std::vector<signed char> q(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = std::rint(x[i] / scale);
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
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
    const float m = device_absmax(x, n, dm, stream);
    const float scale = m > 0.0f ? m / 127.0f : 1.0f;
    cudaMemcpy(scale_out, &scale, sizeof(float), cudaMemcpyHostToDevice);
    quantize_kernel<<<(unsigned)grid_for(n), kBlock, 0, (cudaStream_t)stream>>>(
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
    const float zero = 0.0f;
    cudaMemcpy(dm.p, &zero, sizeof(float), cudaMemcpyHostToDevice);
    qadd_absmax_kernel<<<(unsigned)grid_for(n), kBlock, 0, (cudaStream_t)stream>>>(qa, qb, sa, sb, dm.p, n);
    check_launch("qadd absmax kernel launch");
    float m = 0.0f;
    cudaMemcpy(&m, dm.p, sizeof(float), cudaMemcpyDeviceToHost);
    const float scale = m > 0.0f ? m / 127.0f : 1.0f;
    cudaMemcpy(out_scale, &scale, sizeof(float), cudaMemcpyHostToDevice);
    qadd_kernel<<<(unsigned)grid_for(n), kBlock, 0, (cudaStream_t)stream>>>(qa, qb, sa, sb,
                                                   1.0f / scale, qy, n);
    check_launch("qadd kernel launch");
}

} // namespace fusedtok
