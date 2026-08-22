// Elementwise activations and binary ops.
//
// Structure per operator: a functor with the element formula, a CPU
// reference implementation (ground truth, runs anywhere), and a *_launch
// raw-pointer entry point. The CUDA path vectorizes into float4 loads and
// stores whenever the buffers are 16-byte aligned (torch and cudaMalloc
// allocations always are); a scalar tail kernel handles n % 4 leftovers.

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace fusedtok {

namespace {

// --- elementwise functors ----------------------------------------------------

struct SiluOp {
    __device__ __forceinline__ float operator()(float v) const {
        return v / (1.0f + expf(-v));
    }
};
struct GeluOp {
    __device__ __forceinline__ float operator()(float v) const {
        return 0.5f * v * (1.0f + erff(v / 1.4142135623730951f));
    }
};
struct GeluTanhOp {
    __device__ __forceinline__ float operator()(float v) const {
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        return 0.5f * v * (1.0f + tanhf(inner));
    }
};
struct ReluOp {
    __device__ __forceinline__ float operator()(float v) const {
        return v > 0.0f ? v : 0.0f;
    }
};
struct TanhOp {
    __device__ __forceinline__ float operator()(float v) const { return tanhf(v); }
};
struct SigmoidOp {
    __device__ __forceinline__ float operator()(float v) const {
        return 1.0f / (1.0f + expf(-v));
    }
};
struct TemperatureOp {
    float inv_t;
    __device__ __forceinline__ float operator()(float v) const { return v * inv_t; }
};
struct AxpyOp {
    float a, b;
    __device__ __forceinline__ float operator()(float v) const { return a * v + b; }
};
struct AddOp {
    __device__ __forceinline__ float operator()(float x, float y) const { return x + y; }
};
struct MulOp {
    __device__ __forceinline__ float operator()(float x, float y) const { return x * y; }
};
struct SwigluOp {
    __device__ __forceinline__ float operator()(float g, float u) const {
        return g / (1.0f + expf(-g)) * u;
    }
};

// --- vectorized kernels -------------------------------------------------------

template <typename F>
__global__ void unary_f4_kernel(const float4* __restrict__ in,
                                float4* __restrict__ out, long long n4, F f) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        float4 o;
        o.x = f(v.x);
        o.y = f(v.y);
        o.z = f(v.z);
        o.w = f(v.w);
        out[i] = o;
    }
}

template <typename F>
__global__ void unary_f1_kernel(const float* __restrict__ in,
                                float* __restrict__ out, int n, F f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(in[i]);
}

template <typename F>
__global__ void binary_f4_kernel(const float4* __restrict__ a,
                                 const float4* __restrict__ b,
                                 float4* __restrict__ out, long long n4, F f) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 va = a[i];
        float4 vb = b[i];
        float4 o;
        o.x = f(va.x, vb.x);
        o.y = f(va.y, vb.y);
        o.z = f(va.z, vb.z);
        o.w = f(va.w, vb.w);
        out[i] = o;
    }
}

template <typename F>
__global__ void binary_f1_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ out, int n, F f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(a[i], b[i]);
}

// --- host drivers --------------------------------------------------------------

// Launch a unary elementwise op, vectorized when possible. Pointers must be
// 16-byte aligned for the float4 path; misaligned views fall back to scalar.
template <typename F>
void launch_unary(const float* x, float* y, long long n, F f) {
    if (n <= 0) return;
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) |
                           reinterpret_cast<uintptr_t>(y)) & 0xF) == 0;
    if (aligned && n >= 8) {
        const long long n4 = n / 4;
        unary_f4_kernel<F><<<(unsigned)((n4 + kBlock - 1) / kBlock), kBlock>>>(
            reinterpret_cast<const float4*>(x), reinterpret_cast<float4*>(y), n4, f);
        const int tail = (int)(n - n4 * 4);
        if (tail > 0)
            unary_f1_kernel<F><<<(tail + kBlock - 1) / kBlock, kBlock>>>(
                x + n4 * 4, y + n4 * 4, tail, f);
    } else {
        unary_f1_kernel<F><<<(unsigned)grid_for(n), kBlock>>>(x, y, (int)n, f);
    }
    check_launch("elementwise kernel launch");
}

template <typename F>
void launch_binary(const float* a, const float* b, float* y, long long n, F f) {
    if (n <= 0) return;
    const bool aligned = ((reinterpret_cast<uintptr_t>(a) |
                           reinterpret_cast<uintptr_t>(b) |
                           reinterpret_cast<uintptr_t>(y)) & 0xF) == 0;
    if (aligned && n >= 8) {
        const long long n4 = n / 4;
        binary_f4_kernel<F><<<(unsigned)((n4 + kBlock - 1) / kBlock), kBlock>>>(
            reinterpret_cast<const float4*>(a), reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(y), n4, f);
        const int tail = (int)(n - n4 * 4);
        if (tail > 0)
            binary_f1_kernel<F><<<(tail + kBlock - 1) / kBlock, kBlock>>>(
                a + n4 * 4, b + n4 * 4, y + n4 * 4, tail, f);
    } else {
        binary_f1_kernel<F><<<(unsigned)grid_for(n), kBlock>>>(a, b, y, (int)n, f);
    }
    check_launch("elementwise kernel launch");
}

} // namespace

// --- SiLU ----------------------------------------------------------------------

std::vector<float> silu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
    return y;
}

void silu_launch(const float* x, float* y, long long n) {
    launch_unary(x, y, n, SiluOp{});
}

// --- GeLU (exact erf form) -------------------------------------------------------

std::vector<float> gelu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::erf(v / 1.4142135623730951f));
    }
    return y;
}

void gelu_launch(const float* x, float* y, long long n) {
    launch_unary(x, y, n, GeluOp{});
}

// --- GeLU (tanh approximation) ----------------------------------------------------

std::vector<float> gelu_tanh_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
    return y;
}

void gelu_tanh_launch(const float* x, float* y, long long n) {
    launch_unary(x, y, n, GeluTanhOp{});
}

// --- ReLU --------------------------------------------------------------------------

std::vector<float> relu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return y;
}

void relu_launch(const float* x, float* y, long long n) {
    launch_unary(x, y, n, ReluOp{});
}

// --- Tanh --------------------------------------------------------------------------

std::vector<float> tanh_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = std::tanh(x[i]);
    return y;
}

void tanh_launch(const float* x, float* y, long long n) {
    launch_unary(x, y, n, TanhOp{});
}

// --- Sigmoid ----------------------------------------------------------------------

std::vector<float> sigmoid_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return y;
}

void sigmoid_launch(const float* x, float* y, long long n) {
    launch_unary(x, y, n, SigmoidOp{});
}

// --- Temperature scaling --------------------------------------------------------------

std::vector<float> temperature_cpu(const std::vector<float>& x, float t) {
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] / t;
    return y;
}

void temperature_launch(const float* x, float* y, int n, float t) {
    TemperatureOp op{1.0f / t};
    launch_unary(x, y, n, op);
}

// --- Add / Mul ------------------------------------------------------------------------

std::vector<float> add_cpu(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("add: inputs must have the same size");
    std::vector<float> y(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] + b[i];
    return y;
}

void add_launch(const float* a, const float* b, float* y, long long n) {
    launch_binary(a, b, y, n, AddOp{});
}

std::vector<float> mul_cpu(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("mul: inputs must have the same size");
    std::vector<float> y(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] * b[i];
    return y;
}

void mul_launch(const float* a, const float* b, float* y, long long n) {
    launch_binary(a, b, y, n, MulOp{});
}

// --- SwiGLU --------------------------------------------------------------------------

std::vector<float> swiglu_cpu(const std::vector<float>& gate,
                              const std::vector<float>& up) {
    if (gate.size() != up.size())
        throw std::invalid_argument("gate.size() must equal up.size()");
    std::vector<float> out(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        float g = gate[i];
        out[i] = g / (1.0f + std::exp(-g)) * up[i];
    }
    return out;
}

void swiglu_launch(const float* gate, const float* up, float* y, long long n) {
    launch_binary(gate, up, y, n, SwigluOp{});
}

} // namespace fusedtok
