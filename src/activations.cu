// Elementwise activations and binary ops.
//
// Structure per operator: a __global__ kernel (device code), a *_cpu
// reference implementation (ground truth, runs anywhere), and a *_launch
// raw-pointer entry point used by the bindings for both the staged numpy
// path and the zero-copy torch path.

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void silu_kernel(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v * sigmoidf_(v);
    }
}

// erff is a CUDA math-library intrinsic; accuracy matches CPU std::erf
// closely enough for the 1e-5 parity tolerance used in tests.
__global__ void gelu_kernel(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + erff(v / 1.4142135623730951f));
    }
}

// Tanh-approximation GeLU used by many BERT/GPT checkpoints when the exact
// erf form is too expensive. Max abs error vs exact GeLU is ~1e-3.
__global__ void gelu_tanh_kernel(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

__global__ void relu_kernel(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void tanh_kernel(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

__global__ void sigmoid_kernel(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sigmoidf_(x[i]);
}

__global__ void temperature_kernel(const float* x, float* y, int n, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / t;
}

__global__ void add_kernel(const float* a, const float* b, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

} // namespace

// --- SiLU ------------------------------------------------------------------

std::vector<float> silu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
    return y;
}

void silu_launch(const float* x, float* y, long long n) {
    if (n <= 0) return;
    silu_kernel<<<(unsigned)grid_for(n), kBlock>>>(x, y, n);
    check_launch("silu kernel launch");
}

// --- GeLU (exact) -------------------------------------------------------------

std::vector<float> gelu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::erf(v / 1.4142135623730951f));
    }
    return y;
}

void gelu_launch(const float* x, float* y, long long n) {
    if (n <= 0) return;
    gelu_kernel<<<(unsigned)grid_for(n), kBlock>>>(x, y, n);
    check_launch("gelu kernel launch");
}

// --- GeLU (tanh approximation) ------------------------------------------------

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
    if (n <= 0) return;
    gelu_tanh_kernel<<<(unsigned)grid_for(n), kBlock>>>(x, y, n);
    check_launch("gelu_tanh kernel launch");
}

// --- ReLU ---------------------------------------------------------------------

std::vector<float> relu_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return y;
}

void relu_launch(const float* x, float* y, long long n) {
    if (n <= 0) return;
    relu_kernel<<<(unsigned)grid_for(n), kBlock>>>(x, y, n);
    check_launch("relu kernel launch");
}

// --- Tanh ---------------------------------------------------------------------

std::vector<float> tanh_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = std::tanh(x[i]);
    return y;
}

void tanh_launch(const float* x, float* y, long long n) {
    if (n <= 0) return;
    tanh_kernel<<<(unsigned)grid_for(n), kBlock>>>(x, y, n);
    check_launch("tanh kernel launch");
}

// --- Sigmoid ------------------------------------------------------------------

std::vector<float> sigmoid_cpu(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return y;
}

void sigmoid_launch(const float* x, float* y, long long n) {
    if (n <= 0) return;
    sigmoid_kernel<<<(unsigned)grid_for(n), kBlock>>>(x, y, n);
    check_launch("sigmoid kernel launch");
}

// --- Temperature scaling --------------------------------------------------------

std::vector<float> temperature_cpu(const std::vector<float>& x, float t) {
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] / t;
    return y;
}

void temperature_launch(const float* x, float* y, int n, float t) {
    if (n <= 0) return;
    temperature_kernel<<<(unsigned)grid_for(n), kBlock>>>(x, y, n, t);
    check_launch("temperature kernel launch");
}

// --- Add / Mul ------------------------------------------------------------------

std::vector<float> add_cpu(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("add: inputs must have the same size");
    std::vector<float> y(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] + b[i];
    return y;
}

void add_launch(const float* a, const float* b, float* y, long long n) {
    if (n <= 0) return;
    add_kernel<<<(unsigned)grid_for(n), kBlock>>>(a, b, y, n);
    check_launch("add kernel launch");
}

std::vector<float> mul_cpu(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("mul: inputs must have the same size");
    std::vector<float> y(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] * b[i];
    return y;
}

void mul_launch(const float* a, const float* b, float* y, long long n) {
    if (n <= 0) return;
    mul_kernel<<<(unsigned)grid_for(n), kBlock>>>(a, b, y, n);
    check_launch("mul kernel launch");
}

} // namespace fusedtok
