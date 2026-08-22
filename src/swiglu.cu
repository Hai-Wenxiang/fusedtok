// SwiGLU activation: out = silu(gate) * up.

#include "fusedtok/fusedtok.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

// sigmoid via 1 / (1 + exp(-x)); expf overflow-safe for large |x| because
// expf(-x) -> 0 for large positive x and -> inf denominator for large
// negative x, both giving the correct limit.
__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// One thread per element: out[i] = silu(gate[i]) * up[i].
__global__ void swiglu_kernel(const float* gate, const float* up, float* out,
                              long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = g * sigmoidf_(g) * up[i];
    }
}

} // namespace

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
    if (n <= 0) return;
    swiglu_kernel<<<(unsigned)grid_for(n), kBlock>>>(gate, up, y, n);
    check_launch("swiglu kernel launch");
}

} // namespace fusedtok
