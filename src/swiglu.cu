#include "fusedtok/fusedtok.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace fusedtok {

namespace {

// sigmoid via 1 / (1 + exp(-x)); expf overflow-safe for large |x| because
// expf(-|x|) -> 0 for large positive x and -> inf denominator for large
// negative x, both giving the correct limit.
__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void swiglu_check(const std::vector<float>& gate, const std::vector<float>& up) {
    if (gate.size() != up.size())
        throw std::invalid_argument("gate.size() must equal up.size()");
}

} // namespace

// One thread per element: out[i] = silu(gate[i]) * up[i].
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = g * sigmoidf_(g) * up[i];
    }
}

std::vector<float> swiglu_cpu(const std::vector<float>& gate,
                              const std::vector<float>& up) {
    swiglu_check(gate, up);
    std::vector<float> out(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        float g = gate[i];
        out[i] = g / (1.0f + std::exp(-g)) * up[i];
    }
    return out;
}

std::vector<float> swiglu_cuda(const std::vector<float>& gate,
                               const std::vector<float>& up) {
    swiglu_check(gate, up);
    const int n = static_cast<int>(gate.size());
    if (n == 0) return {};
    std::vector<float> out(n);

    float *dg = nullptr, *du = nullptr, *dout = nullptr;
    if (cudaMalloc(&dg, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc gate failed");
    if (cudaMalloc(&du, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc up failed");
    if (cudaMalloc(&dout, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc out failed");

    if (cudaMemcpy(dg, gate.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D gate failed");
    if (cudaMemcpy(du, up.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D up failed");

    swiglu_kernel<<<(n + 255) / 256, 256>>>(dg, du, dout, n);

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("swiglu kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));

    if (cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H out failed");

    cudaFree(dg); cudaFree(du); cudaFree(dout);
    return out;
}

}
