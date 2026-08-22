// Repetition penalty (CTRL-style, as used by HF transformers):
//
//   y[i] = x[i]                                if i not in token_ids
//   y[i] = x[i] / penalty                      if x[i] > 0
//   y[i] = x[i] * penalty                      if x[i] < 0
//
// Applied to logits of previously generated tokens before sampling. The
// elementwise pass is parallel; the "which tokens are penalized" lookup is
// a small gather over m ids (m = number of generated tokens, typically tiny
// compared to vocab n).

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace fusedtok {

namespace {

// Copy logits, then one thread per penalized id applies the scale in place.
__global__ void repetition_penalty_kernel(const float* x, const long long* ids,
                                          int n, int m, float penalty,
                                          float* y) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    int id = (int)ids[j];
    if (id < 0 || id >= n) return;   // validated on host; defensive guard
    float v = x[id];
    y[id] = v > 0.0f ? v / penalty : v * penalty;
}

// Straight copy kernel: y = x (used when m == 0 to keep behavior uniform).
__global__ void copy_kernel(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

} // namespace

std::vector<float> repetition_penalty_cpu(const std::vector<float>& logits,
                                          const std::vector<long long>& token_ids,
                                          float penalty) {
    if (!(penalty > 0.0f))
        throw std::invalid_argument("penalty must be > 0");
    for (long long id : token_ids)
        if (id < 0 || id >= (long long)logits.size())
            throw std::invalid_argument("token id out of range");
    std::vector<float> y = logits;
    for (long long id : token_ids) {
        float v = y[(size_t)id];
        y[(size_t)id] = v > 0.0f ? v / penalty : v * penalty;
    }
    return y;
}

void repetition_penalty_launch(const float* logits, const long long* ids,
                               int n, int m, float penalty, float* y) {
    if (n <= 0) return;
    // Non-listed logits pass through unchanged: copy first, then scale the
    // listed ids in place (two stream-ordered launches, still async).
    copy_kernel<<<(unsigned)grid_for(n), kBlock>>>(logits, y, n);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess && m > 0) {
        repetition_penalty_kernel<<<(unsigned)grid_for(m), kBlock>>>(
            logits, ids, n, m, penalty, y);
        err = cudaGetLastError();
    }
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("repetition_penalty kernel launch: ") +
                                 cudaGetErrorString(err));
}

} // namespace fusedtok
