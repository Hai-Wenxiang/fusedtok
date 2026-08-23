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
#include <algorithm>
#include <cmath>
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

// ---------------------------------------------------------------------------
// fused nucleus sampling - CPU reference
// ---------------------------------------------------------------------------

// Same algorithm as the GPU kernel (order, cut rule, RNG hash) so results
// agree up to floating-point rounding: sort logits/T descending with
// earliest-index ties (the packed-key order), accumulate exp(v - row_max)
// in float32 in that order, cut the nucleus at cum >= p * total, and
// inverse-CDF a splitmix-hash uniform scaled to the nucleus mass. CPU uses
// exact exp vs the device __expf, so draws landing exactly on a boundary
// may pick a neighbor token - both are valid samplers of the distribution.
long long sample_topp_cpu(const std::vector<float>& logits,
                          float p, float t, unsigned long long seed) {
    if (logits.empty())
        throw std::invalid_argument("sample of empty logits");
    if (!(p > 0.0f && p <= 1.0f))
        throw std::invalid_argument("p must be in (0, 1]");
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");

    const size_t n = logits.size();
    // order indices by (logit/T desc, index asc) - the packed-key order
    std::vector<unsigned int> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (unsigned int)i;
    const float inv_t = 1.0f / t;
    std::sort(order.begin(), order.end(), [&](unsigned int a, unsigned int b) {
        const float va = logits[a] * inv_t, vb = logits[b] * inv_t;
        if (va != vb) return va > vb;
        return a < b;
    });

    const float row_max = logits[order[0]] * inv_t;
    auto mass_at = [&](size_t i) {
        return std::exp(logits[order[i]] * inv_t - row_max);
    };

    float total = 0.0f;
    for (size_t i = 0; i < n; ++i) total += mass_at(i);

    float cum = 0.0f;
    size_t nucleus = 0;
    float nucleus_mass = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cum += mass_at(i);
        nucleus = i + 1;
        if (cum >= p * total) { nucleus_mass = cum; break; }
    }

    // splitmix64-finalized uniform, identical to the device side
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const float u = (float)((z >> 11) * (1.0 / 9007199254740992.0));

    const float target = u * nucleus_mass;
    cum = 0.0f;
    for (size_t i = 0; i < nucleus; ++i) {
        cum += mass_at(i);
        if (cum >= target) return (long long)order[i];
    }
    return (long long)order[nucleus - 1];   // float rounding fallback
}

} // namespace fusedtok
