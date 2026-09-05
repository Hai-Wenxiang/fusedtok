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
                               int n, int m, float penalty, float* y, std::uintptr_t stream) {
    if (n <= 0) return;
    // Non-listed logits pass through unchanged: copy first, then scale the
    // listed ids in place (two stream-ordered launches, still async).
    copy_kernel<<<(unsigned)grid_for(n), kBlock, 0, (cudaStream_t)stream>>>(logits, y, n);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess && m > 0) {
        repetition_penalty_kernel<<<(unsigned)grid_for(m), kBlock, 0, (cudaStream_t)stream>>>(
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
    const float u = splitmix_uniform(seed);

    const float target = u * nucleus_mass;
    cum = 0.0f;
    for (size_t i = 0; i < nucleus; ++i) {
        cum += mass_at(i);
        if (cum >= target) return (long long)order[i];
    }
    return (long long)order[nucleus - 1];   // float rounding fallback
}

// ---------------------------------------------------------------------------
// fused top-k sampling - CPU reference
// ---------------------------------------------------------------------------

// Same algorithm as sample_topk_launch (order, renormalization, RNG
// hash) so results agree up to floating-point rounding: sort logits/T
// descending with earliest-index ties (the packed-key order), sum
// exp(v - row_max) over the FIRST k entries in float32 in that order,
// and inverse-CDF the splitmix-hash uniform scaled to the k-window
// mass. CPU uses exact exp vs the device __expf, so draws landing
// exactly on a boundary may pick a neighbor token - both are valid
// samplers of the renormalized top-k distribution.
long long sample_topk_cpu(const std::vector<float>& logits, int k, float t,
                          unsigned long long seed) {
    if (logits.empty())
        throw std::invalid_argument("sample of empty logits");
    if (k <= 0)
        throw std::invalid_argument("k must be >= 1");
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");
    const size_t n = logits.size();
    if ((size_t)k > n) k = (int)n;   // full-vocab sampling

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

    float window_mass = 0.0f;
    for (int i = 0; i < k; ++i) window_mass += mass_at((size_t)i);

    // splitmix64-finalized uniform, identical to the device side
    const float u = splitmix_uniform(seed);

    const float target = u * window_mass;
    float cum = 0.0f;
    for (int i = 0; i < k; ++i) {
        cum += mass_at((size_t)i);
        if (cum >= target) return (long long)order[(size_t)i];
    }
    return (long long)order[(size_t)k - 1];   // float rounding fallback
}

// ---------------------------------------------------------------------------
// fused min-p sampling - CPU reference (v1.3)
// ---------------------------------------------------------------------------

// Same algorithm as sample_minp_launch: keep every token whose
// probability is at least min_p times the maximum probability - in the
// max-normalized exp column that is a prefix cut at the first element
// with exp < min_p (exps[0] == 1.0 by construction), renormalize
// within that nucleus and inverse-CDF the splitmix-hash uniform.
// Identical accumulation order to the device serial kernel; CPU exact
// exp vs device __expf gives the usual neighboring-draw caveat on
// exp-rounding boundaries.
long long sample_minp_cpu(const std::vector<float>& logits, float min_p,
                          float t, unsigned long long seed) {
    if (logits.empty())
        throw std::invalid_argument("sample of empty logits");
    if (!(min_p > 0.0f && min_p <= 1.0f))
        throw std::invalid_argument("min_p must be in (0, 1]");
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

    // nucleus: prefix while exp >= min_p (mass_at(0) == 1.0 >= min_p,
    // so the nucleus is never empty for a valid min_p)
    float nucleus_mass = 0.0f;
    size_t nucleus = 0;
    for (size_t i = 0; i < n; ++i) {
        const float e = mass_at(i);
        if (e < min_p) { nucleus = i; break; }
        nucleus_mass += e;
        nucleus = i + 1;
    }

    // splitmix64-finalized uniform, identical to the device side
    const float u = splitmix_uniform(seed);

    const float target = u * nucleus_mass;
    float cum = 0.0f;
    for (size_t i = 0; i < nucleus; ++i) {
        cum += mass_at(i);
        if (cum >= target) return (long long)order[i];
    }
    return (long long)order[nucleus - 1];     // float rounding fallback
}

// ---------------------------------------------------------------------------
// batched CPU references (v1.4): the row-wise singles verbatim - each
// row's token is bit-identical to calling the single-row reference on
// that row, by construction (same functions, same order).
// ---------------------------------------------------------------------------

std::vector<long long> sample_topp_batched_cpu(
    const std::vector<float>& logits, int rows, int n, float p, float t,
    const std::vector<unsigned long long>& seeds) {
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_topp_cpu(std::vector<float>(row, row + n), p,
                                      t, seeds[r]));
    }
    return out;
}

std::vector<long long> sample_topk_batched_cpu(
    const std::vector<float>& logits, int rows, int n, int k, float t,
    const std::vector<unsigned long long>& seeds) {
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_topk_cpu(std::vector<float>(row, row + n), k,
                                      t, seeds[r]));
    }
    return out;
}

std::vector<long long> sample_minp_batched_cpu(
    const std::vector<float>& logits, int rows, int n, float min_p,
    float t, const std::vector<unsigned long long>& seeds) {
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_minp_cpu(std::vector<float>(row, row + n),
                                      min_p, t, seeds[r]));
    }
    return out;
}

// Batched fused decode step (v1.5): the row-wise composition
// repetition_penalty -> sample_topp, exactly the single-row wrapper's
// CPU path (python/fusedtok/__init__.py composes the same two
// references), so per-row equality with decode_step on the CPU side
// is bit-exact by construction.
std::vector<long long> decode_step_batched_cpu(
    const std::vector<float>& logits, int rows, int n,
    const std::vector<long long>& ids,
    const std::vector<long long>& offs, float penalty, float p, float t,
    const std::vector<unsigned long long>& seeds) {
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    if ((int)offs.size() != rows + 1)
        throw std::invalid_argument(
            "sampled_ids offsets must have rows + 1 entries");
    if (offs.front() != 0 || offs.back() != (long long)ids.size())
        throw std::invalid_argument(
            "sampled_ids offsets must start at 0 and end at its length");
    for (size_t i = 1; i < offs.size(); ++i)
        if (offs[i] < offs[i - 1])
            throw std::invalid_argument(
                "sampled_ids offsets must be non-decreasing");
    if (!(penalty > 0.0f))
        throw std::invalid_argument("penalty must be > 0");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        std::vector<float> logits_row(row, row + n);
        if (!ids.empty())
            logits_row = repetition_penalty_cpu(
                logits_row,
                std::vector<long long>(
                    ids.begin() + offs[r], ids.begin() + offs[r + 1]),
                penalty);
        out.push_back(sample_topp_cpu(logits_row, p, t, seeds[r]));
    }
    return out;
}

} // namespace fusedtok
