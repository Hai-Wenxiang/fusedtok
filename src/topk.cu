// Top-k / top-p selection and greedy decoding helpers.
//
// All selection kernels are deliberately naive (single-thread selection
// loops); they establish an honest, deterministic baseline. Indices are
// int64 end-to-end so results map directly onto torch tensors.

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

namespace fusedtok {

namespace {

void topk_check(const std::vector<float>& x, int k) {
    if (k < 0)
        throw std::invalid_argument("k must be >= 0");
    if (static_cast<size_t>(k) > x.size())
        throw std::invalid_argument("k must not exceed x.size()");
}

} // namespace

// Maximally naive GPU top-k: a SINGLE thread runs the same k-pass selection
// as the CPU reference. Zero parallelism on purpose - the point is a correct
// GPU-side baseline to compare fancier selection kernels against later.
__global__ void topk_kernel(const float* x, float* vals, long long* idxs,
                            int n, int k) {
    for (int sel = 0; sel < k; ++sel) {
        int best = -1;
        for (int i = 0; i < n; ++i) {
            // 'taken' is encoded in idxs[0..sel-1]; skip those indices
            bool used = false;
            for (int s = 0; s < sel; ++s)
                if (idxs[s] == i) { used = true; break; }
            if (used) continue;
            // strict '>' keeps the earliest index on ties (deterministic)
            if (best == -1 || x[i] > x[best]) best = i;
        }
        idxs[sel] = best;
        vals[sel] = x[best];
    }
}

std::pair<std::vector<float>, std::vector<long long>>
topk_cpu(const std::vector<float>& x, int k) {
    topk_check(x, k);
    std::vector<float> vals;
    std::vector<long long> idxs;
    std::vector<char> taken(x.size(), 0);
    for (int sel = 0; sel < k; ++sel) {
        int best = -1;
        for (size_t i = 0; i < x.size(); ++i) {
            if (taken[i]) continue;
            if (best == -1 || x[i] > x[best]) best = (int)i;
        }
        taken[best] = 1;
        idxs.push_back(best);
        vals.push_back(x[best]);
    }
    return {std::move(vals), std::move(idxs)};
}

void topk_launch(const float* x, float* vals, long long* idxs, int n, int k) {
    if (k <= 0) return;
    topk_kernel<<<1, 1>>>(x, vals, idxs, n, k);
    check_launch("topk kernel launch");
}

// ---------------------------------------------------------------------------
// top-p (nucleus) selection
// ---------------------------------------------------------------------------

namespace {

void topp_check(const std::vector<float>& probs, float p) {
    if (!(p > 0.0f && p <= 1.0f))
        throw std::invalid_argument("p must be in (0, 1]");
}

} // namespace

// Single thread computes, over the descending-sorted values, how many
// elements are needed for the cumulative sum to reach p (inclusive of the
// crossing element). Writes the count to out_count[0].
__global__ void topp_count_kernel(const float* sorted_vals, int n, float p,
                                  int* out_count) {
    float cum = 0.0f;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        cum += sorted_vals[i];
        count = i + 1;
        if (cum >= p) break;
    }
    out_count[0] = count;
}

std::pair<std::vector<float>, std::vector<long long>>
topp_cpu(const std::vector<float>& probs, float p) {
    topp_check(probs, p);
    auto [vals, idxs] = topk_cpu(probs, static_cast<int>(probs.size()));
    float cum = 0.0f;
    size_t keep = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
        cum += vals[i];
        keep = i + 1;
        if (cum >= p) break;
    }
    vals.resize(keep);
    idxs.resize(keep);
    return {std::move(vals), std::move(idxs)};
}

void topp_count_launch(const float* sorted_vals, int n, float p, int* out_count) {
    if (n <= 0) {
        out_count[0] = 0;
        return;
    }
    topp_count_kernel<<<1, 1>>>(sorted_vals, n, p, out_count);
    check_launch("topp_count kernel launch");
}

// ---------------------------------------------------------------------------
// argmax / temperature
// ---------------------------------------------------------------------------

// Single thread performs a linear argmax scan - the greedy decoding op.
// Strict '>' keeps the earliest index on ties (deterministic, matches CPU).
__global__ void argmax_kernel(const float* x, int n, int* out) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    out[0] = best;
}

long long argmax_cpu(const std::vector<float>& x) {
    if (x.empty())
        throw std::invalid_argument("argmax of empty vector");
    long long best = 0;
    for (size_t i = 1; i < x.size(); ++i)
        if (x[i] > x[best]) best = (long long)i;
    return best;
}

void argmax_launch(const float* x, int n, int* out) {
    if (n <= 0) return;
    argmax_kernel<<<1, 1>>>(x, n, out);
    check_launch("argmax kernel launch");
}

// temperature_launch is implemented in activations.cu.

} // namespace fusedtok
