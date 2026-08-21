#include "fusedtok/activations.hpp"

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
__global__ void topk_kernel(const float* x, float* vals, int* idxs, int n, int k) {
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

std::pair<std::vector<float>, std::vector<int>> topk_cpu(const std::vector<float>& x, int k) {
    topk_check(x, k);
    std::vector<float> vals;
    std::vector<int> idxs;
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

std::pair<std::vector<float>, std::vector<int>> topk_cuda(const std::vector<float>& x, int k) {
    topk_check(x, k);
    std::vector<float> vals(k);
    std::vector<int> idxs(k);
    if (k == 0) return {{}, {}};

    const int n = static_cast<int>(x.size());
    float *dx = nullptr, *dv = nullptr;
    int* di = nullptr;
    if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc x failed");
    if (cudaMalloc(&dv, k * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc vals failed");
    if (cudaMalloc(&di, k * sizeof(int)) != cudaSuccess) throw std::runtime_error("cudaMalloc idxs failed");
    if (cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D x failed");

    topk_kernel<<<1, 1>>>(dx, dv, di, n, k);

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("topk kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));
    if (cudaMemcpy(vals.data(), dv, k * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H vals failed");
    if (cudaMemcpy(idxs.data(), di, k * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H idxs failed");

    cudaFree(dx); cudaFree(dv); cudaFree(di);
    return {std::move(vals), std::move(idxs)};
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
__global__ void topp_count_kernel(const float* sorted_vals, int n, float p, int* out_count) {
    float cum = 0.0f;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        cum += sorted_vals[i];
        count = i + 1;
        if (cum >= p) break;
    }
    out_count[0] = count;
}

std::pair<std::vector<float>, std::vector<int>> topp_cpu(const std::vector<float>& probs, float p) {
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

std::pair<std::vector<float>, std::vector<int>> topp_cuda(const std::vector<float>& probs, float p) {
    topp_check(probs, p);
    const int n = static_cast<int>(probs.size());
    // Full sort via the naive top-k, then count the nucleus prefix on device
    auto [d_vals, d_idxs] = topk_cuda(probs, n);
    if (n == 0) return {{}, {}};

    float* dsorted = nullptr;
    int* dcount = nullptr;
    int count[1] = {0};
    if (cudaMalloc(&dsorted, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc sorted failed");
    if (cudaMalloc(&dcount, sizeof(int)) != cudaSuccess) throw std::runtime_error("cudaMalloc count failed");
    if (cudaMemcpy(dsorted, d_vals.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D sorted failed");

    topp_count_kernel<<<1, 1>>>(dsorted, n, p, dcount);

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("topp kernel failed: " + std::string(cudaGetErrorString(cudaGetLastError())));
    if (cudaMemcpy(count, dcount, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H count failed");

    cudaFree(dsorted); cudaFree(dcount);
    d_vals.resize(count[0]);
    d_idxs.resize(count[0]);
    return {std::move(d_vals), std::move(d_idxs)};
}

}
