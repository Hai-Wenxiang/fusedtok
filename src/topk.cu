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

}
