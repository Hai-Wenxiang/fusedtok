// Top-k / top-p selection and greedy decoding helpers.
//
// Parallel selection via order-preserving packed keys:
//   key = (monotonic_float_bits << 32) | (0xFFFFFFFF - index)
// A single integer max therefore resolves BOTH the largest value and, on
// ties, the smallest index - deterministic like the naive version.
//
// Main path: a single cooperative-groups kernel runs all k selection
// rounds on device (scan -> warp/block reduce -> atomicMax -> grid.sync ->
// block 0 finalizes the winner and marks the bitmap), so there are no
// per-round host round-trips. Machines without cooperative launch fall
// back to a host-driven round loop with identical semantics.
//
// NaN inputs are not order-preserving here (undefined result), matching
// common library behavior.

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fusedtok {

namespace {

namespace cg = cooperative_groups;

void topk_check(const std::vector<float>& x, int k) {
    if (k < 0)
        throw std::invalid_argument("k must be >= 0");
    if (static_cast<size_t>(k) > x.size())
        throw std::invalid_argument("k must not exceed x.size()");
}

constexpr int kSelBlock = 256;
constexpr int kSelWarps = kSelBlock / 32;

// Grid-stride scan of one selection round; each warp reduces its best key
// and warp leaders atomicMax into the global slot. Used by the cooperative
// kernel body and the fallback round kernel alike.
__device__ __forceinline__ void select_scan_round(
    const float* __restrict__ x,
    const unsigned long long* __restrict__ taken,
    unsigned long long* best, int n, unsigned long long* warp_best) {
    unsigned long long local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (taken && ((taken[i >> 6] >> (i & 63)) & 1ULL)) continue;
        unsigned long long key =
            ((unsigned long long)fkey(x[i]) << 32) | (0xFFFFFFFFULL - (unsigned)i);
        if (key > local) local = key;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local = max(local, __shfl_down_sync(0xffffffffu, local, off));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_best[warp] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long b = warp_best[0];
        #pragma unroll
        for (int w = 1; w < kSelWarps; ++w) b = max(b, warp_best[w]);
        atomicMax(best, b);
    }
}

// Cooperative kernel: max_rounds selection rounds on device. When p_stop
// > 0 (nucleus mode), block 0 tracks the cumulative mass and sets
// stop_flag once it reaches p_stop; count_out then receives the number of
// selected elements. In top-k mode (p_stop == 0) exactly max_rounds
// elements are written and count_out is not touched (may be null).
__global__ void select_coop_kernel(const float* __restrict__ x,
                                   float* __restrict__ vals,
                                   long long* __restrict__ idxs,
                                   int* __restrict__ count_out,
                                   int n, int max_rounds, float p_stop,
                                   unsigned long long* __restrict__ best,
                                   unsigned long long* __restrict__ taken,
                                   int* __restrict__ stop_flag) {
    __shared__ unsigned long long warp_best[kSelWarps];
    cg::grid_group grid = cg::this_grid();

    // Initialization: block 0 zeroes the bitmap and the control slots;
    // a grid-wide barrier publishes them to every block.
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < (n + 63) / 64; i += blockDim.x)
            taken[i] = 0ULL;
        if (threadIdx.x == 0) {
            *best = 0ULL;
            *stop_flag = 0;
        }
    }
    grid.sync();

    float cumulative = 0.0f;   // maintained by block 0 thread 0 only
    int selected = 0;

    for (int round = 0; round < max_rounds; ++round) {
        if (*stop_flag) break;

        select_scan_round(x, taken, best, n, warp_best);
        grid.sync();

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            const unsigned long long key = *best;
            const int idx = (int)(0xFFFFFFFFu - (unsigned)(key & 0xFFFFFFFFu));
            const float value = unfkey((unsigned)(key >> 32));
            if (vals) vals[round] = value;
            if (idxs) idxs[round] = idx;
            taken[idx >> 6] |= 1ULL << (idx & 63);
            *best = 0ULL;
            ++selected;
            if (p_stop > 0.0f) {
                cumulative += value;
                if (cumulative >= p_stop) *stop_flag = 1;
            }
        }
        grid.sync();
    }

    if (count_out && blockIdx.x == 0 && threadIdx.x == 0)
        *count_out = selected;
}

// ---------------------------------------------------------------------------
// Fallback path for devices without cooperative launch: one plain kernel
// per round, host reads back the winner each round.
// ---------------------------------------------------------------------------

__global__ void select_round_kernel(const float* __restrict__ x,
                                    const unsigned long long* __restrict__ taken,
                                    unsigned long long* best, int n) {
    __shared__ unsigned long long warp_best[kSelWarps];
    unsigned long long local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (taken && ((taken[i >> 6] >> (i & 63)) & 1ULL)) continue;
        unsigned long long key =
            ((unsigned long long)fkey(x[i]) << 32) | (0xFFFFFFFFULL - (unsigned)i);
        if (key > local) local = key;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local = max(local, __shfl_down_sync(0xffffffffu, local, off));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_best[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = (threadIdx.x < kSelWarps) ? warp_best[lane] : 0ULL;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local = max(local, __shfl_down_sync(0xffffffffu, local, off));
        if (lane == 0) atomicMax(best, local);
    }
}

__global__ void mark_taken_kernel(unsigned long long* taken, int index) {
    atomicOr(&taken[index >> 6], 1ULL << (index & 63));
}

// Process-wide device scratch for the selection ops, grown on demand and
// never freed (bounded by the largest input seen; a 512k-entry vocabulary
// needs 64KB). Replacing per-call cudaMalloc/cudaFree matters because those
// calls synchronize the device and would serialize repeated invocations.
// Layout: [0] = best key slot, [1] = stop flag (int stored in 64 bits),
// [2..] = taken bitmap. Not safe for concurrent selection launches on
// different streams (documented limitation).
unsigned long long* selection_scratch(size_t bitmap_words) {
    static unsigned long long* buf = nullptr;
    static size_t capacity = 0;
    static std::mutex mu;
    const size_t words = bitmap_words + 2;
    std::lock_guard<std::mutex> lock(mu);
    if (words > capacity) {
        unsigned long long* nb = nullptr;
        if (cudaMalloc(&nb, words * sizeof(unsigned long long)) != cudaSuccess) {
            throw std::runtime_error(std::string("selection scratch alloc failed: ") +
                                     cudaGetErrorString(cudaGetLastError()));
        }
        if (buf) cudaFree(buf);
        buf = nb;
        capacity = words;
    }
    return buf;
}

// RAII-free view over the shared scratch for one selection run.
struct SelectScratch {
    unsigned long long* best = nullptr;
    unsigned long long* taken = nullptr;
    int* stop_flag = nullptr;
    size_t bitmap_words = 0;

    explicit SelectScratch(int n) : bitmap_words(((size_t)n + 63) / 64) {
        unsigned long long* base = selection_scratch(bitmap_words);
        best = base;
        stop_flag = reinterpret_cast<int*>(base + 1);
        taken = base + 2;
    }
};

bool coop_supported() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0, support = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&support, cudaDevAttrCooperativeLaunch, dev);
        cached = support;
    }
    return cached != 0;
}

void launch_cooperative(const float* x, float* vals, long long* idxs,
                        int* count_out, int n, int max_rounds, float p_stop,
                        SelectScratch& s) {
    int dev = 0, num_sms = 0, blocks_per_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, select_coop_kernel, kSelBlock, 0);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    const long long want = (n + kSelBlock - 1) / kSelBlock;
    const int grid = (int)std::min<long long>((long long)num_sms * blocks_per_sm, want);
    const float* x_ = x;
    float* vals_ = vals;
    long long* idxs_ = idxs;
    int* count_ = count_out;
    int n_ = n, rounds_ = max_rounds;
    float p_ = p_stop;
    unsigned long long* best_ = s.best;
    unsigned long long* taken_ = s.taken;
    int* stop_ = s.stop_flag;
    void* args[] = {&x_, &vals_, &idxs_, &count_, &n_, &rounds_, &p_,
                    &best_, &taken_, &stop_};
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)select_coop_kernel, dim3((unsigned)grid), dim3(kSelBlock), args, 0, nullptr);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cooperative selection launch failed: ") +
                                 cudaGetErrorString(err));
}

// Host-driven fallback rounds (semantics identical to the cooperative path).
int select_rounds_host(const float* dx, float* vals, long long* idxs,
                       int* count_out, int n, int max_rounds, float p_stop,
                       SelectScratch& s) {
    cudaMemset(s.taken, 0, s.bitmap_words * sizeof(unsigned long long));
    const int grid = (int)((n + kSelBlock - 1) / kSelBlock);
    float cumulative = 0.0f;
    int selected = 0;
    for (int sel = 0; sel < max_rounds; ++sel) {
        cudaMemset(s.best, 0, sizeof(unsigned long long));
        select_round_kernel<<<grid, kSelBlock>>>(dx, s.taken, s.best, n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("selection round launch failed: ") +
                                     cudaGetErrorString(err));
        unsigned long long key = 0;
        if (cudaMemcpy(&key, s.best, sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("selection winner readback failed");
        const int idx = (int)(0xFFFFFFFFu - (unsigned)(key & 0xFFFFFFFFu));
        unsigned int u = (unsigned)(key >> 32);
        unsigned int bits = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
        float value;
        std::memcpy(&value, &bits, sizeof(float));
        if (vals) vals[sel] = value;
        if (idxs) idxs[sel] = idx;
        mark_taken_kernel<<<1, 1>>>(s.taken, idx);
        ++selected;
        if (p_stop > 0.0f) {
            cumulative += value;
            if (cumulative >= p_stop) break;
        }
    }
    if (count_out)
        cudaMemcpy(count_out, &selected, sizeof(int), cudaMemcpyHostToDevice);
    return selected;
}

} // namespace

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
    if (k <= 0 || n <= 0) return;
    SelectScratch s(n);
    if (coop_supported()) {
        launch_cooperative(x, vals, idxs, nullptr, n, k, 0.0f, s);
    } else {
        select_rounds_host(x, vals, idxs, nullptr, n, k, 0.0f, s);
    }
}

std::pair<std::vector<float>, std::vector<long long>>
topp_cpu(const std::vector<float>& probs, float p) {
    if (!(p > 0.0f && p <= 1.0f))
        throw std::invalid_argument("p must be in (0, 1]");
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

// Early-exiting nucleus selection: rounds stop as soon as the cumulative
// mass reaches p, so cost scales with the nucleus size, not the vocabulary.
void topp_select_launch(const float* x, float* vals, long long* idxs,
                        int n, float p, int* count_out) {
    if (n <= 0) {
        if (count_out) {
            int zero = 0;
            cudaMemcpy(count_out, &zero, sizeof(int), cudaMemcpyHostToDevice);
        }
        return;
    }
    SelectScratch s(n);
    if (coop_supported()) {
        launch_cooperative(x, vals, idxs, count_out, n, n, p, s);
    } else {
        select_rounds_host(x, vals, idxs, count_out, n, n, p, s);
    }
}

// ---------------------------------------------------------------------------
// argmax / temperature
// ---------------------------------------------------------------------------

// Parallel greedy argmax in ONE kernel: blocks scan + atomicMax their best
// key, then the last block to arrive (tracked by an arrival counter in the
// shared scratch) decodes the winner index. Earliest index wins ties via
// the key layout.
__global__ void argmax_kernel(const float* __restrict__ x,
                              unsigned long long* __restrict__ best,
                              unsigned int* __restrict__ counter,
                              int n, int* __restrict__ out) {
    __shared__ unsigned long long warp_best[kSelWarps];
    select_scan_round(x, nullptr, best, n, warp_best);

    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned int arrived = atomicAdd(counter, 1u);
        if (arrived == gridDim.x - 1) {   // last block: publish the answer
            __threadfence();              // acquire: see every block's atomicMax
            *counter = 0u;                // reset for the next launch
            *out = (int)(0xFFFFFFFFu - (unsigned)((*best) & 0xFFFFFFFFu));
        }
    }
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
    SelectScratch s(n);   // best + counter live in the shared cached scratch
    // Zero the best-key slot and the arrival counter in one 16-byte clear;
    // the kernel resets the counter itself for subsequent launches.
    cudaMemsetAsync(s.best, 0, 2 * sizeof(unsigned long long));
    const int grid = (int)((n + kSelBlock - 1) / kSelBlock);
    argmax_kernel<<<grid, kSelBlock>>>(
        x, s.best, reinterpret_cast<unsigned int*>(s.best + 1), n, out);
    check_launch("argmax kernel launch");
}

} // namespace fusedtok
