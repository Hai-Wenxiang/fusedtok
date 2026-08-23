// Top-k / top-p selection and greedy decoding helpers.
//
// Selection contract (identical across all paths):
//   - results sorted descending by value
//   - ties resolved toward the smallest index (deterministic)
//
// Mechanism: order-preserving packed 64-bit keys
//   key = (monotonic_float_bits << 32) | (0xFFFFFFFF - index)
// Keys are unique (index in the low bits) and their total order IS the
// required result order, so selecting and sorting keys yields both the
// ordering and the deterministic tie resolution for free.
//
// GPU main path - a single cooperative kernel per call:
//   phase 1: 8 rounds of 256-bin radix refinement over the packed keys
//            (byte 7 down to 0). Each round histograms one byte of the
//            candidates still matching the prefix, block 0 scans bins from
//            the top to find which bin contains the k-th largest key, and
//            the prefix tightens. After 8 rounds the k-th largest key
//            K_min is known exactly, plus `remaining` = how many keys equal
//            to K_min belong to the result.
//   phase 2: single emit scan. Keys > K_min take slots via an atomic
//            counter; keys == K_min fill the tail slots (their relative
//            order is restored by the sort below). Exactly k slots written.
//   phase 3: bitonic sort of the k keys. k <= 2048 uses a shared-memory
//            fast path inside one block; larger k (e.g. nucleus mode with
//            k = n) sorts in global memory with grid participation.
//   phase 4 (nucleus mode only): cumulative sum over the sorted values
//            until mass >= p, written to count_out.
//
// Workspace (histogram, counters, key buffer) is process-cached and grown
// on demand: per-call cudaMalloc/cudaFree would synchronize the device
// and serialize repeated invocations (and break CUDA graph capture).
//
// Devices without cooperative launch fall back to a host-driven per-round
// selection loop with identical semantics.
//
// NaN inputs are not order-preserving (undefined result), matching common
// library behavior.

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <climits>
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
// Per-block sort chunk (shared-memory bitonic): 2048 keys = 16KB shared.
// Powers of two keep the chunk-merge arithmetic exact.
constexpr int kSelSortChunk = 2048;
// per-block scan chunk for the parallel nucleus count
constexpr int kSelScanChunk = 2048;

// merge tile: one co-rank search + shared staging per this many outputs
constexpr int kSelMergeTile = 256;

// ---------------------------------------------------------------------------
// process-cached workspace
// ---------------------------------------------------------------------------

// Layout: [0..255] histogram, [256] emit counter, [257] tie counter,
// [258] radix prefix, [259] radix remaining, [260..] key buffer (grown to
// the largest padded size seen). Guarded by a mutex; never freed (bounded
// by the largest call). Not safe for concurrent selection launches on
// different streams (documented limitation).
unsigned long long* selection_workspace(size_t key_capacity) {
    static unsigned long long* buf = nullptr;
    static size_t capacity = 0;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    const size_t words = 260 + key_capacity;
    if (words > capacity) {
        unsigned long long* nb = nullptr;
        if (cudaMalloc(&nb, words * sizeof(unsigned long long)) != cudaSuccess)
            throw std::runtime_error(std::string("selection workspace alloc failed: ") +
                                     cudaGetErrorString(cudaGetLastError()));
        if (buf) cudaFree(buf);
        buf = nb;
        capacity = words;
    }
    return buf;
}

inline __device__ unsigned long long pack_key(float v, int i) {
    return ((unsigned long long)fkey(v) << 32) | (0xFFFFFFFFULL - (unsigned)i);
}

// Separate process cache for the fallback path's taken-bitmap: sized by n
// and independent of the key buffer, so the radix layout stays fixed.
unsigned long long* fallback_bitmap(size_t bitmap_words) {
    static unsigned long long* buf = nullptr;
    static size_t capacity = 0;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (bitmap_words > capacity) {
        unsigned long long* nb = nullptr;
        if (cudaMalloc(&nb, bitmap_words * sizeof(unsigned long long)) != cudaSuccess)
            throw std::runtime_error(std::string("fallback bitmap alloc failed: ") +
                                     cudaGetErrorString(cudaGetLastError()));
        if (buf) cudaFree(buf);
        buf = nb;
        capacity = bitmap_words;
    }
    return buf;
}

// ---------------------------------------------------------------------------
// cooperative radix-select + sort kernel
// ---------------------------------------------------------------------------

// Zero the workspace head (histogram + counters) and the key-buffer padding
// region [k, m) so bitonic sorting sees well-defined pad keys (0 = smallest).
//
// Modes driven by the trailing parameters:
//   selection (top-k / top-p): p_stop > 0 requests the nucleus count in
//     count_out; sample mode additionally samples a token inside the nucleus.
//   sample mode (sample_seed != 0): the input is treated as RAW LOGITS scaled
//     by inv_T (temperature); phase 4 converts to softmax probabilities,
//     finds the nucleus at mass p, then inverse-CDF samples within it using
//     a hash-derived uniform draw (deterministic per seed). The winning
//     token goes to token_out.
// Kernel argument bundle: passing ONE struct by pointer through the
// cooperative-launch args array sidesteps the runtime's per-parameter
// marshalling (which silently dropped the tail parameter for this
// 12-argument signature on CUDA 13 / Windows).
struct RadixArgs {
    const float* x;
    float* vals;
    long long* idxs;
    int* count_out;
    int* token_out;
    int n, k, m;
    float p_stop;
    float inv_t;
    unsigned long long sample_seed;
    int sample;                                  // 1 = sampling mode (seed may be 0)
    unsigned long long* ws;
};

// NOTE: count_out / token_out carry NO __restrict__ - they may point INTO
// the workspace (the sample path stores its token slot at ws+256), which
// aliases ws. Promising non-aliasing via __restrict__ would be undefined
// behavior.
__global__ void radix_topk_kernel(const RadixArgs a) {
    const int grid_blocks = gridDim.x;
    cg::grid_group grid = cg::this_grid();

    __shared__ unsigned long long sh_hist[256];    // per-block histogram stage
    unsigned long long* hist = a.ws;                 // 256 bins
    unsigned long long* emit_cnt = a.ws + 256;
    unsigned long long* tie_cnt = a.ws + 257;
    unsigned long long* g_prefix = a.ws + 258;       // refinement state in GLOBAL
    unsigned long long* g_remaining = a.ws + 259;    // memory: every block must
    unsigned long long* keys = a.ws + 260;           // see the same prefix

    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < 260; i += blockDim.x) a.ws[i] = 0ULL;
        if (threadIdx.x == 0) *g_remaining = (unsigned long long)a.k;
        // count_out is an external buffer: preset to INT_MAX so the
        // atomicMin-based nucleus count starts from a clean sentinel
        // (sample mode overwrites it directly in phase 4)
        if (threadIdx.x == 0 && a.count_out) *a.count_out = INT_MAX;
        // zero the pads of BOTH key buffers: merges read both, and every
        // real key is > 0 (index bits), so 0 pads always sink to the tail
        for (long long i = a.k + threadIdx.x; i < a.m; i += blockDim.x) {
            keys[i] = 0ULL;
            (a.ws + 260 + a.m)[i] = 0ULL;
        }
    }
    grid.sync();

    // ---- phase 1: 8-round radix refinement -----------------------------------
    for (int level = 7; level >= 0; --level) {
        const unsigned long long prefix = *g_prefix;    // stable until the scan
        const long long remaining = (long long)*g_remaining;
        // histogram one byte of every key whose bytes ABOVE this level
        // already match the prefix. At level 7 there are no higher bytes,
        // so every key participates (mask 0). For level < 7 the mask
        // covers bytes level+1 .. 7. In sample mode the key is built from
        // the temperature-scaled logit (identical ordering for T > 0).
        const unsigned long long topmask =
            (level == 7) ? 0ULL : ~((1ULL << (8 * (level + 1))) - 1ULL);
        // two-stage histogram: accumulate into per-block SHARED bins, then
        // merge once per block - a global atomic per element serializes on
        // hot bins, this keeps the hot loop shared-memory-only
        for (int b = threadIdx.x; b < 256; b += kSelBlock) sh_hist[b] = 0ULL;
        __syncthreads();
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.n;
             i += grid_blocks * blockDim.x) {
            const unsigned long long key = pack_key(a.x[i] * a.inv_t, i);
            if ((key & topmask) != prefix) continue;
            atomicAdd(&sh_hist[(key >> (8 * level)) & 0xFF], 1ULL);
        }
        __syncthreads();
        for (int b = threadIdx.x; b < 256; b += kSelBlock)
            if (sh_hist[b]) atomicAdd(&hist[b], sh_hist[b]);
        grid.sync();

        // block 0 thread 0 scans bins top-down to locate the boundary bin
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            unsigned long long acc = 0;
            for (int b = 255; b >= 0; --b) {
                const unsigned long long c = hist[b];
                if (acc + c >= (unsigned long long)remaining) {
                    *g_prefix = prefix | ((unsigned long long)b << (8 * level));
                    *g_remaining = (unsigned long long)remaining - acc;
                    break;
                }
                acc += c;
            }
        }
        grid.sync();                               // publish the new prefix

        // clear histogram for the next level
        if (blockIdx.x == 0) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0ULL;
        }
        grid.sync();
    }
    const unsigned long long k_min = *g_prefix;    // a.k-th largest key
    const long long tie_take = (long long)*g_remaining;  // keys == k_min to include

    // ---- phase 2: emit selected keys -----------------------------------------
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.n;
         i += grid_blocks * blockDim.x) {
        const unsigned long long key = pack_key(a.x[i] * a.inv_t, i);
        if (key > k_min) {
            const unsigned long long pos = atomicAdd(emit_cnt, 1ULL);
            if (pos < (unsigned long long)a.k) keys[pos] = key;
        } else if (key == k_min && tie_take > 0) {
            const unsigned long long pos = atomicAdd(tie_cnt, 1ULL);
            if (pos < (unsigned long long)tie_take)
                keys[(unsigned long long)a.k - tie_take + pos] = key;
        }
    }
    grid.sync();

    // ---- phase 3: parallel chunk sort + pairwise merge ------------------------
    // v0.3: replaces the global bitonic (O(log^2 m) grid.sync barriers, most
    // blocks idle at high strides) with two structured stages:
    //   3a) every block independently sorts one kSelSortChunk-key chunk in
    //       shared memory (no grid participation inside the sort)
    //   3b) log2(nb) pairwise merge levels; each output element finds its
    //       source via a merge-path co-rank binary search over the two runs
    //       (arrays are strictly descending with unique keys, so the search
    //       predicate is monotone). Buffers ping-pong between the primary
    //       key array and the scratch array; pads are 0 and every real key
    //       is > 0 (index bits), so zeros always merge to the tail.
    {
        __shared__ unsigned long long sk[kSelSortChunk];
        __shared__ unsigned long long sA[kSelMergeTile];
        __shared__ unsigned long long sB[kSelMergeTile];
        unsigned long long* keys_a = keys;                 // primary buffer
        unsigned long long* keys_b = a.ws + 260 + a.m;       // scratch buffer

        // 3a) per-block chunk sort (blocks beyond nb_sort only wait)
        const int c = kSelSortChunk;
        const int nb_sort = (a.m + c - 1) / c;
        // blocks may FEWER than chunks (grid is capped by cooperative-launch
        // occupancy): stride blocks over chunks so every chunk gets sorted
        for (int chunk = blockIdx.x; chunk < nb_sort; chunk += grid_blocks) {
            const int base = chunk * c;
            // one chunk covers all of m -> sort exactly m (a power of two);
            // running the full 2048 network on tiny k wastes ~4x passes
            const int len = (nb_sort == 1) ? a.m : c;
            for (int i = threadIdx.x; i < len; i += kSelBlock)
                sk[i] = keys[base + i];
            __syncthreads();
            for (int size = 2; size <= len; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    __syncthreads();
                    for (int i = threadIdx.x; i < len; i += kSelBlock) {
                        const int j = i ^ stride;
                        if (j > i && j < len) {
                            const bool up = (i & size) == 0;   // descending
                            const unsigned long long v0 = sk[i];
                            const unsigned long long v1 = sk[j];
                            if ((up && v0 < v1) || (!up && v0 > v1)) {
                                sk[i] = v1;
                                sk[j] = v0;
                            }
                        }
                    }
                }
            }
            __syncthreads();
            for (int i = threadIdx.x; i < len; i += kSelBlock)
                if (base + i < a.m) keys[base + i] = sk[i];
            __syncthreads();   // sk reused by the next chunk of this block
        }
        grid.sync();

        // 3b) pairwise merges, tile-based merge path: each block handles
        // tiles of kSelMergeTile outputs. ONE global-memory co-rank binary
        // search per tile locates the (i, j) split of the two source runs;
        // the tile's inputs are staged into shared memory and each thread
        // resolves its element with a cheap in-shared search. Global reads
        // stay coalesced and per-element binary search over global memory
        // (the v0.3 first cut, 3x slower than bitonic at 131k) is avoided.
        unsigned long long* src_buf = keys_a;
        unsigned long long* dst_buf = keys_b;
        int run = c;
        while (run < a.m) {
            grid.sync();   // all reads target the previous level's output
            const long long ntiles = a.m / kSelMergeTile;
            for (long long tile = blockIdx.x; tile < ntiles;
                 tile += grid_blocks) {
                const long long p0 = tile * kSelMergeTile;
                const long long pair = p0 / (2 * run);
                const long long lo0 = p0 - pair * 2 * run;
                const long long a_base = pair * 2 * run;
                const long long b_base = a_base + run;
                // co-rank for the tile start: largest i with
                // (i == 0 || A[i-1] > B[lo0-i])
                long long l = lo0 > run ? lo0 - run : 0;
                long long h = lo0 < run ? lo0 : run;
                while (l < h) {
                    const long long mid = (l + h + 1) >> 1;
                    const bool pred =
                        mid == 0 ||
                        src_buf[a_base + mid - 1] > src_buf[b_base + lo0 - mid];
                    if (pred) l = mid; else h = mid - 1;
                }
                const long long i0 = l, j0 = lo0 - i0;
                // stage up to kSelMergeTile inputs from each run (0-padded;
                // real keys are > 0 so pads sort to the tail)
                for (int u = threadIdx.x; u < kSelMergeTile; u += kSelBlock) {
                    sA[u] = (i0 + u < run) ? src_buf[a_base + i0 + u] : 0ULL;
                    sB[u] = (j0 + u < run) ? src_buf[b_base + j0 + u] : 0ULL;
                }
                __syncthreads();
                // in-tile co-rank per thread (shared memory, log T steps)
                for (int u = threadIdx.x; u < kSelMergeTile; u += kSelBlock) {
                    long long l2 = u > kSelMergeTile ? 0 : 0;   // clamp below
                    l2 = 0;
                    long long h2 = u < kSelMergeTile ? u : kSelMergeTile;
                    while (l2 < h2) {
                        const long long mid = (l2 + h2 + 1) >> 1;
                        const bool pred =
                            mid == 0 || sA[mid - 1] > sB[u - mid];
                        if (pred) l2 = mid; else h2 = mid - 1;
                    }
                    const long long i = l2, j = u - i;
                    const bool take_a =
                        (i < kSelMergeTile) &&
                        (j >= kSelMergeTile || sA[i] > sB[j]);
                    dst_buf[p0 + u] = take_a ? sA[i] : sB[j];
                }
                __syncthreads();
            }
            unsigned long long* tmp = src_buf; src_buf = dst_buf; dst_buf = tmp;
            run <<= 1;
        }
        grid.sync();

        // 3c) ensure the sorted result lives in the primary buffer (decode
        // below reads `keys`); an odd number of merge levels ends in scratch
        if (src_buf != keys_a) {
            for (int i = blockIdx.x * kSelBlock + threadIdx.x; i < a.k;
                 i += grid_blocks * kSelBlock)
                keys[i] = keys_b[i];
            grid.sync();
        }
    }
    // ---- phase 4: decode / nucleus count / sample ------------------------------
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < a.k; i += blockDim.x) {
            const unsigned long long key = keys[i];
            const int idx = (int)(0xFFFFFFFFu - (unsigned)(key & 0xFFFFFFFFu));
            if (a.vals) a.vals[i] = unfkey((unsigned)(key >> 32));
            if (a.idxs) a.idxs[i] = idx;
        }
        if (a.sample != 0 && threadIdx.x == 0) {
            // fused nucleus sampling over the sorted keys:
            //   probs = softmax(logits * a.inv_t); keys are already sorted by
            //   logit, and the running softmax numerator exp(v - a.m) shares
            //   the row max a.m = key[0]'s value, so probabilities come out as
            //   plain exp of the gap. Pass 1 totals the mass and the nucleus
            //   size (cum >= a.p_stop); pass 2 inverse-CDFs a hash-uniform
            //   draw scaled to the nucleus mass.
            const float row_max = unfkey((unsigned)(keys[0] >> 32));
            float total = 0.0f;
            for (int i = 0; i < a.k; ++i)
                total += __expf(unfkey((unsigned)(keys[i] >> 32)) - row_max);
            float cum = 0.0f;
            int nucleus = 0;
            float nucleus_mass = 0.0f;
            bool covered = false;
            for (int i = 0; i < a.k; ++i) {
                cum += __expf(unfkey((unsigned)(keys[i] >> 32)) - row_max);
                nucleus = i + 1;
                if (cum >= a.p_stop * total) { nucleus_mass = cum; covered = true; break; }
            }
            if (!covered) return;   // window too small; host retries wider
            // splitmix64-finalized uniform in [0, 1): deterministic per seed
            unsigned long long z = a.sample_seed + 0x9E3779B97F4A7C15ULL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z ^= z >> 31;
            const float u = (float)((z >> 11) * (1.0 / 9007199254740992.0));
            float target = u * nucleus_mass;
            cum = 0.0f;
            for (int i = 0; i < nucleus; ++i) {
                cum += __expf(unfkey((unsigned)(keys[i] >> 32)) - row_max);
                if (cum >= target) {
                    *a.token_out = (int)(0xFFFFFFFFu -
                                       (unsigned)(keys[i] & 0xFFFFFFFFu));
                    break;
                }
            }
            if (a.count_out) *a.count_out = nucleus;
        }
    }

    // ---- phase 4b: parallel nucleus count (ALL blocks participate) ------
    // Must live OUTSIDE the block-0-only decode scope: the grid scan calls
    // grid.sync(), which deadlocks unless every block reaches it.
    if (a.sample == 0 && a.p_stop > 0.0f) {
        float* bsums = reinterpret_cast<float*>(a.ws + 260 + 2 * a.m);
        const int scan_len = a.k;
        const long long per =
            (scan_len + (long long)grid_blocks - 1) / grid_blocks;
        const long long b0 = (long long)blockIdx.x * per;
        const long long b1 = min((long long)scan_len, b0 + per);

        float bsum = 0.0f;
        for (long long i = b0 + threadIdx.x; i < b1; i += kSelBlock)
            bsum += unfkey((unsigned)(keys[i] >> 32));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            bsum += __shfl_down_sync(0xffffffffu, bsum, off);
        __shared__ float sh_warp[kSelWarps];
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) sh_warp[warp] = bsum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < kSelWarps; ++w) s += sh_warp[w];
            bsums[blockIdx.x] = s;
        }
        grid.sync();

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            float acc = 0.0f;
            for (int b = 0; b < (int)grid_blocks; ++b) {
                const float s = bsums[b];
                bsums[b] = acc;
                acc += s;
            }
        }
        grid.sync();

        const float carry = bsums[blockIdx.x];
        const int cnt = (int)(b1 - b0);
        __shared__ float loc[2][kSelScanChunk];
        __shared__ int sh_first;
        if (threadIdx.x == 0) sh_first = INT_MAX;
        __syncthreads();
        if (cnt > kSelScanChunk) {
            // only when the grid is tiny; serial fallback stays correct.
            // A block whose slice never crosses p must NOT publish its
            // end-of-slice index - only report an actual crossing.
            if (threadIdx.x == 0) {
                float cum = carry;
                bool found = false;
                for (int i = 0; i < cnt; ++i) {
                    cum += unfkey((unsigned)(keys[b0 + i] >> 32));
                    if (cum >= a.p_stop) {
                        atomicMin(a.count_out, (int)b0 + i + 1);
                        found = true;
                        break;
                    }
                }
                (void)found;
            }
        } else if (cnt > 0) {
            // double-buffered Hillis-Steele: within one round, address x is
            // WRITTEN (as dst[i]) and READ (as src[i-stride]) by different
            // threads - a single shared array races (verified by
            // compute-sanitizer racecheck). Ping-pong between loc0/loc1.
            float* cur = loc[0];
            float* nxt = loc[1];
            for (int i = threadIdx.x; i < cnt; i += kSelBlock)
                cur[i] = unfkey((unsigned)(keys[b0 + i] >> 32));
            __syncthreads();
            for (int stride = 1; stride < cnt; stride <<= 1) {
                for (int i = threadIdx.x; i < cnt; i += kSelBlock)
                    nxt[i] = cur[i] + ((i >= stride) ? cur[i - stride] : 0.0f);
                __syncthreads();
                float* tmp = cur; cur = nxt; nxt = tmp;
            }
            for (int i = threadIdx.x; i < cnt; i += kSelBlock) {
                if (carry + cur[i] >= a.p_stop) {
                    atomicMin(&sh_first, (int)(b0 + i));
                    break;   // later i in this thread are larger
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && sh_first != INT_MAX)
                atomicMin(a.count_out, sh_first + 1);
        }
        grid.sync();
        // no crossing anywhere (p > total mass): the nucleus is everything
        if (blockIdx.x == 0 && threadIdx.x == 0 && *a.count_out == INT_MAX)
            *a.count_out = scan_len;
    }
}

// ---------------------------------------------------------------------------
// fallback path for devices without cooperative launch: one plain kernel
// per round, host reads back the winner each round. Kept from v0.1.
// ---------------------------------------------------------------------------

__global__ void select_round_kernel(const float* __restrict__ x,
                                    const unsigned long long* __restrict__ taken,
                                    unsigned long long* best, int n) {
    __shared__ unsigned long long warp_best[kSelWarps];
    unsigned long long local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (taken && ((taken[i >> 6] >> (i & 63)) & 1ULL)) continue;
        unsigned long long key = pack_key(x[i], i);
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

int select_rounds_host(const float* dx, float* vals, long long* idxs,
                       int* count_out, int n, int max_rounds, float p_stop,
                       unsigned long long* best, unsigned long long* taken,
                       size_t bitmap_words) {
    cudaMemset(taken, 0, bitmap_words * sizeof(unsigned long long));
    const int grid = (int)((n + kSelBlock - 1) / kSelBlock);
    float cumulative = 0.0f;
    int selected = 0;
    for (int sel = 0; sel < max_rounds; ++sel) {
        cudaMemset(best, 0, sizeof(unsigned long long));
        select_round_kernel<<<grid, kSelBlock>>>(dx, taken, best, n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("selection round launch failed: ") +
                                     cudaGetErrorString(err));
        unsigned long long key = 0;
        if (cudaMemcpy(&key, best, sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("selection winner readback failed");
        const int idx = (int)(0xFFFFFFFFu - (unsigned)(key & 0xFFFFFFFFu));
        // host-side mirror of unfkey(): reconstruct the float from the
        // monotone-mapped bits
        const unsigned int u = (unsigned)(key >> 32);
        const unsigned int bits = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
        float value;
        std::memcpy(&value, &bits, sizeof(float));
        if (vals) vals[sel] = value;
        if (idxs) idxs[sel] = idx;
        mark_taken_kernel<<<1, 1>>>(taken, idx);
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

void launch_radix_topk(const float* x, float* vals, long long* idxs,
                       int* count_out, int* token_out,
                       int n, int k, float p_stop, float inv_t,
                       unsigned long long sample_seed, int sample) {
    int m = 1;
    while (m < k) m <<= 1;                        // bitonic pad size
    // two m-word key buffers (ping-pong) + 256 words of block-sum scratch
    // for the parallel nucleus count (writes past 2*m otherwise)
    unsigned long long* ws = selection_workspace(2 * (size_t)m + 256);

    int dev = 0, num_sms = 0, blocks_per_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, radix_topk_kernel, kSelBlock, 0);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    const long long want = (n + kSelBlock - 1) / kSelBlock;
    const int grid = (int)std::min<long long>((long long)num_sms * blocks_per_sm,
                                              std::max<long long>(want, 1));

    RadixArgs args{x, vals, idxs, count_out, token_out,
                   n, k, m, p_stop, inv_t, sample_seed, sample, ws};
    void* arg_ptrs[] = {&args};
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)radix_topk_kernel, dim3((unsigned)grid), dim3(kSelBlock),
        arg_ptrs, 0, nullptr);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("radix selection launch failed: ") +
                                 cudaGetErrorString(err));
}

void select_common(const float* x, float* vals, long long* idxs,
                   int* count_out, int n, int k, float p_stop) {
    if (coop_supported()) {
        launch_radix_topk(x, vals, idxs, count_out, nullptr, n, k, p_stop,
                          1.0f, 0ULL, /*sample=*/0);
    } else {
        const size_t bitmap_words = ((size_t)n + 63) / 64;
        unsigned long long* ws = selection_workspace(0);
        select_rounds_host(x, vals, idxs, count_out, n, k, p_stop,
                           ws + 256, fallback_bitmap(bitmap_words),
                           bitmap_words);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// public entry points (signatures unchanged from v0.1)
// ---------------------------------------------------------------------------

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
    select_common(x, vals, idxs, nullptr, n, k, 0.0f);
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

void topp_select_launch(const float* x, float* vals, long long* idxs,
                        int n, float p, int* count_out) {
    if (n <= 0) {
        if (count_out) {
            int zero = 0;
            cudaMemcpy(count_out, &zero, sizeof(int), cudaMemcpyHostToDevice);
        }
        return;
    }
    select_common(x, vals, idxs, count_out, n, n, p);
}

// ---------------------------------------------------------------------------
// fused nucleus sampling: softmax(logits/T) -> nucleus(p) -> inverse-CDF
// draw from a hash-derived uniform, all inside the cooperative kernel.
// Returns the sampled token via a one-int host readback.
// (CPU reference: sample_topp_cpu in sampling.cu, defined in activations.hpp)
// ---------------------------------------------------------------------------

long long sample_topp_launch(const float* x, int n, float p, float t,
                             unsigned long long seed) {
    if (n <= 0)
        throw std::invalid_argument("sample of empty logits");
    if (!(p > 0.0f && p <= 1.0f))
        throw std::invalid_argument("p must be in (0, 1]");
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");
    if (!coop_supported()) {
        // rare: device without cooperative launch - do the whole thing on
        // the host (readback + CPU reference) so the API still works
        std::vector<float> host(n);
        if (cudaMemcpy(host.data(), x, n * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("sample fallback readback failed");
        return sample_topp_cpu(host, p, t, seed);
    }
    // Widening-window strategy: sort only the top-M candidates (shared-
    // memory bitonic below 2048-class sizes, no global-sort grid.sync
    // storm), sample within them; if the nucleus is not covered by the
    // window, retry with 4x as many candidates. Typical distributions are
    // covered by the first window.
    int window = 4096;
    if (window > n) window = n;
    for (;;) {
        int m = 1;
        while (m < window) m <<= 1;
        // Grow the workspace to the FULL launch size (two m-word key
        // buffers + block-sum scratch) BEFORE grabbing the token slot: the
        // launch would grow it otherwise and free the old buffer below it.
        unsigned long long* ws = selection_workspace(2 * (size_t)m + 256);
        int* token_out = reinterpret_cast<int*>(ws + 256);
        int token = -1;
        cudaMemcpy(token_out, &token, sizeof(int), cudaMemcpyHostToDevice);
        launch_radix_topk(x, nullptr, nullptr, nullptr, token_out, n, window,
                          p, 1.0f / t, seed, /*sample=*/1);
        cudaError_t err = cudaDeviceSynchronize();   // surface kernel faults
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("sample kernel failed: ") +
                                     cudaGetErrorString(err));
        if (cudaMemcpy(&token, token_out, sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("sample result readback failed");
        if (token >= 0)
            return token;                  // nucleus covered, token sampled
        if (window == n)
            throw std::runtime_error("sample nucleus not covered");
        window = std::min(n, window * 4);  // widen and retry
    }
}

// ---------------------------------------------------------------------------
// argmax: single plain kernel (packed-key max + arrival-counter finalize)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void argmax_scan(
    const float* __restrict__ x,
    unsigned long long* __restrict__ best, int n,
    unsigned long long* warp_best) {
    unsigned long long local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        unsigned long long key = pack_key(x[i], i);
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

__global__ void argmax_kernel(const float* __restrict__ x,
                              unsigned long long* __restrict__ best,
                              unsigned int* __restrict__ counter,
                              int n, int* __restrict__ out) {
    __shared__ unsigned long long warp_best[kSelWarps];
    argmax_scan(x, best, n, warp_best);

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
    unsigned long long* ws = selection_workspace(0);
    unsigned long long* best = ws + 256;           // counter slot doubles as reset
    // Zero the best-key slot and the arrival counter in one 16-byte clear;
    // the kernel resets the counter itself for subsequent launches.
    cudaMemsetAsync(best, 0, 2 * sizeof(unsigned long long));
    const int grid = (int)((n + kSelBlock - 1) / kSelBlock);
    argmax_kernel<<<grid, kSelBlock>>>(
        x, best, reinterpret_cast<unsigned int*>(best + 1), n, out);
    check_launch("argmax kernel launch");
}

} // namespace fusedtok
