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
// GPU path - a pipeline of PLAIN kernels on the caller's stream (v0.4).
// The v0.1-v0.3 single cooperative kernel synchronized every radix round
// with grid.sync() barriers (24 per call); on many-SM GPUs the barrier
// storms dominated the runtime (bare topk 0.31x vs torch's CUB on a
// 36-SM part). The v0.4 pipeline removes cooperative launch entirely:
//
//   stage 1 - radix refinement: one small kernel per byte level (7..0).
//     Every block histograms its share of the surviving candidates into
//     shared memory, merges the nonzero bins into a global histogram
//     (device atomics), then bumps an arrival ticket (release fence
//     before the ticket). The LAST block to arrive - and only it - scans
//     the 256 bins top-down, tightens the prefix, and resets the
//     histogram + ticket for the next launch. Inter-kernel visibility
//     comes from stream ordering; intra-kernel cross-block visibility
//     comes from the fence+ticket release/acquire pair. No block ever
//     waits: the decider is simply the last one to arrive.
//   stage 2 - early exit: when a round's boundary bin holds at most
//     kSelEarlyOut candidates, the next launch skips histogramming and
//     instead COMPACTS the survivors into a small buffer; its last block
//     bitonic-sorts them in shared memory and publishes the final k-th
//     key directly. Remaining round launches self-disable (cheap no-ops).
//     Typical spread data exits after two rounds regardless of k.
//   stage 3 - emit: a parallel pass writes key > k_min into result slots
//     via an atomic counter (keys are unique, so exactly one key equals
//     k_min; it fills the last slot through the tie counter).
//   stage 4a - k <= kSelEarlyOut: the emit kernel's last block sorts the
//     k keys in shared memory, decodes values/indices, and (in nucleus /
//     sampling mode) runs the serial mass scan - one launch finishes.
//   stage 4b - k > kSelEarlyOut: chunk sort (per-block kSelSortChunk-key
//     shared bitonic) + a merge ladder with ONE plain launch per level
//     (merge-path co-rank tiles), then an elementwise decode kernel and,
//     for top-p, a two-kernel parallel nucleus count.
//
// Workspace (histogram, tickets, counters, key buffers) is process-cached
// and grown on demand: per-call cudaMalloc/cudaFree would synchronize the
// device and serialize repeated invocations (and break CUDA graph
// capture). A fixed kWsHead-word head is zeroed with one async memset per
// call; all later state flows through that region.
//
// Devices without cooperative launch: no longer special-cased - plain
// launches run everywhere (the v0.1 host-rounds fallback is gone).
//
// NaN inputs are not order-preserving (undefined result), matching common
// library behavior.

#include "fusedtok/activations.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace fusedtok {

namespace {

void topk_check(const std::vector<float>& x, int k) {
    if (k < 0)
        throw std::invalid_argument("k must be >= 0");
    if (static_cast<size_t>(k) > x.size())
        throw std::invalid_argument("k must not exceed x.size()");
}

constexpr int kSelBlock = 256;
constexpr int kSelWarps = kSelBlock / 32;
// Per-block sort chunk (shared-memory bitonic): 1024 keys = 8KB shared.
// Powers of two keep the chunk-merge arithmetic exact. 1024 (v1.0, was
// 2048): the sort is the mid-k bottleneck and a 2048-key bitonic runs in
// ONE block - halving the chunk halves its serial span and buys one
// parallel merge level instead (the ladder costs ~2us per level inside
// the cached graph; the single-block sort cost ~20us at k=2048).
constexpr int kSelSortChunk = 1024;
// Early-exit threshold: a radix boundary bin with at most this many
// survivors is resolved by an in-block sort instead of further rounds.
// 1024 (v1.0, was 2048): a single block bitonic-sorting 2048 keys keeps
// every SM but one idle, while the parallel chunk+merge tail spreads the
// same work over the whole device - measured as the entire mid-k
// regression window on both test GPUs.
constexpr int kSelEarlyOut = 1024;
// merge tile: one co-rank search + shared staging per this many outputs
constexpr int kSelMergeTile = 256;
// grid cap for the pipelined kernels (ticket scans stay short)
constexpr int kMaxGrid = 1024;
// Nucleus-scan scratch reserved between the key buffers and the args
// block: topp_partial_kernel writes ONE float per block and its grid is
// capped at kMaxGrid blocks, so the reserve must hold kMaxGrid floats
// (kMaxGrid/2 u64 words). The old 256-word reserve (512 floats) was
// exactly enough for n = 512*256 = 131072 - the suite's largest vocab -
// and vocabularies beyond that (e.g. Qwen's 152064) overflowed into the
// SelArgs block and corrupted the output pointers (fixed in 1.2.1).
constexpr int kWsScanWords =
    kMaxGrid * (int)sizeof(float) / (int)sizeof(unsigned long long);

// Workspace head layout (unsigned long long words). The head is zeroed
// by one async memset at the start of every call; the tail (candidates +
// key buffers + scan scratch) is rewritten before every read.
constexpr int kWsTicket = 256;      // arrival ticket (current stage)
constexpr int kWsEmit = 257;        // emit counter
constexpr int kWsTie = 258;         // tie counter
constexpr int kWsPrefix = 259;      // refinement prefix / final k_min
constexpr int kWsRemaining = 260;   // keys still needed / tie_take
constexpr int kWsStage = 261;       // 0 refine | 1 compact next | 2 done
constexpr int kWsCandCnt = 262;     // compaction counter
constexpr int kWsToken = 263;       // sampled token slot (int)
constexpr int kWsExpMax = 264;      // global logit max, fkey bits (sample)
constexpr int kWsTotal = 265;       // global softmax total (float, sample)
// cumulated softmax mass of the whole (failed) sampling window: written
// by the sampling tail whenever the nucleus is not covered, read by the
// host to sharpen the widening jump (widen_window). Lives next to
// kWsTotal so one 8-byte readback fetches both.
constexpr int kWsCumW = 266;        // window cum mass (sample, float)
constexpr int kWsLevelDone = 267;   // level of the last completed round
// argmax-dedicated slots (NOT shared with the selection pipeline): the
// selection ticket/emit counters can be non-zero after a call, so argmax
// cannot rely on the per-call head memset alone. These two words hold the
// invariant "zero at kernel entry": the buffer is zeroed once at (re)alloc
// and argmax's own finalize resets both words before the kernel exits,
// which makes the per-call cudaMemsetAsync redundant (one launch saved on
// every argmax call - the dominant cost of this op on submission-bound
// hosts; WDDM measured ~20-30us per extra launch).
constexpr int kWsArgBest = 268;     // argmax packed-key max (self-resetting)
constexpr int kWsArgCnt = 269;      // argmax arrival counter (self-resetting)
constexpr int kWsHead = 270;
constexpr int kWsCand = kWsHead;               // candidates [0, kSelEarlyOut)
constexpr int kWsKeys = kWsHead + kSelEarlyOut;  // key buffer A (m words)

// ---------------------------------------------------------------------------
// process-cached workspace
// ---------------------------------------------------------------------------

// Layout: [0..kWsHead) control head (zeroed per call), [kWsCand..+
// kSelEarlyOut) early-exit candidates, then key buffer A (m words), key
// buffer B (m
// words, merge scratch), then 1024 floats of block-sum scratch for the
// big-path nucleus count. Guarded by a mutex; never freed (bounded by the
// largest call). Not safe for concurrent selection launches on different
// streams (documented limitation).
unsigned long long* selection_workspace(size_t extra_words) {
    static unsigned long long* buf = nullptr;
    static size_t capacity = 0;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    const size_t words = kWsHead + extra_words;
    if (words > capacity) {
        unsigned long long* nb = nullptr;
        if (cudaMalloc(&nb, words * sizeof(unsigned long long)) != cudaSuccess)
            throw std::runtime_error(std::string("selection workspace alloc failed: ") +
                                     cudaGetErrorString(cudaGetLastError()));
        // Zero the fresh buffer once. cudaMalloc memory is NOT guaranteed
        // zero, and the argmax kernel's self-reset contract (kWsArgBest /
        // kWsArgCnt must be zero at kernel entry) starts at the very first
        // call after a (re)allocation. One synchronous memset per growth
        // event only; selection calls keep their own per-call head memset
        // and never depend on this.
        cudaMemset(nb, 0, words * sizeof(unsigned long long));
        if (buf) cudaFree(buf);
        buf = nb;
        capacity = words;
    }
    return buf;
}

inline __device__ unsigned long long pack_key(float v, int i) {
    return ((unsigned long long)fkey(v) << 32) | (0xFFFFFFFFULL - (unsigned)i);
}

// Repetition-penalty context for the fused decode_step path: a vocab
// bitmap (built once per step from the sampled ids) plus the penalty.
// use == 0 selects the plain path (no bitmap traffic). The penalty
// applies to the RAW logit before the temperature scale, matching the
// composed repetition_penalty -> temperature -> sample reference order.
struct PenCtx {
    const unsigned long long* bm;
    float penalty;
    int use;
};

__device__ __forceinline__ float step_logit(const float* __restrict__ x,
                                            int i, float inv_t, PenCtx pen) {
    float v = x[i];
    if (pen.use) {
        if ((pen.bm[i >> 6] >> (i & 63)) & 1ULL)
            v = v > 0.0f ? v / pen.penalty : v * pen.penalty;
    }
    return v * inv_t;
}

// Marks the sampled ids in the vocab bitmap (one bit per token).
__global__ void penalty_bitmap_kernel(const long long* __restrict__ ids,
                                      int m,
                                      unsigned long long* __restrict__ bm) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const int id = (int)ids[j];
    atomicOr(&bm[id >> 6], 1ULL << (id & 63));
}

// ---------------------------------------------------------------------------
// per-call pointer block (graph-indirect arguments)
// ---------------------------------------------------------------------------

// The selection pipeline is captured into a process-cached CUDA graph so
// the whole multi-kernel sequence submits as ONE launch (per-kernel
// submission costs 2-8us depending on platform and otherwise dominates
// the pipeline). A graph bakes pointer VALUES into its nodes, so the
// per-call pointers (input x, outputs vals/idxs/count_out) travel
// through this small block instead: the host writes a pinned mirror,
// one async H2D copy ships it to a fixed workspace slot, and the kernels
// dereference it at entry. Raw launches (first call / outer capture)
// use the exact same path.
struct SelArgs {
    const float* x;
    float* vals;
    long long* idxs;
    int* count_out;
    float p_stop;
};

// Device-side arg slot for a given pad size m (after the scan scratch).
inline size_t sel_args_off(int m) {
    return kWsHead + (size_t)kSelEarlyOut + 2 * (size_t)m + kWsScanWords;
}

// Pinned host mirror as a rotating ring: the CPU may rewrite the args
// while a PREVIOUS call's async H2D copy is still queued (host-ahead
// submission), so each call gets its own slot and a slot is reused only
// after an event proves its prior copy executed. 32 slots keep steady
// decode loops running without CPU stalls.
constexpr int kArgRing = 32;

// Ship one call's pointers: guard a ring slot (its previous H2D copy
// must have EXECUTED before the CPU overwrites it), write the args,
// issue the copy, then record the completion event AFTER the copy so the
// event truly fences the read. Used on the internal-graph path (never
// during an outer capture).
void ship_args(cudaStream_t cs, SelArgs* dargs, const float* x,
               float* vals, long long* idxs, int* count_out, float p_stop) {
    struct Ring {
        SelArgs* slots = nullptr;      // one pinned block, kArgRing entries
        cudaEvent_t ev[kArgRing];
        int cur = 0;
        std::mutex mu;
        Ring() {
            if (cudaHostAlloc(&slots, kArgRing * sizeof(SelArgs),
                              cudaHostAllocDefault) != cudaSuccess) {
                slots = nullptr;
                throw std::runtime_error(
                    std::string("pinned arg ring alloc failed: ") +
                    cudaGetErrorString(cudaGetLastError()));
            }
            for (int i = 0; i < kArgRing; ++i)
                if (cudaEventCreate(&ev[i]) != cudaSuccess)
                    throw std::runtime_error("arg ring event create failed");
        }
        ~Ring() {
            if (slots) cudaFreeHost(slots);
            for (int i = 0; i < kArgRing; ++i)
                if (ev[i]) cudaEventDestroy(ev[i]);
        }
    };
    static Ring ring;
    std::lock_guard<std::mutex> lock(ring.mu);
    const int i = ring.cur;
    ring.cur = (i + 1) % kArgRing;
    // wait until the copy that last read this slot has executed
    cudaError_t st = cudaEventQuery(ring.ev[i]);
    if (st == cudaErrorNotReady)
        cudaEventSynchronize(ring.ev[i]);
    else if (st != cudaSuccess)
        cudaGetLastError();          // clear a sticky error, keep going
    SelArgs& a = ring.slots[i];
    a.x = x;
    a.vals = vals;
    a.idxs = idxs;
    a.count_out = count_out;
    a.p_stop = p_stop;
    cudaMemcpyAsync(dargs, &a, sizeof(SelArgs), cudaMemcpyHostToDevice, cs);
    cudaEventRecord(ring.ev[i], cs);   // fences the copy for the next reuse
}

// Descending shared-memory bitonic sort over `len` (power of two) keys.
// Within one stride pass each index joins exactly one (i, i^stride) pair,
// so plain shared loads/stores race nowhere. Call from all block threads.
__device__ __forceinline__ void bitonic_desc_shared(unsigned long long* sk,
                                                    int len) {
    for (int size = 2; size <= len; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < len; i += blockDim.x) {
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
}

// ---------------------------------------------------------------------------
// stage 1/2: radix refinement rounds + early-exit compaction
// ---------------------------------------------------------------------------

// One launch per byte level (7..0), plus one extra "finalize" launch with
// level == -1. Behavior is driven by the stage word so the host can issue
// the full fixed sequence unconditionally (CUDA-graph friendly):
//   stage 0 (refine)  - histogram byte `level` of the candidates whose
//                       bytes above this level match the prefix; the last
//                       arriving block scans the global histogram and
//                       tightens the prefix. level == -1 refines nothing:
//                       the prefix is already the full k-th key.
//   stage 1/2         - return immediately: compaction and the k_min
//                       publish belong to select_finalize_kernel (kept
//                       out of this kernel so the rounds run with 2KB of
//                       shared memory and only the finalize pays for the
//                       candidate buffer); post-settled rounds are no-ops.
// `remaining0` carries k for the first round (the head memset left the
// remaining slot at zero). `inv_t` scales logits in sampling mode; any
// positive scale preserves the key order.
__global__ void select_round_kernel(const SelArgs* __restrict__ a,
                                    unsigned long long* __restrict__ ws,
                                    int n, int level,
                                    unsigned long long remaining0,
                                    float inv_t, PenCtx pen) {
    const float* __restrict__ x = a->x;
    __shared__ unsigned long long sh_hist[256];
    __shared__ int sh_ticket;

    const unsigned long long stage = ws[kWsStage];
    if (stage != 0ULL) return;                       // settle/compact elsewhere
    if (level < 0) return;                           // finalize: not here

    unsigned long long* hist = ws;
    unsigned long long* ticket = ws + kWsTicket;

    const unsigned long long prefix = ws[kWsPrefix];          // 0 at level 7
    const unsigned long long remaining =
        (level == 7) ? remaining0 : ws[kWsRemaining];
    // Histogram byte `level` of every key whose bytes ABOVE this level
    // already match the prefix. At level 7 there are no higher bytes, so
    // every key participates (mask 0). In sampling mode the key is built
    // from the temperature-scaled logit (identical ordering for T > 0).
    const unsigned long long topmask =
        (level == 7) ? 0ULL : ~((1ULL << (8 * (level + 1))) - 1ULL);
    // two-stage histogram: accumulate into per-block SHARED bins, then
    // merge once per block - a global atomic per element serializes on
    // hot bins, this keeps the hot loop shared-memory-only. Within a
    // warp, lanes hitting the same bin aggregate FIRST (__match_any_sync)
    // and only the group leader adds once: concentrated distributions
    // would otherwise serialize every lane on a handful of shared
    // addresses (measured 3-5x of the whole kernel on byte-7 rounds).
    for (int b = threadIdx.x; b < 256; b += kSelBlock) sh_hist[b] = 0ULL;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const unsigned long long key = pack_key(step_logit(x, i, inv_t, pen), i);
        if ((key & topmask) != prefix) continue;
        const int bin = (int)((key >> (8 * level)) & 0xFF);
        const unsigned grp = __match_any_sync(__activemask(), bin);
        if (__ffs(grp) - 1 == (int)(threadIdx.x & 31))
            atomicAdd(&sh_hist[bin], (unsigned long long)__popc(grp));
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += kSelBlock)
        if (sh_hist[b]) atomicAdd(&hist[b], sh_hist[b]);
    __syncthreads();                                // my atomics are issued
    if (threadIdx.x == 0) {
        __threadfence();                            // release: hist visible
        sh_ticket = (int)atomicAdd(ticket, 1ULL);
    }
    __syncthreads();
    if (sh_ticket != (int)gridDim.x - 1) return;    // decider = last arrival
    __threadfence();                                // acquire: hist visible

    // Stage the histogram through shared memory with a COOPERATIVE
    // volatile load: one volatile read per thread (~600ns L2 latency,
    // fully parallel) instead of 256 serial reads by one thread, which
    // latency-dominated the whole pipeline.
    const volatile unsigned long long* vhist =
        (const volatile unsigned long long*)hist;
    for (int b = threadIdx.x; b < 256; b += kSelBlock)
        sh_hist[b] = vhist[b];
    __syncthreads();

    // Scan bins top-down (shared memory now) to locate the boundary bin:
    // the first bin where the accumulated count reaches `remaining`.
    if (threadIdx.x == 0) {
        unsigned long long acc = 0;
        for (int b = 255; b >= 0; --b) {
            const unsigned long long c = sh_hist[b];
            if (acc + c >= remaining) {
                ws[kWsPrefix] = prefix | ((unsigned long long)b << (8 * level));
                ws[kWsRemaining] = remaining - acc;
                ws[kWsLevelDone] = (unsigned long long)level;
                // Small boundary bin: the finalize launch compacts + sorts
                // these survivors instead of histogramming again.
                if (c <= (unsigned long long)kSelEarlyOut) ws[kWsStage] = 1ULL;
                break;
            }
            acc += c;
        }
    }
    // reset histogram + ticket for the next round launch (cooperative)
    for (int b = threadIdx.x; b < 256; b += kSelBlock) hist[b] = 0ULL;
    if (threadIdx.x == 0) *ticket = 0ULL;
}

// ---------------------------------------------------------------------------
// stage 3/4a: emit (+ in-block finish for k <= kSelEarlyOut)
// ---------------------------------------------------------------------------

// Emit selected keys: keys > k_min take slots through the emit counter;
// the single key == k_min fills the last slot through the tie counter.
// Exactly k slots are written (full-key uniqueness).
//
// Counting is TWO-LEVEL: selections stage into a shared-memory buffer (a
// bounded batch of blockDim.x per grid-stride step), then each block
// grabs ONE global slot range with a single atomicAdd of its batch count.
// A flat per-key global atomic serializes the whole grid on one L2
// address (~260us at 131k keys); the two-level scheme costs one atomic
// per block. The k == n full-selection case skips counting entirely -
// every key ranks, so the element index IS the slot.
__device__ __forceinline__ void emit_selected(
    const float* __restrict__ x, unsigned long long* __restrict__ ws,
    int n, int k, float inv_t, PenCtx pen) {
    const unsigned long long k_min = ws[kWsPrefix];
    const unsigned long long tie_take = ws[kWsRemaining];
    unsigned long long* keys = ws + kWsKeys;
    unsigned long long* emit_cnt = ws + kWsEmit;
    unsigned long long* tie_cnt = ws + kWsTie;

    if (k == n) {
        // full selection: every key ranks (k_min is the minimum key)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x)
            keys[i] = pack_key(step_logit(x, i, inv_t, pen), i);
        return;
    }

    __shared__ unsigned long long sh_keys[kSelBlock];
    __shared__ unsigned long long sh_cnt;
    __shared__ unsigned long long sh_base;
    if (threadIdx.x == 0) sh_cnt = 0ULL;
    __syncthreads();
    // grid-stride in batches of blockDim.x elements per block: each batch
    // stages at most blockDim.x keys before the flush
    for (int base = blockIdx.x * blockDim.x; base < n;
         base += gridDim.x * blockDim.x) {
        const int i = base + threadIdx.x;
        if (i < n) {
            const unsigned long long key = pack_key(step_logit(x, i, inv_t, pen), i);
            if (key > k_min) {
                const unsigned long long pos = atomicAdd(&sh_cnt, 1ULL);
                if (pos < (unsigned long long)kSelBlock) sh_keys[pos] = key;
            } else if (key == k_min && tie_take > 0) {
                const unsigned long long pos = atomicAdd(tie_cnt, 1ULL);
                if (pos < tie_take) keys[k - tie_take + pos] = key;
            }
        }
        __syncthreads();
        const unsigned long long cnt = sh_cnt;
        if (cnt > 0ULL) {
            if (threadIdx.x == 0) {
                __threadfence();        // stage stores before the claim
                sh_base = atomicAdd(emit_cnt, cnt);
            }
            __syncthreads();
            const unsigned long long out = sh_base + threadIdx.x;
            if (threadIdx.x < cnt && out < (unsigned long long)k)
                keys[out] = sh_keys[threadIdx.x];
        }
        __syncthreads();
        if (threadIdx.x == 0) sh_cnt = 0ULL;
        __syncthreads();
    }
}

// Modes driven by the trailing parameters (identical to v0.3 semantics):
//   selection: p_stop > 0 requests the nucleus count in count_out
//   sampling:  sample != 0 treats the input as RAW LOGITS scaled by
//     inv_T; the finisher converts the sorted keys to softmax
//     probabilities, finds the nucleus at mass p, then inverse-CDF
//     samples within it using a hash-derived uniform (deterministic per
//     seed). The winning token goes to token_out; an uncovered nucleus
//     leaves it untouched (host retries with a wider window).
// The serial mass scans deliberately accumulate in the SAME order as the
// CPU reference so per-seed GPU/CPU token parity is bit-stable away from
// exact mass-boundary draws.
__global__ void emit_finish_kernel(const SelArgs* __restrict__ a,
                                   unsigned long long* __restrict__ ws,
                                   int n, int k, float inv_t,
                                   unsigned long long seed, int sample,
                                   PenCtx pen) {
    const float p_stop = a->p_stop;
    const float* __restrict__ x = a->x;
    float* vals = a->vals;
    long long* idxs = a->idxs;
    int* count_out = a->count_out;
    int* token_out = reinterpret_cast<int*>(ws + kWsToken);
    __shared__ int sh_ticket;
    emit_selected(x, ws, n, k, inv_t, pen);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();                            // publish my key stores
        sh_ticket = (int)atomicAdd(ws + kWsTicket, 1ULL);
    }
    __syncthreads();
    if (sh_ticket != (int)gridDim.x - 1) return;    // finisher = last block
    __threadfence();                                // acquire: keys visible

    // sort the k emitted keys in shared memory (0-pad to a power of two)
    __shared__ unsigned long long sk[kSelEarlyOut];
    const volatile unsigned long long* vkeys =
        (const volatile unsigned long long*)(ws + kWsKeys);
    int len = 1;
    while (len < k) len <<= 1;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        sk[i] = (i < k) ? vkeys[i] : 0ULL;
    __syncthreads();
    bitonic_desc_shared(sk, len);

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const unsigned long long key = sk[i];
        const int idx = (int)(0xFFFFFFFFu - (unsigned)(key & 0xFFFFFFFFu));
        if (vals) vals[i] = unfkey((unsigned)(key >> 32));
        if (idxs) idxs[i] = idx;
    }

    if (threadIdx.x != 0) return;
    if (sample != 0) {
        // fused nucleus sampling over the sorted window keys (see header
        // note). The threshold uses the GLOBAL softmax total computed by
        // exptotal_kernel: cum is a prefix of the global CDF, so the
        // crossing may fall beyond the window - that is the host's cue
        // to widen (the token stays untouched).
        const float row_max = unfkey((unsigned)(sk[0] >> 32));
        const float total =
            *reinterpret_cast<const float*>(&ws[kWsTotal]);
        float cum = 0.0f;
        int nucleus = 0;
        float nucleus_mass = 0.0f;
        bool covered = false;
        for (int i = 0; i < k; ++i) {
            cum += __expf(unfkey((unsigned)(sk[i] >> 32)) - row_max);
            nucleus = i + 1;
            if (cum >= p_stop * total) { nucleus_mass = cum; covered = true; break; }
        }
        if (!covered) {
            if (k < n) {      // window too small; host retries wider -
                // leave the window's whole cum mass for the host's
                // next-jump bound (widen_window)
                *reinterpret_cast<float*>(&ws[kWsCumW]) = cum;
                return;
            }
            nucleus = k;            // full window IS the vocabulary
            nucleus_mass = cum;
        }
        // splitmix64-finalized uniform in [0, 1): deterministic per seed
        unsigned long long z = seed + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        const float u = (float)((z >> 11) * (1.0 / 9007199254740992.0));
        const float target = u * nucleus_mass;
        cum = 0.0f;
        for (int i = 0; i < nucleus; ++i) {
            cum += __expf(unfkey((unsigned)(sk[i] >> 32)) - row_max);
            if (cum >= target) {
                *token_out = (int)(0xFFFFFFFFu -
                                   (unsigned)(sk[i] & 0xFFFFFFFFu));
                break;
            }
        }
        if (count_out) *count_out = nucleus;
        return;
    }
    if (p_stop > 0.0f) {
        // nucleus count over the sorted probabilities: first prefix whose
        // mass reaches p_stop (crossing element included)
        float cum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cum += unfkey((unsigned)(sk[i] >> 32));
            if (cum >= p_stop) {
                *count_out = i + 1;
                return;
            }
        }
        *count_out = k;    // p exceeds the total mass (float rounding)
    }
}

// ---------------------------------------------------------------------------
// stage 3/4b: big-k pipeline (emit, chunk sort, merge ladder, decode)
// ---------------------------------------------------------------------------

__global__ void emit_kernel(const SelArgs* __restrict__ a,
                            unsigned long long* __restrict__ ws,
                            int n, int k, float inv_t, PenCtx pen) {
    emit_selected(a->x, ws, n, k, inv_t, pen);
}

// Sort one kSelSortChunk-key chunk per block (shared-memory bitonic).
// Elements beyond k load as zero pads; real keys are always > 0 (index
// bits), so pads sink to the tail of the descending order.
__global__ void chunk_sort_kernel(unsigned long long* __restrict__ keys,
                                  int k, int m) {
    __shared__ unsigned long long sk[kSelSortChunk];
    const int chunks = (m + kSelSortChunk - 1) / kSelSortChunk;
    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        const int base = chunk * kSelSortChunk;
        const int len = min(kSelSortChunk, m - base);
        for (int i = threadIdx.x; i < len; i += blockDim.x)
            sk[i] = (base + i < k) ? keys[base + i] : 0ULL;
        __syncthreads();
        bitonic_desc_shared(sk, len);
        for (int i = threadIdx.x; i < len; i += blockDim.x)
            keys[base + i] = sk[i];
        __syncthreads();    // sk reused by the next chunk of this block
    }
}

// One merge level of the ladder: pairwise merges of `run`-sized sorted
// runs, tile-based merge path. ONE global-memory co-rank binary search
// per tile locates the (i, j) split of the two source runs; the tile's
// inputs are staged into shared memory and each thread resolves its
// element with a cheap in-shared search. Global reads stay coalesced.
// Buffers ping-pong; the host tracks which side holds the result.
__global__ void merge_level_kernel(const unsigned long long* __restrict__ src,
                                   unsigned long long* __restrict__ dst,
                                   int m, int run) {
    __shared__ unsigned long long sA[kSelMergeTile];
    __shared__ unsigned long long sB[kSelMergeTile];
    const long long ntiles = m / kSelMergeTile;
    for (long long tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
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
                mid == 0 || src[a_base + mid - 1] > src[b_base + lo0 - mid];
            if (pred) l = mid; else h = mid - 1;
        }
        const long long i0 = l, j0 = lo0 - i0;
        // stage up to kSelMergeTile inputs from each run (0-padded;
        // real keys are > 0 so pads sort to the tail)
        for (int u = threadIdx.x; u < kSelMergeTile; u += kSelBlock) {
            sA[u] = (i0 + u < run) ? src[a_base + i0 + u] : 0ULL;
            sB[u] = (j0 + u < run) ? src[b_base + j0 + u] : 0ULL;
        }
        __syncthreads();
        // in-tile co-rank per thread (shared memory, log T steps)
        for (int u = threadIdx.x; u < kSelMergeTile; u += kSelBlock) {
            long long l2 = 0;
            long long h2 = u < kSelMergeTile ? u : kSelMergeTile;
            while (l2 < h2) {
                const long long mid = (l2 + h2 + 1) >> 1;
                const bool pred = mid == 0 || sA[mid - 1] > sB[u - mid];
                if (pred) l2 = mid; else h2 = mid - 1;
            }
            const long long i = l2, j = u - i;
            const bool take_a =
                (i < kSelMergeTile) &&
                (j >= kSelMergeTile || sA[i] > sB[j]);
            dst[p0 + u] = take_a ? sA[i] : sB[j];
        }
        __syncthreads();
    }
}

__global__ void decode_kernel(const SelArgs* __restrict__ a,
                              const unsigned long long* __restrict__ keys,
                              int k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k;
         i += gridDim.x * blockDim.x) {
        const unsigned long long key = keys[i];
        if (a->vals) a->vals[i] = unfkey((unsigned)(key >> 32));
        if (a->idxs) a->idxs[i] = (int)(0xFFFFFFFFu - (unsigned)(key & 0xFFFFFFFFu));
    }
}

// ---------------------------------------------------------------------------
// stage 4b tail: parallel nucleus count (two kernels)
// ---------------------------------------------------------------------------

// Per-block partial sums of the sorted probabilities.
__global__ void topp_partial_kernel(const unsigned long long* __restrict__ keys,
                                    float* __restrict__ partials, int k,
                                    int per) {
    const long long b0 = (long long)blockIdx.x * per;
    const long long b1 = min((long long)k, b0 + per);
    float s = 0.0f;
    for (long long i = b0 + threadIdx.x; i < b1; i += kSelBlock)
        s += unfkey((unsigned)(keys[i] >> 32));
    s = warp_reduce_sum(s);
    __shared__ float sh_warp[kSelWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) sh_warp[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int w = 0; w < kSelWarps; ++w) t += sh_warp[w];
        partials[blockIdx.x] = t;
    }
}

// Single-block decision: walk the block partials to the block containing
// the p_stop crossing, then resolve the exact index inside that block's
// slice. The float accumulation order differs from the serial CPU loop
// (block-tree partials), so boundary draws may differ by a couple of
// elements at 131k-scale vocabularies - same contract as v0.3.
__global__ void topp_crossing_kernel(const SelArgs* __restrict__ a,
                                     const unsigned long long* __restrict__ keys,
                                     const float* __restrict__ partials,
                                     int k, int per, int nblocks) {
    int* count_out = a->count_out;
    const float p_stop = a->p_stop;
    __shared__ float sh_p[kMaxGrid];
    for (int i = threadIdx.x; i < nblocks; i += kSelBlock) sh_p[i] = partials[i];
    __syncthreads();
    if (threadIdx.x != 0) return;
    float carry = 0.0f;
    int tb = -1;
    for (int b = 0; b < nblocks; ++b) {
        if (carry + sh_p[b] >= p_stop) { tb = b; break; }
        carry += sh_p[b];
    }
    if (tb < 0) { *count_out = k; return; }   // p > total mass
    const long long b0 = (long long)tb * per;
    const long long b1 = min((long long)k, b0 + per);
    float cum = carry;
    for (long long i = b0; i < b1; ++i) {
        cum += unfkey((unsigned)(keys[i] >> 32));
        if (cum >= p_stop) { *count_out = (int)i + 1; return; }
    }
    *count_out = k;    // partial-sum said crossing, serial scan didn't
}

// ---------------------------------------------------------------------------
// stage 4b tail: serial sampling scan (window > kSelEarlyOut)
// ---------------------------------------------------------------------------

// Parallel pre-pass for the sampling scans: materialize exp(v - row_max)
// for the whole sorted window into `exps` (a free ping-pong key buffer).
// The serial walkers below then only do sequential float adds - no expf
// in the single-threaded path, which is what made flat-distribution
// windows (up to 131k entries) cost milliseconds. row_max comes from
// keys[0] (descending order), the exact same value the walkers used to
// derive, so the stored values are bit-identical to the old on-the-fly
// __expf and the accumulation order of every walk is UNCHANGED - per-seed
// tokens are exactly what the pre-1.1 serial scans produced.
__global__ void exp_window_kernel(const unsigned long long* __restrict__ keys,
                                  float* __restrict__ exps, int k) {
    const float row_max = unfkey((unsigned)(keys[0] >> 32));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k;
         i += gridDim.x * blockDim.x) {
        exps[i] = __expf(unfkey((unsigned)(keys[i] >> 32)) - row_max);
    }
}

// Latency-hiding helper for the contract-serial walks below (v1.2).
// The accumulation MUST stay strictly sequential (cum_{i+1} = cum_i +
// e_i in index order - that exact fadd sequence IS the CPU-parity
// determinism contract), but the LOADS are free to run ahead: a batch
// of kWalkBatch independent reads is issued together, then consumed by
// serial adds one at a time. On a full-vocabulary window (n=131072,
// flat distributions) the naive one-load-one-add walk was pure L2
// latency (~160 cycles/element, 12.8ms in one kernel, 97% of the whole
// flat case); with 32 outstanding loads the same order-identical walk
// runs memory-throughput-bound instead (~10 cycles/element).
namespace {
constexpr int kWalkBatch = 32;    // floats per batch = 8 x float4 loads
// Walk 1: scan exps[0..count), return the crossing index (cum >= thr
// first reached there) or -1; *cum_out carries the running total.
__device__ int walk_until(const float* __restrict__ exps, int count,
                          float thr, float* cum_out) {
    float cum = 0.0f;
    int i = 0;
    // scalar prologue to a 16-byte boundary (the exps buffers are at
    // least 4-byte aligned floats, but their word offset inside the
    // workspace is not guaranteed even), then the vector-load body.
    // Scheduling note: the consume side must stay BRANCH-FREE inside a
    // batch (nvcc refuses to hoist loads above the per-element early
    // exits, which serializes one LDG.128 per ~200 cycles - measured
    // 12.8ms at n=131072). The adds run strictly in index order with a
    // predicated `any` flag; on the (single) batch where the threshold
    // is crossed the exact first index is REDONE element-by-element
    // with the identical serial adds, so every returned index, cum and
    // token is bit-identical to the naive scalar walk.
    const int a = (int)(((16u - (unsigned)(uintptr_t)exps) & 15u) >> 2);
    for (; i < a && i < count; ++i) {
        cum += exps[i];
        if (cum >= thr) { *cum_out = cum; return i; }
    }
    for (; i + kWalkBatch <= count; ) {
        const float4* q = reinterpret_cast<const float4*>(exps + i);
        float4 r[kWalkBatch / 4];
        #pragma unroll
        for (int j = 0; j < kWalkBatch / 4; ++j) r[j] = q[j];
        const float base = cum;
        bool any = false;
        #pragma unroll
        for (int j = 0; j < kWalkBatch / 4; ++j) {
            cum += r[j].x; any |= (cum >= thr);
            cum += r[j].y; any |= (cum >= thr);
            cum += r[j].z; any |= (cum >= thr);
            cum += r[j].w; any |= (cum >= thr);
        }
        if (any) {
            // threshold crossed somewhere in THIS batch: replay just
            // these 32 adds (identical order - identical bits, hot in
            // L1) to pin the exact first index
            float c = base;
            for (int k = i; k < i + kWalkBatch; ++k) {
                c += exps[k];
                if (c >= thr) { *cum_out = c; return k; }
            }
        }
        i += kWalkBatch;
    }
    for (; i < count; ++i) {                          // scalar tail
        cum += exps[i];
        if (cum >= thr) { *cum_out = cum; return i; }
    }
    *cum_out = cum;
    return -1;
}
} // namespace

// Identical arithmetic and accumulation order to the in-finisher sample
// scan (and to the CPU reference): one thread walks the sorted keys. The
// serial walk is what keeps per-seed CPU/GPU token parity stable; big
// windows only occur for flat distributions (the host widening loop).
// `exps` carries the precomputed exp column (exp_window_kernel); the
// walk is two sequential passes - total mass, then inverse-CDF - both
// batched by walk_until (load pipelining only; the fadd order is
// untouched, so every token is bit-identical to the scalar walk).
__global__ void sample_serial_kernel(const unsigned long long* __restrict__ keys,
                                     const float* __restrict__ exps,
                                     unsigned long long* __restrict__ ws,
                                     int k, int n, float p_stop,
                                     unsigned long long seed,
                                     int* token_out, int* count_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // threshold against the GLOBAL softmax total (exptotal_kernel) so a
    // window that cannot contain the nucleus reports "not covered"
    // instead of silently renormalizing (see exptotal_kernel note).
    // window == n with float-drift cum < p*total: the nucleus is
    // everything by definition - force coverage (matches the CPU
    // reference, which compares an identical serial accumulation).
    const float total = *reinterpret_cast<const float*>(&ws[kWsTotal]);
    float cum = 0.0f;
    const int edge = walk_until(exps, k, p_stop * total, &cum);
    if (edge < 0) {
        if (k < n) {              // window too small; host retries wider -
            // leave the window's whole cum mass for the host's
            // next-jump bound (widen_window)
            *reinterpret_cast<float*>(&ws[kWsCumW]) = cum;
            return;
        }
    }
    const int nucleus = (edge < 0) ? k : edge + 1;
    // walk_until reports cum AT the crossing (covers the edge < 0 case
    // too: full-window cum, exactly what the forced-coverage path used)
    const float nucleus_mass = cum;
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const float u = (float)((z >> 11) * (1.0 / 9007199254740992.0));
    const float target = u * nucleus_mass;
    cum = 0.0f;
    const int hit = walk_until(exps, nucleus, target, &cum);
    if (hit >= 0)
        *token_out = (int)(0xFFFFFFFFu -
                           (unsigned)(keys[hit] & 0xFFFFFFFFu));
    if (count_out) *count_out = nucleus;
}

// ---------------------------------------------------------------------------
// sample-mode prerequisites: GLOBAL softmax mass
// ---------------------------------------------------------------------------

// Global max of the scaled logits as fkey bits (monotone unsigned order,
// so a plain integer atomicMax finds the float max of any-sign values).
__global__ void expmax_kernel(const float* __restrict__ x,
                              unsigned long long* __restrict__ ws,
                              int n, float inv_t, PenCtx pen) {
    __shared__ unsigned int warp_best[kSelWarps];
    unsigned int local = 0u;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const unsigned int fk = fkey(step_logit(x, i, inv_t, pen));
        if (fk > local) local = fk;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local = max(local, __shfl_down_sync(0xffffffffu, local, off));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_best[warp] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int b = warp_best[0];
        #pragma unroll
        for (int w = 1; w < kSelWarps; ++w) b = max(b, warp_best[w]);
        atomicMax(&ws[kWsExpMax], (unsigned long long)b);
    }
}

// Global softmax total: sum of __expf(v - max) over ALL n logits. The
// nucleus threshold must use the GLOBAL mass - a window-local total
// silently renormalizes the softmax and shrinks the nucleus whenever the
// distribution is flat enough that the tail carries real mass (bug found
// by the v0.4 widening test; v0.3 had the same flaw). Launched right
// after expmax_kernel: the kernel boundary publishes the max.
__global__ void exptotal_kernel(const float* __restrict__ x,
                                unsigned long long* __restrict__ ws,
                                int n, float inv_t, PenCtx pen) {
    const float row_max = unfkey((unsigned)ws[kWsExpMax]);
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        s += __expf(step_logit(x, i, inv_t, pen) - row_max);
    __shared__ float warp_sum[kSelWarps];
    s = warp_reduce_sum(s);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int w = 0; w < kSelWarps; ++w) t += warp_sum[w];
        atomicAdd(reinterpret_cast<float*>(&ws[kWsTotal]), t);
    }
}

// Finalize launch (after the eight round launches):
//   stage 2 - nothing to do (early exit already settled k_min)
//   stage 0 - eight full rounds completed: the prefix already IS the
//             full k-th key and remaining already collapsed to 1 (keys
//             are unique) - just publish the settled state
//   stage 1 - early exit fired: COMPACT the survivors (every key matching
//             the refined prefix) into the candidate buffer; the last
//             block sorts them in shared memory and publishes k_min /
//             tie_take = 1. Keeping this stage out of the round kernel
//             lets the rounds run with 2KB of static shared memory
//             (higher occupancy) and only this kernel pays the 16KB.
__global__ void select_finalize_kernel(const SelArgs* __restrict__ a,
                                       unsigned long long* __restrict__ ws,
                                       int n, float inv_t, PenCtx pen) {
    __shared__ int sh_ticket;
    const unsigned long long stage = ws[kWsStage];
    if (stage == 2ULL) return;
    if (stage == 0ULL) {
        if (threadIdx.x == 0 && blockIdx.x == 0) ws[kWsStage] = 2ULL;
        return;
    }

    const float* __restrict__ x = a->x;
    unsigned long long* ticket = ws + kWsTicket;
    const unsigned long long prefix = ws[kWsPrefix];
    const int level_done = (int)ws[kWsLevelDone];
    // prefix covers bytes [level_done..7]; match exactly those bytes
    // (shift by 8*level_done: 0 for a fully refined prefix, 56 for a
    // single high byte - no undefined shifts either way)
    const unsigned long long mask =
        ~((1ULL << (8 * level_done)) - 1ULL);
    unsigned long long* cand = ws + kWsCand;
    unsigned long long* cand_cnt = ws + kWsCandCnt;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const unsigned long long key = pack_key(step_logit(x, i, inv_t, pen), i);
        if ((key & mask) == prefix) {
            const unsigned long long pos = atomicAdd(cand_cnt, 1ULL);
            // survivors <= kSelEarlyOut is guaranteed by the early-exit
            // condition; the bound keeps a bug from smashing memory
            if (pos < (unsigned long long)kSelEarlyOut) cand[pos] = key;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();                            // publish my stores
        sh_ticket = (int)atomicAdd(ticket, 1ULL);
    }
    __syncthreads();
    if (sh_ticket != (int)gridDim.x - 1) return;    // finisher = last block
    __threadfence();                                // acquire: cand visible

    int cnt = (int)(
        *(const volatile unsigned long long*)&ws[kWsCandCnt]);
    if (cnt > kSelEarlyOut) cnt = kSelEarlyOut;     // defensive clamp
    const unsigned long long remaining = ws[kWsRemaining];
    __shared__ unsigned long long sk[kSelEarlyOut];
    int len = 1;
    while (len < cnt) len <<= 1;
    const volatile unsigned long long* vcand =
        (const volatile unsigned long long*)cand;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        sk[i] = (i < cnt) ? vcand[i] : 0ULL;
    __syncthreads();
    bitonic_desc_shared(sk, len);
    // The remaining-th largest survivor is the k-th largest key overall;
    // every full key is unique, so exactly one key equals it.
    ws[kWsPrefix] = sk[remaining - 1];
    ws[kWsRemaining] = 1ULL;
    ws[kWsStage] = 2ULL;
    *ticket = 0ULL;                                  // defensive reset
}

// Device-side argument loader: writes the per-call pointers into the
// args block. Used ONLY on the outer-capture path, where the pointers
// ride as kernel parameters and get baked into the CALLER'S graph (the
// torch.cuda.graph contract: captures use fixed tensors, replays mutate
// them in place). One thread, one 32-byte store - effectively free.
__global__ void set_args_kernel(SelArgs* __restrict__ dargs,
                                const float* x, float* vals,
                                long long* idxs, int* count_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dargs->x = x;
        dargs->vals = vals;
        dargs->idxs = idxs;
        dargs->count_out = count_out;
    }
}

// ---------------------------------------------------------------------------
// shared launcher plumbing
// ---------------------------------------------------------------------------

// ~4 elements per thread: small inputs launch ONE resident wave (the
// arrival-ticket tail dominates when blocks are cheap and many), large
// inputs cap at kMaxGrid with grid-stride loops covering the rest.
int selection_grid(int n) {
    long long want = (n + 4 * kSelBlock - 1) / (4 * kSelBlock);
    return (int)std::max<long long>(1LL, std::min<long long>(want, kMaxGrid));
}

// The full pipeline as plain stream-ordered launches (device args block
// already populated). Used directly when the caller's stream is being
// captured by an OUTER graph (torch.cuda.graph) - in that case `src` is
// non-null and a one-thread loader kernel bakes this call's pointers
// into the caller's graph as its leading node - and once per (n, k,
// mode) to record our internal cached graph (src == nullptr; the args
// arrive through the per-call H2D ring copy instead).
constexpr PenCtx kNoPen{nullptr, 1.0f, 0};

void select_raw_pipeline(const SelArgs* dargs, unsigned long long* ws,
                         int n, int k, int m, bool with_nucleus,
                         cudaStream_t cs, const SelArgs* src = nullptr,
                         PenCtx pen = kNoPen) {
    if (src)
        set_args_kernel<<<1, 32, 0, cs>>>(
            const_cast<SelArgs*>(dargs), src->x, src->vals, src->idxs,
            src->count_out);
    const int grid = selection_grid(n);
    // per-call state reset
    cudaMemsetAsync(ws, 0, kWsHead * sizeof(unsigned long long), cs);
    for (int level = 7; level >= 0; --level)
        select_round_kernel<<<grid, kSelBlock, 0, cs>>>(
            dargs, ws, n, level, (unsigned long long)k, 1.0f, pen);
    // finalize: compacts when the early exit fired, otherwise just
    // publishes the settled state (prefix == full k-th key)
    select_finalize_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n, 1.0f, pen);
    check_launch("selection round launch");

    if (k <= kSelEarlyOut) {
        emit_finish_kernel<<<grid, kSelBlock, 0, cs>>>(
            dargs, ws, n, k, 1.0f, 0ULL, /*sample=*/0, pen);
        check_launch("selection emit+finish launch");
        return;
    }

    emit_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n, k, 1.0f, pen);
    chunk_sort_kernel<<<(int)((m + kSelSortChunk - 1) / kSelSortChunk),
                        kSelBlock, 0, cs>>>(ws + kWsKeys, k, m);
    check_launch("selection chunk sort launch");
    unsigned long long* bufs[2] = {ws + kWsKeys,
                                   ws + kWsKeys + (size_t)m};
    int cur = 0;
    for (int run = kSelSortChunk; run < m; run <<= 1) {
        const long long ntiles = m / kSelMergeTile;
        const int gmerge = (int)std::max<long long>(
            1LL, std::min<long long>(ntiles, kMaxGrid));
        merge_level_kernel<<<gmerge, kSelBlock, 0, cs>>>(bufs[cur],
                                                         bufs[cur ^ 1],
                                                         m, run);
        check_launch("selection merge launch");
        cur ^= 1;
    }
    decode_kernel<<<selection_grid(k), kSelBlock, 0, cs>>>(dargs, bufs[cur],
                                                           k);
    check_launch("selection decode launch");

    if (with_nucleus) {
        float* partials = reinterpret_cast<float*>(
            ws + kWsKeys + 2 * (size_t)m);
        const int gsum = (int)std::max<long long>(
            1LL, std::min<long long>((k + kSelBlock - 1) / kSelBlock,
                                     kMaxGrid));
        const int per = (k + gsum - 1) / gsum;
        topp_partial_kernel<<<gsum, kSelBlock, 0, cs>>>(bufs[cur], partials,
                                                        k, per);
        topp_crossing_kernel<<<1, kSelBlock, 0, cs>>>(dargs, bufs[cur],
                                                      partials, k, per,
                                                      gsum);
        check_launch("selection nucleus count launch");
    }
}

// ---------------------------------------------------------------------------
// internal graph cache
// ---------------------------------------------------------------------------

// One cached executable graph per (n, k, mode, workspace). Capture runs
// on a private side stream (capturing on the caller's stream is illegal
// when it is the legacy default stream); replay always targets the
// caller's stream, which orders it with surrounding work. The graph
// nodes reference only fixed workspace addresses plus the device args
// block, so replays pick up fresh pointers through the per-call H2D
// copy. The stage-machine kernels read their state from the workspace at
// runtime, so a fixed launch sequence stays correct for every input.
struct SelGraph {
    cudaGraphExec_t exec = nullptr;
};
using SelGraphKey = std::tuple<int, int, int, unsigned long long*>;

cudaStream_t capture_stream() {
    static cudaStream_t s = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (cudaStreamCreate(&s) != cudaSuccess)
            throw std::runtime_error(std::string("capture stream create failed: ") +
                                     cudaGetErrorString(cudaGetLastError()));
    });
    return s;
}

// Bounded cache: decode loops use a handful of shapes; clear everything
// if pathological call patterns ever exceed the cap.
cudaGraphExec_t cached_graph(const SelGraphKey& key,
                             const std::function<void(cudaStream_t)>& record) {
    static std::mutex mu;
    static std::map<SelGraphKey, SelGraph> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second.exec;
    if (cache.size() >= 64) {
        for (auto& kv : cache)
            if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        cache.clear();
    }
    cudaStream_t cs = capture_stream();
    if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        throw std::runtime_error(std::string("graph capture begin failed: ") +
                                 cudaGetErrorString(cudaGetLastError()));
    record(cs);
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(cs, &graph) != cudaSuccess || !graph)
        throw std::runtime_error(std::string("graph capture end failed: ") +
                                 cudaGetErrorString(cudaGetLastError()));
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess || !exec) {
        cudaGraphDestroy(graph);
        throw std::runtime_error(std::string("graph instantiate failed: ") +
                                 cudaGetErrorString(cudaGetLastError()));
    }
    cudaGraphDestroy(graph);
    cache.emplace(key, SelGraph{exec});
    return exec;
}

// Rounds + finalize + the right emit/sort/decode tail for the given k.
// p_stop > 0 requests the nucleus count (top-p mode; vals are already
// probabilities, so the mass scan sums them directly). Sampling lives in
// sample_topp_launch, which drives its own (windowed) sequence.
void select_common(const float* x, float* vals, long long* idxs,
                   int* count_out, int n, int k, float p_stop,
                   std::uintptr_t stream) {
    cudaStream_t cs = (cudaStream_t)stream;
    int m = 1;
    while (m < k) m <<= 1;                        // sort pad size
    // candidates + two m-word key buffers + kMaxGrid floats of scan
    // scratch + the per-call args block
    unsigned long long* ws =
        selection_workspace((size_t)kSelEarlyOut + 2 * (size_t)m + kWsScanWords +
                            sizeof(SelArgs) / sizeof(unsigned long long));
    SelArgs* dargs = reinterpret_cast<SelArgs*>(ws + sel_args_off(m));
    // outer capture in progress (torch.cuda.graph): contribute the raw
    // kernels to the caller's graph, with a loader node that re-installs
    // the capture-time pointers on every replay (torch's fixed-tensor
    // contract). No host allocations or event queries happen here - both
    // are illegal mid-capture.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cs, &cap) != cudaSuccess)
        cap = cudaStreamCaptureStatusNone;
    if (cap == cudaStreamCaptureStatusActive) {
        const SelArgs tmp{x, vals, idxs, count_out, p_stop};
        select_raw_pipeline(dargs, ws, n, k, m, p_stop > 0.0f, cs, &tmp);
        return;
    }
    ship_args(cs, dargs, x, vals, idxs, count_out, p_stop);

    const int mode = p_stop > 0.0f ? 1 : 0;
    cudaGraphExec_t exec = cached_graph(
        SelGraphKey(n, k, mode, ws),
        [&](cudaStream_t rec) {
            select_raw_pipeline(dargs, ws, n, k, m, mode == 1, rec);
        });
    if (cudaGraphLaunch(exec, cs) != cudaSuccess)
        throw std::runtime_error(std::string("selection graph launch failed: ") +
                                 cudaGetErrorString(cudaGetLastError()));
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

void topk_launch(const float* x, float* vals, long long* idxs, int n, int k, std::uintptr_t stream) {
    if (k <= 0 || n <= 0) return;
    select_common(x, vals, idxs, nullptr, n, k, 0.0f, stream);
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
                        int n, float p, int* count_out, std::uintptr_t stream) {
    if (n <= 0) {
        if (count_out) {
            int zero = 0;
            cudaMemcpy(count_out, &zero, sizeof(int), cudaMemcpyHostToDevice);
        }
        return;
    }
    select_common(x, vals, idxs, count_out, n, n, p, stream);
}

// Adaptive widening jump (v1.2). After a window attempt fails to cover
// the nucleus, two floats the attempt already produced steer the next
// window size (one 8-byte readback of the adjacent kWsTotal/kWsCumW
// words):
//   T = GLOBAL softmax total (exptotal_kernel, max-normalized: the
//       largest exp is exactly 1.0f)
//   C = whole-window cum mass (stored by the sampling tail on failure)
// Every element past rank W is at most C/W (the smallest of the top-W
// exps bounds the rest, and it is at most their average), so covering
// the remaining p*T - C mass needs at least (p*T - C)*W/C further
// elements: w >= ceil(W * p * T / C) - a necessary lower bound that is
// TIGHT for flat distributions (all exps ~ 1 -> w ~ p*n, one jump to
// (nearly) the full vocabulary) and still useful on heavy tails like a
// plain randn (max-normalization flattens everything but the extreme
// tail, so the nucleus is surprisingly wide). The x8 ladder step stays
// as the floor, so mid-tailed distributions never regress. The bound is
// rounded up to a power of two for slack against float drift - a
// hair-miss would otherwise cost one more full attempt. The sampled
// token is independent of the jump schedule (the threshold uses the
// global mass and the draw renormalizes inside the nucleus): only the
// retry count changes, which is what this saves.
static int widen_window(int window, int n, float p,
                        const unsigned long long* ws) {
    float mass[2] = {0.0f, 0.0f};           // [total, window cum]
    if (cudaMemcpy(mass, ws + kWsTotal, 2 * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("sample mass readback failed");
    const double total = (double)mass[0];
    const double cw = (double)mass[1];
    long long lb;
    if (cw > 0.0 && p * total > cw)         // else keep the plain bound
        lb = (long long)std::ceil((double)window * p * total / cw) + 1;
    else
        lb = (long long)std::ceil(p * total) + 1;
    const long long want = std::max<long long>((long long)window * 8, lb);
    if (want >= n) return n;
    int mp = 1;                       // pow2 headroom over the bound
    while (mp < want) mp <<= 1;
    return std::min(n, mp);
}

// ---------------------------------------------------------------------------
// fused nucleus sampling: softmax(logits/T) -> nucleus(p) -> inverse-CDF
// draw from a hash-derived uniform. The token comes back through a
// workspace slot with a host readback (this launcher is intentionally
// NOT CUDA-graph capturable - the widening loop needs the result).
// (CPU reference: sample_topp_cpu in sampling.cu, declared in
// activations.hpp)
// ---------------------------------------------------------------------------

long long sample_topp_launch(const float* x, int n, float p, float t,
                             unsigned long long seed, std::uintptr_t stream) {
    if (n <= 0)
        throw std::invalid_argument("sample of empty logits");
    if (!(p > 0.0f && p <= 1.0f))
        throw std::invalid_argument("p must be in (0, 1]");
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");
    // Widening-window strategy: sort only the top-M candidates (M starts
    // at the early-exit/in-block-sort size), sample within them; if the
    // nucleus is not covered by the window, retry wider - adaptively via
    // the p*T mass bound (widen_window). Typical distributions are
    // covered by the first window.
    int window = std::min(kSelEarlyOut, n);
    for (;;) {
        int m = 1;
        while (m < window) m <<= 1;
        // Grow the workspace BEFORE presetting the token slot: the growth
        // would otherwise free the buffer the slot lives in.
        unsigned long long* ws =
            selection_workspace((size_t)kSelEarlyOut + 2 * (size_t)m + kWsScanWords +
                                sizeof(SelArgs) / sizeof(unsigned long long));
        cudaStream_t cs = (cudaStream_t)stream;
        SelArgs* dargs = reinterpret_cast<SelArgs*>(ws + sel_args_off(m));
        int* token_out = reinterpret_cast<int*>(ws + kWsToken);
        int token = -1;
        // reset the control head FIRST, then preset the token sentinel
        // (the memset would otherwise wipe it), then ship the args
        cudaMemsetAsync(ws, 0, kWsHead * sizeof(unsigned long long), cs);
        cudaMemcpyAsync(token_out, &token, sizeof(int), cudaMemcpyHostToDevice, cs);
        ship_args(cs, dargs, x, nullptr, nullptr, nullptr, p);
        const int grid = selection_grid(n);
        // Full-vocabulary fast path (v1.2): when the window covers the
        // whole vocabulary every key survives, so the radix rounds and
        // the finalize are pure waste - emit_kernel's k==n branch (a
        // plain parallel pack) is all the "selection" that is needed.
        // The sort / exp / serial-walk tail below is unchanged, so the
        // accumulation order (and therefore every sampled token) stays
        // exactly what the full pipeline produced.
        const bool full = (window == n);
        if (!full) {
            for (int level = 7; level >= 0; --level)
                select_round_kernel<<<grid, kSelBlock, 0, cs>>>(
                    dargs, ws, n, level, (unsigned long long)window, 1.0f / t,
                    kNoPen);
            select_finalize_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n,
                                                               1.0f / t,
                                                               kNoPen);
        }
        // global softmax mass (max, then total): the nucleus threshold
        // must be global, not window-local (see exptotal_kernel)
        expmax_kernel<<<grid, kSelBlock, 0, cs>>>(x, ws, n, 1.0f / t, kNoPen);
        exptotal_kernel<<<grid, kSelBlock, 0, cs>>>(x, ws, n, 1.0f / t, kNoPen);
        check_launch("sample round launch");
        if (!full && window <= kSelEarlyOut) {
            emit_finish_kernel<<<grid, kSelBlock, 0, cs>>>(
                dargs, ws, n, window, 1.0f / t, seed, /*sample=*/1, kNoPen);
            check_launch("sample emit+finish launch");
        } else {
            // partial window: two-level emit; full window: emit's k==n
            // branch degenerates to the parallel pack (see above)
            emit_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n, window,
                                                    1.0f / t, kNoPen);
            chunk_sort_kernel<<<(m + kSelSortChunk - 1) / kSelSortChunk,
                                kSelBlock, 0, cs>>>(ws + kWsKeys, window, m);
            unsigned long long* bufs[2] = {ws + kWsKeys,
                                           ws + kWsKeys + (size_t)m};
            int cur = 0;
            for (int run = kSelSortChunk; run < m; run <<= 1) {
                const long long ntiles = m / kSelMergeTile;
                const int gmerge = (int)std::max<long long>(
                    1LL, std::min<long long>(ntiles, kMaxGrid));
                merge_level_kernel<<<gmerge, kSelBlock, 0, cs>>>(bufs[cur],
                                                                 bufs[cur ^ 1],
                                                                 m, run);
                cur ^= 1;
            }
            // parallel exp precompute; the serial walk then only adds
            float* exps = reinterpret_cast<float*>(bufs[cur ^ 1]);
            exp_window_kernel<<<selection_grid(window), kSelBlock, 0, cs>>>(
                bufs[cur], exps, window);
            sample_serial_kernel<<<1, 32, 0, cs>>>(bufs[cur], exps, ws,
                                                   window, n, p, seed,
                                                   token_out, nullptr);
            check_launch("sample tail launch");
        }
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
        window = widen_window(window, n, p, ws);  // widen and retry
    }
}

// ---------------------------------------------------------------------------
// fused decode step: repetition penalty -> temperature -> nucleus sample
// ---------------------------------------------------------------------------

// One call from raw logits to the next token. The penalty set becomes a
// vocab bitmap (one kernel), then the SAME selection pipeline runs with
// a penalty-aware key-packing: every site that reads a logit applies
// penalty (to the raw value, matching the composed reference order)
// then the temperature scale. No cached graphs here - the penalty rides
// as a kernel parameter, and this launcher has an inherent host
// readback anyway (same contract as sample_topp).
long long decode_step_launch(const float* x, const long long* ids,
                             int n, int m, float penalty, float p, float t,
                             unsigned long long seed, std::uintptr_t stream) {
    if (n <= 0)
        throw std::invalid_argument("decode_step of empty logits");
    if (m < 0)
        throw std::invalid_argument("sampled_ids size must be >= 0");
    if (!(penalty > 0.0f))
        throw std::invalid_argument("penalty must be > 0");
    if (!(p > 0.0f && p <= 1.0f))
        throw std::invalid_argument("p must be in (0, 1]");
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");
    cudaStream_t cs = (cudaStream_t)stream;
    const size_t bitmap_words = ((size_t)n + 63) / 64;
    const int use_pen = (m > 0 && penalty != 1.0f) ? 1 : 0;

    int window = std::min(kSelEarlyOut, n);
    for (;;) {
        int mm = 1;
        while (mm < window) mm <<= 1;
        unsigned long long* ws = selection_workspace(
            (size_t)kSelEarlyOut + 2 * (size_t)mm + kWsScanWords + bitmap_words +
            sizeof(SelArgs) / sizeof(unsigned long long));
        // bitmap lives right after the args block; it must be zeroed on
        // every (re)try because the marking kernel ORs bits in - zero
        // the whole span from the head through the bitmap (the buffers
        // in between do not need it, but one contiguous clear is cheaper
        // than two async memsets)
        unsigned long long* bm = ws + sel_args_off(mm) +
                                 sizeof(SelArgs) / sizeof(unsigned long long);
        SelArgs* dargs = reinterpret_cast<SelArgs*>(ws + sel_args_off(mm));
        int* token_out = reinterpret_cast<int*>(ws + kWsToken);
        int token = -1;
        cudaMemsetAsync(ws, 0,
                        (size_t)(bm - ws + bitmap_words) *
                            sizeof(unsigned long long),
                        cs);
        cudaMemcpyAsync(token_out, &token, sizeof(int),
                        cudaMemcpyHostToDevice, cs);
        ship_args(cs, dargs, x, nullptr, nullptr, nullptr, p);
        const PenCtx pen{bm, penalty, use_pen};
        if (use_pen)
            penalty_bitmap_kernel<<<(int)((m + kSelBlock - 1) / kSelBlock),
                                    kSelBlock, 0, cs>>>(ids, m, bm);
        const int grid = selection_grid(n);
        const float inv_t = 1.0f / t;
        // full-vocabulary fast path: same rationale as sample_topp (the
        // radix rounds cannot discard anything when window == n)
        const bool full = (window == n);
        if (!full) {
            for (int level = 7; level >= 0; --level)
                select_round_kernel<<<grid, kSelBlock, 0, cs>>>(
                    dargs, ws, n, level, (unsigned long long)window, inv_t,
                    pen);
            select_finalize_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n,
                                                               inv_t, pen);
        }
        expmax_kernel<<<grid, kSelBlock, 0, cs>>>(x, ws, n, inv_t, pen);
        exptotal_kernel<<<grid, kSelBlock, 0, cs>>>(x, ws, n, inv_t, pen);
        check_launch("decode step round launch");
        if (!full && window <= kSelEarlyOut) {
            emit_finish_kernel<<<grid, kSelBlock, 0, cs>>>(
                dargs, ws, n, window, inv_t, seed, /*sample=*/1, pen);
            check_launch("decode step finish launch");
        } else {
            // partial window: two-level emit; full window: emit's k==n
            // branch degenerates to the parallel pack
            emit_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n, window,
                                                    inv_t, pen);
            chunk_sort_kernel<<<(mm + kSelSortChunk - 1) / kSelSortChunk,
                                kSelBlock, 0, cs>>>(ws + kWsKeys, window, mm);
            unsigned long long* bufs[2] = {ws + kWsKeys,
                                           ws + kWsKeys + (size_t)mm};
            int cur = 0;
            for (int run = kSelSortChunk; run < mm; run <<= 1) {
                const long long ntiles = mm / kSelMergeTile;
                const int gmerge = (int)std::max<long long>(
                    1LL, std::min<long long>(ntiles, kMaxGrid));
                merge_level_kernel<<<gmerge, kSelBlock, 0, cs>>>(
                    bufs[cur], bufs[cur ^ 1], mm, run);
                cur ^= 1;
            }
            // parallel exp precompute; the serial walk then only adds
            float* exps = reinterpret_cast<float*>(bufs[cur ^ 1]);
            exp_window_kernel<<<selection_grid(window), kSelBlock, 0, cs>>>(
                bufs[cur], exps, window);
            sample_serial_kernel<<<1, 32, 0, cs>>>(bufs[cur], exps, ws,
                                                   window, n, p, seed,
                                                   token_out, nullptr);
            check_launch("decode step tail launch");
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("decode step kernel failed: ") +
                                     cudaGetErrorString(err));
        if (cudaMemcpy(&token, token_out, sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("decode step readback failed");
        if (token >= 0)
            return token;
        if (window == n)
            throw std::runtime_error("decode step nucleus not covered");
        window = widen_window(window, n, p, ws);  // same adaptive jump
    }
}

// ---------------------------------------------------------------------------
// fused top-k sampling: softmax temperature-scale, top-k truncation,
// renormalize WITHIN the k survivors, inverse-CDF draw - one readback
// ---------------------------------------------------------------------------

// Serial draw over the sorted top-k keys (one thread, one pass): the
// top-k window is ALWAYS covered by construction (the distribution is
// renormalized inside it), so - unlike the nucleus sampler - there is no
// global-mass threshold, no coverage check and no widening loop. Same
// arithmetic and accumulation order as the CPU reference: exp walk
// seeded at the row max, splitmix64 hash of the seed, serial inverse
// CDF - per-seed CPU/GPU token parity is bit-stable away from draws
// landing exactly on an exp-rounding boundary.
__global__ void sample_topk_serial_kernel(
    const unsigned long long* __restrict__ keys,
    const float* __restrict__ exps, int* __restrict__ token_out,
    int k, unsigned long long seed) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // both walks batched by walk_until: load pipelining only, the fadd
    // order (the CPU-parity contract) is untouched (v1.2)
    float window_mass = 0.0f;
    walk_until(exps, k, INFINITY, &window_mass);   // sum without crossing
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const float u = (float)((z >> 11) * (1.0 / 9007199254740992.0));
    const float target = u * window_mass;
    float cum = 0.0f;
    const int hit = walk_until(exps, k, target, &cum);
    const int idx = (hit >= 0) ? hit : k - 1;      // rounding fallback
    *token_out = (int)(0xFFFFFFFFu -
                       (unsigned)(keys[idx] & 0xFFFFFFFFu));
}

long long sample_topk_launch(const float* x, int n, int k, float t,
                             unsigned long long seed, std::uintptr_t stream) {
    if (n <= 0)
        throw std::invalid_argument("sample of empty logits");
    if (k <= 0)
        throw std::invalid_argument("k must be >= 1");
    if (!(t > 0.0f))
        throw std::invalid_argument("temperature must be > 0");
    if (k > n) k = n;                             // full-vocab sampling
    cudaStream_t cs = (cudaStream_t)stream;
    int mm = 1;
    while (mm < k) mm <<= 1;                      // sort pad size
    unsigned long long* ws =
        selection_workspace((size_t)kSelEarlyOut + 2 * (size_t)mm + kWsScanWords +
                            sizeof(SelArgs) / sizeof(unsigned long long));
    SelArgs* dargs = reinterpret_cast<SelArgs*>(ws + sel_args_off(mm));
    int* token_out = reinterpret_cast<int*>(ws + kWsToken);
    int token = -1;
    cudaMemsetAsync(ws, 0, kWsHead * sizeof(unsigned long long), cs);
    cudaMemcpyAsync(token_out, &token, sizeof(int), cudaMemcpyHostToDevice,
                    cs);
    ship_args(cs, dargs, x, nullptr, nullptr, nullptr, 0.0f);
    const int grid = selection_grid(n);
    const float inv_t = 1.0f / t;
    for (int level = 7; level >= 0; --level)
        select_round_kernel<<<grid, kSelBlock, 0, cs>>>(
            dargs, ws, n, level, (unsigned long long)k, inv_t, kNoPen);
    select_finalize_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n,
                                                       inv_t, kNoPen);
    check_launch("sample topk round launch");
    // the sort tail is unified for every k: emit the k survivors, sort
    // them (one or several 1024-key chunks + the merge ladder), draw.
    // No mass threshold anywhere - the window renormalizes by definition.
    emit_kernel<<<grid, kSelBlock, 0, cs>>>(dargs, ws, n, k, inv_t, kNoPen);
    chunk_sort_kernel<<<(int)((mm + kSelSortChunk - 1) / kSelSortChunk),
                        kSelBlock, 0, cs>>>(ws + kWsKeys, k, mm);
    unsigned long long* bufs[2] = {ws + kWsKeys,
                                   ws + kWsKeys + (size_t)mm};
    int cur = 0;
    for (int run = kSelSortChunk; run < mm; run <<= 1) {
        const long long ntiles = mm / kSelMergeTile;
        const int gmerge = (int)std::max<long long>(
            1LL, std::min<long long>(ntiles, kMaxGrid));
        merge_level_kernel<<<gmerge, kSelBlock, 0, cs>>>(bufs[cur],
                                                         bufs[cur ^ 1],
                                                         mm, run);
        cur ^= 1;
    }
    // parallel exp precompute; the serial walk then only adds
    float* exps = reinterpret_cast<float*>(bufs[cur ^ 1]);
    exp_window_kernel<<<selection_grid(k), kSelBlock, 0, cs>>>(
        bufs[cur], exps, k);
    sample_topk_serial_kernel<<<1, 32, 0, cs>>>(bufs[cur], exps, token_out,
                                                k, seed);
    check_launch("sample topk tail launch");
    cudaError_t err = cudaDeviceSynchronize();    // surface kernel faults
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("sample topk kernel failed: ") +
                                 cudaGetErrorString(err));
    if (cudaMemcpy(&token, token_out, sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("sample topk readback failed");
    if (token < 0)
        throw std::runtime_error("sample topk produced no token");
    return token;
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
            *out = (int)(0xFFFFFFFFu - (unsigned)((*best) & 0xFFFFFFFFu));
            *best = 0ULL;                 // self-reset: next launch needs no memset
            *counter = 0u;
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

void argmax_launch(const float* x, int n, int* out, std::uintptr_t stream) {
    if (n <= 0) return;
    unsigned long long* ws = selection_workspace(0);
    // Dedicated self-resetting slots (see kWsArgBest): zero at entry by
    // contract - zeroed once at workspace (re)alloc and reset by every
    // kernel's finalize - so no per-call cudaMemsetAsync is needed. The
    // saved launch is the dominant cost of this op on submission-bound
    // hosts. Selection calls memset the whole head (covering these slots)
    // at their own start, which keeps the invariant across mixed call
    // sequences on the same stream.
    unsigned long long* best = ws + kWsArgBest;
    const int grid = (int)((n + kSelBlock - 1) / kSelBlock);
    argmax_kernel<<<grid, kSelBlock, 0, (cudaStream_t)stream>>>(
        x, best, reinterpret_cast<unsigned int*>(ws + kWsArgCnt), n, out);
    check_launch("argmax kernel launch");
}

} // namespace fusedtok
