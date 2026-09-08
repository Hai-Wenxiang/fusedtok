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
#include <limits>
#include <map>
#include <mutex>
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

// ---------------------------------------------------------------------------
// HF-style combined logit penalties (repetition / presence / frequency):
//
//   c[i] = number of occurrences of id i in token_ids
//   y[i] = x[i]                                                   if c[i] == 0
//   y[i]: v = x[i]
//         if repetition != 1: v = v > 0 ? v / repetition : v * repetition
//         if presence    != 0: v -= presence
//         if frequency   != 0: v -= c[i] * frequency
//         y[i] = v                                        (applied once per id)
//
// The composition order (repetition scale, then presence shift, then
// count-weighted frequency shift) matches the HF processors and is shared
// verbatim by the CPU reference, so the two paths agree bit-exactly: the
// counts are integers, no output value is touched by more than one thread,
// and every arithmetic op is a correctly rounded IEEE single op on both
// sides. Duplicate ids must not stack: the histogram counts each id once
// and the apply pass reads the ORIGINAL logit, so every duplicate computes
// the identical value (the 1.5.2 per-distinct-id lesson, kernel side).
// ---------------------------------------------------------------------------

// One thread per id: bump the id's histogram bucket. Host-side validation
// already rejected out-of-range ids; the guard keeps a bad device-resident
// tensor (trusted, not synced - see _ids_arg) from corrupting memory.
__global__ void penalty_count_kernel(const long long* ids, int n, int m,
                                     int* counts) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    long long id = ids[j];
    if (id < 0 || id >= (long long)n) return;   // validated on host; defensive guard
    atomicAdd(&counts[id], 1);
}

// One thread per vocab slot: penalized slots rewrite the logit in the HF
// composition order, every other slot passes through. Reads x (never y),
// so in-place out == logits stays safe on the cached-workspace path.
__global__ void logit_penalties_kernel(const float* x, const int* counts,
                                       int n, float repetition, float presence,
                                       float frequency, float* y) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    int c = counts[i];
    if (c > 0) {
        if (repetition != 1.0f) v = v > 0.0f ? v / repetition : v * repetition;
        if (presence != 0.0f) v -= presence;
        // __fmul_rn/__fsub_rn keep both roundings separate: ptxas would
        // otherwise fuse this into one FFMA (single rounding) and the
        // host reference (two roundings) could differ by 1 ulp
        if (frequency != 0.0f)
            v = __fsub_rn(v, __fmul_rn((float)c, frequency));
    }
    y[i] = v;
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
    // apply each DISTINCT id exactly once: the GPU kernel reads the
    // original logits and writes y[id] once per occurrence (duplicate
    // threads write the identical value), so scaling per occurrence
    // here would diverge from the GPU on repeated ids
    for (long long id : token_ids) {
        float v = logits[(size_t)id];
        y[(size_t)id] = v > 0.0f ? v / penalty : v * penalty;
    }
    return y;
}

std::vector<float> logit_penalties_cpu(const std::vector<float>& logits,
                                       const std::vector<long long>& token_ids,
                                       float repetition, float presence,
                                       float frequency) {
    if (!(repetition > 0.0f))
        throw std::invalid_argument("repetition must be > 0");
    for (long long id : token_ids)
        if (id < 0 || id >= (long long)logits.size())
            throw std::invalid_argument("token id out of range");
    std::vector<float> y = logits;
    // one entry per DISTINCT id: the apply formula depends on the total
    // count only, mirroring the GPU histogram (per-occurrence stacking
    // would diverge on repeated ids - see repetition_penalty_cpu)
    std::map<long long, int> counts;
    for (long long id : token_ids) counts[id] += 1;
    for (const auto& kv : counts) {
        float v = logits[(size_t)kv.first];
        if (repetition != 1.0f) v = v > 0.0f ? v / repetition : v * repetition;
        if (presence != 0.0f) v -= presence;
        if (frequency != 0.0f) v -= (float)kv.second * frequency;
        y[(size_t)kv.first] = v;
    }
    return y;
}

namespace {

// Per-vocab-size histogram buffers (int per vocab slot, <= 512 KiB at the
// 131072 cap). Allocated OUTSIDE stream captures (the attention-workspace
// pattern); a first use that races an active capture runs the histogram
// through the caller's output buffer instead - same byte count (n * 4),
// cleared before the histogram and rewritten by the apply pass, so the
// buffer is back to logits by the time the caller reads it. That fallback
// borrows y, hence the out != logits requirement there.
std::mutex& pen_ws_mutex() {
    static std::mutex m;
    return m;
}
std::map<int, int*>& pen_ws_cache() {
    static std::map<int, int*> c;
    return c;
}

} // namespace

void logit_penalties_launch(const float* logits, const long long* ids,
                            int n, int m, float repetition, float presence,
                            float frequency, float* y, std::uintptr_t stream) {
    if (n <= 0) return;
    if (!(repetition > 0.0f))
        throw std::invalid_argument("repetition must be > 0");
    cudaStream_t cs = (cudaStream_t)stream;
    int* counts = nullptr;
    {
        std::lock_guard<std::mutex> lock(pen_ws_mutex());
        auto it = pen_ws_cache().find(n);
        if (it != pen_ws_cache().end()) {
            counts = it->second;
        } else if (stream_is_capturing(cs)) {
            // first use raced an active capture: histogram through the
            // output buffer for THIS call; later uncaptured calls
            // populate the cache
            if (y == logits)
                throw std::invalid_argument(
                    "logit_penalties: out must not alias logits when capture "
                    "races first use; call once outside capture first");
            counts = reinterpret_cast<int*>(y);
        } else {
            int* buf = nullptr;
            if (cudaMalloc(&buf, (size_t)n * sizeof(int)) != cudaSuccess) {
                cudaGetLastError();
                // allocation failed: same output-buffer fallback, stay correct
                if (y == logits)
                    throw std::invalid_argument(
                        "logit_penalties: out must not alias logits when the "
                        "counts workspace cannot be allocated");
                counts = reinterpret_cast<int*>(y);
            } else {
                counts = buf;
                pen_ws_cache().emplace(n, buf);
            }
        }
    }
    // zero the histogram, count the ids, then rewrite the logits (all
    // stream-ordered, still async)
    cudaError_t err = cudaMemsetAsync(counts, 0, (size_t)n * sizeof(int), cs);
    if (err == cudaSuccess && m > 0)
        penalty_count_kernel<<<(unsigned)grid_for(m), kBlock, 0, cs>>>(
            ids, n, m, counts);
    if (err == cudaSuccess)
        logit_penalties_kernel<<<(unsigned)grid_for(n), kBlock, 0, cs>>>(
            logits, counts, n, repetition, presence, frequency, y);
    if (err == cudaSuccess)
        err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("logit_penalties kernel launch: ") +
                                 cudaGetErrorString(err));
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
// Batched HF-style combined logit penalties (v1.7): the single-row trio
// above, one call for a whole [rows, n] batch. Rows carry ragged id
// histories (flat ids + rows+1 offsets, decode_step_batched's layout);
// the semantics are per-row IDENTICAL to logit_penalties - the CTRL
// repetition scale, then the presence shift, then c * frequency with c
// the id's count IN THAT ROW - so each row's output is bit-identical to
// running the single-row op on the row (integer counts, no output
// atomics, same-order IEEE float ops; the batched CPU reference simply
// calls the single-row one per row).
// ---------------------------------------------------------------------------

// One block per row: bump the row's histogram buckets for its id range.
// rows with empty ranges contribute nothing; out-of-range ids are host-
// validated on every path - the guard keeps a bad device-resident id
// array (trusted, not synced) from corrupting memory.
namespace {

__global__ void penalty_count_b_kernel(const long long* ids,
                                       const long long* offs, int n,
                                       int* counts) {
    const int row = blockIdx.x;
    const long long begin = offs[row];
    const long long end = offs[row + 1];
    const long long m = end - begin;
    for (long long j = threadIdx.x; j < m; j += blockDim.x) {
        const long long id = ids[begin + j];
        if (id >= 0 && id < (long long)n)
            atomicAdd(&counts[(size_t)row * n + id], 1);
    }
}

// One thread per (row, vocab slot): the single-row apply formula at a
// flat row-major index. grid_for rounds the grid up, so threads past
// the total exit early. Reads x (never y), so in-place out == logits
// stays safe on the cached-workspace path.
__global__ void logit_penalties_b_kernel(const float* x, const int* counts,
                                         long long total, int n,
                                         float repetition, float presence,
                                         float frequency, float* y) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = (int)(i / n);
    const int col = (int)(i % (long long)n);
    float v = x[i];
    const int c = counts[(size_t)row * n + col];
    if (c > 0) {
        if (repetition != 1.0f) v = v > 0.0f ? v / repetition : v * repetition;
        if (presence != 0.0f) v -= presence;
        // separate roundings so the host reference (mul then sub)
        // stays bit-identical - see the single-row kernel note
        if (frequency != 0.0f)
            v = __fsub_rn(v, __fmul_rn((float)c, frequency));
    }
    y[i] = v;
}

} // namespace

std::vector<float> logit_penalties_batched_cpu(
    const std::vector<float>& logits, int rows, int n,
    const std::vector<long long>& token_ids,
    const std::vector<long long>& offs, float repetition, float presence,
    float frequency) {
    if (!(repetition > 0.0f))
        throw std::invalid_argument("repetition must be > 0");
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
    if (offs.size() != (size_t)rows + 1 || offs.size() == 0 ||
        offs.front() != 0 ||
        offs.back() != (long long)token_ids.size())
        throw std::invalid_argument(
            "token_ids offsets must have rows + 1 entries, start at 0 "
            "and end at the id count");
    for (size_t i = 1; i < offs.size(); ++i)
        if (offs[i] < offs[i - 1])
            throw std::invalid_argument(
                "token_ids offsets must be non-decreasing");
    for (long long id : token_ids)
        if (id < 0 || id >= (long long)n)
            throw std::invalid_argument("token id out of range");
    std::vector<float> y((size_t)rows * n);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        // per-row semantics are the single-row op verbatim: dispatch to
        // it so the bit-exact contract holds by construction
        const std::vector<long long> row_ids(
            token_ids.begin() + offs[r], token_ids.begin() + offs[r + 1]);
        std::vector<float> out = logit_penalties_cpu(
            std::vector<float>(row, row + n), row_ids, repetition,
            presence, frequency);
        std::copy(out.begin(), out.end(), y.begin() + (size_t)r * n);
    }
    return y;
}

namespace {

// Per-(rows, n) histogram buffers (rows * n ints, 4 MiB at the standard
// [8, 131072] bench shape). Allocated OUTSIDE stream captures (the
// attention-workspace pattern); a first use that races an active
// capture - or an allocation failure - falls back to running the
// single-row kernels per row with the row's own output slice as the
// histogram scratch (same byte count, cleared then rewritten), which
// needs no extra memory and no allocation.
std::mutex& bpen_ws_mutex() {
    static std::mutex m;
    return m;
}
std::map<std::pair<int, int>, int*>& bpen_ws_cache() {
    static std::map<std::pair<int, int>, int*> c;
    return c;
}

} // namespace

// Workspace-free fallback shared by the capture-race and allocation-
// failure paths: clear the WHOLE output buffer, histogram the ids into
// it (an int view - counts fit exactly because counts and logits have
// the same element count), then let the apply pass read those counts
// and overwrite the buffer with the result. Never touches offs on the
// host, needs no allocation, and every launch is capture-safe.
void borrow_output_fallback(const float* logits, const long long* ids,
                            const long long* offs, int rows, int n,
                            float repetition, float presence,
                            float frequency, float* y, cudaStream_t cs) {
    const long long total = (long long)rows * n;
    cudaError_t err = cudaMemsetAsync(y, 0, (size_t)total * sizeof(int), cs);
    if (err == cudaSuccess)
        penalty_count_b_kernel<<<rows, kBlock, 0, cs>>>(
            ids, offs, n, reinterpret_cast<int*>(y));
    if (err == cudaSuccess)
        logit_penalties_b_kernel<<<(unsigned)grid_for(total), kBlock,
                                   0, cs>>>(
            logits, reinterpret_cast<const int*>(y), total, n, repetition,
            presence, frequency, y);
    if (err == cudaSuccess)
        err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("logit_penalties_batched kernel launch: ") +
            cudaGetErrorString(err));
}

void logit_penalties_batched_launch(const float* logits,
                                    const long long* ids,
                                    const long long* offs, int rows, int n,
                                    float repetition, float presence,
                                    float frequency, float* y,
                                    std::uintptr_t stream) {
    if (rows <= 0 || n <= 0) return;
    if (!(repetition > 0.0f))
        throw std::invalid_argument("repetition must be > 0");
    cudaStream_t cs = (cudaStream_t)stream;
    int* counts = nullptr;
    {
        std::lock_guard<std::mutex> lock(bpen_ws_mutex());
        const auto key = std::make_pair(rows, n);
        auto it = bpen_ws_cache().find(key);
        if (it != bpen_ws_cache().end()) {
            counts = it->second;
        } else if (stream_is_capturing(cs)) {
            // first use raced an active capture: borrow the WHOLE
            // output buffer as the histogram scratch for THIS call
            // (same byte count as counts, cleared then rewritten by
            // the apply pass). Aliased in-place calls have no safe
            // form here; every non-capturing later call populates the
            // cache
            if (y == logits)
                throw std::invalid_argument(
                    "logit_penalties_batched: out must not alias logits "
                    "when capture races first use; call once outside "
                    "capture first");
            borrow_output_fallback(logits, ids, offs, rows, n, repetition,
                                   presence, frequency, y, cs);
            return;
        } else {
            int* buf = nullptr;
            if (cudaMalloc(&buf, (size_t)rows * n * sizeof(int))
                    != cudaSuccess) {
                cudaGetLastError();
                // allocation failed: same borrowed-output fallback
                if (y == logits)
                    throw std::invalid_argument(
                        "logit_penalties_batched: out must not alias "
                        "logits when the counts workspace cannot be "
                        "allocated");
                borrow_output_fallback(logits, ids, offs, rows, n,
                                       repetition, presence, frequency, y,
                                       cs);
                return;
            }
            counts = buf;
            bpen_ws_cache().emplace(key, buf);
        }
    }
    // zero the histograms, count the ids, then rewrite the logits (all
    // stream-ordered, still async)
    cudaError_t err = cudaMemsetAsync(counts, 0,
                                      (size_t)rows * n * sizeof(int), cs);
    if (err == cudaSuccess)
        penalty_count_b_kernel<<<rows, kBlock, 0, cs>>>(ids, offs, n,
                                                        counts);
    if (err == cudaSuccess)
        logit_penalties_b_kernel<<<(unsigned)grid_for((long long)rows * n),
                                   kBlock, 0, cs>>>(
            logits, counts, (long long)rows * n, n, repetition, presence,
            frequency, y);
    if (err == cudaSuccess)
        err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("logit_penalties_batched kernel launch: ") +
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

// top-a sampling: keep every token with p_i >= top_a * p_max^2, the
// top-a rule from the open-source sampling stack. Prefix cut on a VALUE
// threshold of the normalized probability (p_max = 1 / total in the
// max-normalized exp column, so the cutoff in exp units is
// top_a / total - the same quantity the device serial kernel derives
// from the float total; here everything runs in double, giving the
// usual neighboring-draw caveat on rounding boundaries). The cutoff is
// bounded by p_max for any top_a <= 1, so the nucleus is never empty.
long long sample_topa_cpu(const std::vector<float>& logits, float top_a,
                          float t, unsigned long long seed) {
    if (logits.empty())
        throw std::invalid_argument("sample of empty logits");
    if (!(top_a > 0.0f && top_a <= 1.0f))
        throw std::invalid_argument("top_a must be in (0, 1]");
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

    // total in double: both the cutoff and the membership comparisons
    // derive from it
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) total += mass_at(i);
    const double cutoff_p = (double)top_a / (total * total);   // a * p_max^2

    // nucleus: prefix while p_i >= cutoff (p of the rank-0 element is
    // the max probability, which the cutoff never exceeds for a valid
    // top_a <= 1, so the nucleus is never empty)
    float nucleus_mass = 0.0f;
    size_t nucleus = 0;
    for (size_t i = 0; i < n; ++i) {
        const double p = mass_at(i) / total;
        if (p < cutoff_p) { nucleus = i; break; }
        nucleus_mass += mass_at(i);
        nucleus = i + 1;
    }
    if (nucleus == 0) nucleus = 1;   // min-token guard against drift

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

// top-n-sigma sampling (Shi et al. 2024, "Top-n sigma: Not All Logits
// Are You Need"): keep every token whose temperature-scaled logit is
// at least mu - nsigma * sigma (mu / sigma: the row's mean and
// standard deviation over the scaled logits themselves), renormalize
// within that prefix and inverse-CDF the splitmix-hash uniform. The
// moments run in double here while the device derives them from float
// accumulators - the usual neighboring-draw caveat on rounding
// boundaries applies. A near-flat row (sigma -> 0, cutoff -> every
// logit) keeps the whole vocabulary.
long long sample_nsigma_cpu(const std::vector<float>& logits,
                            float nsigma, float t,
                            unsigned long long seed) {
    if (logits.empty())
        throw std::invalid_argument("sample of empty logits");
    if (!(nsigma > 0.0f))
        throw std::invalid_argument("nsigma must be > 0");
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
    // max-shifted moments in double: the cutoff derives from them
    double s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)(logits[order[i]] * inv_t) - row_max;
        s1 += d;
        s2 += d * d;
    }
    const double mu_d = s1 / (double)n;
    const double sigma =
        std::sqrt(std::fmax(s2 / (double)n - mu_d * mu_d, 0.0));
    const double cutoff = row_max + mu_d - (double)nsigma * sigma;

    // nucleus: prefix while the scaled logit stays at or above the
    // cutoff (the rank-0 logit equals the max, which the cutoff never
    // exceeds, so the nucleus is never empty)
    float nucleus_mass = 0.0f;
    size_t nucleus = 0;
    auto mass_at = [&](size_t i) {
        return std::exp(logits[order[i]] * inv_t - row_max);
    };
    for (size_t i = 0; i < n; ++i) {
        if ((double)(logits[order[i]] * inv_t) < cutoff) {
            nucleus = i;
            break;
        }
        nucleus_mass += mass_at(i);
        nucleus = i + 1;
    }
    if (nucleus == 0) nucleus = 1;   // min-token guard against drift

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

// eta-cutoff sampling (v1.6, Hewitt et al. 2022): keep every token with
// p_i >= eta * min(1, exp(-H)), H the distribution entropy in nats,
// renormalize within the kept prefix and inverse-CDF the splitmix-hash
// uniform. Prefix cut on a VALUE threshold of the normalized
// probability (p_i = exp / total), so the structure mirrors
// sample_minp_cpu; the entropy uses exact exp/log here while the device
// derives it from __expf accumulators - the usual neighboring-draw
// caveat on rounding boundaries applies, and the accumulated H itself
// can drift ~ulps between paths (same class as top-p's atomic total).
long long sample_eta_cpu(const std::vector<float>& logits, float eta,
                         float t, unsigned long long seed) {
    if (logits.empty())
        throw std::invalid_argument("sample of empty logits");
    if (!(eta > 0.0f && eta <= 1.0f))
        throw std::invalid_argument("eta must be in (0, 1]");
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

    // total and entropy accumulator (double: the entropy is the one
    // quantity both the cutoff and the widening bound derive from)
    double total = 0.0;
    double s_acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double e = mass_at(i);
        const double lv = (double)(logits[order[i]] * inv_t) - row_max;
        total += e;
        s_acc += e * lv;
    }
    const double h = std::log(total) - s_acc / total;
    const double cutoff_p =
        (double)eta * std::fmin(1.0, std::exp(-h));   // absolute prob

    // nucleus: prefix while p_i >= cutoff (p of the rank-0 element is
    // the max probability, which bounds the weighted geometric mean
    // cutoff, so the nucleus is never empty for a valid eta)
    float nucleus_mass = 0.0f;
    size_t nucleus = 0;
    for (size_t i = 0; i < n; ++i) {
        const double p = mass_at(i) / total;
        if (p < cutoff_p) { nucleus = i; break; }
        nucleus_mass += mass_at(i);
        nucleus = i + 1;
    }
    if (nucleus == 0) nucleus = 1;   // min-token guard against drift

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

// locally typical sampling (v1.6, Meister et al. 2022): keep the
// smallest value-ordered band whose mass reaches typical * total. The
// shifted surprise |lv_i - m| (lv the max-shifted logit, m its
// p-weighted mean) is U-shaped along the descending-p order, so the
// band is a contiguous slice found by expanding from the valley with
// two pointers; the draw renormalizes inside the band replaying its
// cumsum in ascending index order - identical structure to the device
// serial kernel, exact exp/log here (neighboring-draw caveat on
// rounding boundaries as usual).
long long sample_typical_cpu(const std::vector<float>& logits, float typical,
                             float t, unsigned long long seed) {
    if (logits.empty())
        throw std::invalid_argument("sample of empty logits");
    if (!(typical > 0.0f && typical <= 1.0f))
        throw std::invalid_argument("typical must be in (0, 1]");
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

    double total = 0.0;
    double s_acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double e = mass_at(i);
        const double lv = (double)(logits[order[i]] * inv_t) - row_max;
        total += e;
        s_acc += e * lv;
    }
    const double m = s_acc / total;   // p-mean of lv; shifted = |lv - m|

    auto shifted_at = [&](size_t i) {
        return std::fabs((double)(logits[order[i]] * inv_t) - row_max - m);
    };

    // the valley seeds the band; two-pointer grows it in ascending
    // shifted order (merging the two monotone arms) until the band mass
    // reaches typical * total
    size_t amin = 0;
    double best = shifted_at(0);
    for (size_t i = 1; i < n; ++i) {
        const double d = shifted_at(i);
        if (d < best) { best = d; amin = i; }
    }
    size_t lo = amin, hi = amin;
    double mass = (double)mass_at(amin);
    const double need = (double)typical * total;
    while (mass < need && (lo > 0 || hi < n - 1)) {
        const double dl = (lo > 0) ? shifted_at(lo - 1)
                                   : std::numeric_limits<double>::infinity();
        const double dr = (hi < n - 1)
                              ? shifted_at(hi + 1)
                              : std::numeric_limits<double>::infinity();
        if (dl <= dr) { --lo; mass += (double)mass_at(lo); }
        else { ++hi; mass += (double)mass_at(hi); }
    }

    // splitmix64-finalized uniform, identical to the device side
    const float u = splitmix_uniform(seed);

    const double target = (double)u * mass;
    double cum = 0.0;
    for (size_t i = lo; i <= hi; ++i) {
        cum += (double)mass_at(i);
        if (cum >= target) return (long long)order[i];
    }
    return (long long)order[hi];              // float rounding fallback
}

std::vector<long long> sample_eta_batched_cpu(
    const std::vector<float>& logits, int rows, int n, float eta, float t,
    const std::vector<unsigned long long>& seeds) {
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_eta_cpu(std::vector<float>(row, row + n),
                                      eta, t, seeds[r]));
    }
    return out;
}

std::vector<long long> sample_typical_batched_cpu(
    const std::vector<float>& logits, int rows, int n, float typical,
    float t, const std::vector<unsigned long long>& seeds) {
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_typical_cpu(std::vector<float>(row, row + n),
                                          typical, t, seeds[r]));
    }
    return out;
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
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
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
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
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
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_minp_cpu(std::vector<float>(row, row + n),
                                      min_p, t, seeds[r]));
    }
    return out;
}

std::vector<long long> sample_topa_batched_cpu(
    const std::vector<float>& logits, int rows, int n, float top_a,
    float t, const std::vector<unsigned long long>& seeds) {
    if (rows < 0)
        throw std::invalid_argument("rows must be >= 0");
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_topa_cpu(std::vector<float>(row, row + n),
                                      top_a, t, seeds[r]));
    }
    return out;
}

std::vector<long long> sample_nsigma_batched_cpu(
    const std::vector<float>& logits, int rows, int n, float nsigma,
    float t, const std::vector<unsigned long long>& seeds) {
    if (rows < 0)
        throw std::invalid_argument("rows must be >= 0");
    if ((int)seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
    std::vector<long long> out;
    out.reserve((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + (size_t)r * n;
        out.push_back(sample_nsigma_cpu(std::vector<float>(row, row + n),
                                        nsigma, t, seeds[r]));
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
    if ((long long)logits.size() < (long long)rows * n)
        throw std::invalid_argument(
            "logits size must be at least rows * n");
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

std::vector<long long> argmax_batched_cpu(const std::vector<float>& x,
                                          int rows, int n) {
    if (rows < 0)
        throw std::invalid_argument("rows must be >= 0");
    if (n <= 0)
        throw std::invalid_argument("argmax of empty vector");
    if ((long long)x.size() < (long long)rows * n)
        throw std::invalid_argument("x size must be at least rows * n");
    std::vector<long long> out((size_t)rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = x.data() + (size_t)r * n;
        long long best = 0;
        for (int i = 1; i < n; ++i)
            if (row[i] > row[best]) best = i;
        out[r] = best;
    }
    return out;
}

} // namespace fusedtok
