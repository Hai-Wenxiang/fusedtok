// Fused decode-step attention (v0.5): single-token causal attention with
// GQA over a contiguous kv-cache.
//
//   out[b, h] = softmax(q . K[b,kv(h)]^T / sqrt(D)) . V[b,kv(h)]
//
// Two execution strategies behind one launcher:
//
//  1. Single-kernel path (short caches / already-saturated grids): one
//     block per (batch, q head). The block's 8 warps stride the key/value
//     rows; each warp keeps a running ONLINE softmax state in registers -
//     running score max m, denominator l, and the [D] output accumulator -
//     rescaled as new maxima arrive, so scores are never materialized to
//     global memory. A shared-memory merge folds the eight warp partials
//     (weighted by exp(m_warp - m_global)) into the final normalized row.
//
//  2. Flash-decoding split path (long caches): the sequence is cut into
//     slices; stage 1 launches one block per (batch, kv head, slice) that
//     computes the online-softmax partials of ALL q heads of the GQA group
//     over its slice and writes them to a per-shape cached workspace;
//     stage 2 reduces the per-slice partials (max-rescale merge) into the
//     output rows. The grid grows with the sequence length instead of
//     being pinned at B*Hq blocks, which is what keeps long caches
//     bandwidth-saturated on small batches.
//
// Lanes cover the head dimension in float4 chunks (D is a multiple of 4;
// every row base stays 16B aligned since the caller's buffers are
// torch/cudaMalloc allocations and row strides are multiples of 16B).
//
// Sequences with len == 0 (or an empty cache) write zero rows. No host
// round trips: stream-ordered on the caller's stream and CUDA-graph
// capturable (the workspace is allocated OUTSIDE captures on first use;
// a capture that races the first call falls back to the single-kernel
// path, which needs no workspace).

#include "fusedtok/attention.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace fusedtok {

namespace {

constexpr int kAttBlock = 256;       // 8 warps striding the sequence
constexpr int kAttWarps = kAttBlock / 32;
constexpr int kAttMaxDim = 512;      // shared accumulator budget per warp
// float4 chunks a single lane can own: ceil((MaxDim/4) / 32)
constexpr int kAttLaneChunks = kAttMaxDim / 4 / 32;
constexpr int kAttMinSlice = 256;    // do not slice shorter spans
constexpr int kAttMaxSlices = 32;    // workspace cap on the split grid

void attention_check(int batch, int hq, int hkv, int t_seq, int dim) {
    if (batch < 0 || hq < 1 || hkv < 1 || t_seq < 0 || dim < 1)
        throw std::invalid_argument(
            "attention shapes must satisfy batch>=0, heads>=1, rows>=0, dim>=1");
    if (hq % hkv != 0)
        throw std::invalid_argument(
            "q heads must be a multiple of kv heads (contiguous GQA groups)");
    if (dim % 4 != 0 || dim > kAttMaxDim)
        throw std::invalid_argument("dim must be a multiple of 4 and at most 512");
}

__global__ void attn_decode_kernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const int* __restrict__ lens,
                                   float* __restrict__ out,
                                   int hq, int hkv, int t_seq, int dim) {
    const int bi = blockIdx.x / hq;
    const int h = blockIdx.x % hq;
    const int len = lens ? lens[bi] : t_seq;

    float4* o4 = reinterpret_cast<float4*>(
        out + ((size_t)bi * hq + h) * dim);
    const int chunks = dim / 4;

    // empty sequence: an empty softmax is defined as a zero row
    if (len == 0) {
        for (int c = threadIdx.x; c < chunks; c += kAttBlock)
            o4[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // GQA: q heads form contiguous groups over kv heads
    const int kv = (int)((long long)h * hkv / hq);
    const float scale = 1.0f / sqrtf((float)dim);
    const float4* q4 = reinterpret_cast<const float4*>(
        q + ((size_t)bi * hq + h) * dim);
    const float4* k4 = reinterpret_cast<const float4*>(
        k + (((size_t)bi * hkv + kv) * t_seq) * dim);
    const float4* v4 = reinterpret_cast<const float4*>(
        v + (((size_t)bi * hkv + kv) * t_seq) * dim);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // per-lane q chunks (lane owns chunk c for c = lane, lane+32, ...)
    // plus the running online-softmax accumulator over this warp's keys
    float4 qv[kAttLaneChunks];
    float4 acc[kAttLaneChunks];
    int nc = 0;
    for (int c = lane; c < chunks; c += 32, ++nc) {
        qv[nc] = q4[c];
        acc[nc] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float m = -INFINITY;   // running max of this warp's scaled scores
    float l = 0.0f;        // running softmax denominator

    // 8 warps stride the sequence, each with an independent online
    // softmax; warp-local maxima merge later in shared memory
    for (int t = warp; t < len; t += kAttWarps) {
        const float4* krow = k4 + (size_t)t * chunks;
        float dot = 0.0f;
        int j = 0;
        for (int c = lane; c < chunks; c += 32, ++j) {
            const float4 kk = krow[c];
            dot += qv[j].x * kk.x + qv[j].y * kk.y
                 + qv[j].z * kk.z + qv[j].w * kk.w;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        const float s = __shfl_sync(0xffffffffu, dot, 0) * scale;

        // rescale the running state to the new max (exact 1.0 when the
        // max is unchanged; exp(-inf) = 0 handles the first key)
        const float m_new = fmaxf(m, s);
        const float rescale = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * rescale + p;
        j = 0;
        for (int c = lane; c < chunks; c += 32, ++j) {
            const float4 vv = v4[(size_t)t * chunks + c];
            acc[j].x = acc[j].x * rescale + p * vv.x;
            acc[j].y = acc[j].y * rescale + p * vv.y;
            acc[j].z = acc[j].z * rescale + p * vv.z;
            acc[j].w = acc[j].w * rescale + p * vv.w;
        }
        m = m_new;
    }

    // merge the 8 warp partials: sm_acc rows are padded-free (stride
    // MaxDim*4B = 2KB, 16B aligned) so float4 views stay legal
    __shared__ float sm_acc[kAttWarps][kAttMaxDim];
    __shared__ float sm_ml[kAttWarps][2];   // [warp] -> (m, l)
    if (lane == 0) {
        sm_ml[warp][0] = m;
        sm_ml[warp][1] = l;
    }
    {
        int j = 0;
        for (int c = lane; c < chunks; c += 32, ++j)
            *reinterpret_cast<float4*>(&sm_acc[warp][c * 4]) = acc[j];
    }
    __syncthreads();

    // global max over the warps that actually saw keys (l > 0); len > 0
    // guarantees at least one such warp
    float m_all = -INFINITY;
    for (int w = 0; w < kAttWarps; ++w)
        if (sm_ml[w][1] > 0.0f)
            m_all = fmaxf(m_all, sm_ml[w][0]);
    float denom = 0.0f;
    for (int w = 0; w < kAttWarps; ++w)
        if (sm_ml[w][1] > 0.0f)
            denom += sm_ml[w][1] * expf(sm_ml[w][0] - m_all);

    // each lane rescales its own chunks across all warp partials
    int j = 0;
    for (int c = lane; c < chunks; c += 32, ++j) {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int w = 0; w < kAttWarps; ++w) {
            if (sm_ml[w][1] > 0.0f) {
                const float wgt = expf(sm_ml[w][0] - m_all);
                const float4* src =
                    reinterpret_cast<const float4*>(&sm_acc[w][c * 4]);
                s.x += src->x * wgt;
                s.y += src->y * wgt;
                s.z += src->z * wgt;
                s.w += src->w * wgt;
            }
        }
        o4[c] = make_float4(s.x / denom, s.y / denom, s.z / denom,
                            s.w / denom);
    }
}

// ---------------------------------------------------------------------------
// split path, stage 1: block per (batch, kv head, slice).
//
// Computes the online-softmax partials of ALL G q heads sharing this kv
// head over rows [s0, min(len, s0 + slice_len)) and stores them into the
// workspace: one (m, l, acc[D]) triple per (b, kv, slice, q head). The
// key row is loaded once per lane and reused across the G heads; the
// per-warp states are merged through shared memory one head at a time
// (the shared tile is reused across heads, keeping the budget at one
// warp-partial table regardless of G). Empty slices (past len) store
// l = 0 and are skipped by stage 2. G is a compile-time template
// parameter so the per-head register state stays in registers; runtime
// groups larger than 16 stay on the single-kernel path.
// ---------------------------------------------------------------------------

template <int G>
__global__ void attn_split_kernel(const float* __restrict__ q,
                                  const float* __restrict__ k,
                                  const float* __restrict__ v,
                                  const int* __restrict__ lens,
                                  float* __restrict__ ws_ml,
                                  float* __restrict__ ws_acc,
                                  int hq, int hkv, int t_seq, int dim,
                                  int slices, int slice_len) {
    const int slice = blockIdx.x % slices;
    const int kv = (blockIdx.x / slices) % hkv;
    const int bi = blockIdx.x / (slices * hkv);
    const int len = lens ? lens[bi] : t_seq;
    const int t0 = slice * slice_len;
    const int t1 = min(len, t0 + slice_len);

    const float scale = 1.0f / sqrtf((float)dim);
    const int chunks = dim / 4;
    const float4* k4 = reinterpret_cast<const float4*>(
        k + (((size_t)bi * hkv + kv) * t_seq) * dim);
    const float4* v4 = reinterpret_cast<const float4*>(
        v + (((size_t)bi * hkv + kv) * t_seq) * dim);
    const float4* q4 = reinterpret_cast<const float4*>(
        q + ((size_t)bi * hq + (size_t)kv * G) * dim);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // per-lane q chunks for each head of the group + running state.
    // Lanes that own no chunk (chunks < 32, e.g. D=4) keep the empty
    // online state, which the merge skips via l == 0.
    float4 qv[G][kAttLaneChunks];
    float4 acc[G][kAttLaneChunks];
    float m[G], l[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    int nc = 0;
    for (int c = lane; c < chunks; c += 32, ++nc)
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            qv[g][nc] = q4[(size_t)g * chunks + c];
            acc[g][nc] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

    // warps stride the slice; the k/v rows are read once and reused
    // across the whole GQA group
    for (int t = t0 + warp; t < t1; t += kAttWarps) {
        const float4* krow = k4 + (size_t)t * chunks;
        const float4* vrow = v4 + (size_t)t * chunks;
        float4 kc[kAttLaneChunks], vc[kAttLaneChunks];
        int nj = 0;
        for (int c = lane; c < chunks; c += 32, ++nj) {
            kc[nj] = krow[c];
            vc[nj] = vrow[c];
        }
        for (int g = 0; g < G; ++g) {
            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < kAttLaneChunks; ++j) {
                if (j >= nj) break;
                dot += qv[g][j].x * kc[j].x + qv[g][j].y * kc[j].y
                     + qv[g][j].z * kc[j].z + qv[g][j].w * kc[j].w;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            const float s = __shfl_sync(0xffffffffu, dot, 0) * scale;
            const float m_new = fmaxf(m[g], s);
            const float rescale = expf(m[g] - m_new);
            const float p = expf(s - m_new);
            l[g] = l[g] * rescale + p;
            #pragma unroll
            for (int j = 0; j < kAttLaneChunks; ++j) {
                if (j >= nj) break;
                acc[g][j].x = acc[g][j].x * rescale + p * vc[j].x;
                acc[g][j].y = acc[g][j].y * rescale + p * vc[j].y;
                acc[g][j].z = acc[g][j].z * rescale + p * vc[j].z;
                acc[g][j].w = acc[g][j].w * rescale + p * vc[j].w;
            }
            m[g] = m_new;
        }
    }

    // merge warp partials per head, reusing one shared table across heads
    __shared__ float sm_acc[kAttWarps][kAttMaxDim];
    __shared__ float sm_ml[kAttWarps][2];
    for (int g = 0; g < G; ++g) {
        if (lane == 0) {
            sm_ml[warp][0] = m[g];
            sm_ml[warp][1] = l[g];
        }
        {
            int j = 0;
            for (int c = lane; c < chunks; c += 32, ++j)
                *reinterpret_cast<float4*>(&sm_acc[warp][c * 4]) = acc[g][j];
        }
        __syncthreads();
        const size_t pidx =
            (((size_t)bi * hkv + kv) * slices + slice) * G + g;
        float m_all = -INFINITY;
        for (int w = 0; w < kAttWarps; ++w)
            if (sm_ml[w][1] > 0.0f)
                m_all = fmaxf(m_all, sm_ml[w][0]);
        float l_all = 0.0f;
        for (int w = 0; w < kAttWarps; ++w)
            if (sm_ml[w][1] > 0.0f)
                l_all += sm_ml[w][1] * expf(sm_ml[w][0] - m_all);
        if (threadIdx.x == 0) {
            ws_ml[pidx * 2] = m_all;       // m stays -inf for empty slices
            ws_ml[pidx * 2 + 1] = l_all;   // 0 marks "skipped" in stage 2
        }
        int j = 0;
        for (int c = lane; c < chunks; c += 32, ++j) {
            float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int w = 0; w < kAttWarps; ++w) {
                if (sm_ml[w][1] > 0.0f) {
                    const float wgt = expf(sm_ml[w][0] - m_all);
                    const float4* src =
                        reinterpret_cast<const float4*>(&sm_acc[w][c * 4]);
                    s.x += src->x * wgt;
                    s.y += src->y * wgt;
                    s.z += src->z * wgt;
                    s.w += src->w * wgt;
                }
            }
            *reinterpret_cast<float4*>(&ws_acc[pidx * dim + c * 4]) = s;
        }
        __syncthreads();                   // sm reused by the next head
    }
}

// ---------------------------------------------------------------------------
// split path, stage 2: block per (batch, q head).
//
// Streams the slice partials of this q head and folds them with the same
// max-rescale rule the online softmax uses (first live partial adopted
// verbatim, later ones rescaled). All-skip (len == 0) writes zero rows.
// Every thread reads the same partial rows - L1 broadcast territory, no
// cross-thread communication needed until the final divide.
// ---------------------------------------------------------------------------

__global__ void attn_reduce_kernel(const float* __restrict__ ws_ml,
                                   const float* __restrict__ ws_acc,
                                   float* __restrict__ out,
                                   int hq, int hkv, int dim,
                                   int group, int slices) {
    const int bi = blockIdx.x / hq;
    const int h = blockIdx.x % hq;
    const int kv = h / group;
    const int gi = h % group;
    const int chunks = dim / 4;
    const int lane = threadIdx.x & 31;

    float4* o4 = reinterpret_cast<float4*>(
        out + ((size_t)bi * hq + h) * dim);

    float4 o[kAttLaneChunks];
    float m = -INFINITY, l = 0.0f;
    int nc = 0;
    for (int c = lane; c < chunks; c += 32, ++nc)
        o[nc] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const size_t base = (((size_t)bi * hkv + kv) * slices) * group + gi;
    for (int s = 0; s < slices; ++s) {
        const size_t p = base + (size_t)s * group;
        const float ms = ws_ml[p * 2];
        const float ls = ws_ml[p * 2 + 1];
        if (ls <= 0.0f)
            continue;                      // empty slice (past len)
        const float4* pa = reinterpret_cast<const float4*>(
            &ws_acc[p * dim]);
        if (l == 0.0f) {                   // first live partial: adopt
            m = ms;
            l = ls;
            int j = 0;
            for (int c = lane; c < chunks; c += 32, ++j)
                o[j] = pa[c];
            continue;
        }
        const float m_new = fmaxf(m, ms);
        const float r_old = expf(m - m_new), r_new = expf(ms - m_new);
        l = l * r_old + ls * r_new;
        int j = 0;
        for (int c = lane; c < chunks; c += 32, ++j) {
            const float4 a = pa[c];
            o[j].x = o[j].x * r_old + a.x * r_new;
            o[j].y = o[j].y * r_old + a.y * r_new;
            o[j].z = o[j].z * r_old + a.z * r_new;
            o[j].w = o[j].w * r_old + a.w * r_new;
        }
        m = m_new;
    }
    if (l <= 0.0f) {                       // every slice empty: zero row
        for (int c = threadIdx.x; c < chunks; c += kAttBlock)
            o4[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    int j = 0;
    for (int c = lane; c < chunks; c += 32, ++j)
        o4[c] = make_float4(o[j].x / l, o[j].y / l, o[j].z / l, o[j].w / l);
}

// ---------------------------------------------------------------------------
// per-shape workspace cache for the split path (process lifetime, like
// the selection pipeline's scratch). Allocations happen OUTSIDE stream
// captures; a first call that races an active capture falls back to the
// single-kernel path instead of allocating mid-capture.
// ---------------------------------------------------------------------------

struct AttWs {
    float* ml;
    float* acc;
};

using AttWsKey = std::tuple<int, int, int, int>;  // (batch*kv, slices, group, dim)

std::mutex& att_ws_mutex() {
    static std::mutex m;
    return m;
}
std::map<AttWsKey, AttWs>& att_ws_cache() {
    static std::map<AttWsKey, AttWs> c;
    return c;
}

// Slice-count heuristic: aim for ~4 waves of split blocks, never slice
// spans below kAttMinSlice and never beyond kAttMaxSlices.
int att_choose_slices(int batch, int hkv, int t_seq, int sm_count) {
    if (t_seq < 2 * kAttMinSlice)
        return 1;                          // too short to be worth it
    const int target = std::max(1, (4 * sm_count) / std::max(1, batch * hkv));
    if (target <= 1)
        return 1;                          // the naive grid is already full
    const int slice_len = std::max(kAttMinSlice, (t_seq + target - 1) / target);
    const int slices = (t_seq + slice_len - 1) / slice_len;
    return std::min(slices, kAttMaxSlices);
}

int sm_multi_processor_count() {
    static int cached = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev)
                != cudaSuccess) {
            cudaGetLastError();
            return 28;                     // sane default if the query fails
        }
        return n;
    }();
    return cached;
}

} // namespace

// ---------------------------------------------------------------------------
// CPU reference: plain two-pass softmax attention, float32 throughout
// (same expf as the kernel; only summation order differs)
// ---------------------------------------------------------------------------

std::vector<float> attention_decode_cpu(const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v,
                                        const std::vector<int>* lens,
                                        int batch, int hq, int hkv,
                                        int t_seq, int dim) {
    attention_check(batch, hq, hkv, t_seq, dim);
    if (q.size() < (size_t)batch * hq * dim ||
        k.size() < (size_t)batch * hkv * t_seq * dim ||
        v.size() < (size_t)batch * hkv * t_seq * dim)
        throw std::invalid_argument("attention operand size mismatch");
    if (lens && lens->size() < (size_t)batch)
        throw std::invalid_argument("lens must have batch entries");
    if (lens) {
        for (int i = 0; i < batch; ++i)
            if ((*lens)[i] < 0 || (*lens)[i] > t_seq)
                throw std::invalid_argument(
                    "lens entries must be within [0, cache rows]");
    }

    const int group = hq / hkv;
    const float scale = 1.0f / sqrtf((float)dim);
    std::vector<float> out((size_t)batch * hq * dim, 0.0f);
    std::vector<float> scores(t_seq > 0 ? t_seq : 1);
    for (int bi = 0; bi < batch; ++bi) {
        const int len = lens ? (*lens)[bi] : t_seq;
        for (int h = 0; h < hq; ++h) {
            if (len == 0)
                continue;              // zero row convention (out is zeroed)
            const int kv = h / group;
            const float* qp = q.data() + ((size_t)bi * hq + h) * dim;
            const float* kp = k.data() +
                (((size_t)bi * hkv + kv) * t_seq) * dim;
            const float* vp = v.data() +
                (((size_t)bi * hkv + kv) * t_seq) * dim;
            float m = -INFINITY;
            for (int t = 0; t < len; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d)
                    dot += qp[d] * kp[(size_t)t * dim + d];
                scores[t] = dot * scale;
                m = fmaxf(m, scores[t]);
            }
            float l = 0.0f;
            for (int t = 0; t < len; ++t)
                l += expf(scores[t] - m);
            float* op = out.data() + ((size_t)bi * hq + h) * dim;
            for (int t = 0; t < len; ++t) {
                const float p = expf(scores[t] - m) / l;
                for (int d = 0; d < dim; ++d)
                    op[d] += p * vp[(size_t)t * dim + d];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// launcher: single-kernel path for short caches / saturated grids /
// unusual group sizes, flash-decoding split path (stage1 partials +
// stage2 reduce) for long caches. Stream-ordered and graph-capturable on
// both paths (the workspace is allocated outside captures on first use).
// ---------------------------------------------------------------------------

void attention_decode_launch(const float* q, const float* k, const float* v,
                             const int* lens, float* out,
                             int batch, int hq, int hkv, int t_seq, int dim,
                             std::uintptr_t stream) {
    attention_check(batch, hq, hkv, t_seq, dim);
    if (batch == 0)
        return;                        // nothing to compute
    cudaStream_t cs = (cudaStream_t)stream;
    const int group = hq / hkv;

    // the split kernels are templated over the GQA group width; unusual
    // divisors (3, 5, 6, ...) stay on the single-kernel path
    const bool splittable = group == 1 || group == 2 || group == 4 ||
                            group == 8 || group == 16;
    int slices = 1;
    if (splittable)
        slices = att_choose_slices(batch, hkv, t_seq,
                                   sm_multi_processor_count());
    if (slices <= 1) {
        attn_decode_kernel<<<batch * hq, kAttBlock, 0, cs>>>(
            q, k, v, lens, out, hq, hkv, t_seq, dim);
        check_launch("attention decode kernel launch");
        return;
    }

    // split path: find (or allocate, outside captures) the workspace
    const AttWsKey key(batch * hkv, slices, group, dim);
    AttWs ws{nullptr, nullptr};
    {
        std::lock_guard<std::mutex> lock(att_ws_mutex());
        auto it = att_ws_cache().find(key);
        if (it != att_ws_cache().end()) {
            ws = it->second;
        } else if (stream_is_capturing(cs)) {
            // first use raced an active capture: the single-kernel path
            // needs no workspace, use it for THIS launch (and for the
            // whole capture); later uncaptured calls populate the cache
            attn_decode_kernel<<<batch * hq, kAttBlock, 0, cs>>>(
                q, k, v, lens, out, hq, hkv, t_seq, dim);
            check_launch("attention decode kernel launch");
            return;
        } else {
            const size_t npart = (size_t)batch * hkv * slices * group;
            float* ml = nullptr;
            float* acc = nullptr;
            if (cudaMalloc(&ml, npart * 2 * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&acc, npart * (size_t)dim * sizeof(float))
                    != cudaSuccess) {
                cudaGetLastError();
                // allocation failed: fall back, stay correct
                attn_decode_kernel<<<batch * hq, kAttBlock, 0, cs>>>(
                    q, k, v, lens, out, hq, hkv, t_seq, dim);
                check_launch("attention decode kernel launch");
                return;
            }
            ws = AttWs{ml, acc};
            att_ws_cache().emplace(key, ws);
        }
    }

    const int grid1 = batch * hkv * slices;
    const int slice_len = (t_seq + slices - 1) / slices;
    if (group == 1)
        attn_split_kernel<1><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else if (group == 2)
        attn_split_kernel<2><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else if (group == 4)
        attn_split_kernel<4><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else if (group == 8)
        attn_split_kernel<8><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else                          // splittable already pinned group == 16
        attn_split_kernel<16><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    check_launch("attention split kernel launch");
    attn_reduce_kernel<<<batch * hq, kAttBlock, 0, cs>>>(
        ws.ml, ws.acc, out, hq, hkv, dim, group, slices);
    check_launch("attention reduce kernel launch");
}

} // namespace fusedtok
