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
// Lanes cover the head dimension in 4-element chunks (D is a multiple of
// 4; every row base stays aligned for the storage width since the
// caller's buffers are torch/cudaMalloc allocations and row strides are
// multiples of the chunk size).
//
// STORAGE DTYPES (v1.1): q/k/v/out may be float32, bfloat16 or float16 -
// the kernels are templated on the storage type and compute entirely in
// float32 (loads widen at the boundary, stores narrow back
// round-to-nearest). bf16/fp16 cache rows are 8B per 4-element chunk vs
// f32's 16B: half the global bytes on the bandwidth-bound decode path.
// The workspace and every softmax/accumulator state stay float32 on all
// paths, so numerics are identical across storage dtypes up to the
// input rounding.
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
#include <cuda_fp16.h>

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
// prefill lane-group slices: chunks/LPR per lane is at most 8 for
// every (dim band, LPR) combination the launcher picks
constexpr int kAttPrefChunks = 8;

// ---------------------------------------------------------------------------
// storage-dtype chunk traits: one 4-element chunk of q/k/v/out per access,
// widened to float4 for compute and narrowed back on store. float keeps
// the native float4 (16B); bf16/fp16 pack 4 elements into a uint2 (8B) -
// half the bytes per cache row, which is exactly the win the
// bandwidth-bound decode path is after. Alignment: D % 4 == 0 makes every
// row offset a multiple of the 8B chunk, and torch/cudaMalloc bases are
// far stricter than that.
// ---------------------------------------------------------------------------
template <typename T> struct AttChunk;

template <> struct AttChunk<float> {
    using Vec = float4;
    __device__ __forceinline__ static float4 ld(const float* p) {
        return *reinterpret_cast<const float4*>(p);
    }
    __device__ __forceinline__ static void st(float* p, float4 v) {
        *reinterpret_cast<float4*>(p) = v;
    }
};

template <> struct AttChunk<__nv_bfloat16> {
    using Vec = uint2;
    __device__ __forceinline__ static float4 ld(const __nv_bfloat16* p) {
        const uint2 u = *reinterpret_cast<const uint2*>(p);
        const float2 lo = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 hi = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&u.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    __device__ __forceinline__ static void st(__nv_bfloat16* p, float4 v) {
        const __nv_bfloat162 lo = __float22bfloat162_rn(make_float2(v.x, v.y));
        const __nv_bfloat162 hi = __float22bfloat162_rn(make_float2(v.z, v.w));
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&lo);
        u.y = *reinterpret_cast<const unsigned*>(&hi);
        *reinterpret_cast<uint2*>(p) = u;
    }
};

template <> struct AttChunk<__half> {
    using Vec = uint2;
    __device__ __forceinline__ static float4 ld(const __half* p) {
        const uint2 u = *reinterpret_cast<const uint2*>(p);
        const float2 lo = __half22float2(
            *reinterpret_cast<const __half2*>(&u.x));
        const float2 hi = __half22float2(
            *reinterpret_cast<const __half2*>(&u.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    __device__ __forceinline__ static void st(__half* p, float4 v) {
        const __half2 lo = __float22half2_rn(make_float2(v.x, v.y));
        const __half2 hi = __float22half2_rn(make_float2(v.z, v.w));
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&lo);
        u.y = *reinterpret_cast<const unsigned*>(&hi);
        *reinterpret_cast<uint2*>(p) = u;
    }
};

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

template <typename T>
__global__ void attn_decode_kernel(const T* __restrict__ q,
                                   const T* __restrict__ k,
                                   const T* __restrict__ v,
                                   const int* __restrict__ lens,
                                   T* __restrict__ out,
                                   int hq, int hkv, int t_seq, int dim) {
    using C = AttChunk<T>;
    const int bi = blockIdx.x / hq;
    const int h = blockIdx.x % hq;
    const int len = lens ? lens[bi] : t_seq;

    T* oraw = out + ((size_t)bi * hq + h) * dim;
    const int chunks = dim / 4;

    // empty sequence: an empty softmax is defined as a zero row
    if (len == 0) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int c = threadIdx.x; c < chunks; c += kAttBlock)
            C::st(oraw + c * 4, zero);
        return;
    }

    // GQA: q heads form contiguous groups over kv heads
    const int kv = (int)((long long)h * hkv / hq);
    const float scale = 1.0f / sqrtf((float)dim);
    const T* qp = q + ((size_t)bi * hq + h) * dim;
    const T* kp = k + (((size_t)bi * hkv + kv) * t_seq) * dim;
    const T* vp = v + (((size_t)bi * hkv + kv) * t_seq) * dim;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // per-lane q chunks (lane owns chunk c for c = lane, lane+32, ...)
    // plus the running online-softmax accumulator over this warp's keys;
    // `j` is the per-lane chunk index used consistently from load to
    // store (a second name for the same mapping hides the invariant)
    float4 qv[kAttLaneChunks];
    float4 acc[kAttLaneChunks];
    int j = 0;
    for (int c = lane; c < chunks; c += 32, ++j)
        qv[j] = C::ld(qp + c * 4);
    for (int i = 0; i < kAttLaneChunks; ++i)
        acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float m = -INFINITY;   // running max of this warp's scaled scores
    float l = 0.0f;        // running softmax denominator

    // 8 warps stride the sequence, each with an independent online
    // softmax; warp-local maxima merge later in shared memory
    for (int t = warp; t < len; t += kAttWarps) {
        const T* krow = kp + (size_t)t * dim;
        float dot = 0.0f;
        j = 0;
        for (int c = lane; c < chunks; c += 32, ++j) {
            const float4 kk = C::ld(krow + c * 4);
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
            const float4 vv = C::ld(vp + ((size_t)t * chunks + c) * 4);
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
        j = 0;
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
    j = 0;
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
        C::st(oraw + c * 4, make_float4(s.x / denom, s.y / denom,
                                        s.z / denom, s.w / denom));
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

template <int G, typename T>
__global__ void attn_split_kernel(const T* __restrict__ q,
                                  const T* __restrict__ k,
                                  const T* __restrict__ v,
                                  const int* __restrict__ lens,
                                  float* __restrict__ ws_ml,
                                  float* __restrict__ ws_acc,
                                  int hq, int hkv, int t_seq, int dim,
                                  int slices, int slice_len) {
    using C = AttChunk<T>;
    const int slice = blockIdx.x % slices;
    const int kv = (blockIdx.x / slices) % hkv;
    const int bi = blockIdx.x / (slices * hkv);
    const int len = lens ? lens[bi] : t_seq;
    const int t0 = slice * slice_len;
    const int t1 = min(len, t0 + slice_len);

    const float scale = 1.0f / sqrtf((float)dim);
    const int chunks = dim / 4;
    const T* kp = k + (((size_t)bi * hkv + kv) * t_seq) * dim;
    const T* vp = v + (((size_t)bi * hkv + kv) * t_seq) * dim;
    const T* qp = q + ((size_t)bi * hq + (size_t)kv * G) * dim;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // per-lane q chunks for each head of the group + running state.
    // Lanes that own no chunk (chunks < 32, e.g. D=4) keep the empty
    // online state, which the merge skips via l == 0. `nj` is the
    // per-lane chunk count; `j` below walks the same mapping.
    float4 qv[G][kAttLaneChunks];
    float4 acc[G][kAttLaneChunks];
    float m[G], l[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    int nj = 0;
    for (int c = lane; c < chunks; c += 32, ++nj)
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            qv[g][nj] = C::ld(qp + (size_t)g * dim + c * 4);
            acc[g][nj] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

    // warps stride the slice; the k/v rows are read once and reused
    // across the whole GQA group
    for (int t = t0 + warp; t < t1; t += kAttWarps) {
        const T* krow = kp + (size_t)t * dim;
        const T* vrow = vp + (size_t)t * dim;
        float4 kc[kAttLaneChunks], vc[kAttLaneChunks];
        for (int j = 0; j < nj; ++j) {
            kc[j] = C::ld(krow + (lane + (size_t)j * 32) * 4);
            vc[j] = C::ld(vrow + (lane + (size_t)j * 32) * 4);
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

template <typename T>
__global__ void attn_reduce_kernel(const float* __restrict__ ws_ml,
                                   const float* __restrict__ ws_acc,
                                   T* __restrict__ out,
                                   int hq, int hkv, int dim,
                                   int group, int slices) {
    using C = AttChunk<T>;
    const int bi = blockIdx.x / hq;
    const int h = blockIdx.x % hq;
    const int kv = h / group;
    const int gi = h % group;
    const int chunks = dim / 4;
    const int lane = threadIdx.x & 31;

    T* oraw = out + ((size_t)bi * hq + h) * dim;

    float4 o[kAttLaneChunks];
    float m = -INFINITY, l = 0.0f;
    // per-lane partial chunks (lane owns chunk c for c = lane, lane+32,
    // ...); `j` walks the same mapping from reset to store
    int j = 0;
    for (int c = lane; c < chunks; c += 32, ++j)
        o[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

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
            j = 0;
            for (int c = lane; c < chunks; c += 32, ++j)
                o[j] = pa[c];
            continue;
        }
        const float m_new = fmaxf(m, ms);
        const float r_old = expf(m - m_new), r_new = expf(ms - m_new);
        l = l * r_old + ls * r_new;
        j = 0;
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
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int c = threadIdx.x; c < chunks; c += kAttBlock)
            C::st(oraw + c * 4, zero);
        return;
    }
    j = 0;
    for (int c = lane; c < chunks; c += 32, ++j)
        C::st(oraw + c * 4, make_float4(o[j].x / l, o[j].y / l,
                                        o[j].z / l, o[j].w / l));
}

// ---------------------------------------------------------------------------
// prefill kernel: fresh-sequence attention over S query rows (the
// lightweight v0.5 prefill - no tensor cores, bandwidth-first).
//
// One block per (batch, q head, tile of QTILE query rows); K/V stream
// through KVTILE-row shared chunks so every global load is reused by
// the whole tile. The latency-critical inner dot avoids the classic
// 10-shuffle warp reduction: lanes split into RPW row groups of
// LANES_PER_ROW = 32 / RPW lanes, each lane owning an equal slice of
// the head dimension (8 float4 chunks for every supported shape), so a
// row's dot needs only log2(LANES_PER_ROW) xor-shuffles with no
// broadcast and no cross-row serialization. q slices and accumulators
// live in registers; different rows in a warp share the same k/v
// shared reads for free (broadcast).
//
//   dim <= 128: 64-row tile, 8 rows/warp,  4 lanes/row
//   dim <= 256: 32-row tile, 4 rows/warp,  8 lanes/row
//   dim <= 512: 16-row tile, 2 rows/warp, 16 lanes/row
//   dim  <  32: the dim<=128 band, with the surplus lanes of a row
//               group contributing zero (bounds-guarded loads/stores)
// Causality falls out of each row's own limit (row i stops at key i+1).
// ---------------------------------------------------------------------------

template <int QTILE, int KVTILE, int LPR, typename T>
__global__ void attn_prefill_kernel(const T* __restrict__ q,
                                    const T* __restrict__ k,
                                    const T* __restrict__ v,
                                    T* __restrict__ out,
                                    int hq, int hkv, int seq, int dim,
                                    int tiles, int causal) {
    using C = AttChunk<T>;
    constexpr int RPW = 32 / LPR;      // query rows per warp
    const int tile = blockIdx.x % tiles;
    const int h = (blockIdx.x / tiles) % hq;
    const int bi = blockIdx.x / (tiles * hq);
    const int group = hq / hkv;
    const int kv = h / group;
    const int row0 = tile * QTILE;
    const int chunks = dim / 4;        // per row
    // ceil: the last lane of a row may own a partial slice (chunk
    // counts not divisible by LPR, e.g. dim=36 -> 9 chunks over 4
    // lanes); out-of-range chunks read as zero and never store
    const int lane_chunks = (chunks + LPR - 1) / LPR;
    const float scale = 1.0f / sqrtf((float)dim);

    const T* kp = k + (((size_t)bi * hkv + kv) * seq) * dim;
    const T* vp = v + (((size_t)bi * hkv + kv) * seq) * dim;
    const T* qp = q + (((size_t)bi * hq + h) * seq) * dim;

    // dynamic shared layout: [QTILE x dim q tile][KVTILE x dim k][v]
    extern __shared__ float sm[];
    float* q_s = sm;
    float* k_s = sm + QTILE * dim;
    float* v_s = k_s + KVTILE * dim;

    // stage the query tile (padding rows past seq with zeros)
    for (int idx = threadIdx.x; idx < QTILE * chunks; idx += kAttBlock) {
        const int r = idx / chunks, c = idx % chunks;
        const int row = row0 + r;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4*>(&q_s[r * dim + c * 4]) =
            (row < seq) ? C::ld(qp + ((size_t)row * chunks + c) * 4)
                        : zero;
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r_in_warp = lane / LPR;              // row within the warp
    const int i_in_row = lane % LPR;               // lane's slice of dim
    const int row_w = row0 + warp * RPW;           // warp's first row

    // register-resident q slice + online accumulator for OUR row (each
    // lane holds lane_chunks float4 of one row; lanes of a row all
    // track identical (m, l) - redundant but broadcast-free)
    float4 qs[kAttPrefChunks];
    float4 acc[kAttPrefChunks];
    float m = -INFINITY, l = 0.0f;
    const int row_abs = row_w + r_in_warp;
    const bool live = row_abs < seq;
    if (live) {
        const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4* qrow = reinterpret_cast<const float4*>(
            &q_s[(warp * RPW + r_in_warp) * dim]);
        // compile-time trip count keeps qs/acc in registers (a runtime
        // bound demotes the arrays to local memory)
        #pragma unroll
        for (int j = 0; j < kAttPrefChunks; ++j) {
            if (j >= lane_chunks) break;
            const int c = i_in_row + j * LPR;
            qs[j] = (c < chunks) ? qrow[c] : zero4;
            acc[j] = zero4;
        }
    }
    const int lim = causal ? row_abs + 1 : seq;

    // this block's last needed key row (rows past it are never visible)
    const int row_end = min(seq, causal ? row0 + QTILE : seq);
    for (int t0 = 0; t0 < row_end; t0 += KVTILE) {
        // stage one k/v chunk (zero-padded past seq)
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int idx = threadIdx.x; idx < KVTILE * chunks;
             idx += kAttBlock) {
            const int r = idx / chunks, c = idx % chunks;
            const int row = t0 + r;
            const bool klive = row < seq;
            *reinterpret_cast<float4*>(&k_s[r * dim + c * 4]) =
                klive ? C::ld(kp + ((size_t)row * chunks + c) * 4) : zero;
            *reinterpret_cast<float4*>(&v_s[r * dim + c * 4]) =
                klive ? C::ld(vp + ((size_t)row * chunks + c) * 4) : zero;
        }
        __syncthreads();

        if (live) {
            const int t_stop = min(t0 + KVTILE, lim);
            for (int tt = t0; tt < t_stop; ++tt) {
                const float4* krow = reinterpret_cast<const float4*>(
                    &k_s[(tt - t0) * dim]);
                const float4* vrow = reinterpret_cast<const float4*>(
                    &v_s[(tt - t0) * dim]);
                float dot = 0.0f;
                #pragma unroll
                for (int j = 0; j < kAttPrefChunks; ++j) {
                    if (j >= lane_chunks) break;
                    const int c = i_in_row + j * LPR;
                    if (c < chunks) {
                        const float4 kk = krow[c];
                        dot += qs[j].x * kk.x + qs[j].y * kk.y
                             + qs[j].z * kk.z + qs[j].w * kk.w;
                    }
                }
                // reduce within the row's lane group only: log2(LPR)
                // xor-shuffles, every lane keeps the total. The mask
                // must cover JUST the row's contiguous lane group -
                // rows of the same warp can have different `live`, and
                // a full-warp mask with only some lanes arriving
                // deadlocks the shuffle.
                const unsigned row_mask =
                    ((1u << LPR) - 1u) << (r_in_warp * LPR);
                if (LPR > 1) {
                    #pragma unroll
                    for (int off = LPR / 2; off > 0; off >>= 1)
                        dot += __shfl_xor_sync(row_mask, dot, off);
                }
                const float s = dot * scale;
                const float m_new = fmaxf(m, s);
                // __expf (SFU approximation, ~2 ulp): this is the hot
                // two-instruction path of the prefill kernel; the
                // parity tests' 1e-4 tolerances absorb the difference
                const float rescale = __expf(m - m_new);
                const float p = __expf(s - m_new);
                l = l * rescale + p;
                #pragma unroll
                for (int j = 0; j < kAttPrefChunks; ++j) {
                    if (j >= lane_chunks) break;
                    const int c = i_in_row + j * LPR;
                    if (c < chunks) {
                        const float4 vv = vrow[c];
                        acc[j].x = acc[j].x * rescale + p * vv.x;
                        acc[j].y = acc[j].y * rescale + p * vv.y;
                        acc[j].z = acc[j].z * rescale + p * vv.z;
                        acc[j].w = acc[j].w * rescale + p * vv.w;
                    }
                }
                m = m_new;
            }
        }
        __syncthreads();               // before restaging k/v
    }

    if (live) {
        // lim >= 1 always holds, so l > 0 (the row saw key 0)
        T* orow = out + (((size_t)bi * hq + h) * seq + (size_t)row_abs) * dim;
        #pragma unroll
        for (int j = 0; j < kAttPrefChunks; ++j) {
            if (j >= lane_chunks) break;
            const int c = i_in_row + j * LPR;
            if (c < chunks)
                C::st(orow + c * 4, make_float4(acc[j].x / l, acc[j].y / l,
                                                acc[j].z / l, acc[j].w / l));
        }
    }
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
// Templated on the storage dtype; float32 / bfloat16 / float16 entry
// points share everything but the kernel instantiation (the workspace is
// float32 on every path, so its cache is shared across dtypes).
// ---------------------------------------------------------------------------

template <typename T>
void attention_decode_launch_t(const T* q, const T* k, const T* v,
                               const int* lens, T* out,
                               int batch, int hq, int hkv, int t_seq,
                               int dim, std::uintptr_t stream) {
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
        attn_decode_kernel<T><<<batch * hq, kAttBlock, 0, cs>>>(
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
            attn_decode_kernel<T><<<batch * hq, kAttBlock, 0, cs>>>(
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
                attn_decode_kernel<T><<<batch * hq, kAttBlock, 0, cs>>>(
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
        attn_split_kernel<1, T><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else if (group == 2)
        attn_split_kernel<2, T><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else if (group == 4)
        attn_split_kernel<4, T><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else if (group == 8)
        attn_split_kernel<8, T><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    else                          // splittable already pinned group == 16
        attn_split_kernel<16, T><<<grid1, kAttBlock, 0, cs>>>(
            q, k, v, lens, ws.ml, ws.acc, hq, hkv, t_seq, dim,
            slices, slice_len);
    check_launch("attention split kernel launch");
    attn_reduce_kernel<T><<<batch * hq, kAttBlock, 0, cs>>>(
        ws.ml, ws.acc, out, hq, hkv, dim, group, slices);
    check_launch("attention reduce kernel launch");
}

void attention_decode_launch(const float* q, const float* k, const float* v,
                             const int* lens, float* out,
                             int batch, int hq, int hkv, int t_seq, int dim,
                             std::uintptr_t stream) {
    attention_decode_launch_t(q, k, v, lens, out, batch, hq, hkv,
                              t_seq, dim, stream);
}

void attention_decode_launch_bf16(const void* q, const void* k,
                                  const void* v, const int* lens, void* out,
                                  int batch, int hq, int hkv, int t_seq,
                                  int dim, std::uintptr_t stream) {
    attention_decode_launch_t(static_cast<const __nv_bfloat16*>(q),
                              static_cast<const __nv_bfloat16*>(k),
                              static_cast<const __nv_bfloat16*>(v),
                              lens, static_cast<__nv_bfloat16*>(out),
                              batch, hq, hkv, t_seq, dim, stream);
}

void attention_decode_launch_fp16(const void* q, const void* k,
                                  const void* v, const int* lens, void* out,
                                  int batch, int hq, int hkv, int t_seq,
                                  int dim, std::uintptr_t stream) {
    attention_decode_launch_t(static_cast<const __half*>(q),
                              static_cast<const __half*>(k),
                              static_cast<const __half*>(v),
                              lens, static_cast<__half*>(out),
                              batch, hq, hkv, t_seq, dim, stream);
}

// ---------------------------------------------------------------------------
// prefill CPU reference: masked two-pass softmax attention
// ---------------------------------------------------------------------------

std::vector<float> attention_prefill_cpu(const std::vector<float>& q,
                                         const std::vector<float>& k,
                                         const std::vector<float>& v,
                                         int batch, int hq, int hkv,
                                         int seq, int dim, bool causal) {
    attention_check(batch, hq, hkv, seq, dim);
    const size_t qn = (size_t)batch * hq * seq * dim;
    const size_t kvn = (size_t)batch * hkv * seq * dim;
    if (q.size() < qn || k.size() < kvn || v.size() < kvn)
        throw std::invalid_argument("attention operand size mismatch");
    const int group = hq / hkv;
    const float scale = 1.0f / sqrtf((float)dim);
    std::vector<float> out(qn, 0.0f);
    std::vector<float> scores(seq > 0 ? seq : 1);
    for (int bi = 0; bi < batch; ++bi)
        for (int h = 0; h < hq; ++h) {
            const int kv = h / group;
            const float* qp = q.data() +
                (((size_t)bi * hq + h) * seq) * dim;
            const float* kp = k.data() +
                (((size_t)bi * hkv + kv) * seq) * dim;
            const float* vp = v.data() +
                (((size_t)bi * hkv + kv) * seq) * dim;
            for (int i = 0; i < seq; ++i) {
                const int lim = causal ? i + 1 : seq;
                const float* qr = qp + (size_t)i * dim;
                float m = -INFINITY;
                for (int t = 0; t < lim; ++t) {
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d)
                        dot += qr[d] * kp[(size_t)t * dim + d];
                    scores[t] = dot * scale;
                    m = fmaxf(m, scores[t]);
                }
                float l = 0.0f;
                for (int t = 0; t < lim; ++t)
                    l += expf(scores[t] - m);
                float* op = out.data() +
                    (((size_t)bi * hq + h) * seq + i) * dim;
                for (int t = 0; t < lim; ++t) {
                    const float p = expf(scores[t] - m) / l;
                    for (int d = 0; d < dim; ++d)
                        op[d] += p * vp[(size_t)t * dim + d];
                }
            }
        }
    return out;
}

// ---------------------------------------------------------------------------
// prefill launcher: one block per (batch, q head, 16-row tile);
// stream-ordered, no workspace, CUDA-graph capturable as-is. float32 /
// bfloat16 / float16 entry points share the tile heuristics and differ
// only in the kernel instantiation (staging shared memory stays float32).
// ---------------------------------------------------------------------------

template <typename T>
void attention_prefill_launch_t(const T* q, const T* k, const T* v, T* out,
                                int batch, int hq, int hkv, int seq, int dim,
                                bool causal, std::uintptr_t stream) {
    attention_check(batch, hq, hkv, seq, dim);
    if (batch == 0 || seq == 0)
        return;                        // nothing to compute
    cudaStream_t cs = (cudaStream_t)stream;
    // Tile shape by head size: bigger tiles for smaller heads. The old
    // dim<32 warp-per-row fallback kernel is GONE (v1.1): the tiled
    // kernel handles tiny heads correctly through its zero-padded
    // staging and per-lane bounds guards, in every storage dtype - the
    // dedicated fallback miscompiled under dtype templating and bought
    // nothing but a second code path.
    int qtile, kvtile, lpr;
    if (dim <= 128) { qtile = 64; kvtile = 16; lpr = 4; }
    else if (dim <= 256) { qtile = 32; kvtile = 8; lpr = 8; }
    else { qtile = 16; kvtile = 4; lpr = 16; }
    const int smem = (qtile + 2 * kvtile) * dim * (int)sizeof(float);
    const int tiles = (seq + qtile - 1) / qtile;
    dim3 grid((unsigned)(batch * hq * tiles));
    if (lpr == 4)
        attn_prefill_kernel<64, 16, 4, T><<<grid, kAttBlock, smem, cs>>>(
            q, k, v, out, hq, hkv, seq, dim, tiles, causal ? 1 : 0);
    else if (lpr == 8)
        attn_prefill_kernel<32, 8, 8, T><<<grid, kAttBlock, smem, cs>>>(
            q, k, v, out, hq, hkv, seq, dim, tiles, causal ? 1 : 0);
    else
        attn_prefill_kernel<16, 4, 16, T><<<grid, kAttBlock, smem, cs>>>(
            q, k, v, out, hq, hkv, seq, dim, tiles, causal ? 1 : 0);
    check_launch("attention prefill kernel launch");
}

void attention_prefill_launch(const float* q, const float* k,
                              const float* v, float* out,
                              int batch, int hq, int hkv, int seq, int dim,
                              bool causal, std::uintptr_t stream) {
    attention_prefill_launch_t(q, k, v, out, batch, hq, hkv, seq, dim,
                               causal, stream);
}

void attention_prefill_launch_bf16(const void* q, const void* k,
                                   const void* v, void* out,
                                   int batch, int hq, int hkv, int seq,
                                   int dim, bool causal,
                                   std::uintptr_t stream) {
    attention_prefill_launch_t(static_cast<const __nv_bfloat16*>(q),
                               static_cast<const __nv_bfloat16*>(k),
                               static_cast<const __nv_bfloat16*>(v),
                               static_cast<__nv_bfloat16*>(out),
                               batch, hq, hkv, seq, dim, causal, stream);
}

void attention_prefill_launch_fp16(const void* q, const void* k,
                                   const void* v, void* out,
                                   int batch, int hq, int hkv, int seq,
                                   int dim, bool causal,
                                   std::uintptr_t stream) {
    attention_prefill_launch_t(static_cast<const __half*>(q),
                               static_cast<const __half*>(k),
                               static_cast<const __half*>(v),
                               static_cast<__half*>(out),
                               batch, hq, hkv, seq, dim, causal, stream);
}

} // namespace fusedtok
