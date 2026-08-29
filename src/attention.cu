// Fused decode-step attention (v0.5): single-token causal attention with
// GQA over a contiguous kv-cache.
//
//   out[b, h] = softmax(q[b,h] . K[b,kv(h)]^T / sqrt(D)) . V[b,kv(h)]
//
// One block per (batch, q head). The block's 8 warps stride the key/value
// rows; each warp keeps a running ONLINE softmax state in registers -
// running score max m, denominator l, and the [D] output accumulator -
// rescaled as new maxima arrive, so scores are never materialized to
// global memory. Lanes cover the head dimension in float4 chunks (D is a
// multiple of 4; every row base stays 16B aligned since the caller's
// buffers are torch/cudaMalloc allocations and row strides are multiples
// of 16B). A shared-memory merge folds the eight warp partials
// (weighted by exp(m_warp - m_global)) into the final normalized row.
//
// Sequences with len == 0 (or an empty cache) write zero rows. The
// kernel is one launch, reads q/K/V exactly once, performs no host
// round trips, no allocations and no syncs: stream-ordered on the
// caller's stream and CUDA-graph capturable as-is.

#include "fusedtok/attention.hpp"
#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace fusedtok {

namespace {

constexpr int kAttBlock = 256;       // 8 warps striding the sequence
constexpr int kAttWarps = kAttBlock / 32;
constexpr int kAttMaxDim = 512;      // shared accumulator budget per warp
// float4 chunks a single lane can own: ceil((MaxDim/4) / 32)
constexpr int kAttLaneChunks = kAttMaxDim / 4 / 32;

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
// launcher: one block per (batch, q head), stream-ordered, graph-safe
// ---------------------------------------------------------------------------

void attention_decode_launch(const float* q, const float* k, const float* v,
                             const int* lens, float* out,
                             int batch, int hq, int hkv, int t_seq, int dim,
                             std::uintptr_t stream) {
    attention_check(batch, hq, hkv, t_seq, dim);
    if (batch == 0)
        return;                        // nothing to compute
    cudaStream_t cs = (cudaStream_t)stream;
    attn_decode_kernel<<<batch * hq, kAttBlock, 0, cs>>>(
        q, k, v, lens, out, hq, hkv, t_seq, dim);
    check_launch("attention decode kernel launch");
}

} // namespace fusedtok
