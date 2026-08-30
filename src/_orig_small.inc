__global__ void attn_prefill_small_kernel_f32(const float* __restrict__ q,
                                          const float* __restrict__ k,
                                          const float* __restrict__ v,
                                          float* __restrict__ out,
                                          int hq, int hkv, int seq,
                                          int dim, int tiles, int causal) {
    constexpr int QTILE = 16;
    const int tile = blockIdx.x % tiles;
    const int h = (blockIdx.x / tiles) % hq;
    const int bi = blockIdx.x / (tiles * hq);
    const int group = hq / hkv;
    const int kv = h / group;
    const int row0 = tile * QTILE;
    const int chunks = dim / 4;
    const float scale = 1.0f / sqrtf((float)dim);

    const float4* k4 = reinterpret_cast<const float4*>(
        k + (((size_t)bi * hkv + kv) * seq) * dim);
    const float4* v4 = reinterpret_cast<const float4*>(
        v + (((size_t)bi * hkv + kv) * seq) * dim);
    const float4* q4 = reinterpret_cast<const float4*>(
        q + (((size_t)bi * hq + h) * seq) * dim);
    float4* o4 = reinterpret_cast<float4*>(
        out + (((size_t)bi * hq + h) * seq) * dim);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_a = row0 + warp * 2, row_b = row0 + warp * 2 + 1;
    const bool has_a = row_a < seq, has_b = row_b < seq;
    const int lim_a = causal ? row_a + 1 : seq;
    const int lim_b = causal ? row_b + 1 : seq;
    const int tmax = max(has_a ? lim_a : 0, has_b ? lim_b : 0);

    // one float4 chunk per lane (chunks <= 8 on this path)
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 qc_a = (has_a && lane < chunks)
        ? q4[(size_t)row_a * chunks + lane] : zero;
    const float4 qc_b = (has_b && lane < chunks)
        ? q4[(size_t)row_b * chunks + lane] : zero;

    float m_a = -INFINITY, l_a = 0.0f, m_b = -INFINITY, l_b = 0.0f;
    float4 acc_a = zero, acc_b = zero;
    for (int t = 0; t < tmax; ++t) {
        const float4 kc = (lane < chunks)
            ? k4[(size_t)t * chunks + lane] : zero;
        const float4 vc = (lane < chunks)
            ? v4[(size_t)t * chunks + lane] : zero;
        if (has_a && t < lim_a) {
            float dot = qc_a.x * kc.x + qc_a.y * kc.y
                      + qc_a.z * kc.z + qc_a.w * kc.w;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            const float s = __shfl_sync(0xffffffffu, dot, 0) * scale;
            const float m_new = fmaxf(m_a, s);
            const float rescale = expf(m_a - m_new);
            const float p = expf(s - m_new);
            l_a = l_a * rescale + p;
            m_a = m_new;
            acc_a.x = acc_a.x * rescale + p * vc.x;
            acc_a.y = acc_a.y * rescale + p * vc.y;
            acc_a.z = acc_a.z * rescale + p * vc.z;
            acc_a.w = acc_a.w * rescale + p * vc.w;
        }
        if (has_b && t < lim_b) {
            float dot = qc_b.x * kc.x + qc_b.y * kc.y
                      + qc_b.z * kc.z + qc_b.w * kc.w;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            const float s = __shfl_sync(0xffffffffu, dot, 0) * scale;
            const float m_new = fmaxf(m_b, s);
            const float rescale = expf(m_b - m_new);
            const float p = expf(s - m_new);
            l_b = l_b * rescale + p;
            m_b = m_new;
            acc_b.x = acc_b.x * rescale + p * vc.x;
            acc_b.y = acc_b.y * rescale + p * vc.y;
            acc_b.z = acc_b.z * rescale + p * vc.z;
            acc_b.w = acc_b.w * rescale + p * vc.w;
        }
    }
    if (has_a && lane < chunks)
        o4[(size_t)row_a * chunks + lane] =
            make_float4(acc_a.x / l_a, acc_a.y / l_a,
                        acc_a.z / l_a, acc_a.w / l_a);
    if (has_b && lane < chunks)
        o4[(size_t)row_b * chunks + lane] =
            make_float4(acc_b.x / l_b, acc_b.y / l_b,
                        acc_b.z / l_b, acc_b.w / l_b);
}