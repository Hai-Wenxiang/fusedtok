#pragma once

// Attention operators (v0.5): decode-step causal attention with GQA over
// a contiguous kv-cache. Kernel launchers live in cuda_launch.hpp.

#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// attention (decode step): single-token causal attention with GQA.
//
//   q:       [B, Hq, D]       queries of the NEW token (one per sequence)
//   k_cache: [B, Hkv, T, D]   key cache; per sequence only rows [0, len_b)
//   v_cache: [B, Hkv, T, D]   value cache   (rows past len_b are padding)
//   lens:    [B] or null      valid cache length per sequence (null = all T)
//
//   out:     [B, Hq, D] = softmax(q . K^T / sqrt(D)) . V
//
// GQA mapping: q head h attends with kv head h * Hkv / Hq, i.e. q heads
// form contiguous groups of (Hq / Hkv) over each kv head (Hq == Hkv is
// plain MHA). Hq must be a multiple of Hkv. A sequence with len == 0
// yields a zero output row (an empty softmax has no direction; defined
// as zero so variable-length batches share one cache tensor). D must be
// a multiple of 4 and at most 512.
// ---------------------------------------------------------------------------

std::vector<float> attention_decode_cpu(const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v,
                                        const std::vector<int>* lens,
                                        int batch, int q_heads, int kv_heads,
                                        int cache_rows, int dim);

// ---------------------------------------------------------------------------
// attention (decode step) over a PAGED kv-cache (v1.2): the cache is a
// pool of fixed-size token blocks instead of a per-sequence contiguous
// span - the vLLM-style layout that keeps memory fragmentation out of
// the cache.
//
//   q:       [B, Hq, D]          queries of the NEW token
//   k_pool:  [Nb, Hkv, P, D]     key pool (P = tokens per block)
//   v_pool:  [Nb, Hkv, P, D]     value pool
//   table:   [B, S] ints         block table: sequence b's token t lives
//                                 in block table[b, t / P] at offset t % P
//   lens:    [B] or null         valid length per sequence (null = S*P)
//
//   out:     [B, Hq, D] = softmax(q . K^T / sqrt(D)) . V
//
// Same GQA mapping, zero-row-for-len-0 convention and dim limits as the
// contiguous op. The paged GPU path supports GQA group sizes 1/2/4/8/16
// (other divisors: use the contiguous op); table VALUES are validated on
// the CPU/staged paths (ValueError) and trusted on the zero-copy path
// (like data_ptr: a device table is not host-readable without a sync).
// ---------------------------------------------------------------------------

std::vector<float> attention_decode_paged_cpu(
    const std::vector<float>& q, const std::vector<float>& k_pool,
    const std::vector<float>& v_pool, const std::vector<int>& table,
    const std::vector<int>* lens, int batch, int q_heads, int kv_heads,
    int page, int tbl_width, int num_blocks, int dim);

// ---------------------------------------------------------------------------
// attention (prefill): fresh-sequence self-attention over S query rows.
//
//   q: [B, Hq, S, D], k: [B, Hkv, S, D], v: [B, Hkv, S, D]
//   out: [B, Hq, S, D]
//
// causal = true (default): query row i attends to key rows [0, i] (the
// prefill diagonal of a fresh sequence). causal = false attends to all
// S rows (bidirectional / encoder style). Same GQA grouping convention
// as the decode op (q head h -> kv head h * Hkv / Hq). D must be a
// multiple of 4 and at most 512; S and B are arbitrary.
// ---------------------------------------------------------------------------

std::vector<float> attention_prefill_cpu(const std::vector<float>& q,
                                         const std::vector<float>& k,
                                         const std::vector<float>& v,
                                         int batch, int q_heads,
                                         int kv_heads, int seq, int dim,
                                         bool causal);

} // namespace fusedtok
