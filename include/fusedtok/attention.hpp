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
