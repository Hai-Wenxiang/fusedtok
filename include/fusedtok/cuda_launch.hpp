#pragma once

// Raw device-pointer kernel launchers (the "_launch" family).
//
// Contract (applies to every function below):
//   - All float*/int* pointers are valid CUDA device pointers owned by the
//     caller; sizes and shapes are validated by the caller before calling.
//   - Functions are asynchronous: they enqueue kernels on the caller's CUDA
//     stream (trailing `stream` argument; 0 = the default stream) and
//     return without host synchronization. Callers that need the
//     results on the host must synchronize (or use a blocking memcpy).
//   - Errors are reported as std::runtime_error (Python RuntimeError) via
//     cudaGetLastError after launch.
//
// These launchers power the zero-copy torch-CUDA path: Python allocates
// torch tensors on the GPU and passes data_ptr() addresses straight into
// the kernels with no staging copies.

#include <cstdint>
#include <vector>

// bf16 storage type (compute stays float32)
struct __nv_bfloat16;

namespace fusedtok {

// --- elementwise unary activations -----------------------------------------
void silu_launch(const float* x, float* y, long long n, std::uintptr_t stream = 0);
void gelu_launch(const float* x, float* y, long long n, std::uintptr_t stream = 0);       // exact erf form
void gelu_tanh_launch(const float* x, float* y, long long n, std::uintptr_t stream = 0);  // tanh approx
void relu_launch(const float* x, float* y, long long n, std::uintptr_t stream = 0);
void tanh_launch(const float* x, float* y, long long n, std::uintptr_t stream = 0);
void sigmoid_launch(const float* x, float* y, long long n, std::uintptr_t stream = 0);
void axpy_launch(const float* x, float* y, long long n, float a, float b, std::uintptr_t stream = 0);

// --- elementwise binary ------------------------------------------------------
void add_launch(const float* a, const float* b, float* y, long long n, std::uintptr_t stream = 0);
void mul_launch(const float* a, const float* b, float* y, long long n, std::uintptr_t stream = 0);
void swiglu_launch(const float* gate, const float* up, float* y, long long n, std::uintptr_t stream = 0);

// --- row-wise normalization / softmax ---------------------------------------
// x/r/y: [rows * cols] row-major, w: [cols]. r may be null (no residual).
void rmsnorm_launch(const float* x, const float* w, const float* r,
                    float* y, int rows, int cols, float eps, std::uintptr_t stream = 0);
// x/y: [rows * cols], w/b: [cols].
void layernorm_launch(const float* x, const float* w, const float* b,
                      float* y, int rows, int cols, float eps, std::uintptr_t stream = 0);
// x/y: [rows * cols].
void softmax_launch(const float* x, float* y, int rows, int cols, std::uintptr_t stream = 0);

// --- RoPE --------------------------------------------------------------------
// x/y: [seq * dim]. Positions covered are [pos_offset, pos_offset + seq).
// pos_offset supports kv-cache decoding where new tokens continue an
// existing sequence instead of starting at position 0.
void rope_launch(const float* x, float* y, int seq, int dim, float theta,
                 int pos_offset, std::uintptr_t stream = 0);
void rope_neox_launch(const float* x, float* y, int seq, int dim, float theta,
                      int pos_offset, std::uintptr_t stream = 0);

// --- sampling / logits post-processing --------------------------------------
// Descending top-k; earliest index wins ties (deterministic). idxs is int64.
// Implemented as k parallel packed-key selection rounds.
void topk_launch(const float* x, float* vals, long long* idxs, int n, int k, std::uintptr_t stream = 0);
// Nucleus selection with early exit at cumulative mass >= p; writes the
// selected prefix into vals/idxs and the count into count_out.
void topp_select_launch(const float* x, float* vals, long long* idxs,
                        int n, float p, int* count_out, std::uintptr_t stream = 0);
// Fused nucleus sampling over raw logits with temperature: the selection
// pipeline softmaxes (global-mass threshold over the sorted keys),
// truncates to the p-nucleus and inverse-CDF samples with a
// hash-uniform of `seed`. Returns the token.
long long sample_topp_launch(const float* x, int n, float p, float t,
                              unsigned long long seed, std::uintptr_t stream = 0);

// Fused top-k sampling: temperature, top-k truncation, renormalize
// WITHIN the k survivors, inverse-CDF draw - one call, one readback.
// Deterministic per seed (same RNG as sample_topp_launch).
long long sample_topk_launch(const float* x, int n, int k, float t,
                             unsigned long long seed,
                             std::uintptr_t stream = 0);

// Fused min-p sampling (v1.3): temperature, keep every token with
// probability >= min_p * p_max, renormalize within the nucleus,
// inverse-CDF draw. Deterministic per seed (same RNG). No global-mass
// reduction - the nucleus is a value-threshold prefix.
long long sample_minp_launch(const float* x, int n, float min_p, float t,
                             unsigned long long seed,
                             std::uintptr_t stream = 0);

// Fused decode step: repetition penalty over the sampled ids (vocab
// bitmap), temperature, then nucleus sampling - one call, one readback.
// Returns the sampled token id.
long long decode_step_launch(const float* x, const long long* ids,
                             int n, int m, float penalty, float p, float t,
                             unsigned long long seed,
                             std::uintptr_t stream = 0);

// Batched samplers (v1.4): x is [rows, n] row-major, seeds holds one
// seed per row; returns rows tokens. Each row runs the single-row
// pipeline verbatim (same arithmetic, same accumulation order - see the
// exptotal arrival-order ulp note in topk.cu). Like the single-row
// samplers these synchronize per attempt and are NOT CUDA-graph
// capturable. rows == 0 returns an empty vector.
std::vector<long long> sample_topp_batched_launch(
    const float* x, int rows, int n, float p, float t,
    const std::vector<unsigned long long>& seeds,
    std::uintptr_t stream = 0);
std::vector<long long> sample_topk_batched_launch(
    const float* x, int rows, int n, int k, float t,
    const std::vector<unsigned long long>& seeds,
    std::uintptr_t stream = 0);
std::vector<long long> sample_minp_batched_launch(
    const float* x, int rows, int n, float min_p, float t,
    const std::vector<unsigned long long>& seeds,
    std::uintptr_t stream = 0);

// Batched fused decode step (v1.5): x is [rows, n], ids is the ragged
// history array (flat, row r's slice is ids[offs[r] .. offs[r+1]),
// offs holds rows + 1 non-decreasing entries starting at 0 and ending
// at ids.size()). Each row runs the single-row decode_step pipeline -
// repetition penalty on the RAW logit, then temperature, then nucleus
// sampling - with per-row parity up to the documented exptotal
// arrival-order ulp boundary. Id values must be validated by the
// caller (host-origin data); rows == 0 returns an empty vector. Same
// per-attempt synchronization, so not CUDA-graph capturable.
std::vector<long long> decode_step_batched_launch(
    const float* x, int rows, int n, const std::vector<long long>& ids,
    const std::vector<long long>& offs, float penalty, float p, float t,
    const std::vector<unsigned long long>& seeds,
    std::uintptr_t stream = 0);

// INT8 quantization: q = clamp(round(x * (1/scale))), scale = absmax/127
// (written to scale_out device float). n elements.
void quantize_int8_launch(const float* x, signed char* q,
                          float* scale_out, long long n, std::uintptr_t stream = 0);
// x[i] = q[i] * scale.
void dequantize_int8_launch(const signed char* q, float* x,
                            float scale, long long n, std::uintptr_t stream = 0);
// Fused dequant-add-requant: y = clamp(round((qa*sa + qb*sb) * (1/scale_y)))
// with scale_y = absmax(qa*sa + qb*sb)/127 (written to out_scale).
void qadd_int8_launch(const signed char* qa, const signed char* qb,
                      float sa, float sb, signed char* qy,
                      float* out_scale, long long n, std::uintptr_t stream = 0);

// INT8 matmul: y[M,N] = (A_q[M,K] . B_q[N,K]^T) int32-exact * (sa*sb).
// Both operands row-major along K (LLM layout: activations @ weight.T).
// M == 1 dispatches to a warp-per-row GEMV kernel.
void qgemm_launch(const signed char* aq, const signed char* bq,
                  float* y, int m, int n, int k, float sa, float sb,
                  std::uintptr_t stream = 0);

// Per-output-channel weight scales (SmoothQuant-style W8A8):
// y[i,j] = (A_q . B_q^T) int32-exact * f32(sa * sb_vec[j]). sb_vec has
// n entries (one per output row of B_q). Same exactness contract as
// qgemm_launch: the f32 scale composes once, the product applies once.
void qgemm_perchannel_launch(const signed char* aq, const signed char* bq,
                             const float* sb_vec, float* y,
                             int m, int n, int k, float sa,
                             std::uintptr_t stream = 0);

// Greedy argmax; earliest index wins ties. Single parallel selection round.
void argmax_launch(const float* x, int n, int* out, std::uintptr_t stream = 0);
// y[i] = x[i] / t.
void temperature_launch(const float* x, float* y, long long n, float t, std::uintptr_t stream = 0);
// For each id in ids[0..m): y[id] = x[id] > 0 ? x[id]/penalty : x[id]*penalty.
// logits/y: [n], ids: [m] unique token ids.
void repetition_penalty_launch(const float* logits, const long long* ids,
                               int n, int m, float penalty, float* y, std::uintptr_t stream = 0);


// --- attention (decode step) -------------------------------------------------
// Single-token causal attention with GQA over a contiguous kv-cache:
// out[B,Hq,D] = softmax(q . K^T / sqrt(D)) . V with q heads in contiguous
// groups over kv heads (h -> h*Hkv/Hq). k/v: [B,Hkv,T,D]; lens may be null
// (all rows valid) else per-sequence valid lengths in [0, T] (zero length
// writes zero rows). dim: multiple of 4, at most 512. Short caches run as
// one kernel launch; long caches split flash-decoding style into a slice
// pass plus a reduce pass over a per-shape workspace (allocated outside
// captures). Either way: no per-call syncs, stream-ordered and
// CUDA-graph capturable.
void attention_decode_launch(const float* q, const float* k, const float* v,
                             const int* lens, float* out,
                             int batch, int q_heads, int kv_heads,
                             int cache_rows, int dim,
                             std::uintptr_t stream = 0);

// bfloat16 / float16 storage variants: q/k/v/out are half-precision
// (out matches the input dtype); softmax and every accumulator still
// run in float32, so numerics match the float32 path up to input
// rounding. Pointers alias half-width buffers (out: [B,Hq,D] elements
// of the same dtype).
void attention_decode_launch_bf16(const void* q, const void* k,
                                  const void* v, const int* lens, void* out,
                                  int batch, int q_heads, int kv_heads,
                                  int cache_rows, int dim,
                                  std::uintptr_t stream = 0);
void attention_decode_launch_fp16(const void* q, const void* k,
                                  const void* v, const int* lens, void* out,
                                  int batch, int q_heads, int kv_heads,
                                  int cache_rows, int dim,
                                  std::uintptr_t stream = 0);

// Prefill (fresh-sequence) attention: q [B,Hq,S,D] attends over k/v
// [B,Hkv,S,D]; causal=true masks query row i to key rows [0, i] (the
// prefill diagonal), causal=false attends everywhere (bidirectional).
// Same GQA grouping and dim constraints as attention_decode. One tiled
// kernel (16 query rows resident per block), no workspace: stream-
// ordered and CUDA-graph capturable.
void attention_prefill_launch(const float* q, const float* k,
                              const float* v, float* out,
                              int batch, int q_heads, int kv_heads,
                              int seq, int dim, bool causal,
                              std::uintptr_t stream = 0);

// bfloat16 / float16 storage variants: q/k/v/out are half-precision
// (out matches the input dtype); staging and accumulators stay float32.
void attention_prefill_launch_bf16(const void* q, const void* k,
                                   const void* v, void* out,
                                   int batch, int q_heads, int kv_heads,
                                   int seq, int dim, bool causal,
                                   std::uintptr_t stream = 0);
void attention_prefill_launch_fp16(const void* q, const void* k,
                                   const void* v, void* out,
                                   int batch, int q_heads, int kv_heads,
                                   int seq, int dim, bool causal,
                                   std::uintptr_t stream = 0);

// --- attention (decode step) over a PAGED kv-cache (v1.2) --------------------
// Same math as attention_decode_launch, but the kv-cache is a block pool
// [Nb, Hkv, P, D] reached through a per-sequence block table [B, S] (token
// t of sequence b lives at pool[table[b, t / P], kv, t % P, :]). lens may
// be null (every sequence uses its full table width). The split path's
// float32 workspace is shared with the contiguous op; warm a shape up
// OUTSIDE a CUDA-graph capture before capturing it (the workspace must
// pre-exist; there is no allocation-free fallback path here). GQA group
// must be one of 1/2/4/8/16. dim: multiple of 4, at most 512.
void attention_decode_paged_launch(const float* q, const float* k_pool,
                                   const float* v_pool, const int* table,
                                   const int* lens, float* out,
                                   int batch, int q_heads, int kv_heads,
                                   int page, int tbl_width, int dim,
                                   std::uintptr_t stream = 0);
void attention_decode_paged_launch_bf16(const void* q, const void* k_pool,
                                        const void* v_pool, const int* table,
                                        const int* lens, void* out,
                                        int batch, int q_heads, int kv_heads,
                                        int page, int tbl_width, int dim,
                                        std::uintptr_t stream = 0);
void attention_decode_paged_launch_fp16(const void* q, const void* k_pool,
                                        const void* v_pool, const int* table,
                                        const int* lens, void* out,
                                        int batch, int q_heads, int kv_heads,
                                        int page, int tbl_width, int dim,
                                        std::uintptr_t stream = 0);

// --- paged kv-cache append (v1.2) -------------------------------------------
// Scatter ONE fresh token's k/v rows per sequence into the pool blocks:
// k_new/v_new are [B, Hkv, D], the write position of sequence b is its
// CURRENT length lens[b] (required), landing in block table[b, lens/P]
// at offset lens%P. IN-PLACE on the pools; the scheduler owns the table
// (this never writes table entries). Same dtype family and dim rules as
// the attention ops; one tiny kernel, stream-ordered and capturable.
void kv_append_paged_launch(const float* k_new, const float* v_new,
                            const int* table, const int* lens, float* k_pool,
                            float* v_pool, int batch, int hkv, int dim,
                            int page, int tbl_width,
                            std::uintptr_t stream = 0);
void kv_append_paged_launch_bf16(const void* k_new, const void* v_new,
                                 const int* table, const int* lens,
                                 void* k_pool, void* v_pool, int batch,
                                 int hkv, int dim, int page, int tbl_width,
                                 std::uintptr_t stream = 0);
void kv_append_paged_launch_fp16(const void* k_new, const void* v_new,
                                 const int* table, const int* lens,
                                 void* k_pool, void* v_pool, int batch,
                                 int hkv, int dim, int page, int tbl_width,
                                 std::uintptr_t stream = 0);
void kv_append_paged_cpu(const std::vector<float>& k_new,
                         const std::vector<float>& v_new,
                         const std::vector<int>& table,
                         const std::vector<int>& lens,
                         std::vector<float>& k_pool,
                         std::vector<float>& v_pool, int batch, int hkv,
                         int dim, int page, int tbl_width, int num_blocks);

// Contiguous-cache twin (v1.3): scatter ONE fresh token's k/v rows per
// sequence to cache row lens[b] of the [B, Hkv, T, D] caches. Same
// conventions as the paged op (in-place, lens required and trusted on
// the zero-copy path, one tiny capturable kernel).
void kv_append_launch(const float* k_new, const float* v_new,
                      const int* lens, float* k_cache, float* v_cache,
                      int batch, int hkv, int dim, int t_rows,
                      std::uintptr_t stream = 0);
void kv_append_launch_bf16(const void* k_new, const void* v_new,
                           const int* lens, void* k_cache, void* v_cache,
                           int batch, int hkv, int dim, int t_rows,
                           std::uintptr_t stream = 0);
void kv_append_launch_fp16(const void* k_new, const void* v_new,
                           const int* lens, void* k_cache, void* v_cache,
                           int batch, int hkv, int dim, int t_rows,
                           std::uintptr_t stream = 0);
void kv_append_cpu(const std::vector<float>& k_new,
                   const std::vector<float>& v_new,
                   const std::vector<int>& lens,
                   std::vector<float>& k_cache, std::vector<float>& v_cache,
                   int batch, int hkv, int dim, int t_rows);

// --- bf16 variants: float32 compute, bf16 storage ---------------------------
// Weight/bias parameters stay float32 (norm weights are commonly kept fp32
// in bf16 checkpoints). Available for: elementwise unary/binary, norms,
// softmax, RoPE. Sampling/selection ops remain float32 (logits are f32).
void silu_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void gelu_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void gelu_tanh_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void relu_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void tanh_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void sigmoid_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void add_launch_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void mul_launch_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void swiglu_launch_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                        __nv_bfloat16* y, long long n, std::uintptr_t stream = 0);
void rmsnorm_launch_bf16(const __nv_bfloat16* x, const float* w,
                         const __nv_bfloat16* r, __nv_bfloat16* y,
                         int rows, int cols, float eps, std::uintptr_t stream = 0);
void layernorm_launch_bf16(const __nv_bfloat16* x, const float* w,
                           const float* b, __nv_bfloat16* y,
                           int rows, int cols, float eps, std::uintptr_t stream = 0);
void softmax_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y,
                         int rows, int cols, std::uintptr_t stream = 0);
void rope_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int seq,
                      int dim, float theta, int pos_offset, std::uintptr_t stream = 0);
void rope_neox_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int seq,
                           int dim, float theta, int pos_offset, std::uintptr_t stream = 0);

} // namespace fusedtok
