#pragma once

// Raw device-pointer kernel launchers (the "_launch" family).
//
// Contract (applies to every function below):
//   - All float*/int* pointers are valid CUDA device pointers owned by the
//     caller; sizes and shapes are validated by the caller before calling.
//   - Functions are asynchronous: they enqueue kernels on the CUDA default
//     stream and return without host synchronization. Callers that need the
//     results on the host must synchronize (or use a blocking memcpy).
//   - Errors are reported as std::runtime_error (Python RuntimeError) via
//     cudaGetLastError after launch.
//
// These launchers power the zero-copy torch-CUDA path: Python allocates
// torch tensors on the GPU and passes data_ptr() addresses straight into
// the kernels with no staging copies.

#include <cstdint>

// bf16 storage type (compute stays float32)
struct __nv_bfloat16;

namespace fusedtok {

// --- elementwise unary activations -----------------------------------------
void silu_launch(const float* x, float* y, long long n);
void gelu_launch(const float* x, float* y, long long n);       // exact erf form
void gelu_tanh_launch(const float* x, float* y, long long n);  // tanh approx
void relu_launch(const float* x, float* y, long long n);
void tanh_launch(const float* x, float* y, long long n);
void sigmoid_launch(const float* x, float* y, long long n);
void axpy_launch(const float* x, float* y, long long n, float a, float b);

// --- elementwise binary ------------------------------------------------------
void add_launch(const float* a, const float* b, float* y, long long n);
void mul_launch(const float* a, const float* b, float* y, long long n);
void swiglu_launch(const float* gate, const float* up, float* y, long long n);

// --- row-wise normalization / softmax ---------------------------------------
// x/r/y: [rows * cols] row-major, w: [cols]. r may be null (no residual).
void rmsnorm_launch(const float* x, const float* w, const float* r,
                    float* y, int rows, int cols, float eps);
// x/y: [rows * cols], w/b: [cols].
void layernorm_launch(const float* x, const float* w, const float* b,
                      float* y, int rows, int cols, float eps);
// x/y: [rows * cols].
void softmax_launch(const float* x, float* y, int rows, int cols);

// --- RoPE --------------------------------------------------------------------
// x/y: [seq * dim]. Positions covered are [pos_offset, pos_offset + seq).
// pos_offset supports kv-cache decoding where new tokens continue an
// existing sequence instead of starting at position 0.
void rope_launch(const float* x, float* y, int seq, int dim, float theta,
                 int pos_offset);
void rope_neox_launch(const float* x, float* y, int seq, int dim, float theta,
                      int pos_offset);

// --- sampling / logits post-processing --------------------------------------
// Descending top-k; earliest index wins ties (deterministic). idxs is int64.
// Implemented as k parallel packed-key selection rounds.
void topk_launch(const float* x, float* vals, long long* idxs, int n, int k);
// Nucleus selection with early exit at cumulative mass >= p; writes the
// selected prefix into vals/idxs and the count into count_out.
void topp_select_launch(const float* x, float* vals, long long* idxs,
                        int n, float p, int* count_out);
// Fused nucleus sampling over raw logits with temperature: one cooperative
// kernel softmaxes (via sorted exp accumulation), truncates to the p-nucleus
// and inverse-CDF samples with a hash-uniform of `seed`. Returns the token.
long long sample_topp_launch(const float* x, int n, float p, float t,
                             unsigned long long seed);
// Greedy argmax; earliest index wins ties. Single parallel selection round.
void argmax_launch(const float* x, int n, int* out);
// y[i] = x[i] / t.
void temperature_launch(const float* x, float* y, int n, float t);
// For each id in ids[0..m): y[id] = x[id] > 0 ? x[id]/penalty : x[id]*penalty.
// logits/y: [n], ids: [m] unique token ids.
void repetition_penalty_launch(const float* logits, const long long* ids,
                               int n, int m, float penalty, float* y);


// --- bf16 variants: float32 compute, bf16 storage ---------------------------
// Weight/bias parameters stay float32 (norm weights are commonly kept fp32
// in bf16 checkpoints). Available for: elementwise unary/binary, norms,
// softmax, RoPE. Sampling/selection ops remain float32 (logits are f32).
void silu_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n);
void gelu_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n);
void gelu_tanh_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n);
void relu_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n);
void tanh_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n);
void sigmoid_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, long long n);
void add_launch_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     __nv_bfloat16* y, long long n);
void mul_launch_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     __nv_bfloat16* y, long long n);
void swiglu_launch_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                        __nv_bfloat16* y, long long n);
void rmsnorm_launch_bf16(const __nv_bfloat16* x, const float* w,
                         const __nv_bfloat16* r, __nv_bfloat16* y,
                         int rows, int cols, float eps);
void layernorm_launch_bf16(const __nv_bfloat16* x, const float* w,
                           const float* b, __nv_bfloat16* y,
                           int rows, int cols, float eps);
void softmax_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y,
                         int rows, int cols);
void rope_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int seq,
                      int dim, float theta, int pos_offset);
void rope_neox_launch_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int seq,
                           int dim, float theta, int pos_offset);

} // namespace fusedtok
