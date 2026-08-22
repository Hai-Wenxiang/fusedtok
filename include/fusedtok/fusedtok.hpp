#pragma once

#include <utility>
#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// axpy: trivial skeleton operator, y = a * x + b (demo / smoke-test path)
// ---------------------------------------------------------------------------

std::vector<float> axpy_cpu(const std::vector<float>& x, float a, float b);

// Returns true if at least one CUDA device is usable.
bool cuda_available();

// ---------------------------------------------------------------------------
// RMSNorm (with optional residual)
//
//   y = (x + r) * rsqrt(mean((x + r)^2) + eps) * w
//
// x and residual are flattened row-major [rows, cols]; weight is [cols].
// r may be nullptr for plain RMSNorm. This is the normalization used by
// LLaMA / Qwen style transformers.
// ---------------------------------------------------------------------------

std::vector<float> rmsnorm_cpu(const std::vector<float>& x,
                               const std::vector<float>& w,
                               int rows, int cols, float eps,
                               const std::vector<float>* residual);

// ---------------------------------------------------------------------------
// SwiGLU activation
//
//   out = silu(gate) * up,   silu(v) = v * sigmoid(v)
//
// gate and up are equal-length vectors (the two halves of a SwiGLU MLP
// projection). Used by LLaMA / Qwen style feed-forward blocks.
// ---------------------------------------------------------------------------

std::vector<float> swiglu_cpu(const std::vector<float>& gate,
                              const std::vector<float>& up);

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embedding), interleaved-pair variant (original
// RoFormer formulation).
//
// For position m and pair index j (pairs are (2j, 2j+1) within each row of
// width dim):
//
//   angle   = m * theta^(-2j / dim)
//   x2j'    = x2j  * cos(angle) - x2j+1 * sin(angle)
//   x2j+1'  = x2j  * sin(angle) + x2j+1 * cos(angle)
//
// q (and optionally k) are flattened row-major [seq, dim]; dim must be even.
// pos_offset shifts the position of the first row (kv-cache decoding where
// the batch starts mid-sequence). Returns {q_rotated, k_rotated}; the second
// element is empty when no k was supplied.
// ---------------------------------------------------------------------------

std::pair<std::vector<float>, std::vector<float>>
rope_cpu(const std::vector<float>& q, const std::vector<float>* k,
         int seq, int dim, float theta, int pos_offset = 0);

// ---------------------------------------------------------------------------
// RoPE, "rotate_half" (GPT-NeoX / LLaMA-HF) variant.
//
// Same frequencies as the interleaved variant, but pairs are formed across
// the row halves instead of adjacent elements:
//
//   x1 = x[0 .. dim/2),  x2 = x[dim/2 .. dim)
//   x1'[j] = x1[j] * cos(angle) - x2[j] * sin(angle)
//   x2'[j] = x1[j] * sin(angle) + x2[j] * cos(angle)
//
// Both variants produce permutations of each other; models are trained with
// one specific layout, so the library offers both.
// ---------------------------------------------------------------------------

std::pair<std::vector<float>, std::vector<float>>
rope_neox_cpu(const std::vector<float>& q, const std::vector<float>* k,
              int seq, int dim, float theta, int pos_offset = 0);

} // namespace fusedtok
