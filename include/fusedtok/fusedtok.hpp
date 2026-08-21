#pragma once

#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// axpy: trivial skeleton operator, y = a * x + b
// ---------------------------------------------------------------------------

// CPU reference implementation: y[i] = a * x[i] + b.
// Serves as ground truth for GPU parity tests and runs anywhere.
std::vector<float> axpy_cpu(const std::vector<float>& x, float a, float b);

// GPU implementation: naive one-thread-per-element kernel.
// Host buffers in / host buffers out; all transfers handled internally.
std::vector<float> axpy_cuda(const std::vector<float>& x, float a, float b);

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

// CPU reference implementation.
std::vector<float> rmsnorm_cpu(const std::vector<float>& x,
                               const std::vector<float>& w,
                               int rows, int cols, float eps,
                               const std::vector<float>* residual);

// GPU implementation, naive version:
//   kernel 1: one thread per row, serial loop accumulates sum of squares
//   kernel 2: one thread per element, applies scale
std::vector<float> rmsnorm_cuda(const std::vector<float>& x,
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

// CPU reference implementation.
std::vector<float> swiglu_cpu(const std::vector<float>& gate,
                              const std::vector<float>& up);

// GPU implementation: naive one-thread-per-element kernel.
std::vector<float> swiglu_cuda(const std::vector<float>& gate,
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
// Returns {q_rotated, k_rotated}; the second element is nullptr-equivalent
// (empty) when no k was supplied.
// ---------------------------------------------------------------------------

// CPU reference implementation. k may be nullptr to rotate q only.
std::pair<std::vector<float>, std::vector<float>>
rope_cpu(const std::vector<float>& q, const std::vector<float>* k,
         int seq, int dim, float theta);

// GPU implementation: naive one-thread-per-pair kernel.
std::pair<std::vector<float>, std::vector<float>>
rope_cuda(const std::vector<float>& q, const std::vector<float>* k,
          int seq, int dim, float theta);

// ---------------------------------------------------------------------------
// RoPE, "rotate_half" (GPT-NeoX / LLaMA-HF) variant.
//
// Same frequencies as the interleaved variant, but pairs are formed across
// the row halves instead of adjacent elements:
//
//   x1 = x[0 .. dim/2),  x2 = x[dim/2 .. dim)
//   angle(m, j) = m * theta^(-2j / dim)
//   x1'[j] = x1[j] * cos - x2[j] * sin
//   x2'[j] = x1[j] * sin + x2[j] * cos
//
// Both variants produce permutations of each other; models are trained with
// one specific layout, so the library offers both.
// ---------------------------------------------------------------------------

std::pair<std::vector<float>, std::vector<float>>
rope_neox_cpu(const std::vector<float>& q, const std::vector<float>* k,
              int seq, int dim, float theta);

std::pair<std::vector<float>, std::vector<float>>
rope_neox_cuda(const std::vector<float>& q, const std::vector<float>* k,
               int seq, int dim, float theta);

}
