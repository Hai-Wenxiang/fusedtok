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

}
