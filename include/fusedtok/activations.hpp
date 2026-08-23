#pragma once

#include <utility>
#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// Elementwise unary activations
//
//   silu(v)   = v * sigmoid(v)                       (aka Swish)
//   gelu(v)   = 0.5 * v * (1 + erf(v / sqrt(2)))     (exact formulation)
//   gelu_tanh(v) = 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 v^3)))
//   relu(v)   = max(v, 0)
//   tanh(v), sigmoid(v) = the usual functions
//
// CPU reference implementations; CUDA paths go through the *_launch
// entry points declared in cuda_launch.hpp.
// ---------------------------------------------------------------------------

std::vector<float> silu_cpu(const std::vector<float>& x);
std::vector<float> gelu_cpu(const std::vector<float>& x);
std::vector<float> gelu_tanh_cpu(const std::vector<float>& x);
std::vector<float> relu_cpu(const std::vector<float>& x);
std::vector<float> tanh_cpu(const std::vector<float>& x);
std::vector<float> sigmoid_cpu(const std::vector<float>& x);

// ---------------------------------------------------------------------------
// Elementwise binary ops: add(a, b) = a + b, mul(a, b) = a * b.
// "add" doubles as the fused add + residual pattern of inference stacks.
// ---------------------------------------------------------------------------

std::vector<float> add_cpu(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> mul_cpu(const std::vector<float>& a, const std::vector<float>& b);

// ---------------------------------------------------------------------------
// top-k selection
//
// Returns the k largest elements of x with their indices, sorted descending.
// Naive algorithm: k passes over the data, each extracting the current max
// and marking it as visited. O(n * k). Ties resolve to the earliest index
// (deterministic). Returns {values, indices} with indices as int64.
// ---------------------------------------------------------------------------

std::pair<std::vector<float>, std::vector<long long>>
topk_cpu(const std::vector<float>& x, int k);

// ---------------------------------------------------------------------------
// top-p (nucleus) selection
//
// Given a probability vector (summing to ~1), returns the smallest set of
// highest-probability elements whose cumulative mass reaches p (the element
// that crosses the threshold is included). Returns {values, indices}
// sorted descending. Built on top of top-k with k = n.
// ---------------------------------------------------------------------------

std::pair<std::vector<float>, std::vector<long long>>
topp_cpu(const std::vector<float>& probs, float p);

// ---------------------------------------------------------------------------
// Sampling helpers
//
// argmax(x)      -> index of the largest element (earliest index on ties)
// temperature(x) -> x[i] / t elementwise (t > 0; t < 1 sharpens, t > 1 flattens)
// repetition_penalty(logits, ids, p) -> logits with every listed token id
//     scaled by 1/p if positive, p if negative (CTRL-style penalty applied
//     to previously generated tokens before sampling).
// ---------------------------------------------------------------------------

long long argmax_cpu(const std::vector<float>& x);

std::vector<float> temperature_cpu(const std::vector<float>& x, float t);

std::vector<float> repetition_penalty_cpu(const std::vector<float>& logits,
                                          const std::vector<long long>& token_ids,
                                          float penalty);

// ---------------------------------------------------------------------------
// Fused nucleus sampling (deterministic per seed):
//   probs = softmax(logits / t); nucleus = smallest top-p prefix;
//   u = uniform hash of seed; return the token where the nucleus cumulative
//   probability reaches u. The RNG is a splitmix-style hash - deterministic
//   and reproducible, NOT cryptographically secure.
// Returns the sampled token id.
// ---------------------------------------------------------------------------

long long sample_topp_cpu(const std::vector<float>& logits,
                          float p, float t, unsigned long long seed);

} // namespace fusedtok
