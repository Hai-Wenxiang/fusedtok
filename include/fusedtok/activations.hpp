#pragma once

#include <utility>
#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// Elementwise activations
//
// silu(v) = v * sigmoid(v)                  (aka Swish)
// gelu(v) = 0.5 * v * (1 + erf(v / sqrt(2)))  (exact formulation)
// ---------------------------------------------------------------------------

// SiLU
std::vector<float> silu_cpu(const std::vector<float>& x);
std::vector<float> silu_cuda(const std::vector<float>& x);

// GeLU (exact erf form)
std::vector<float> gelu_cpu(const std::vector<float>& x);
std::vector<float> gelu_cuda(const std::vector<float>& x);

// ReLU
std::vector<float> relu_cpu(const std::vector<float>& x);
std::vector<float> relu_cuda(const std::vector<float>& x);

// Tanh
std::vector<float> tanh_cpu(const std::vector<float>& x);
std::vector<float> tanh_cuda(const std::vector<float>& x);

// ---------------------------------------------------------------------------
// top-k selection (naive)
//
// Returns the k largest elements of x with their indices, sorted descending.
// Naive algorithm: k passes over the data, each extracting the current max
// and marking it as visited. O(n * k) - fine for small k, educational.
// Returns {values, indices}.
// ---------------------------------------------------------------------------

std::pair<std::vector<float>, std::vector<int>> topk_cpu(const std::vector<float>& x, int k);
std::pair<std::vector<float>, std::vector<int>> topk_cuda(const std::vector<float>& x, int k);

}
