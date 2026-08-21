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

}
