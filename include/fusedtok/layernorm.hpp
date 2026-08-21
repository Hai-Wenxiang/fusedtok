#pragma once

#include <utility>
#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// LayerNorm
//
//   y = (x - mean(row)) / sqrt(var(row) + eps) * w + b
//
// var is the biased (population) variance: mean of squared deviations.
// x is flattened row-major [rows, cols]; weight and bias are [cols].
// ---------------------------------------------------------------------------

// CPU reference implementation.
std::vector<float> layernorm_cpu(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 const std::vector<float>& b,
                                 int rows, int cols, float eps);

// GPU implementation, naive version: one thread per row, serial loops for
// mean, variance, and the normalized write.
std::vector<float> layernorm_cuda(const std::vector<float>& x,
                                  const std::vector<float>& w,
                                  const std::vector<float>& b,
                                  int rows, int cols, float eps);

}
