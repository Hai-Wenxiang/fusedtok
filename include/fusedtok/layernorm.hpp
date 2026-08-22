#pragma once

#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// LayerNorm
//
//   y = (x - mean(row)) / sqrt(var(row) + eps) * w + b
//
// var is the biased (population) variance: mean of squared deviations.
// x is flattened row-major [rows, cols]; weight and bias are [cols].
// CPU reference; the CUDA path goes through layernorm_launch.
// ---------------------------------------------------------------------------

std::vector<float> layernorm_cpu(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 const std::vector<float>& b,
                                 int rows, int cols, float eps);

} // namespace fusedtok
