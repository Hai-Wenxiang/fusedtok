#pragma once

#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// Softmax (row-wise, max-subtracted for numerical stability)
//
//   y[row, i] = exp(x[row, i] - max(row)) / sum_i exp(x[row, i] - max(row))
//
// x is flattened row-major [rows, cols]. CPU reference; the CUDA path goes
// through softmax_launch (cuda_launch.hpp).
// ---------------------------------------------------------------------------

std::vector<float> softmax_cpu(const std::vector<float>& x, int rows, int cols);

} // namespace fusedtok
