#pragma once

#include <utility>
#include <vector>

namespace fusedtok {

// ---------------------------------------------------------------------------
// Softmax (row-wise, max-subtracted for numerical stability)
//
//   y[row, i] = exp(x[row, i] - max(row)) / sum_i exp(x[row, i] - max(row))
//
// x is flattened row-major [rows, cols].
// ---------------------------------------------------------------------------

// CPU reference implementation.
std::vector<float> softmax_cpu(const std::vector<float>& x, int rows, int cols);

// GPU implementation, naive version: one thread per row, three serial loops
// (max, sum of exp, write). Same kernel structure as the CPU reference.
std::vector<float> softmax_cuda(const std::vector<float>& x, int rows, int cols);

}
