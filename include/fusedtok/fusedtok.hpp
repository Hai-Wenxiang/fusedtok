#pragma once

#include <vector>

namespace fusedtok {

// CPU reference implementation: y[i] = a * x[i] + b.
// Serves as ground truth for GPU parity tests and runs anywhere.
std::vector<float> axpy_cpu(const std::vector<float>& x, float a, float b);

// GPU implementation: naive one-thread-per-element kernel.
// Host buffers in / host buffers out; all transfers handled internally.
std::vector<float> axpy_cuda(const std::vector<float>& x, float a, float b);

// Returns true if at least one CUDA device is usable.
bool cuda_available();

}
