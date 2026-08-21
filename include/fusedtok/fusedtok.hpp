#pragma once
#include <vector>

namespace fusedtok {

std::vector<float> axpy_cpu(const std::vector<float>& x, float a, float b);
std::vector<float> axpy_cuda(const std::vector<float>& x, float a, float b);
bool cuda_available();

}
