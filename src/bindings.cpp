#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fusedtok/fusedtok.hpp"

namespace py = pybind11;

// Thin binding layer: argument forwarding only, no logic here.
// std::vector<float> converts to/from Python lists automatically.
PYBIND11_MODULE(_fusedtok, m) {
    m.doc() = "fusedtok: fused CUDA kernels for LLM inference";

    m.def("axpy", [](const std::vector<float>& x, float a, float b, bool use_cuda) {
        return use_cuda ? fusedtok::axpy_cuda(x, a, b) : fusedtok::axpy_cpu(x, a, b);
    }, py::arg("x"), py::arg("a"), py::arg("b"), py::arg("cuda") = false,
       "Compute y = a * x + b element-wise (CPU or CUDA).");

    m.def("cuda_available", &fusedtok::cuda_available,
          "True if a CUDA device is available.");
}
