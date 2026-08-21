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

    // RMSNorm with optional residual: y = (x + r) * rsqrt(mean((x+r)^2) + eps) * w.
    // x/r are flattened row-major [rows, cols], w is [cols]. r may be None.
    m.def("rmsnorm", [](const std::vector<float>& x, const std::vector<float>& w,
                        int rows, int cols, float eps, py::object residual, bool use_cuda) {
        const std::vector<float>* r = nullptr;
        std::vector<float> keep_alive;
        if (!residual.is_none()) {
            keep_alive = residual.cast<std::vector<float>>();
            r = &keep_alive;
        }
        return use_cuda ? fusedtok::rmsnorm_cuda(x, w, rows, cols, eps, r)
                        : fusedtok::rmsnorm_cpu(x, w, rows, cols, eps, r);
    }, py::arg("x"), py::arg("w"), py::arg("rows"), py::arg("cols"),
       py::arg("eps"), py::arg("residual") = py::none(), py::arg("cuda") = false,
       "RMSNorm (optionally fused with residual add).");
}
