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

    // SwiGLU activation: out = silu(gate) * up.
    m.def("swiglu", [](const std::vector<float>& gate, const std::vector<float>& up, bool use_cuda) {
        return use_cuda ? fusedtok::swiglu_cuda(gate, up)
                        : fusedtok::swiglu_cpu(gate, up);
    }, py::arg("gate"), py::arg("up"), py::arg("cuda") = false,
       "SwiGLU activation: silu(gate) * up.");

    // RoPE (interleaved pairs). q/k flattened row-major [seq, dim], dim even.
    // Returns (q_rotated, k_rotated); the second element is None if k is None.
    m.def("rope", [](const std::vector<float>& q, py::object k,
                     int seq, int dim, float theta, bool use_cuda) {
        const std::vector<float>* kp = nullptr;
        std::vector<float> keep_alive;
        if (!k.is_none()) {
            keep_alive = k.cast<std::vector<float>>();
            kp = &keep_alive;
        }
        auto result = use_cuda ? fusedtok::rope_cuda(q, kp, seq, dim, theta)
                               : fusedtok::rope_cpu(q, kp, seq, dim, theta);
        py::object k_out = result.second.empty() && !kp ? py::none() : py::cast(result.second);
        return py::make_tuple(py::cast(result.first), k_out);
    }, py::arg("q"), py::arg("k"), py::arg("seq"), py::arg("dim"),
       py::arg("theta") = 10000.0f, py::arg("cuda") = false,
       "Rotary position embedding on (q, k); returns (q', k').");
}
