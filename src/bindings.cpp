#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fusedtok/fusedtok.hpp"
#include "fusedtok/activations.hpp"
#include "fusedtok/softmax.hpp"
#include "fusedtok/layernorm.hpp"

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

    // Elementwise activations
    m.def("silu", [](const std::vector<float>& x, bool use_cuda) {
        return use_cuda ? fusedtok::silu_cuda(x) : fusedtok::silu_cpu(x);
    }, py::arg("x"), py::arg("cuda") = false,
       "SiLU activation: v * sigmoid(v).");
    m.def("gelu", [](const std::vector<float>& x, bool use_cuda) {
        return use_cuda ? fusedtok::gelu_cuda(x) : fusedtok::gelu_cpu(x);
    }, py::arg("x"), py::arg("cuda") = false,
       "GeLU activation (exact erf form).");
    m.def("relu", [](const std::vector<float>& x, bool use_cuda) {
        return use_cuda ? fusedtok::relu_cuda(x) : fusedtok::relu_cpu(x);
    }, py::arg("x"), py::arg("cuda") = false,
       "ReLU activation.");
    m.def("tanh", [](const std::vector<float>& x, bool use_cuda) {
        return use_cuda ? fusedtok::tanh_cuda(x) : fusedtok::tanh_cpu(x);
    }, py::arg("x"), py::arg("cuda") = false,
       "Tanh activation.");

    // Top-k selection: returns (values, indices), descending, deterministic
    // (earliest index wins ties).
    m.def("topk", [](const std::vector<float>& x, int k, bool use_cuda) {
        return use_cuda ? fusedtok::topk_cuda(x, k) : fusedtok::topk_cpu(x, k);
    }, py::arg("x"), py::arg("k"), py::arg("cuda") = false,
       "Return the k largest elements and their indices (descending).");

    // Top-p (nucleus) selection over a probability vector.
    m.def("topp", [](const std::vector<float>& probs, float p, bool use_cuda) {
        return use_cuda ? fusedtok::topp_cuda(probs, p) : fusedtok::topp_cpu(probs, p);
    }, py::arg("probs"), py::arg("p"), py::arg("cuda") = false,
       "Smallest set of top probabilities with cumulative mass >= p.");

    // Row-wise softmax over a flattened [rows, cols] tensor.
    m.def("softmax", [](const std::vector<float>& x, int rows, int cols, bool use_cuda) {
        return use_cuda ? fusedtok::softmax_cuda(x, rows, cols)
                        : fusedtok::softmax_cpu(x, rows, cols);
    }, py::arg("x"), py::arg("rows"), py::arg("cols"), py::arg("cuda") = false,
       "Row-wise numerically stable softmax.");

    // LayerNorm with affine weight/bias over a flattened [rows, cols] tensor.
    m.def("layernorm", [](const std::vector<float>& x, const std::vector<float>& w,
                          const std::vector<float>& b, int rows, int cols, float eps,
                          bool use_cuda) {
        return use_cuda ? fusedtok::layernorm_cuda(x, w, b, rows, cols, eps)
                        : fusedtok::layernorm_cpu(x, w, b, rows, cols, eps);
    }, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("rows"), py::arg("cols"),
       py::arg("eps"), py::arg("cuda") = false,
       "LayerNorm with learned affine transform.");
}
