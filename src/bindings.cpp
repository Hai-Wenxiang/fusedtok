// pybind11 binding layer for the fusedtok native module.
//
// Three entry styles per operator:
//   1. "<op>_cpu"  - numpy in / numpy out, runs the std::vector CPU reference
//      implementations. These are the ground truth for parity tests and run
//      on CI machines without a GPU.
//   2. "<op>"      - staged CUDA: numpy in / numpy out. Device memory is
//      managed here (DevBuf RAII), transfers in/out, launch, synchronize.
//   3. "<op>_launch" - raw CUDA entry taking int64 device pointers, no
//      staging and no synchronization. The Python layer drives these with
//      torch CUDA tensors (torch.empty + data_ptr()), so kernels read and
//      write torch's own buffers with zero copies.
//
// Error contract (whole library): shape problems raise std::invalid_argument
// (surfaces as Python ValueError), CUDA problems raise std::runtime_error
// (surfaces as Python RuntimeError).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fusedtok/fusedtok.hpp"
#include "fusedtok/activations.hpp"
#include "fusedtok/softmax.hpp"
#include "fusedtok/layernorm.hpp"
#include "fusedtok/attention.hpp"
#include "fusedtok/cuda_launch.hpp"

namespace py = pybind11;
namespace ft = fusedtok;

namespace {

[[noreturn]] void throw_cuda(const char* what) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(cudaGetLastError()));
}

// RAII device buffer holding raw bytes. Empty allocations stay null.
class DevBuf {
public:
    explicit DevBuf(size_t bytes) {
        if (bytes == 0) return;
        if (cudaMalloc(&p_, bytes) != cudaSuccess) {
            p_ = nullptr;
            throw_cuda("cudaMalloc failed");
        }
    }
    ~DevBuf() { if (p_) cudaFree(p_); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    void* get() const { return p_; }
    float* fget() const { return static_cast<float*>(p_); }

private:
    void* p_ = nullptr;
};

void h2d(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        throw_cuda("H2D copy failed");
}
void d2h(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        throw_cuda("D2H copy failed");
}
void sync_device(const char* what) {
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(std::string(what) + " failed: " +
                                 cudaGetErrorString(cudaGetLastError()));
}

// ---------------------------------------------------------------------------
// numpy helpers
// ---------------------------------------------------------------------------

// forcecast converts other dtypes to float32 and c_style guarantees
// contiguity (making a temporary copy when needed) - convenient and safe,
// at the cost of a hidden copy for exotic inputs.
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::vector<py::ssize_t> shape_of(const py::array_t<float>& a) {
    return std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim());
}

bool same_shape(const py::array_t<float>& a, const py::array_t<float>& b) {
    return shape_of(a) == shape_of(b);
}

// Flatten to [rows, cols] where cols is the last dimension. 1-D input is
// treated as a single row.
void rows_cols_of(const FArray& a, long long& rows, long long& cols) {
    if (a.ndim() == 1) {
        rows = 1;
        cols = a.shape(0);
    } else if (a.ndim() == 2) {
        rows = a.shape(0);
        cols = a.shape(1);
    } else {
        throw std::invalid_argument("expected a 1-D or 2-D array");
    }
}

std::vector<float> to_vec(const FArray& a) {
    return std::vector<float>(a.data(), a.data() + a.size());
}

py::array_t<float> wrap_vec(const std::vector<float>& v,
                            const std::vector<py::ssize_t>& shape) {
    py::array_t<float> out(shape);
    if (!v.empty())
        std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
    return out;
}

py::array_t<long long> wrap_ivec(const std::vector<long long>& v) {
    py::array_t<long long> out(v.size());
    if (!v.empty())
        std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(long long));
    return out;
}

// Interpret Python ints (torch data_ptr()) as device pointers.
const float* df(py::int_ p) { return reinterpret_cast<const float*>((uintptr_t)p); }

// Batched-sampling seed conversion (v1.4): a contiguous int64 host
// array to unsigned long long row seeds. No forcecast on the array
// type - a float dtype is a caller bug, not something to round away.
// The launcher's per-attempt synchronization makes the async upload of
// the temporary copy safe.
using I64Array = py::array_t<long long, py::array::c_style>;

std::vector<unsigned long long> seeds_vec(const I64Array& a) {
    const long long* p = a.data();
    return std::vector<unsigned long long>(
        reinterpret_cast<const unsigned long long*>(p),
        reinterpret_cast<const unsigned long long*>(p + a.size()));
}

// Batched-sampler validation (v1.4.1): the checks every entry style
// shares, collapsed into one place after the 1.4.0 bindings drifted
// apart - the staged trio missed the shape-vs-buffer check (an
// oversized rows*n read past the numpy buffer) and the _cpu trio
// missed n <= 0 (an inverted pointer range is UB). The Python layer
// derives rows/n from the shape, but _fusedtok is a supported direct
// surface and gets the same contract.
void check_batch_rows_n(int rows, int n) {
    if (rows < 0)
        throw std::invalid_argument("rows must be >= 0");
    if (n <= 0)
        throw std::invalid_argument("sample of empty logits");
}

void check_batch_host(const FArray& logits, int rows, int n) {
    if (logits.ndim() != 2)
        throw std::invalid_argument("logits must be 2-D");
    check_batch_rows_n(rows, n);
    if (logits.shape(0) != rows || logits.shape(1) != n)
        throw std::invalid_argument("logits shape must be [rows, n]");
}

void check_batch_seeds(const I64Array& seeds, int rows) {
    if (seeds.size() != rows)
        throw std::invalid_argument("seeds must have one entry per row");
}

void check_batch_unit(const char* what, double v) {
    if (!(v > 0.0 && v <= 1.0))
        throw std::invalid_argument(std::string(what) +
                                    " must be in (0, 1]");
}

void check_batch_temp(double t) {
    if (!(t > 0.0))
        throw std::invalid_argument("temperature must be > 0");
}

// decode_step_batched ids/offsets validation (v1.5): offsets are the
// ragged-history contract (rows + 1 non-decreasing entries, 0 to
// length), and HOST-ORIGIN id values are range-checked - the values
// are visible here, and the id arrays ride a host upload anyway, so
// unlike device-resident lens/table data there is no sync to avoid.
void check_batch_ids(const I64Array& ids, const I64Array& offs, int rows,
                     int n) {
    if (offs.size() != rows + 1)
        throw std::invalid_argument(
            "sampled_ids offsets must have rows + 1 entries");
    if (offs.size() == 0 || offs.at(0) != 0 ||
        offs.at(offs.size() - 1) != (long long)ids.size())
        throw std::invalid_argument(
            "sampled_ids offsets must start at 0 and end at its length");
    for (py::ssize_t i = 1; i < offs.size(); ++i)
        if (offs.at(i) < offs.at(i - 1))
            throw std::invalid_argument(
                "sampled_ids offsets must be non-decreasing");
    const long long* p = ids.data();
    for (py::ssize_t i = 0; i < ids.size(); ++i)
        if (p[i] < 0 || p[i] >= n)
            throw std::invalid_argument(
                "sampled_ids entries must be in [0, vocab)");
}
float* dfm(py::int_ p) { return reinterpret_cast<float*>((uintptr_t)p); }
const long long* dll(py::int_ p) { return reinterpret_cast<const long long*>((uintptr_t)p); }
long long* dllm(py::int_ p) { return reinterpret_cast<long long*>((uintptr_t)p); }
int* dim_(py::int_ p) { return reinterpret_cast<int*>((uintptr_t)p); }
const int* dic(py::int_ p) {
    return reinterpret_cast<const int*>((uintptr_t)p);
}

// Untyped device pointers for the dtype-templated attention bindings
// (bf16/fp16 launchers take void* and cast on the C++ side).
const void* dvoid(py::int_ p) {
    return reinterpret_cast<const void*>((uintptr_t)p);
}
void* dvoidm(py::int_ p) { return reinterpret_cast<void*>((uintptr_t)p); }

// Optional pointer from an optional tensor-like argument: py::none (or
// an omitted default) becomes nullptr. Used for the attention `lens`
// and friends, whose "not provided" case means "use the full cache".
template <typename T>
const T* opt_ptr(const py::object& o) {
    if (o.is_none())
        return nullptr;
    return reinterpret_cast<const T*>((uintptr_t)py::int_(o));
}
const int* opt_int_ptr(const py::object& o) { return opt_ptr<int>(o); }

// ---------------------------------------------------------------------------
// Staged CUDA drivers for elementwise ops (numpy in -> numpy out)
// ---------------------------------------------------------------------------

using UnaryLauncher = void (*)(const float*, float*, long long,
                               std::uintptr_t);
using BinaryLauncher = void (*)(const float*, const float*, float*, long long,
                                std::uintptr_t);

// Templated so captured lambdas (parameterized launches) work too.
template <typename F>
py::array_t<float> staged_unary(const FArray& x, F launch) {
    const long long n = x.size();
    py::array_t<float> y(shape_of(x));
    if (n == 0) return y;
    DevBuf dx(n * sizeof(float)), dy(n * sizeof(float));
    h2d(dx.get(), x.data(), n * sizeof(float));
    launch(dx.fget(), dy.fget(), n, 0);
    d2h(y.mutable_data(), dy.get(), n * sizeof(float));
    sync_device("elementwise kernel");
    return y;
}

py::array_t<float> staged_binary(const FArray& a, const FArray& b,
                                 BinaryLauncher launch) {
    if (!same_shape(a, b))
        throw std::invalid_argument("inputs must have the same shape");
    const long long n = a.size();
    py::array_t<float> y(shape_of(a));
    if (n == 0) return y;
    DevBuf da(n * sizeof(float)), db(n * sizeof(float)), dy(n * sizeof(float));
    h2d(da.get(), a.data(), n * sizeof(float));
    h2d(db.get(), b.data(), n * sizeof(float));
    launch(da.fget(), db.fget(), dy.fget(), n, 0);
    d2h(y.mutable_data(), dy.get(), n * sizeof(float));
    sync_device("elementwise kernel");
    return y;
}

} // namespace

PYBIND11_MODULE(_fusedtok, m) {
    m.doc() = "fusedtok native module: CPU reference, staged CUDA, and raw "
              "device-pointer launchers";

    m.def("cuda_available", &ft::cuda_available,
          "True if a CUDA device context can be created.");

    // ==================================================================
    // axpy (skeleton/demo operator): y = a * x + b
    // ==================================================================
    m.def("axpy_cpu", [](FArray x, float a, float b) {
        return wrap_vec(ft::axpy_cpu(to_vec(x), a, b), shape_of(x));
    }, py::arg("x"), py::arg("a"), py::arg("b"));
    m.def("axpy", [](FArray x, float a, float b) {
        return staged_unary(x, [a, b](const float* in, float* out, long long n,
                                      std::uintptr_t stream) {
            ft::axpy_launch(in, out, n, a, b, stream);
        });
    }, py::arg("x"), py::arg("a"), py::arg("b"));

    // ==================================================================
    // Elementwise unary activations
    // ==================================================================
    m.def("silu_cpu", [](FArray x) { return wrap_vec(ft::silu_cpu(to_vec(x)), shape_of(x)); },
          py::arg("x"));
    m.def("silu", [](FArray x) { return staged_unary(x, ft::silu_launch); }, py::arg("x"));

    m.def("gelu_cpu", [](FArray x) { return wrap_vec(ft::gelu_cpu(to_vec(x)), shape_of(x)); },
          py::arg("x"));
    m.def("gelu", [](FArray x) { return staged_unary(x, ft::gelu_launch); }, py::arg("x"));

    m.def("gelu_tanh_cpu",
          [](FArray x) {
              return wrap_vec(ft::gelu_tanh_cpu(to_vec(x)), shape_of(x));
          }, py::arg("x"));
    m.def("gelu_tanh",
          [](FArray x) { return staged_unary(x, ft::gelu_tanh_launch); },
          py::arg("x"));

    m.def("relu_cpu", [](FArray x) { return wrap_vec(ft::relu_cpu(to_vec(x)), shape_of(x)); },
          py::arg("x"));
    m.def("relu", [](FArray x) { return staged_unary(x, ft::relu_launch); }, py::arg("x"));

    m.def("tanh_cpu", [](FArray x) { return wrap_vec(ft::tanh_cpu(to_vec(x)), shape_of(x)); },
          py::arg("x"));
    m.def("tanh", [](FArray x) { return staged_unary(x, ft::tanh_launch); }, py::arg("x"));

    m.def("sigmoid_cpu", [](FArray x) { return wrap_vec(ft::sigmoid_cpu(to_vec(x)), shape_of(x)); },
          py::arg("x"));
    m.def("sigmoid", [](FArray x) { return staged_unary(x, ft::sigmoid_launch); }, py::arg("x"));

    m.def("temperature_cpu", [](FArray x, float t) {
        return wrap_vec(ft::temperature_cpu(to_vec(x), t), shape_of(x));
    }, py::arg("x"), py::arg("t"));
    m.def("temperature", [](FArray x, float t) {
        if (!(t > 0.0f)) throw std::invalid_argument("temperature must be > 0");
        return staged_unary(x, [t](const float* in, float* out, long long n,
                                   std::uintptr_t stream) {
            ft::temperature_launch(in, out, n, t, stream);
        });
    }, py::arg("x"), py::arg("t"));

    // ==================================================================
    // Elementwise binary ops
    // ==================================================================
    m.def("add_cpu", [](FArray a, FArray b) {
        return wrap_vec(ft::add_cpu(to_vec(a), to_vec(b)), shape_of(a));
    }, py::arg("a"), py::arg("b"));
    m.def("add", [](FArray a, FArray b) { return staged_binary(a, b, ft::add_launch); },
          py::arg("a"), py::arg("b"));

    m.def("mul_cpu", [](FArray a, FArray b) {
        return wrap_vec(ft::mul_cpu(to_vec(a), to_vec(b)), shape_of(a));
    }, py::arg("a"), py::arg("b"));
    m.def("mul", [](FArray a, FArray b) { return staged_binary(a, b, ft::mul_launch); },
          py::arg("a"), py::arg("b"));

    m.def("swiglu_cpu", [](FArray g, FArray u) {
        if (!same_shape(g, u))
            throw std::invalid_argument("gate and up must have the same shape");
        return wrap_vec(ft::swiglu_cpu(to_vec(g), to_vec(u)), shape_of(g));
    }, py::arg("gate"), py::arg("up"));
    m.def("swiglu", [](FArray g, FArray u) {
        if (!same_shape(g, u))
            throw std::invalid_argument("gate and up must have the same shape");
        return staged_binary(g, u, ft::swiglu_launch);
    }, py::arg("gate"), py::arg("up"));

    // Raw launchers for the zero-copy torch path -------------------------
    m.def("axpy_launch",
          [](py::int_ in, py::int_ out, long long n, float a, float b, std::uintptr_t stream) {
        ft::axpy_launch(df(in), dfm(out), n, a, b, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("a"),
        py::arg("b"), py::arg("stream") = 0);
    m.def("silu_launch", [](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::silu_launch(df(in), dfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("gelu_launch", [](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::gelu_launch(df(in), dfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("gelu_tanh_launch", [](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::gelu_tanh_launch(df(in), dfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("relu_launch", [](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::relu_launch(df(in), dfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("tanh_launch", [](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::tanh_launch(df(in), dfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("sigmoid_launch", [](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::sigmoid_launch(df(in), dfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("temperature_launch",
          [](py::int_ in, py::int_ out, long long n, float t, std::uintptr_t stream) {
        if (!(t > 0.0f))
            throw std::invalid_argument("temperature must be > 0");
        ft::temperature_launch(df(in), dfm(out), n, t, stream); }, py::arg("in"), py::arg("out"),
        py::arg("n"), py::arg("t"), py::arg("stream") = 0);
    m.def("add_launch",
          [](py::int_ a, py::int_ b, py::int_ out, long long n, std::uintptr_t stream) {
        ft::add_launch(df(a), df(b), dfm(out), n, stream); }, py::arg("a"), py::arg("b"),
        py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("mul_launch",
          [](py::int_ a, py::int_ b, py::int_ out, long long n, std::uintptr_t stream) {
        ft::mul_launch(df(a), df(b), dfm(out), n, stream); }, py::arg("a"), py::arg("b"),
        py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("swiglu_launch",
          [](py::int_ g, py::int_ u, py::int_ out, long long n, std::uintptr_t stream) {
        ft::swiglu_launch(df(g), df(u), dfm(out), n, stream); }, py::arg("gate"), py::arg("up"),
        py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    // bf16 zero-copy launchers (torch bf16 tensors; compute stays float32)
    using BF = __nv_bfloat16;
    auto dbf = [](py::int_ p) { return reinterpret_cast<const BF*>((uintptr_t)p); };
    auto dbfm = [](py::int_ p) { return reinterpret_cast<BF*>((uintptr_t)p); };
    m.def("silu_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::silu_launch_bf16(dbf(in), dbfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("gelu_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::gelu_launch_bf16(dbf(in), dbfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("gelu_tanh_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::gelu_tanh_launch_bf16(dbf(in), dbfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("relu_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::relu_launch_bf16(dbf(in), dbfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("tanh_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::tanh_launch_bf16(dbf(in), dbfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("sigmoid_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, long long n, std::uintptr_t stream) {
        ft::sigmoid_launch_bf16(dbf(in), dbfm(out), n, stream);
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("add_launch_bf16",
          [dbf, dbfm](py::int_ a, py::int_ b, py::int_ out, long long n, std::uintptr_t stream) {
        ft::add_launch_bf16(dbf(a), dbf(b), dbfm(out), n, stream); }, py::arg("a"), py::arg("b"),
        py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("mul_launch_bf16",
          [dbf, dbfm](py::int_ a, py::int_ b, py::int_ out, long long n, std::uintptr_t stream) {
        ft::mul_launch_bf16(dbf(a), dbf(b), dbfm(out), n, stream); }, py::arg("a"), py::arg("b"),
        py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("swiglu_launch_bf16",
          [dbf, dbfm](py::int_ g, py::int_ u, py::int_ out, long long n, std::uintptr_t stream) {
        ft::swiglu_launch_bf16(dbf(g), dbf(u), dbfm(out), n, stream);
    }, py::arg("gate"), py::arg("up"),
        py::arg("out"), py::arg("n"), py::arg("stream") = 0);
    m.def("rmsnorm_launch_bf16",
          [dbf, dbfm](py::int_ x, py::int_ w, py::object r,
                      py::int_ out, int rows, int cols, float eps,
                      std::uintptr_t stream) {
        // residual shares x's bf16 storage (the wrapper enforces the
        // dtype match), so it rides the BF* path directly
        const BF* rp = opt_ptr<BF>(r);
        ft::rmsnorm_launch_bf16(dbf(x), df(w), rp, dbfm(out), rows, cols, eps, stream);
    }, py::arg("x"), py::arg("weight"), py::arg("residual"), py::arg("out"),
       py::arg("rows"), py::arg("cols"), py::arg("eps"),
       py::arg("stream") = 0);
    m.def("layernorm_launch_bf16",
          [dbf, dbfm](py::int_ x, py::int_ w, py::int_ b,
                      py::int_ out, int rows, int cols, float eps,
                      std::uintptr_t stream) {
        ft::layernorm_launch_bf16(dbf(x), df(w), df(b), dbfm(out), rows, cols, eps, stream);
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("rows"), py::arg("cols"), py::arg("eps"), py::arg("stream") = 0);
    m.def("softmax_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, int rows, int cols, std::uintptr_t stream) {
        ft::softmax_launch_bf16(dbf(in), dbfm(out), rows, cols, stream);
    }, py::arg("in"), py::arg("out"), py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);
    m.def("rope_launch_bf16", [dbf, dbfm](py::int_ in, py::int_ out, int seq, int dim,
                                          double theta, int pos_offset, std::uintptr_t stream) {
        ft::rope_launch_bf16(dbf(in), dbfm(out), seq, dim, (float)theta, pos_offset, stream);
    }, py::arg("in"), py::arg("out"), py::arg("seq"), py::arg("dim"),
       py::arg("theta"), py::arg("pos_offset"), py::arg("stream") = 0);
    m.def("rope_neox_launch_bf16",
          [dbf, dbfm](py::int_ in, py::int_ out, int seq,
                      int dim, double theta, int pos_offset,
                      std::uintptr_t stream) {
        ft::rope_neox_launch_bf16(dbf(in), dbfm(out), seq, dim, (float)theta, pos_offset, stream);
    }, py::arg("in"), py::arg("out"), py::arg("seq"), py::arg("dim"),
       py::arg("theta"), py::arg("pos_offset"), py::arg("stream") = 0);

    // ==================================================================
    // RMSNorm / LayerNorm / softmax (row-major 2-D)
    // ==================================================================
    m.def("rmsnorm_cpu", [](FArray x, FArray w, py::object residual, float eps) {
        long long rows, cols;
        rows_cols_of(x, rows, cols);
        if (w.ndim() != 1 || w.shape(0) != cols)
            throw std::invalid_argument("weight must be 1-D with length cols");
        std::vector<float> r;
        const std::vector<float>* rp = nullptr;
        if (!residual.is_none()) {
            FArray r_arr = residual.cast<FArray>();
            if (!same_shape(r_arr, x))
                throw std::invalid_argument("residual must have the same shape as x");
            r = to_vec(r_arr);
            rp = &r;
        }
        return wrap_vec(ft::rmsnorm_cpu(to_vec(x), to_vec(w), (int)rows, (int)cols, eps, rp),
                        shape_of(x));
    }, py::arg("x"), py::arg("weight"), py::arg("residual") = py::none(),
       py::arg("eps") = 1e-6f);

    m.def("rmsnorm", [](FArray x, FArray w, py::object residual, float eps) {
        long long rows, cols;
        rows_cols_of(x, rows, cols);
        if (w.ndim() != 1 || w.shape(0) != cols)
            throw std::invalid_argument("weight must be 1-D with length cols");
        const long long n = rows * cols;
        py::array_t<float> y(shape_of(x));
        if (n == 0) return y;
        bool has_r = !residual.is_none();
        FArray r_arr;
        if (has_r) {
            r_arr = residual.cast<FArray>();
            if (!same_shape(r_arr, x))
                throw std::invalid_argument("residual must have the same shape as x");
        }
        DevBuf dx(n * 4), dw(cols * 4), dr(has_r ? n * 4 : 0), dy(n * 4);
        h2d(dx.get(), x.data(), n * 4);
        h2d(dw.get(), w.data(), cols * 4);
        if (has_r) h2d(dr.get(), r_arr.data(), n * 4);
        ft::rmsnorm_launch(dx.fget(), dw.fget(), has_r ? dr.fget() : nullptr,
                           dy.fget(), (int)rows, (int)cols, eps);
        d2h(y.mutable_data(), dy.get(), n * 4);
        sync_device("rmsnorm kernel");
        return y;
    }, py::arg("x"), py::arg("weight"), py::arg("residual") = py::none(),
       py::arg("eps") = 1e-6f);

    m.def("rmsnorm_launch", [](py::int_ x, py::int_ w, py::object r, py::int_ out,
                               int rows, int cols, float eps, std::uintptr_t stream) {
        const float* rp = r.is_none()
            ? nullptr
            : reinterpret_cast<const float*>((uintptr_t)py::int_(r));
        ft::rmsnorm_launch(df(x), df(w), rp, dfm(out), rows, cols, eps, stream);
    }, py::arg("x"), py::arg("weight"), py::arg("residual"), py::arg("out"),
       py::arg("rows"), py::arg("cols"), py::arg("eps"), py::arg("stream") = 0);

    m.def("layernorm_cpu", [](FArray x, FArray w, FArray b, float eps) {
        long long rows, cols;
        rows_cols_of(x, rows, cols);
        if (w.ndim() != 1 || w.shape(0) != cols || b.ndim() != 1 || b.shape(0) != cols)
            throw std::invalid_argument("weight/bias must be 1-D with length cols");
        return wrap_vec(ft::layernorm_cpu(to_vec(x), to_vec(w), to_vec(b),
                                          (int)rows, (int)cols, eps), shape_of(x));
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-6f);

    m.def("layernorm", [](FArray x, FArray w, FArray b, float eps) {
        long long rows, cols;
        rows_cols_of(x, rows, cols);
        if (w.ndim() != 1 || w.shape(0) != cols || b.ndim() != 1 || b.shape(0) != cols)
            throw std::invalid_argument("weight/bias must be 1-D with length cols");
        const long long n = rows * cols;
        py::array_t<float> y(shape_of(x));
        if (n == 0) return y;
        DevBuf dx(n * 4), dw(cols * 4), db(cols * 4), dy(n * 4);
        h2d(dx.get(), x.data(), n * 4);
        h2d(dw.get(), w.data(), cols * 4);
        h2d(db.get(), b.data(), cols * 4);
        ft::layernorm_launch(dx.fget(), dw.fget(), db.fget(), dy.fget(),
                             (int)rows, (int)cols, eps);
        d2h(y.mutable_data(), dy.get(), n * 4);
        sync_device("layernorm kernel");
        return y;
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-6f);

    m.def("layernorm_launch", [](py::int_ x, py::int_ w, py::int_ b, py::int_ out,
                                 int rows, int cols, float eps, std::uintptr_t stream) {
        ft::layernorm_launch(df(x), df(w), df(b), dfm(out), rows, cols, eps, stream);
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("rows"), py::arg("cols"), py::arg("eps"), py::arg("stream") = 0);

    m.def("softmax_cpu", [](FArray x) {
        long long rows, cols;
        rows_cols_of(x, rows, cols);
        return wrap_vec(ft::softmax_cpu(to_vec(x), (int)rows, (int)cols), shape_of(x));
    }, py::arg("x"));

    m.def("softmax", [](FArray x) {
        long long rows, cols;
        rows_cols_of(x, rows, cols);
        const long long n = rows * cols;
        py::array_t<float> y(shape_of(x));
        if (n == 0) return y;
        DevBuf dx(n * 4), dy(n * 4);
        h2d(dx.get(), x.data(), n * 4);
        ft::softmax_launch(dx.fget(), dy.fget(), (int)rows, (int)cols);
        d2h(y.mutable_data(), dy.get(), n * 4);
        sync_device("softmax kernel");
        return y;
    }, py::arg("x"));

    m.def("softmax_launch",
          [](py::int_ in, py::int_ out, int rows, int cols, std::uintptr_t stream) {
        ft::softmax_launch(df(in), dfm(out), rows, cols, stream);
    }, py::arg("in"), py::arg("out"), py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    // ==================================================================
    // RoPE (both layouts, optional position offset for kv-cache decoding)
    // ==================================================================
    m.def("rope_cpu", [](bool neox, FArray q, py::object k, double theta,
                         long long pos_offset) {
        if (q.ndim() != 2)
            throw std::invalid_argument("q must be 2-D [seq, dim]");
        const long long seq = q.shape(0), dim = q.shape(1);
        if (dim % 2 != 0)
            throw std::invalid_argument("dim must be even (RoPE pairs elements)");
        if (pos_offset < 0)
            throw std::invalid_argument("pos_offset must be >= 0");
        const std::vector<float>* kp = nullptr;
        std::vector<float> kv;
        if (!k.is_none()) {
            FArray k_arr = k.cast<FArray>();
            if (!same_shape(k_arr, q))
                throw std::invalid_argument("k must have the same shape as q");
            kv = to_vec(k_arr);
            kp = &kv;
        }
        auto result = neox
            ? ft::rope_neox_cpu(to_vec(q), kp, (int)seq, (int)dim, (float)theta, (int)pos_offset)
            : ft::rope_cpu(to_vec(q), kp, (int)seq, (int)dim, (float)theta, (int)pos_offset);
        // NB: no ternary here - py::none() would implicitly convert to
        // py::array_t and produce a garbage array.
        py::object k_out = py::none();
        if (kp)
            k_out = py::cast<py::array_t<float>>(wrap_vec(result.second, shape_of(q)));
        return py::make_tuple(wrap_vec(result.first, shape_of(q)), k_out);
    }, py::arg("neox"), py::arg("q"), py::arg("k"), py::arg("theta") = 10000.0,
       py::arg("pos_offset") = 0);

    m.def("rope", [](bool neox, FArray q, py::object k, double theta,
                     long long pos_offset) {
        if (q.ndim() != 2)
            throw std::invalid_argument("q must be 2-D [seq, dim]");
        const long long seq = q.shape(0), dim = q.shape(1);
        if (dim % 2 != 0)
            throw std::invalid_argument("dim must be even (RoPE pairs elements)");
        if (pos_offset < 0)
            throw std::invalid_argument("pos_offset must be >= 0");
        bool has_k = !k.is_none();
        FArray k_arr;
        if (has_k) {
            k_arr = k.cast<FArray>();
            if (!same_shape(k_arr, q))
                throw std::invalid_argument("k must have the same shape as q");
        }
        auto rotate_staged = [&](const FArray& in) -> py::array_t<float> {
            const long long n = seq * dim;
            py::array_t<float> out(shape_of(in));
            if (n == 0) return out;
            DevBuf dx(n * 4), dy(n * 4);
            h2d(dx.get(), in.data(), n * 4);
            if (neox)
                ft::rope_neox_launch(dx.fget(), dy.fget(), (int)seq, (int)dim,
                                     (float)theta, (int)pos_offset);
            else
                ft::rope_launch(dx.fget(), dy.fget(), (int)seq, (int)dim,
                                (float)theta, (int)pos_offset);
            d2h(out.mutable_data(), dy.get(), n * 4);
            sync_device("rope kernel");
            return out;
        };
        py::object k_out = py::none();
        if (has_k)
            k_out = rotate_staged(k_arr);
        return py::make_tuple(rotate_staged(q), k_out);
    }, py::arg("neox"), py::arg("q"), py::arg("k"), py::arg("theta") = 10000.0,
       py::arg("pos_offset") = 0);

    m.def("rope_launch", [](bool neox, py::int_ in, py::int_ out, int seq, int dim,
                            double theta, int pos_offset, std::uintptr_t stream) {
        // the raw launcher skips the Python wrapper's checks; without
        // them an odd dim silently leaves the last element of every
        // row unrotated (the same messages the wrapper raises)
        if (dim % 2 != 0)
            throw std::invalid_argument("dim must be even (RoPE pairs elements)");
        if (pos_offset < 0)
            throw std::invalid_argument("pos_offset must be >= 0");
        if (neox)
            ft::rope_neox_launch(df(in), dfm(out), seq, dim, (float)theta, pos_offset, stream);
        else
            ft::rope_launch(df(in), dfm(out), seq, dim, (float)theta, pos_offset, stream);
    }, py::arg("neox"), py::arg("in"), py::arg("out"), py::arg("seq"), py::arg("dim"),
       py::arg("theta"), py::arg("pos_offset"), py::arg("stream") = 0);

    // ==================================================================
    // Sampling / logits post-processing
    // ==================================================================
    m.def("argmax_cpu", [](FArray x) -> long long {
        if (x.ndim() != 1) throw std::invalid_argument("argmax expects 1-D input");
        return ft::argmax_cpu(to_vec(x));
    }, py::arg("x"));

    m.def("argmax", [](FArray x) -> long long {
        if (x.ndim() != 1) throw std::invalid_argument("argmax expects 1-D input");
        const int n = (int)x.size();
        if (n == 0) throw std::invalid_argument("argmax of empty input");
        DevBuf dx(n * 4);
        DevBuf dout(sizeof(int));
        h2d(dx.get(), x.data(), n * 4);
        ft::argmax_launch(dx.fget(), n, static_cast<int*>(dout.get()));
        long long idx = 0;
        d2h(&idx, dout.get(), sizeof(int));
        sync_device("argmax kernel");
        return idx;
    }, py::arg("x"));

    m.def("topk_cpu", [](FArray x, int k) {
        if (x.ndim() != 1) throw std::invalid_argument("topk expects 1-D input");
        if (k < 0 || (py::ssize_t)k > x.size())
            throw std::invalid_argument("k must be in [0, n]");
        auto [vals, idxs] = ft::topk_cpu(to_vec(x), k);
        return py::make_tuple(wrap_vec(vals, {(py::ssize_t)k}), wrap_ivec(idxs));
    }, py::arg("x"), py::arg("k"));

    m.def("topk", [](FArray x, int k) {
        if (x.ndim() != 1) throw std::invalid_argument("topk expects 1-D input");
        const int n = (int)x.size();
        if (k < 0 || k > n) throw std::invalid_argument("k must be in [0, n]");
        py::array_t<float> vals(std::vector<py::ssize_t>{(py::ssize_t)k});
        py::array_t<long long> idxs(std::vector<py::ssize_t>{(py::ssize_t)k});
        if (k == 0) return py::make_tuple(vals, idxs);
        DevBuf dx(n * 4), dv(k * 4), di((size_t)k * sizeof(long long));
        h2d(dx.get(), x.data(), n * 4);
        ft::topk_launch(dx.fget(), dv.fget(),
                        static_cast<long long*>(di.get()), n, k);
        d2h(vals.mutable_data(), dv.get(), (size_t)k * 4);
        d2h(idxs.mutable_data(), di.get(), (size_t)k * sizeof(long long));
        sync_device("topk kernel");
        return py::make_tuple(vals, idxs);
    }, py::arg("x"), py::arg("k"));

    m.def("topp_cpu", [](FArray probs, double p) {
        if (!(p > 0.0 && p <= 1.0))
            throw std::invalid_argument("p must be in (0, 1]");
        if (probs.ndim() != 1) throw std::invalid_argument("topp expects 1-D input");
        auto [vals, idxs] = ft::topp_cpu(to_vec(probs), (float)p);
        return py::make_tuple(wrap_vec(vals, {(py::ssize_t)vals.size()}), wrap_ivec(idxs));
    }, py::arg("probs"), py::arg("p"));

    m.def("topp", [](FArray probs, double p) {
        if (!(p > 0.0 && p <= 1.0))
            throw std::invalid_argument("p must be in (0, 1]");
        if (probs.ndim() != 1) throw std::invalid_argument("topp expects 1-D input");
        const int n = (int)probs.size();
        if (n == 0) {
            py::array_t<float> vals(std::vector<py::ssize_t>{0});
            py::array_t<long long> idxs(std::vector<py::ssize_t>{0});
            return py::make_tuple(vals, idxs);
        }
        DevBuf dx(n * 4), dv(n * 4), di((size_t)n * sizeof(long long)), dc(sizeof(int));
        h2d(dx.get(), probs.data(), n * 4);
        ft::topp_select_launch(dx.fget(), dv.fget(),
                               static_cast<long long*>(di.get()), n, (float)p,
                               static_cast<int*>(dc.get()));
        int count = 0;
        d2h(&count, dc.get(), sizeof(int));
        sync_device("topp kernels");
        py::array_t<float> vals(std::vector<py::ssize_t>{(py::ssize_t)count});
        py::array_t<long long> idxs(std::vector<py::ssize_t>{(py::ssize_t)count});
        if (count > 0) {
            d2h(vals.mutable_data(), dv.get(), (size_t)count * 4);
            d2h(idxs.mutable_data(), di.get(), (size_t)count * sizeof(long long));
        }
        return py::make_tuple(vals, idxs);
    }, py::arg("probs"), py::arg("p"));

    m.def("topk_launch",
          [](py::int_ x, py::int_ vals, py::int_ idxs, int n, int k, std::uintptr_t stream) {
        ft::topk_launch(df(x), dfm(vals), dllm(idxs), n, k, stream);
    }, py::arg("x"), py::arg("vals"), py::arg("idxs"), py::arg("n"),
        py::arg("k"), py::arg("stream") = 0);

    m.def("topp_select_launch", [](py::int_ x, py::int_ vals, py::int_ idxs,
                                   int n, double p, py::int_ count, std::uintptr_t stream) {
        ft::topp_select_launch(df(x), dfm(vals), dllm(idxs), n, (float)p, dim_(count), stream);
    }, py::arg("x"), py::arg("vals"), py::arg("idxs"), py::arg("n"), py::arg("p"),
       py::arg("count"), py::arg("stream") = 0);

    m.def("argmax_launch", [](py::int_ x, py::int_ out, int n, std::uintptr_t stream) {
        ft::argmax_launch(df(x), n, dim_(out), stream);
    }, py::arg("x"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    // ==================================================================
    // repetition penalty
    // ==================================================================
    m.def("repetition_penalty_cpu", [](FArray logits, py::sequence ids, double penalty) {
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        if (!(penalty > 0.0))
            throw std::invalid_argument("penalty must be > 0");
        std::vector<long long> token_ids;
        for (auto item : ids) token_ids.push_back(item.cast<long long>());
        return wrap_vec(ft::repetition_penalty_cpu(to_vec(logits), token_ids, (float)penalty),
                        shape_of(logits));
    }, py::arg("logits"), py::arg("token_ids"), py::arg("penalty"));

    m.def("repetition_penalty", [](FArray logits, py::sequence ids, double penalty) {
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        if (!(penalty > 0.0))
            throw std::invalid_argument("penalty must be > 0");
        std::vector<long long> token_ids;
        for (auto item : ids) token_ids.push_back(item.cast<long long>());
        for (long long id : token_ids)
            if (id < 0 || id >= (long long)logits.size())
                throw std::invalid_argument("token id out of range");
        const int n = (int)logits.size();
        const int m = (int)token_ids.size();
        py::array_t<float> y(std::vector<py::ssize_t>{(py::ssize_t)n});
        if (n == 0) return y;
        DevBuf dxl(n * 4), dids((size_t)m * sizeof(long long)), dy(n * 4);
        h2d(dxl.get(), logits.data(), n * 4);
        if (m > 0)
            h2d(dids.get(), token_ids.data(), (size_t)m * sizeof(long long));
        ft::repetition_penalty_launch(dxl.fget(),
                                      static_cast<const long long*>(dids.get()),
                                      n, m, (float)penalty, dy.fget());
        d2h(y.mutable_data(), dy.get(), n * 4);
        sync_device("repetition_penalty kernel");
        return y;
    }, py::arg("logits"), py::arg("token_ids"), py::arg("penalty"));

    m.def("repetition_penalty_launch", [](py::int_ logits, py::int_ ids, py::int_ out,
                                          int n, int m, float penalty, std::uintptr_t stream) {
        ft::repetition_penalty_launch(df(logits), dll(ids), n, m, penalty, dfm(out), stream);
    }, py::arg("logits"), py::arg("token_ids"), py::arg("out"), py::arg("n"),
       py::arg("m"), py::arg("penalty"), py::arg("stream") = 0);

    // ==================================================================
    // fused nucleus sampling: softmax -> nucleus -> inverse-CDF draw
    // ==================================================================
    m.def("sample_topp_cpu", [](FArray logits, double p, double t,
                                unsigned long long seed) -> long long {
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        return ft::sample_topp_cpu(to_vec(logits), (float)p, (float)t, seed);
    }, py::arg("logits"), py::arg("p"), py::arg("t") = 1.0, py::arg("seed") = 0);

    m.def("sample_topk_cpu", [](FArray logits, int k, double t,
                                unsigned long long seed) -> long long {
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        return ft::sample_topk_cpu(to_vec(logits), k, (float)t, seed);
    }, py::arg("logits"), py::arg("k"), py::arg("t") = 1.0, py::arg("seed") = 0);

    m.def("sample_topk", [](FArray logits, int k, double t,
                            unsigned long long seed) -> long long {
        // staged: copy logits up, run, read the token back
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        if (k <= 0)
            throw std::invalid_argument("k must be >= 1");
        if (!(t > 0.0))
            throw std::invalid_argument("temperature must be > 0");
        const int n = (int)logits.size();
        if (n == 0)
            throw std::invalid_argument("sample of empty logits");
        DevBuf dx(n * 4);
        h2d(dx.get(), logits.data(), n * 4);
        const long long token = ft::sample_topk_launch(dx.fget(), n, k,
                                                       (float)t, seed);
        sync_device("sample topk kernel");
        return token;
    }, py::arg("logits"), py::arg("k"), py::arg("t") = 1.0, py::arg("seed") = 0);

    m.def("sample_topk_launch", [](py::int_ x, int n, int k, double t,
                                   unsigned long long seed,
                                   std::uintptr_t stream) -> long long {
        if (n <= 0)
            throw std::invalid_argument("sample of empty logits");
        if (k <= 0)
            throw std::invalid_argument("k must be >= 1");
        if (!(t > 0.0))
            throw std::invalid_argument("temperature must be > 0");
        return ft::sample_topk_launch(reinterpret_cast<const float*>((uintptr_t)x),
                                      n, k, (float)t, seed, stream);
    }, py::arg("x"), py::arg("n"), py::arg("k"), py::arg("t") = 1.0,
       py::arg("seed") = 0, py::arg("stream") = 0);

    m.def("sample_minp_cpu", [](FArray logits, double min_p, double t,
                                unsigned long long seed) -> long long {
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        return ft::sample_minp_cpu(to_vec(logits), (float)min_p, (float)t,
                                   seed);
    }, py::arg("logits"), py::arg("min_p"), py::arg("t") = 1.0,
       py::arg("seed") = 0);

    m.def("sample_minp", [](FArray logits, double min_p, double t,
                            unsigned long long seed) -> long long {
        // staged: copy logits up, run, read the token back
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        if (!(min_p > 0.0 && min_p <= 1.0))
            throw std::invalid_argument("min_p must be in (0, 1]");
        if (!(t > 0.0))
            throw std::invalid_argument("temperature must be > 0");
        const int n = (int)logits.size();
        if (n == 0)
            throw std::invalid_argument("sample of empty logits");
        DevBuf dx(n * 4);
        h2d(dx.get(), logits.data(), n * 4);
        const long long token = ft::sample_minp_launch(dx.fget(), n,
                                                       (float)min_p,
                                                       (float)t, seed);
        sync_device("sample minp kernel");
        return token;
    }, py::arg("logits"), py::arg("min_p"), py::arg("t") = 1.0,
       py::arg("seed") = 0);

    m.def("sample_minp_launch", [](py::int_ x, int n, double min_p,
                                   double t, unsigned long long seed,
                                   std::uintptr_t stream) -> long long {
        if (n <= 0)
            throw std::invalid_argument("sample of empty logits");
        if (!(min_p > 0.0 && min_p <= 1.0))
            throw std::invalid_argument("min_p must be in (0, 1]");
        if (!(t > 0.0))
            throw std::invalid_argument("temperature must be > 0");
        return ft::sample_minp_launch(reinterpret_cast<const float*>((uintptr_t)x),
                                      n, (float)min_p, (float)t, seed,
                                      stream);
    }, py::arg("x"), py::arg("n"), py::arg("min_p"), py::arg("t") = 1.0,
       py::arg("seed") = 0, py::arg("stream") = 0);

    m.def("sample_topp", [](FArray logits, double p, double t,
                            unsigned long long seed) -> long long {
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        if (!(p > 0.0 && p <= 1.0))
            throw std::invalid_argument("p must be in (0, 1]");
        if (!(t > 0.0))
            throw std::invalid_argument("temperature must be > 0");
        const int n = (int)logits.size();
        if (n == 0)
            throw std::invalid_argument("sample of empty logits");
        DevBuf dx(n * 4);
        h2d(dx.get(), logits.data(), n * 4);
        const long long token = ft::sample_topp_launch(dx.fget(), n,
                                                       (float)p, (float)t, seed);
        sync_device("sample kernel");
        return token;
    }, py::arg("logits"), py::arg("p"), py::arg("t") = 1.0, py::arg("seed") = 0);

    // Zero-copy torch path: device pointer in, token out (the one-int
    // readback is inherent to returning a host value).
    m.def("sample_topp_launch", [](py::int_ x, int n, double p, double t,
                                   unsigned long long seed,
                                   std::uintptr_t stream) -> long long {
        if (n <= 0)
            throw std::invalid_argument("sample of empty logits");
        if (!(p > 0.0 && p <= 1.0))
            throw std::invalid_argument("p must be in (0, 1]");
        if (!(t > 0.0))
            throw std::invalid_argument("temperature must be > 0");
        return ft::sample_topp_launch(df(x), n, (float)p, (float)t, seed, stream);
    }, py::arg("logits"), py::arg("n"), py::arg("p"), py::arg("t"),
       py::arg("seed"), py::arg("stream") = 0);

    // ==================================================================
    // batched sampling (v1.4): [rows, n] logits, one seed per row,
    // one token per row. Same three entry styles as the singles.
    // ==================================================================
    // (seeds_vec lives in the anonymous namespace above: the binding
    // lambdas are capture-less)

    m.def("sample_topp_batched_cpu",
          [](FArray logits, int rows, int n, double p, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        check_batch_unit("p", p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        return wrap_ivec(ft::sample_topp_batched_cpu(
            to_vec(logits), rows, n, (float)p, (float)t,
            seeds_vec(seeds)));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("p"),
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("sample_topk_batched_cpu",
          [](FArray logits, int rows, int n, int k, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        if (k <= 0)
            throw std::invalid_argument("k must be >= 1");
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        return wrap_ivec(ft::sample_topk_batched_cpu(
            to_vec(logits), rows, n, k, (float)t, seeds_vec(seeds)));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("k"),
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("sample_minp_batched_cpu",
          [](FArray logits, int rows, int n, double min_p, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        check_batch_unit("min_p", min_p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        return wrap_ivec(ft::sample_minp_batched_cpu(
            to_vec(logits), rows, n, (float)min_p, (float)t,
            seeds_vec(seeds)));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("min_p"),
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("sample_topp_batched",
          [](FArray logits, int rows, int n, double p, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        check_batch_unit("p", p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        if (rows == 0)
            return wrap_ivec({});
        DevBuf dx((size_t)rows * n * 4);
        h2d(dx.get(), logits.data(), (size_t)rows * n * 4);
        const std::vector<long long> tokens = ft::sample_topp_batched_launch(
            dx.fget(), rows, n, (float)p, (float)t, seeds_vec(seeds));
        sync_device("sample topp batched kernel");
        return wrap_ivec(tokens);
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("p"),
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("sample_topk_batched",
          [](FArray logits, int rows, int n, int k, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        if (k <= 0)
            throw std::invalid_argument("k must be >= 1");
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        if (rows == 0)
            return wrap_ivec({});
        DevBuf dx((size_t)rows * n * 4);
        h2d(dx.get(), logits.data(), (size_t)rows * n * 4);
        const std::vector<long long> tokens = ft::sample_topk_batched_launch(
            dx.fget(), rows, n, k, (float)t, seeds_vec(seeds));
        sync_device("sample topk batched kernel");
        return wrap_ivec(tokens);
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("k"),
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("sample_minp_batched",
          [](FArray logits, int rows, int n, double min_p, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        check_batch_unit("min_p", min_p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        if (rows == 0)
            return wrap_ivec({});
        DevBuf dx((size_t)rows * n * 4);
        h2d(dx.get(), logits.data(), (size_t)rows * n * 4);
        const std::vector<long long> tokens = ft::sample_minp_batched_launch(
            dx.fget(), rows, n, (float)min_p, (float)t, seeds_vec(seeds));
        sync_device("sample minp batched kernel");
        return wrap_ivec(tokens);
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("min_p"),
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("sample_topp_batched_launch",
          [](py::int_ x, int rows, int n, double p, double t,
             const I64Array& seeds,
             std::uintptr_t stream) -> py::array_t<long long> {
        check_batch_rows_n(rows, n);
        check_batch_unit("p", p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        return wrap_ivec(ft::sample_topp_batched_launch(
            df(x), rows, n, (float)p, (float)t, seeds_vec(seeds), stream));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("p"),
       py::arg("t") = 1.0, py::arg("seeds"), py::arg("stream") = 0);

    m.def("sample_topk_batched_launch",
          [](py::int_ x, int rows, int n, int k, double t,
             const I64Array& seeds,
             std::uintptr_t stream) -> py::array_t<long long> {
        check_batch_rows_n(rows, n);
        if (k <= 0)
            throw std::invalid_argument("k must be >= 1");
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        return wrap_ivec(ft::sample_topk_batched_launch(
            df(x), rows, n, k, (float)t, seeds_vec(seeds), stream));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("k"),
       py::arg("t") = 1.0, py::arg("seeds"), py::arg("stream") = 0);

    m.def("sample_minp_batched_launch",
          [](py::int_ x, int rows, int n, double min_p, double t,
             const I64Array& seeds,
             std::uintptr_t stream) -> py::array_t<long long> {
        check_batch_rows_n(rows, n);
        check_batch_unit("min_p", min_p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        return wrap_ivec(ft::sample_minp_batched_launch(
            df(x), rows, n, (float)min_p, (float)t, seeds_vec(seeds),
            stream));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("min_p"),
       py::arg("t") = 1.0, py::arg("seeds"), py::arg("stream") = 0);

    // ==================================================================
    // batched fused decode step (v1.5): [rows, n] logits, ragged
    // per-row histories (flat ids + rows + 1 offsets), one seed per
    // row, one token per row. Same three entry styles as the other
    // batched samplers; ids/offsets are host arrays on every path
    // (they ride a small per-attempt upload), so their values are
    // always validated - only the logits go zero-copy.
    // ==================================================================

    m.def("decode_step_batched_cpu",
          [](FArray logits, int rows, int n, const I64Array& ids,
             const I64Array& offs, double penalty, double p, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        check_batch_ids(ids, offs, rows, n);
        if (!(penalty > 0.0))
            throw std::invalid_argument("penalty must be > 0");
        check_batch_unit("p", p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        const long long* ip = ids.data();
        const long long* op = offs.data();
        return wrap_ivec(ft::decode_step_batched_cpu(
            to_vec(logits), rows, n,
            std::vector<long long>(ip, ip + ids.size()),
            std::vector<long long>(op, op + offs.size()),
            (float)penalty, (float)p, (float)t, seeds_vec(seeds)));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("ids"),
       py::arg("offs"), py::arg("penalty") = 1.0, py::arg("p") = 0.9,
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("decode_step_batched",
          [](FArray logits, int rows, int n, const I64Array& ids,
             const I64Array& offs, double penalty, double p, double t,
             const I64Array& seeds) -> py::array_t<long long> {
        check_batch_host(logits, rows, n);
        check_batch_ids(ids, offs, rows, n);
        if (!(penalty > 0.0))
            throw std::invalid_argument("penalty must be > 0");
        check_batch_unit("p", p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        if (rows == 0)
            return wrap_ivec({});
        DevBuf dx((size_t)rows * n * 4);
        h2d(dx.get(), logits.data(), (size_t)rows * n * 4);
        const long long* ip = ids.data();
        const long long* op = offs.data();
        const std::vector<long long> tokens =
            ft::decode_step_batched_launch(
                dx.fget(), rows, n,
                std::vector<long long>(ip, ip + ids.size()),
                std::vector<long long>(op, op + offs.size()),
                (float)penalty, (float)p, (float)t, seeds_vec(seeds));
        sync_device("decode step batched kernel");
        return wrap_ivec(tokens);
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("ids"),
       py::arg("offs"), py::arg("penalty") = 1.0, py::arg("p") = 0.9,
       py::arg("t") = 1.0, py::arg("seeds"));

    m.def("decode_step_batched_launch",
          [](py::int_ x, int rows, int n, const I64Array& ids,
             const I64Array& offs, double penalty, double p, double t,
             const I64Array& seeds,
             std::uintptr_t stream) -> py::array_t<long long> {
        check_batch_rows_n(rows, n);
        check_batch_ids(ids, offs, rows, n);
        if (!(penalty > 0.0))
            throw std::invalid_argument("penalty must be > 0");
        check_batch_unit("p", p);
        check_batch_temp(t);
        check_batch_seeds(seeds, rows);
        const long long* ip = ids.data();
        const long long* op = offs.data();
        return wrap_ivec(ft::decode_step_batched_launch(
            df(x), rows, n,
            std::vector<long long>(ip, ip + ids.size()),
            std::vector<long long>(op, op + offs.size()),
            (float)penalty, (float)p, (float)t, seeds_vec(seeds),
            stream));
    }, py::arg("logits"), py::arg("rows"), py::arg("n"), py::arg("ids"),
       py::arg("offs"), py::arg("penalty") = 1.0, py::arg("p") = 0.9,
       py::arg("t") = 1.0, py::arg("seeds"), py::arg("stream") = 0);


    // ==================================================================
    // INT8 symmetric per-tensor quantization
    // ==================================================================
    m.def("quantize_int8_cpu", [](FArray x) {
        auto [q, scale] = ft::quantize_int8_cpu(to_vec(x));
        py::array_t<signed char> qout(std::vector<py::ssize_t>{(py::ssize_t)q.size()});
        if (!q.empty())
            std::memcpy(qout.mutable_data(), q.data(), q.size());
        return py::make_tuple(qout, scale);
    }, py::arg("x"));

    m.def("dequantize_int8_cpu", [](py::array_t<signed char, py::array::c_style> q,
                                    float scale) {
        auto info = q.request();
        const signed char* p = static_cast<const signed char*>(info.ptr);
        std::vector<float> x = ft::dequantize_int8_cpu(
            std::vector<signed char>(p, p + q.size()), scale);
        return wrap_vec(x, shape_of(q));
    }, py::arg("q"), py::arg("scale"));

    m.def("quantize_launch",
          [](py::int_ x, py::int_ q, py::int_ scale, long long n, std::uintptr_t stream) {
        ft::quantize_int8_launch(df(x),
                                 reinterpret_cast<signed char*>((uintptr_t)q),
                                 dfm(scale), n, stream);
    }, py::arg("x"), py::arg("q"), py::arg("scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("dequantize_launch",
          [](py::int_ q, py::int_ x, float scale, long long n, std::uintptr_t stream) {
        ft::dequantize_int8_launch(
            reinterpret_cast<const signed char*>((uintptr_t)q),
            dfm(x), scale, n, stream);
    }, py::arg("q"), py::arg("x"), py::arg("scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("qadd_launch", [](py::int_ qa, py::int_ qb, float sa, float sb,
                            py::int_ qy, py::int_ out_scale, long long n, std::uintptr_t stream) {
        ft::qadd_int8_launch(
            reinterpret_cast<const signed char*>((uintptr_t)qa),
            reinterpret_cast<const signed char*>((uintptr_t)qb),
            sa, sb,
            reinterpret_cast<signed char*>((uintptr_t)qy),
            dfm(out_scale), n, stream);
    }, py::arg("qa"), py::arg("qb"), py::arg("sa"), py::arg("sb"),
       py::arg("qy"), py::arg("out_scale"), py::arg("n"), py::arg("stream") = 0);

    // ==================================================================
    // INT8 matmul
    // ==================================================================
    m.def("qgemm_cpu", [](py::array_t<signed char, py::array::c_style> a,
                          py::array_t<signed char, py::array::c_style> b,
                          int m, int n, int k, float sa, float sb) {
        if (a.size() != (py::ssize_t)m * k || b.size() != (py::ssize_t)n * k)
            throw std::invalid_argument("qgemm operand size mismatch");
        auto ia = a.request(), ib = b.request();
        const signed char* pa = static_cast<const signed char*>(ia.ptr);
        const signed char* pb = static_cast<const signed char*>(ib.ptr);
        std::vector<float> y = ft::qgemm_cpu(
            std::vector<signed char>(pa, pa + a.size()),
            std::vector<signed char>(pb, pb + b.size()),
            m, n, k, sa, sb);
        py::array_t<float> out(std::vector<py::ssize_t>{m, n});
        if (!y.empty())
            std::memcpy(out.mutable_data(), y.data(), y.size() * 4);
        return out;
    }, py::arg("a"), py::arg("b"), py::arg("m"), py::arg("n"), py::arg("k"),
       py::arg("sa"), py::arg("sb"));

    // staged qgemm driver shared by the per-tensor and per-channel
    // flavors: identical validation, uploads and readback; only the
    // launcher differs (the per-channel wrapper uploads its own scale
    // vector alongside)
    auto staged_gemm = [&](const auto& a, const auto& b, int m, int n,
                           int k, const char* sync_what, auto&& launch) {
        if (a.size() != (py::ssize_t)m * k || b.size() != (py::ssize_t)n * k)
            throw std::invalid_argument("qgemm operand size mismatch");
        if ((long long)m * n * k > (1LL << 38))
            throw std::invalid_argument("qgemm operands too large");
        DevBuf da(a.size()), db(b.size());
        h2d(da.get(), a.data(), a.size());
        h2d(db.get(), b.data(), b.size());
        DevBuf dy((size_t)m * n * 4);
        // the launcher no-ops empty operands and zero-fills K == 0
        if (m > 0 && n > 0)
            launch(reinterpret_cast<const signed char*>(da.fget()),
                   reinterpret_cast<const signed char*>(db.fget()),
                   dy.fget());
        py::array_t<float> out(std::vector<py::ssize_t>{m, n});
        if ((long long)m * n > 0)
            d2h(out.mutable_data(), dy.get(), (size_t)m * n * 4);
        sync_device(sync_what);
        return out;
    };

    m.def("qgemm", [&](py::array_t<signed char, py::array::c_style> a,
                       py::array_t<signed char, py::array::c_style> b,
                       int m, int n, int k, float sa, float sb) {
        return staged_gemm(
            a, b, m, n, k, "qgemm kernel",
            [&](const signed char* da, const signed char* db, float* dy) {
                ft::qgemm_launch(da, db, dy, m, n, k, sa, sb, 0);
            });
    }, py::arg("a"), py::arg("b"), py::arg("m"), py::arg("n"), py::arg("k"),
       py::arg("sa"), py::arg("sb"));

    m.def("qgemm_launch", [](py::int_ a, py::int_ b, py::int_ y,
                             int m, int n, int k, float sa, float sb,
                             std::uintptr_t stream) {
        ft::qgemm_launch(
            reinterpret_cast<const signed char*>((uintptr_t)a),
            reinterpret_cast<const signed char*>((uintptr_t)b),
            dfm(y), m, n, k, sa, sb, stream);
    }, py::arg("a"), py::arg("b"), py::arg("y"), py::arg("m"), py::arg("n"),
       py::arg("k"), py::arg("sa"), py::arg("sb"), py::arg("stream") = 0);

    // per-channel weight scales (W8A8): sb has n entries, one per output
    // row of B_q; y[i,j] = (A_q . B_q^T) * f32(sa * sb[j])
    m.def("qgemm_perchannel_cpu", [](py::array_t<signed char, py::array::c_style> a,
                                     py::array_t<signed char, py::array::c_style> b,
                                     FArray sb, int m, int n, int k, float sa) {
        if (a.size() != (py::ssize_t)m * k || b.size() != (py::ssize_t)n * k)
            throw std::invalid_argument("qgemm operand size mismatch");
        if (sb.size() != n)
            throw std::invalid_argument("per-channel scale vector must have n entries");
        auto ia = a.request(), ib = b.request(), is = sb.request();
        std::vector<float> y = ft::qgemm_perchannel_cpu(
            std::vector<signed char>(static_cast<const signed char*>(ia.ptr),
                                     static_cast<const signed char*>(ia.ptr) + a.size()),
            std::vector<signed char>(static_cast<const signed char*>(ib.ptr),
                                     static_cast<const signed char*>(ib.ptr) + b.size()),
            std::vector<float>(static_cast<const float*>(is.ptr),
                               static_cast<const float*>(is.ptr) + sb.size()),
            m, n, k, sa);
        py::array_t<float> out(std::vector<py::ssize_t>{m, n});
        if (!y.empty())
            std::memcpy(out.mutable_data(), y.data(), y.size() * 4);
        return out;
    }, py::arg("a"), py::arg("b"), py::arg("sb"), py::arg("m"), py::arg("n"),
       py::arg("k"), py::arg("sa"));

    m.def("qgemm_perchannel", [&](py::array_t<signed char, py::array::c_style> a,
                                  py::array_t<signed char, py::array::c_style> b,
                                  FArray sb, int m, int n, int k, float sa) {
        if (sb.size() != n)
            throw std::invalid_argument("per-channel scale vector must have n entries");
        DevBuf dsb((size_t)n * 4);          // outlives the launch below
        h2d(dsb.get(), sb.data(), (size_t)n * 4);
        return staged_gemm(
            a, b, m, n, k, "qgemm per-channel kernel",
            [&](const signed char* da, const signed char* db, float* dy) {
                ft::qgemm_perchannel_launch(
                    da, db, reinterpret_cast<const float*>(dsb.fget()),
                    dy, m, n, k, sa, 0);
            });
    }, py::arg("a"), py::arg("b"), py::arg("sb"), py::arg("m"), py::arg("n"),
       py::arg("k"), py::arg("sa"));

    m.def("qgemm_perchannel_launch", [](py::int_ a, py::int_ b, py::int_ sb,
                                        py::int_ y, int m, int n, int k,
                                        float sa, std::uintptr_t stream) {
        ft::qgemm_perchannel_launch(
            reinterpret_cast<const signed char*>((uintptr_t)a),
            reinterpret_cast<const signed char*>((uintptr_t)b),
            reinterpret_cast<const float*>((uintptr_t)sb),
            dfm(y), m, n, k, sa, stream);
    }, py::arg("a"), py::arg("b"), py::arg("sb"), py::arg("y"), py::arg("m"),
       py::arg("n"), py::arg("k"), py::arg("sa"), py::arg("stream") = 0);

    // staged decode step: numpy logits + python ids in, token out
    m.def("decode_step", [](FArray logits, py::array_t<long long, py::array::c_style> ids,
                            double penalty, double p, double t,
                            unsigned long long seed) -> long long {
        if (logits.ndim() != 1)
            throw std::invalid_argument("logits must be 1-D");
        const int n = (int)logits.size();
        const int m = (int)ids.size();
        if (n == 0)
            throw std::invalid_argument("decode_step of empty logits");
        auto ii = ids.request();
        const long long* ip = static_cast<const long long*>(ii.ptr);
        for (int j = 0; j < m; ++j)
            if (ip[j] < 0 || ip[j] >= n)
                throw std::invalid_argument("token id out of range");
        DevBuf dx(n * 4), di((size_t)m * 8);
        h2d(dx.get(), logits.data(), n * 4);
        if (m > 0)
            h2d(di.get(), ip, (size_t)m * 8);
        const long long token = ft::decode_step_launch(
            dx.fget(), reinterpret_cast<const long long*>(di.fget()),
            n, m, (float)penalty, (float)p, (float)t, seed, 0);
        sync_device("decode step kernel");
        return token;
    }, py::arg("logits"), py::arg("sampled_ids"), py::arg("penalty") = 1.0,
       py::arg("p") = 0.9, py::arg("t") = 1.0, py::arg("seed") = 0);

    m.def("decode_step_launch", [](py::int_ x, py::int_ ids, int n, int m,
                                   double penalty, double p, double t,
                                   unsigned long long seed,
                                   std::uintptr_t stream) -> long long {
        return ft::decode_step_launch(
            df(x), reinterpret_cast<const long long*>((uintptr_t)ids),
            n, m, (float)penalty, (float)p, (float)t, seed, stream);
    }, py::arg("x"), py::arg("ids"), py::arg("n"), py::arg("m"),
       py::arg("penalty") = 1.0, py::arg("p") = 0.9, py::arg("t") = 1.0,
       py::arg("seed") = 0, py::arg("stream") = 0);

    // ==================================================================
    // attention (decode step): GQA over a contiguous kv-cache
    // ==================================================================
    using IArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

    m.def("attention_decode_cpu", [](FArray q, FArray k, FArray v,
                                     py::object lens, int batch, int hq,
                                     int hkv, int t_seq, int dim) {
        if (q.size() != (py::ssize_t)batch * hq * dim ||
            k.size() != (py::ssize_t)batch * hkv * t_seq * dim ||
            v.size() != (py::ssize_t)batch * hkv * t_seq * dim)
            throw std::invalid_argument("attention operand size mismatch");
        std::vector<int> lv;
        const std::vector<int>* lp = nullptr;
        if (!lens.is_none()) {
            IArray l_arr = lens;
            if (l_arr.size() != (py::ssize_t)batch)
                throw std::invalid_argument("lens must have batch entries");
            lv.assign(l_arr.data(), l_arr.data() + l_arr.size());
            lp = &lv;
        }
        auto out = ft::attention_decode_cpu(to_vec(q), to_vec(k), to_vec(v),
                                            lp, batch, hq, hkv, t_seq, dim);
        return wrap_vec(out, {(py::ssize_t)batch, (py::ssize_t)hq,
                              (py::ssize_t)dim});
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("lens"),
       py::arg("batch"), py::arg("hq"), py::arg("hkv"), py::arg("t_seq"),
       py::arg("dim"));

    m.def("attention_decode", [](FArray q, FArray k, FArray v,
                                 py::object lens, int batch, int hq,
                                 int hkv, int t_seq, int dim) {
        if (q.size() != (py::ssize_t)batch * hq * dim ||
            k.size() != (py::ssize_t)batch * hkv * t_seq * dim ||
            v.size() != (py::ssize_t)batch * hkv * t_seq * dim)
            throw std::invalid_argument("attention operand size mismatch");
        const bool has_lens = !lens.is_none();
        IArray l_arr;
        if (has_lens) {
            l_arr = lens;
            if (l_arr.size() != (py::ssize_t)batch)
                throw std::invalid_argument("lens must have batch entries");
        }
        const size_t kv_bytes = (size_t)batch * hkv * t_seq * dim * 4;
        const size_t out_n = (size_t)batch * hq * dim;
        py::array_t<float> out(
            std::vector<py::ssize_t>{batch, hq, dim});
        if (out_n == 0) return out;
        DevBuf dq(q.size() * 4), dk(kv_bytes), dv(kv_bytes),
              dl(has_lens ? (size_t)batch * 4 : 0), dy(out_n * 4);
        h2d(dq.get(), q.data(), q.size() * 4);
        h2d(dk.get(), k.data(), kv_bytes);
        h2d(dv.get(), v.data(), kv_bytes);
        if (has_lens) h2d(dl.get(), l_arr.data(), (size_t)batch * 4);
        ft::attention_decode_launch(
            dq.fget(), dk.fget(), dv.fget(),
            has_lens ? static_cast<const int*>(dl.get()) : nullptr,
            dy.fget(), batch, hq, hkv, t_seq, dim);
        d2h(out.mutable_data(), dy.get(), out_n * 4);
        sync_device("attention decode kernel");
        return out;
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("lens"),
       py::arg("batch"), py::arg("hq"), py::arg("hkv"), py::arg("t_seq"),
       py::arg("dim"));

    m.def("attention_decode_launch", [](py::int_ q, py::int_ k, py::int_ v,
                                        py::object lens, py::int_ out,
                                        int batch, int hq, int hkv,
                                        int t_seq, int dim,
                                        std::uintptr_t stream) {
        const int* lp = opt_int_ptr(lens);
        ft::attention_decode_launch(df(q), df(k), df(v), lp, dfm(out),
                                    batch, hq, hkv, t_seq, dim, stream);
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("lens"),
       py::arg("out"), py::arg("batch"), py::arg("hq"), py::arg("hkv"),
       py::arg("t_seq"), py::arg("dim"), py::arg("stream") = 0);

    // bfloat16 / float16 zero-copy variants: q/k/v/out are half-width
    // device buffers (out matches the input dtype); lens stays int32
    m.def("attention_decode_launch_bf16",
          [](py::int_ q, py::int_ k, py::int_ v, py::object lens,
             py::int_ out, int batch, int hq, int hkv, int t_seq, int dim,
             std::uintptr_t stream) {
        const int* lp = opt_int_ptr(lens);
        ft::attention_decode_launch_bf16(
            dvoid(q),
            dvoid(k),
            dvoid(v), lp,
            dvoidm(out),
            batch, hq, hkv, t_seq, dim, stream);
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("lens"),
       py::arg("out"), py::arg("batch"), py::arg("hq"), py::arg("hkv"),
       py::arg("t_seq"), py::arg("dim"), py::arg("stream") = 0);

    m.def("attention_decode_launch_fp16",
          [](py::int_ q, py::int_ k, py::int_ v, py::object lens,
             py::int_ out, int batch, int hq, int hkv, int t_seq, int dim,
             std::uintptr_t stream) {
        const int* lp = opt_int_ptr(lens);
        ft::attention_decode_launch_fp16(
            dvoid(q),
            dvoid(k),
            dvoid(v), lp,
            dvoidm(out),
            batch, hq, hkv, t_seq, dim, stream);
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("lens"),
       py::arg("out"), py::arg("batch"), py::arg("hq"), py::arg("hkv"),
       py::arg("t_seq"), py::arg("dim"), py::arg("stream") = 0);

    // --- paged kv-cache decode (v1.2): pools [Nb,Hkv,P,D] + block table ---
    m.def("attention_decode_paged_cpu",
          [](FArray q, FArray k_pool, FArray v_pool, IArray table,
             py::object lens, int batch, int hq, int hkv, int page,
             int tbl_width, int num_blocks, int dim) {
        std::vector<int> lv;
        const std::vector<int>* lp = nullptr;
        if (!lens.is_none()) {
            IArray l_arr = lens;
            if (l_arr.size() != (py::ssize_t)batch)
                throw std::invalid_argument("lens must have batch entries");
            lv.assign(l_arr.data(), l_arr.data() + l_arr.size());
            lp = &lv;
        }
        std::vector<int> tv(table.data(), table.data() + table.size());
        auto out = ft::attention_decode_paged_cpu(
            to_vec(q), to_vec(k_pool), to_vec(v_pool), tv, lp, batch, hq,
            hkv, page, tbl_width, num_blocks, dim);
        return wrap_vec(out, {(py::ssize_t)batch, (py::ssize_t)hq,
                              (py::ssize_t)dim});
    }, py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("table"), py::arg("lens"), py::arg("batch"), py::arg("hq"),
       py::arg("hkv"), py::arg("page"), py::arg("tbl_width"),
       py::arg("num_blocks"), py::arg("dim"));

    m.def("attention_decode_paged",
          [](FArray q, FArray k_pool, FArray v_pool, IArray table,
             py::object lens, int batch, int hq, int hkv, int page,
             int tbl_width, int num_blocks, int dim) {
        if (q.size() != (py::ssize_t)batch * hq * dim ||
            k_pool.size() !=
                (py::ssize_t)num_blocks * hkv * page * dim ||
            v_pool.size() != k_pool.size() ||
            table.size() != (py::ssize_t)batch * tbl_width)
            throw std::invalid_argument("attention operand size mismatch");
        const bool has_lens = !lens.is_none();
        IArray l_arr;
        if (has_lens) {
            l_arr = lens;
            if (l_arr.size() != (py::ssize_t)batch)
                throw std::invalid_argument("lens must have batch entries");
        }
        // host-side table/lens validation (staged inputs are host
        // memory; the zero-copy launch path trusts device values)
        if (page < 1)
            throw std::invalid_argument("page size must be >= 1");
        const int span = tbl_width * page;
        for (py::ssize_t i = 0; i < table.size(); ++i) {
            const int blk = table.data()[i];
            if (blk < 0 || blk >= num_blocks)
                throw std::invalid_argument(
                    "block table entries must be valid pool block ids");
        }
        if (has_lens)
            for (py::ssize_t i = 0; i < l_arr.size(); ++i)
                if (l_arr.data()[i] < 0 || l_arr.data()[i] > span)
                    throw std::invalid_argument(
                        "lens entries must be within [0, table width * page]");
        const size_t kv_bytes =
            (size_t)num_blocks * hkv * page * dim * 4;
        const size_t out_n = (size_t)batch * hq * dim;
        py::array_t<float> out(
            std::vector<py::ssize_t>{batch, hq, dim});
        if (out_n == 0) return out;
        DevBuf dq(q.size() * 4), dk(kv_bytes), dv(kv_bytes),
              dt((size_t)batch * tbl_width * 4),
              dl(has_lens ? (size_t)batch * 4 : 0), dy(out_n * 4);
        h2d(dq.get(), q.data(), q.size() * 4);
        h2d(dk.get(), k_pool.data(), kv_bytes);
        h2d(dv.get(), v_pool.data(), kv_bytes);
        h2d(dt.get(), table.data(), (size_t)batch * tbl_width * 4);
        if (has_lens) h2d(dl.get(), l_arr.data(), (size_t)batch * 4);
        ft::attention_decode_paged_launch(
            dq.fget(), dk.fget(), dv.fget(),
            static_cast<const int*>(dt.get()),
            has_lens ? static_cast<const int*>(dl.get()) : nullptr,
            dy.fget(), batch, hq, hkv, page, tbl_width, dim);
        d2h(out.mutable_data(), dy.get(), out_n * 4);
        sync_device("attention decode paged kernel");
        return out;
    }, py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("table"), py::arg("lens"), py::arg("batch"), py::arg("hq"),
       py::arg("hkv"), py::arg("page"), py::arg("tbl_width"),
       py::arg("num_blocks"), py::arg("dim"));

    m.def("attention_decode_paged_launch",
          [](py::int_ q, py::int_ k_pool, py::int_ v_pool, py::int_ table,
             py::object lens, py::int_ out, int batch, int hq, int hkv,
             int page, int tbl_width, int dim, std::uintptr_t stream) {
         const int* lp = opt_int_ptr(lens);
         ft::attention_decode_paged_launch(
             df(q), df(k_pool), df(v_pool),
             dic(table), lp, dfm(out),
             batch, hq, hkv, page, tbl_width, dim, stream);
    }, py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("table"), py::arg("lens"), py::arg("out"), py::arg("batch"),
       py::arg("hq"), py::arg("hkv"), py::arg("page"),
       py::arg("tbl_width"), py::arg("dim"), py::arg("stream") = 0);

    m.def("attention_decode_paged_launch_bf16",
          [](py::int_ q, py::int_ k_pool, py::int_ v_pool, py::int_ table,
             py::object lens, py::int_ out, int batch, int hq, int hkv,
             int page, int tbl_width, int dim, std::uintptr_t stream) {
        const int* lp = opt_int_ptr(lens);
         ft::attention_decode_paged_launch_bf16(
            dvoid(q),
            dvoid(k_pool),
            dvoid(v_pool),
            dic(table), lp,
            dvoidm(out),
            batch, hq, hkv, page, tbl_width, dim, stream);
    }, py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("table"), py::arg("lens"), py::arg("out"), py::arg("batch"),
       py::arg("hq"), py::arg("hkv"), py::arg("page"),
       py::arg("tbl_width"), py::arg("dim"), py::arg("stream") = 0);

    m.def("attention_decode_paged_launch_fp16",
          [](py::int_ q, py::int_ k_pool, py::int_ v_pool, py::int_ table,
             py::object lens, py::int_ out, int batch, int hq, int hkv,
             int page, int tbl_width, int dim, std::uintptr_t stream) {
        const int* lp = opt_int_ptr(lens);
        ft::attention_decode_paged_launch_fp16(
            dvoid(q),
            dvoid(k_pool),
            dvoid(v_pool),
            dic(table), lp,
            dvoidm(out),
            batch, hq, hkv, page, tbl_width, dim, stream);
    }, py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("table"), py::arg("lens"), py::arg("out"), py::arg("batch"),
       py::arg("hq"), py::arg("hkv"), py::arg("page"),
       py::arg("tbl_width"), py::arg("dim"), py::arg("stream") = 0);

    // --- paged kv-cache append (v1.2): in-place scatter of one token ---
    m.def("kv_append_paged_cpu",
          [](FArray k_new, FArray v_new, IArray table, IArray lens,
             FArray k_pool, FArray v_pool, int batch, int hkv, int dim,
             int page, int tbl_width, int num_blocks) {
        std::vector<int> tv(table.data(), table.data() + table.size());
        std::vector<int> lv(lens.data(), lens.data() + lens.size());
        // copy-in so the in-place host mutation stays out of numpy's view
        // until the wrapper writes back
        std::vector<float> kp(k_pool.data(), k_pool.data() + k_pool.size());
        std::vector<float> vp(v_pool.data(), v_pool.data() + v_pool.size());
        ft::kv_append_paged_cpu(to_vec(k_new), to_vec(v_new), tv, lv, kp,
                                vp, batch, hkv, dim, page, tbl_width,
                                num_blocks);
        std::copy(kp.begin(), kp.end(), k_pool.mutable_data());
        std::copy(vp.begin(), vp.end(), v_pool.mutable_data());
    }, py::arg("k_new"), py::arg("v_new"), py::arg("table"),
       py::arg("lens"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("batch"), py::arg("hkv"), py::arg("dim"), py::arg("page"),
       py::arg("tbl_width"), py::arg("num_blocks"));

    m.def("kv_append_paged",
          [](FArray k_new, FArray v_new, IArray table, IArray lens,
             FArray k_pool, FArray v_pool, int batch, int hkv, int dim,
             int page, int tbl_width, int num_blocks) {
        if (k_new.size() != (py::ssize_t)batch * hkv * dim ||
            v_new.size() != k_new.size() ||
            table.size() != (py::ssize_t)batch * tbl_width ||
            lens.size() != (py::ssize_t)batch ||
            k_pool.size() != (py::ssize_t)num_blocks * hkv * page * dim ||
            v_pool.size() != k_pool.size())
            throw std::invalid_argument("kv append operand size mismatch");
        if (page < 1)
            throw std::invalid_argument("page size must be >= 1");
        // host-side lens/table validation, matching the CPU reference
        // exactly: each row's write lands at table[bi*tw + pos/page],
        // so a bad pos or a bad USED block id is a persistent OOB
        // device write (the in-place scatter makes it stick). The
        // zero-copy launch path trusts device values, same boundary
        // as the attention twins. Unused table slots are not checked
        // - the CPU reference accepts them, and rejecting them here
        // would make the staged path stricter than the CPU path.
        for (py::ssize_t bi = 0; bi < batch; ++bi) {
            const int pos = (int)lens.data()[bi];
            if (pos < 0 || pos >= tbl_width * page)
                throw std::invalid_argument(
                    "lens entries must be within [0, table width * page)");
            const int blk = table.data()[bi * tbl_width + pos / page];
            if (blk < 0 || blk >= num_blocks)
                throw std::invalid_argument(
                    "block table entries must be valid pool block ids");
        }
        const size_t kv_bytes =
            (size_t)num_blocks * hkv * page * dim * 4;
        DevBuf dk_new(k_new.size() * 4), dv_new(v_new.size() * 4),
              dt((size_t)batch * tbl_width * 4),
              dl((size_t)batch * 4),
              dkp(kv_bytes), dvp(kv_bytes);
        h2d(dk_new.get(), k_new.data(), k_new.size() * 4);
        h2d(dv_new.get(), v_new.data(), v_new.size() * 4);
        h2d(dt.get(), table.data(), (size_t)batch * tbl_width * 4);
        h2d(dl.get(), lens.data(), (size_t)batch * 4);
        h2d(dkp.get(), k_pool.data(), kv_bytes);
        h2d(dvp.get(), v_pool.data(), kv_bytes);
        ft::kv_append_paged_launch(
            dk_new.fget(), dv_new.fget(),
            static_cast<const int*>(dt.get()),
            static_cast<const int*>(dl.get()), dkp.fget(), dvp.fget(),
            batch, hkv, dim, page, tbl_width);
        d2h(k_pool.mutable_data(), dkp.get(), kv_bytes);
        d2h(v_pool.mutable_data(), dvp.get(), kv_bytes);
        sync_device("kv append paged kernel");
    }, py::arg("k_new"), py::arg("v_new"), py::arg("table"),
       py::arg("lens"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("batch"), py::arg("hkv"), py::arg("dim"), py::arg("page"),
       py::arg("tbl_width"), py::arg("num_blocks"));

    m.def("kv_append_paged_launch",
          [](py::int_ k_new, py::int_ v_new, py::int_ table,
             py::int_ lens, py::int_ k_pool, py::int_ v_pool, int batch,
             int hkv, int dim, int page, int tbl_width,
             std::uintptr_t stream) {
        ft::kv_append_paged_launch(
            df(k_new), df(v_new),
            dic(table),
            dic(lens), dfm(k_pool),
            dfm(v_pool), batch, hkv, dim, page, tbl_width, stream);
    }, py::arg("k_new"), py::arg("v_new"), py::arg("table"),
       py::arg("lens"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("batch"), py::arg("hkv"), py::arg("dim"), py::arg("page"),
       py::arg("tbl_width"), py::arg("stream") = 0);

    m.def("kv_append_paged_launch_bf16",
          [](py::int_ k_new, py::int_ v_new, py::int_ table,
             py::int_ lens, py::int_ k_pool, py::int_ v_pool, int batch,
             int hkv, int dim, int page, int tbl_width,
             std::uintptr_t stream) {
        ft::kv_append_paged_launch_bf16(
            dvoid(k_new),
            dvoid(v_new),
            dic(table),
            dic(lens),
            dvoidm(k_pool),
            dvoidm(v_pool),
            batch, hkv, dim, page, tbl_width, stream);
    }, py::arg("k_new"), py::arg("v_new"), py::arg("table"),
       py::arg("lens"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("batch"), py::arg("hkv"), py::arg("dim"), py::arg("page"),
       py::arg("tbl_width"), py::arg("stream") = 0);

    m.def("kv_append_paged_launch_fp16",
          [](py::int_ k_new, py::int_ v_new, py::int_ table,
             py::int_ lens, py::int_ k_pool, py::int_ v_pool, int batch,
             int hkv, int dim, int page, int tbl_width,
             std::uintptr_t stream) {
        ft::kv_append_paged_launch_fp16(
            dvoid(k_new),
            dvoid(v_new),
            dic(table),
            dic(lens),
            dvoidm(k_pool),
            dvoidm(v_pool),
            batch, hkv, dim, page, tbl_width, stream);
    }, py::arg("k_new"), py::arg("v_new"), py::arg("table"),
       py::arg("lens"), py::arg("k_pool"), py::arg("v_pool"),
       py::arg("batch"), py::arg("hkv"), py::arg("dim"), py::arg("page"),
       py::arg("tbl_width"), py::arg("stream") = 0);

    // --- contiguous kv-cache append (v1.3): in-place scatter of one token ---
    m.def("kv_append_cpu",
          [](FArray k_new, FArray v_new, IArray lens, FArray k_cache,
             FArray v_cache, int batch, int hkv, int dim, int t_rows) {
        std::vector<int> lv(lens.data(), lens.data() + lens.size());
        // copy-in so the in-place host mutation stays out of numpy's view
        // until the wrapper writes back
        std::vector<float> kc(k_cache.data(), k_cache.data() + k_cache.size());
        std::vector<float> vc(v_cache.data(), v_cache.data() + v_cache.size());
        ft::kv_append_cpu(to_vec(k_new), to_vec(v_new), lv, kc, vc, batch,
                          hkv, dim, t_rows);
        std::copy(kc.begin(), kc.end(), k_cache.mutable_data());
        std::copy(vc.begin(), vc.end(), v_cache.mutable_data());
    }, py::arg("k_new"), py::arg("v_new"), py::arg("lens"),
       py::arg("k_cache"), py::arg("v_cache"), py::arg("batch"),
       py::arg("hkv"), py::arg("dim"), py::arg("t_rows"));

    m.def("kv_append",
          [](FArray k_new, FArray v_new, IArray lens, FArray k_cache,
             FArray v_cache, int batch, int hkv, int dim, int t_rows) {
        if (k_new.size() != (py::ssize_t)batch * hkv * dim ||
            v_new.size() != k_new.size() ||
            lens.size() != (py::ssize_t)batch ||
            k_cache.size() != (py::ssize_t)batch * hkv * t_rows * dim ||
            v_cache.size() != k_cache.size())
            throw std::invalid_argument("kv append operand size mismatch");
        // host-side lens validation, matching the CPU reference: the
        // in-place scatter writes at row lens[bi], so a bad value is a
        // persistent OOB device write (zero-copy trusts device values)
        for (py::ssize_t bi = 0; bi < batch; ++bi) {
            const int pos = (int)lens.data()[bi];
            if (pos < 0 || pos >= t_rows)
                throw std::invalid_argument(
                    "lens entries must be within [0, cache rows)");
        }
        const size_t cache_bytes = (size_t)batch * hkv * t_rows * dim * 4;
        DevBuf dk_new(k_new.size() * 4), dv_new(v_new.size() * 4),
              dl((size_t)batch * 4), dkc(cache_bytes), dvc(cache_bytes);
        h2d(dk_new.get(), k_new.data(), k_new.size() * 4);
        h2d(dv_new.get(), v_new.data(), v_new.size() * 4);
        h2d(dl.get(), lens.data(), (size_t)batch * 4);
        h2d(dkc.get(), k_cache.data(), cache_bytes);
        h2d(dvc.get(), v_cache.data(), cache_bytes);
        ft::kv_append_launch(dk_new.fget(), dv_new.fget(),
                             static_cast<const int*>(dl.get()), dkc.fget(),
                             dvc.fget(), batch, hkv, dim, t_rows);
        d2h(k_cache.mutable_data(), dkc.get(), cache_bytes);
        d2h(v_cache.mutable_data(), dvc.get(), cache_bytes);
        sync_device("kv append kernel");
    }, py::arg("k_new"), py::arg("v_new"), py::arg("lens"),
       py::arg("k_cache"), py::arg("v_cache"), py::arg("batch"),
       py::arg("hkv"), py::arg("dim"), py::arg("t_rows"));

    m.def("kv_append_launch",
          [](py::int_ k_new, py::int_ v_new, py::int_ lens, py::int_ k_cache,
             py::int_ v_cache, int batch, int hkv, int dim, int t_rows,
             std::uintptr_t stream) {
        ft::kv_append_launch(df(k_new), df(v_new),
                             dic(lens),
                             dfm(k_cache), dfm(v_cache), batch, hkv, dim,
                             t_rows, stream);
    }, py::arg("k_new"), py::arg("v_new"), py::arg("lens"),
       py::arg("k_cache"), py::arg("v_cache"), py::arg("batch"),
       py::arg("hkv"), py::arg("dim"), py::arg("t_rows"),
       py::arg("stream") = 0);

    m.def("kv_append_launch_bf16",
          [](py::int_ k_new, py::int_ v_new, py::int_ lens, py::int_ k_cache,
             py::int_ v_cache, int batch, int hkv, int dim, int t_rows,
             std::uintptr_t stream) {
        ft::kv_append_launch_bf16(
            dvoid(k_new),
            dvoid(v_new),
            dic(lens),
            dvoidm(k_cache),
            dvoidm(v_cache),
            batch, hkv, dim, t_rows, stream);
    }, py::arg("k_new"), py::arg("v_new"), py::arg("lens"),
       py::arg("k_cache"), py::arg("v_cache"), py::arg("batch"),
       py::arg("hkv"), py::arg("dim"), py::arg("t_rows"),
       py::arg("stream") = 0);

    m.def("kv_append_launch_fp16",
          [](py::int_ k_new, py::int_ v_new, py::int_ lens, py::int_ k_cache,
             py::int_ v_cache, int batch, int hkv, int dim, int t_rows,
             std::uintptr_t stream) {
        ft::kv_append_launch_fp16(
            dvoid(k_new),
            dvoid(v_new),
            dic(lens),
            dvoidm(k_cache),
            dvoidm(v_cache),
            batch, hkv, dim, t_rows, stream);
    }, py::arg("k_new"), py::arg("v_new"), py::arg("lens"),
       py::arg("k_cache"), py::arg("v_cache"), py::arg("batch"),
       py::arg("hkv"), py::arg("dim"), py::arg("t_rows"),
       py::arg("stream") = 0);

    m.def("attention_prefill_cpu", [](FArray q, FArray k, FArray v,
                                      int batch, int hq, int hkv, int seq,
                                      int dim, bool causal) {
        if (q.size() != (py::ssize_t)batch * hq * seq * dim ||
            k.size() != (py::ssize_t)batch * hkv * seq * dim ||
            v.size() != (py::ssize_t)batch * hkv * seq * dim)
            throw std::invalid_argument("attention operand size mismatch");
        auto out = ft::attention_prefill_cpu(to_vec(q), to_vec(k),
                                             to_vec(v), batch, hq, hkv,
                                             seq, dim, causal);
        return wrap_vec(out, {(py::ssize_t)batch, (py::ssize_t)hq,
                              (py::ssize_t)seq, (py::ssize_t)dim});
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("batch"),
       py::arg("hq"), py::arg("hkv"), py::arg("seq"), py::arg("dim"),
       py::arg("causal"));

    m.def("attention_prefill", [](FArray q, FArray k, FArray v,
                                  int batch, int hq, int hkv, int seq,
                                  int dim, bool causal) {
        if (q.size() != (py::ssize_t)batch * hq * seq * dim ||
            k.size() != (py::ssize_t)batch * hkv * seq * dim ||
            v.size() != (py::ssize_t)batch * hkv * seq * dim)
            throw std::invalid_argument("attention operand size mismatch");
        const size_t n = (size_t)batch * hq * seq * dim;
        const size_t kvn = (size_t)batch * hkv * seq * dim;
        py::array_t<float> out(std::vector<py::ssize_t>{batch, hq, seq, dim});
        if (n == 0) return out;
        DevBuf dq(n * 4), dk(kvn * 4), dv(kvn * 4), dy(n * 4);
        h2d(dq.get(), q.data(), n * 4);
        h2d(dk.get(), k.data(), kvn * 4);
        h2d(dv.get(), v.data(), kvn * 4);
        ft::attention_prefill_launch(dq.fget(), dk.fget(), dv.fget(),
                                     dy.fget(), batch, hq, hkv, seq, dim,
                                     causal);
        d2h(out.mutable_data(), dy.get(), n * 4);
        sync_device("attention prefill kernel");
        return out;
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("batch"),
       py::arg("hq"), py::arg("hkv"), py::arg("seq"), py::arg("dim"),
       py::arg("causal"));

    m.def("attention_prefill_launch", [](py::int_ q, py::int_ k,
                                         py::int_ v, py::int_ out,
                                         int batch, int hq, int hkv,
                                         int seq, int dim, bool causal,
                                         std::uintptr_t stream) {
        ft::attention_prefill_launch(df(q), df(k), df(v), dfm(out),
                                     batch, hq, hkv, seq, dim, causal,
                                     stream);
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"),
       py::arg("batch"), py::arg("hq"), py::arg("hkv"), py::arg("seq"),
       py::arg("dim"), py::arg("causal"), py::arg("stream") = 0);

    m.def("attention_prefill_launch_bf16",
          [](py::int_ q, py::int_ k, py::int_ v, py::int_ out,
             int batch, int hq, int hkv, int seq, int dim, bool causal,
             std::uintptr_t stream) {
        ft::attention_prefill_launch_bf16(
            dvoid(q),
            dvoid(k),
            dvoid(v),
            dvoidm(out),
            batch, hq, hkv, seq, dim, causal, stream);
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"),
       py::arg("batch"), py::arg("hq"), py::arg("hkv"), py::arg("seq"),
       py::arg("dim"), py::arg("causal"), py::arg("stream") = 0);

    m.def("attention_prefill_launch_fp16",
          [](py::int_ q, py::int_ k, py::int_ v, py::int_ out,
             int batch, int hq, int hkv, int seq, int dim, bool causal,
             std::uintptr_t stream) {
        ft::attention_prefill_launch_fp16(
            dvoid(q),
            dvoid(k),
            dvoid(v),
            dvoidm(out),
            batch, hq, hkv, seq, dim, causal, stream);
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"),
       py::arg("batch"), py::arg("hq"), py::arg("hkv"), py::arg("seq"),
       py::arg("dim"), py::arg("causal"), py::arg("stream") = 0);
}
