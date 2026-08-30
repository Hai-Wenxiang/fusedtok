cmake_minimum_required(VERSION 3.24)
project(fusedtok VERSION 1.1.0 LANGUAGES CXX CUDA)

# C++17 for host code; CUDA follows via nvcc default
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target GPU architectures: native sm_80 (A100) and sm_86 (RTX 30) cubins
# plus a compute_86 PTX embed so newer architectures (RTX 40/50, sm_89+) can
# JIT the kernels with their drivers.
# Set unconditionally (not only when undefined): build frontends may inject
# an empty CMAKE_CUDA_ARCHITECTURES that would otherwise fall back to the
# compiler default. Override with -DFUSEDTOK_CUDA_ARCHITECTURES="..." if
# you need something else.
set(FUSEDTOK_CUDA_ARCHITECTURES "80-real;86-real;86-virtual" CACHE STRING
    "CUDA architectures fusedtok compiles for")
set(CMAKE_CUDA_ARCHITECTURES ${FUSEDTOK_CUDA_ARCHITECTURES})

if(NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-lineinfo")
endif()

find_package(CUDAToolkit REQUIRED)
# pybind11 provided by pip; pass its cmake dir via CMAKE_PREFIX_PATH
# (or it is injected by scikit-build-core during packaged builds)
find_package(pybind11 CONFIG REQUIRED)

# The compiled extension module loaded as `fusedtok._fusedtok`.
# Under scikit-build-core it installs into the python/fusedtok package dir.
pybind11_add_module(_fusedtok
    src/fusedtok.cu
    src/rmsnorm.cu
    src/rope.cu
    src/activations.cu
    src/softmax.cu
    src/layernorm.cu
    src/topk.cu
    src/sampling.cu
    src/quantize.cu
    src/qgemm.cu
    src/attention.cu
    src/bindings.cpp)
target_include_directories(_fusedtok PRIVATE include src)
# Static CUDA runtime: the wheel then carries no libcudart.so dependency,
# letting auditwheel tag it manylinux (PyPI rejects bare linux_x86_64 tags).
target_link_libraries(_fusedtok PRIVATE CUDA::cudart_static)

if(SKBUILD)
    install(TARGETS _fusedtok LIBRARY DESTINATION fusedtok)
endif()

# Suppress C4819 (codepage warning) on MSVC hosts, and force the conforming
# preprocessor (CCCL / cooperative_groups headers refuse the traditional one).
# Options must be routed per-language: nvcc rejects a bare "/Zc:..." or
# "/wd4819" on a CUDA source as a stray input file, so CUDA sources get them
# wrapped in -Xcompiler, CXX sources get them directly.
if(MSVC)
    target_compile_options(_fusedtok PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>"
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>")
endif()
