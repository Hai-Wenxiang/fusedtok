cmake_minimum_required(VERSION 3.24)
project(fusedtok VERSION 1.3.1 LANGUAGES CXX CUDA)

# C++17 for host code and CUDA alike (nvcc defaults to C++17 on CUDA
# 13 but pinning both keeps non-default toolchains from drifting)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures: explicit sm_80 (A100) and sm_86 (RTX 30)
# cubins plus a compute_86 PTX embed so newer architectures (RTX 40/50,
# H100, sm_89+) JIT the kernels with their drivers (Hopper has no native
# cubin here; it JITs the sm_86 PTX like the rest).
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

# Host-compiler warnings: MSVC /W3, GCC/Clang -Wall -Wextra (routed per
# language, CUDA host code via -Xcompiler). Kept visible but not fatal -
# third-party headers (pybind11, CCCL) are the noise floor; the first-
# party sources must compile warning-clean under these levels (any new
# warning from our own code is treated as a build smell).
# C4819 (codepage) is suppressed on MSVC hosts, and the conforming
# preprocessor is forced (CCCL / cooperative_groups headers refuse the
# traditional one).
if(MSVC)
    target_compile_options(_fusedtok PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>"
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>"
        "$<$<COMPILE_LANGUAGE:CXX>:/W3>")
else()
    target_compile_options(_fusedtok PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
endif()
