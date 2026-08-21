cmake_minimum_required(VERSION 3.24)
project(fusedtok VERSION 0.1.0 LANGUAGES CXX CUDA)

# C++17 for host code; CUDA follows via nvcc default
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target GPU architecture: RTX 3060 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
# pybind11 provided by pip; pass its cmake dir via CMAKE_PREFIX_PATH
find_package(pybind11 CONFIG REQUIRED)

# The compiled extension module loaded as `import _fusedtok`
pybind11_add_module(_fusedtok src/fusedtok.cu src/rmsnorm.cu src/swiglu.cu src/rope.cu src/activations.cu src/softmax.cu src/layernorm.cu src/bindings.cpp)
target_include_directories(_fusedtok PRIVATE include)
target_link_libraries(_fusedtok PRIVATE CUDA::cudart)

# Suppress C4819 (codepage warning) on MSVC hosts.
# Options must be routed per-language: nvcc rejects a bare "/wd4819"
# on a CUDA source as a stray input file, so CUDA sources get it
# wrapped in -Xcompiler, CXX sources get it directly.
if(MSVC)
    target_compile_options(_fusedtok PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>"
        "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>")
endif()
