cmake_minimum_required(VERSION 3.24)
project(fusedtok VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fusedtok src/fusedtok.cu src/bindings.cpp)
target_include_directories(_fusedtok PRIVATE include)
target_link_libraries(_fusedtok PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_options(_fusedtok PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>"
        "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>")
endif()
