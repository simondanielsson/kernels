cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ccuda LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# DO NOT set CMAKE_CUDA_COMPILER here when you pass it on the command line.
# DO NOT add clang-only flags for CUDA.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use the GPU arch; CMake translates this to the right -gencode for NVCC.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(TORCH_CUDA_ARCH_LIST All)

add_library(scan_lib src/scan_kernel.cu src/scan.cu)
target_link_libraries(scan_lib "${TORCH_LIBRARIES}")

add_executable(scan src/main.cu)
target_link_libraries(scan scan_lib)
set_property(TARGET scan PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(gemm src/gemm.cu)
target_link_libraries(gemm "${TORCH_LIBRARIES}")
# Line info for source-level profiling (Nsight Systems/Compute)
target_compile_options(gemm PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>"
)

# If you need extra options for NVCC, add them guarded for CUDA language:
# target_compile_options(vec_add PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

