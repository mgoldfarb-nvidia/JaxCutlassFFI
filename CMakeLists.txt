cmake_minimum_required(VERSION 3.28)
project(cutlass_ffi LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

include(FetchContent)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

set (Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Python Executable ${Python3_EXECUTABLE}")

# Get XLA FFI headers
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
  RESULT_VARIABLE XLA_FFI_INCLUDE_DIR_RET
  ECHO_OUTPUT_VARIABLE
  ECHO_ERROR_VARIABLE
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if (NOT XLA_FFI_INCLUDE_DIR_RET EQUAL 0)
  message(FATAL_ERROR "Could not find XLA FFI directory.")
endif()

message(STATUS "XLA_FFI_INCLUDE_DIR = ${XLA_FFI_INCLUDE_DIR}")
include_directories(${XLA_FFI_INCLUDE_DIR})

set(CUTLASS_PATH "" CACHE STRING "Path to CUTLASS.")
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
set(CUTLASS_NVCC_ARCHS "SM100;SM103")
if (NOT CUTLASS_PATH)
  message(STATUS "No CUTLASS_PATH set; defaulting to git.")
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        main # 4.2 has no tag yet
  )
  FetchContent_MakeAvailable(cutlass)
else()
  add_subdirectory(${CUTLASS_PATH})
endif()

pybind11_add_module(cutlass_ffi_kernels src/csrc/kernels.cc
                                        src/csrc/blackwell_group_gemm_block_scaled_sm100.cu
                                        src/csrc/blackwell_group_gemm_block_scaled_sm103.cu)
target_link_libraries(cutlass_ffi_kernels PRIVATE CUDA::cudart CUDA::cuda_driver nvidia::cutlass::cutlass)

list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS -ftemplate-backtrace-limit=0)
#list(APPEND CUTLASS_CUDA_NVCC_FLAGS --gpu-architecture=compute_100a,compute_103a)
#list(APPEND CUTLASS_CUDA_NVCC_FLAGS --gpu-code=sm_100a,compute_100a,sm_103a,compute_103a)

if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -lineinfo)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xptxas=--g-tensor-memory-access-check)
endif()

target_compile_options(cutlass_ffi_kernels PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:${CUTLASS_CUDA_NVCC_FLAGS}>)
install(TARGETS cutlass_ffi_kernels LIBRARY DESTINATION .)
