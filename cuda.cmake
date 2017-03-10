cmake_minimum_required (VERSION 2.8.12)

set (CUDA_VERBOSE_BUILD ON)
find_package (CUDA REQUIRED)

set (SRC_CUDA
    "src/kernels/add.cu"
    "src/device_util.cu"
)
cuda_include_directories (
    "include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "src"
    "third_party/gsl_lite/include"
    "third_party/variant/include"
)
list (APPEND
    CUDA_NVCC_FLAGS "--expt-relaxed-constexpr --default-stream per-thread"
)
cuda_add_library (${core}_cuda
    STATIC ${SRC_CUDA}
    OPTIONS "-gencode arch=compute_30,code=compute_30 -cudart static"
)