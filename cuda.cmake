cmake_minimum_required (VERSION 2.8.12)

set (CUDA_VERBOSE_BUILD ON)
find_package (CUDA REQUIRED)

set (SRC_CUDA
    "src/add.cu"
    "src/device_util.cu"
)
cuda_include_directories (
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gsl_lite/include"
)
cuda_add_library (${core}_cuda
    STATIC ${SRC_CUDA}
    OPTIONS "-gencode arch=compute_30,code=sm_30 -cudart static"
)