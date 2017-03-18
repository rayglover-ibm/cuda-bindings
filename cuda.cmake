cmake_minimum_required (VERSION 2.8.12)

set (CUDA_VERBOSE_BUILD ON)
find_package (CUDA REQUIRED)

set (SRC_CUDA
    "src/kernels/add.cu"
)
get_property (incdirs
    TARGET ${core}
    PROPERTY INCLUDE_DIRECTORIES
)
list (APPEND
    CUDA_NVCC_FLAGS "--expt-relaxed-constexpr --default-stream per-thread"
)
cuda_include_directories (${incdirs})
cuda_add_library (${core}_cuda
    STATIC ${SRC_CUDA}
    OPTIONS "-gencode arch=compute_30,code=compute_30 -cudart static"
)