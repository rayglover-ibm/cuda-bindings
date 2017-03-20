cmake_minimum_required (VERSION 3.0)

set (CUDA_VERBOSE_BUILD ON)
find_package (CUDA REQUIRED)

set (src_kernels
    "src/kernels/add.cu"
)

CUDA_WRAP_SRCS (${core} OBJ
    obj_generated_files ${src_kernels}

    OPTIONS --expt-relaxed-constexpr
            --default-stream per-thread
            -gencode arch=compute_30,code=compute_30

    RELEASE --use_fast_math
)

list (APPEND mwe_generated ${obj_generated_files})
list (APPEND mwe_libs ${CUDA_LIBRARIES})