cmake_minimum_required (VERSION 3.0)

set (CUDA_VERBOSE_BUILD ON)

set (src_kernels
    "src/kernels/add.cu"
)

cuda_wrap_srcs (${core} OBJ
    obj_generated_files ${src_kernels}

    OPTIONS --expt-relaxed-constexpr
            --default-stream per-thread
            -gencode arch=compute_30,code=compute_30
            -std=c++11

    RELEASE --use_fast_math
)

list (APPEND mwe_generated ${obj_generated_files})
list (APPEND mwe_libs ${CUDA_LIBRARIES})
