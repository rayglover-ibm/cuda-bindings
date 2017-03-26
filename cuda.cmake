cmake_minimum_required (VERSION 3.0)

set (CUDA_VERBOSE_BUILD ON)

set (src_kernels
    "src/kernels/add.cu"
)

if (UNIX AND CMAKE_POSITION_INDEPENDENT_CODE)
    list (APPEND CMAKE_CXX_FLAGS "-fPIC")
endif ()

cuda_wrap_srcs (${core} OBJ
    obj_generated_files ${src_kernels}

    OPTIONS --expt-relaxed-constexpr
            --default-stream per-thread
            -gencode arch=compute_30,code=compute_30
            -std=c++11

    RELEASE --use_fast_math
)

target_sources (${core} PRIVATE ${obj_generated_files})
target_link_libraries (${core} PRIVATE ${CUDA_LIBRARIES})
