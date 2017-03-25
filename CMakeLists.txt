cmake_minimum_required (VERSION 3.0)
cmake_policy (SET CMP0048 NEW) # project versioning

# Make the scripts available in the 'cmake' directory available for the
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Name of the core c++ library
set (core mwe)

project (${core}
    LANGUAGES C CXX
    VERSION 0.1.0
)

# cmake options
option (${core}_WITH_TESTS  "Enable unit tests"     ON)
option (${core}_WITH_PYTHON "Enable python binding" OFF)
option (${core}_WITH_NODEJS "Enable nodejs binding" OFF)
option (${core}_WITH_JAVA   "Enable Java binding"   OFF)

set (src
    "src/kernels/add.cpp"
    "src/mwe.cpp"
)

# language requirements/compiler opts
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# kernelpp
add_subdirectory ("third_party/kernelpp")
get_target_property (inc kernelpp
    INCLUDE_DIRECTORIES
)

# include dirs
list (APPEND inc
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gsl_lite/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# cuda
if (kernelpp_WITH_CUDA)
    # add cuda kernels; appends mwe_generated and mwe_libs
    find_package (CUDA REQUIRED)
    cuda_include_directories (${inc})
    include ("cuda.cmake")
endif ()

# create the core library
add_library (${core} STATIC
    ${mwe_generated}
    ${src}
)
set_target_properties (${core} PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)
configure_file (
    "include/${core}_config.h.in"
    "include/${core}_config.h"
)
target_include_directories (${core}
    PUBLIC ${inc}
)
target_link_libraries (${core}
    PUBLIC kernelpp
    PRIVATE ${mwe_libs}
)

# core library unit tests
if (${core}_WITH_TESTS)
    enable_testing ()
    include ("tests.cmake")
endif ()

# language bindings
if (${core}_WITH_PYTHON)
    add_subdirectory ("bindings/python")
endif ()
if (${core}_WITH_NODEJS)
    add_subdirectory ("bindings/nodejs")
endif ()
if (${core}_WITH_JAVA)
    add_subdirectory ("bindings/java")
endif ()
