cmake_minimum_required (VERSION 3.0)
cmake_policy (SET CMP0048 NEW) # project versioning

# Make the scripts available in the 'cmake' directory available for the
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Name of the core c++ library
set (core cufoo)

project (${core}
    LANGUAGES C CXX
    VERSION 0.1.0
)

# cmake options
option (${core}_WITH_TESTS  "Enable unit tests"     ON)
option (${core}_WITH_CUDA   "Enable cuda support"   OFF)
option (${core}_WITH_PYTHON "Enable python binding" OFF)
option (${core}_WITH_NODEJS "Enable nodejs binding" OFF)
option (${core}_WITH_JAVA   "Enable Java binding"   OFF)

set (src
    "src/kernels/add.cpp"
    "src/cufoo.cpp"
)

# language requirements/compiler opts
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_definitions (-std=c++14)
endif ()

# create the core library
add_library (${core} STATIC ${src})
set_target_properties (${core} PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)

# config header
configure_file ("include/${core}_config.h.in"
    "include/${core}_config.h"
)

# project includes
target_include_directories (${core}
    PUBLIC "include"
    PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include"
    PUBLIC "src"
    PUBLIC "third_party/gsl_lite/include"
    PUBLIC "third_party/variant/include"
)

# add cuda
if (${core}_WITH_CUDA)
    include ("cuda.cmake")
    target_link_libraries (${core}
        PUBLIC ${core}_cuda
    )
endif ()

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