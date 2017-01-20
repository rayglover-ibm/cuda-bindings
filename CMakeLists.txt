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
option (${core}_WITH_CUDA   "Enable cuda support"   ON)
option (${core}_WITH_PYTHON "Enable python binding" ON)
option (${core}_WITH_NODEJS "Enable nodejs binding" ON)
option (${core}_WITH_JAVA   "Enable Java binding"   ON)

set (SRC
    "src/add.cpp"
    "src/kernel.cpp"
    "src/cufoo.cpp"
)

# create the library
add_library (${core} STATIC ${SRC})
set_target_properties (${core} PROPERTIES LINKER_LANGUAGE CXX)

# config header
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/${core}_config.h.in"
    "include/${core}_config.h"
)

# project includes
target_include_directories (${core}
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PUBLIC  "${CMAKE_CURRENT_BINARY_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# add cuda
if (${core}_WITH_CUDA)
    include ("cuda.cmake")
    target_link_libraries (${core}
        PUBLIC ${core}_cuda
    )
    export (TARGETS ${core}_cuda
        FILE "${PROJECT_BINARY_DIR}/${core}_cudaTargets.cmake"
    )
endif ()

# core library unit tests
if (${core}_WITH_TESTS)
    enable_testing ()
    add_subdirectory (test)
endif ()

# python binding
if (${core}_WITH_PYTHON)
    add_subdirectory (bindings/python)
endif ()

# nodejs binding
if (${core}_WITH_NODEJS)
    add_subdirectory (bindings/nodejs)
endif ()

# Java binding
if (${core}_WITH_JAVA)
    add_subdirectory (bindings/java)
endif ()

# Add all targets to the build-tree export set
export (TARGETS ${core}
    FILE "${PROJECT_BINARY_DIR}/${core}Targets.cmake"
)

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export (PACKAGE ${core})

# Create the <pkg>Config.cmake and <pkg>ConfigVersion files
configure_file ("cmake/${core}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${core}Config.cmake" @ONLY
)
