cmake_minimum_required (VERSION 2.8.12)

# Make the scripts available in the 'cmake' directory available for the
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set (tgt libsr)

project (${tgt} C CXX)

# cmake options
option (${tgt}_WITH_TESTS  "Enable unit tests" ON)
option (${tgt}_WITH_CUDA   "Enable cuda support" ON)
option (${tgt}_WITH_PYTHON "Enable python binding" ON)
option (${tgt}_WITH_NODEJS "Enable nodejs binding" ON)

# version
set (${tgt}_VERSION_MAJOR 0)
set (${tgt}_VERSION_MINOR 1)

set (SRC
    src/add.cpp
    src/kernels.cpp
    src/libsr.cpp
)

# create the library
add_library (${tgt} STATIC ${SRC})
set_target_properties (${tgt} PROPERTIES LINKER_LANGUAGE CXX)

# config header
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/include/${tgt}_config.h.in
    "include/${tgt}_config.h"
)

# project includes
target_include_directories (${tgt}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC  ${CMAKE_CURRENT_BINARY_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# add cuda
if (${tgt}_WITH_CUDA)
    include("cuda.cmake")
    target_link_libraries (${tgt} 
        PUBLIC ${tgt}_cuda
    )
    export (TARGETS ${tgt}_cuda
        FILE "${PROJECT_BINARY_DIR}/${tgt}_cudaTargets.cmake"
    )
endif ()

# unit tests
if (${tgt}_WITH_TESTS)
    enable_testing ()
    add_subdirectory (test)
endif ()

# python binding
if (${tgt}_WITH_PYTHON)
    add_subdirectory (bindings/python)
endif ()

# nodejs binding
if (${tgt}_WITH_NODEJS)
    add_subdirectory (bindings/nodejs)
endif ()

# Add all targets to the build-tree export set
export (TARGETS ${tgt}
    FILE "${PROJECT_BINARY_DIR}/${tgt}Targets.cmake"
)

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export (PACKAGE ${tgt})

# Create the <pkg>Config.cmake and <pkg>ConfigVersion files
configure_file (cmake/${tgt}Config.cmake.in
    "${PROJECT_BINARY_DIR}/${tgt}Config.cmake" @ONLY
)
