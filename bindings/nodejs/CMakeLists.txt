cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}js"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include (NodeJS)

nodejs_init ()

# Binding package.json and index.js
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/index.js.in" "${PROJECT_NAME}/index.js" @ONLY)
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/package.json.in" "${PROJECT_NAME}/package.json" @ONLY)

# A submodule, "binding", will contain the nodejs binding.
set (tgt "binding-nodejs")
add_nodejs_module (${tgt} "src/binding.cpp")

set_target_properties (${tgt} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}"
    OUTPUT_NAME "binding"
)
target_include_directories (${tgt} 
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/third_party/nan"
)
target_link_libraries (${tgt}
    ${core}
)

# tests
if (cufoo_WITH_TESTS)
    add_subdirectory (test)
endif ()