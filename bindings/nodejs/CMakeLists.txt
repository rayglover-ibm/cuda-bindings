cmake_minimum_required (VERSION 3.1)

project ("libsrjs"
    VERSION "${libsr_VERSION_MAJOR}.${libsr_VERSION_MINOR}.${libsr_VERSION_PATCH}"
)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include (NodeJS)

set (tgt "binding")

nodejs_init ()
add_nodejs_module (${tgt} src/binding.cpp)

set_target_properties (${tgt} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}"
)
target_include_directories (${tgt} 
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/third_party/nan"
)
target_link_libraries (${tgt} libsr)

# module package.json and index.js
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/index.js.in" "${PROJECT_NAME}/index.js" @ONLY)
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/package.json.in" "${PROJECT_NAME}/package.json" @ONLY)

# tests
if (libsr_WITH_TESTS)
    add_subdirectory (test)
endif ()