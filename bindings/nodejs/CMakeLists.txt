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
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/index.js" "${PROJECT_NAME}/index.js" COPYONLY)
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
add_custom_command (TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${tgt}>" "${PROJECT_NAME}"
)

# unit tests
if (cufoo_WITH_TESTS)
    include (CTest)

    set (TEST_FILES
        "src/binding_test.js"
    )
    add_test (
        NAME ${PROJECT_NAME}_test_suite
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMAND node "${TEST_FILES}"
    )
    set_tests_properties (${PROJECT_NAME}_test_suite
        PROPERTIES ENVIRONMENT "NODE_PATH=${PROJECT_BINARY_DIR}"
    )
endif ()