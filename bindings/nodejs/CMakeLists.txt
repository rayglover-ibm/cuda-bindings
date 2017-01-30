cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}-nodejs"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

# import nodejs modules
list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include (NodeJS)

nodejs_init ()

# Binding package.json and index.js
configure_file ("${core}/index.js" "${core}/index.js" COPYONLY)
configure_file ("${core}/package.json.in" "${core}/package.json" @ONLY)

# A submodule, 'binding', will contain the nodejs binding itself.
set (binding "binding-nodejs")
add_nodejs_module (${binding} "${core}/binding.cpp")

set_target_properties (${binding} PROPERTIES 
    OUTPUT_NAME "binding"
)
target_include_directories (${binding} 
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/third_party/nan"
)
target_link_libraries (${binding}
    ${core}
)
add_custom_command (TARGET ${binding} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${binding}>" "${core}"
)

# unit tests
if (cufoo_WITH_TESTS)
    include (CTest)

    set (TEST_SRC
        "${core}/binding_test.js"
    )
    add_test (
        NAME ${PROJECT_NAME}_test_suite
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMAND node "${TEST_SRC}"
    )
    set_tests_properties (${PROJECT_NAME}_test_suite
        PROPERTIES ENVIRONMENT "NODE_PATH=${PROJECT_BINARY_DIR}"
    )
endif ()

# TODO: install/packaging
