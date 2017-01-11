cmake_minimum_required (VERSION 3.2)
include (CTest)

set (TEST_FILES
    basic.js
)
add_test (
    NAME ${PROJECT_NAME}_test_suite
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND node "${TEST_FILES}"
)
set_tests_properties (${PROJECT_NAME}_test_suite
    PROPERTIES ENVIRONMENT "NODE_PATH=${PROJECT_BINARY_DIR}"
)