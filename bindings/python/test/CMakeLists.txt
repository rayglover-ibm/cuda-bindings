cmake_minimum_required (VERSION 3.2)
include (CTest)

set (TEST_FILES
    "basic.py"
)
add_test (
    NAME cufoopy_test_suite
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND python "${TEST_FILES}"
)
set_tests_properties (cufoopy_test_suite
    PROPERTIES ENVIRONMENT "PYTHONPATH=${cufoopy_BINARY_DIR}/$<CONFIGURATION>"
)