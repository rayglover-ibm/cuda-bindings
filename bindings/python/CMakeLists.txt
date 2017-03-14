cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")
set (binding "python")

project ("${core}-${binding}" VERSION "${PROJECT_VERSION}")
add_subdirectory ("thirdparty/pybind11")

set (src      "${core}/binding.cpp")
set (src_test "${core}/binding_test.py")

pybind11_add_module (${binding} ${src})
configure_file ("${core}/__init__.py" "${core}/__init__.py" COPYONLY)

set_target_properties (${binding} PROPERTIES 
    OUTPUT_NAME "binding"
)
target_link_libraries (${binding}
    PRIVATE ${core}
)
add_custom_command (TARGET ${binding} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${binding}>" "${core}"
)

# tests
if (mwe_WITH_TESTS)
    include (CTest)
    add_test (
        NAME ${PROJECT_NAME}_test_suite
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMAND python "-u" "${src_test}"
    )
    set_tests_properties (${PROJECT_NAME}_test_suite
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif ()