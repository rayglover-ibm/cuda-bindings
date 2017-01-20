cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}-python"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

add_subdirectory ("thirdparty/pybind11")

set (binding "binding-python")

configure_file ("${core}/__init__.py" "${core}/__init__.py" COPYONLY)

pybind11_add_module (${binding} 
    "${core}/binding.cpp"
)
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
if (cufoo_WITH_TESTS)
    include (CTest)

    set (TEST_SRC
        "${core}/binding_test.py"
    )
    add_test (
        NAME cufoopy_test_suite
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMAND python "${TEST_SRC}"
    )
    set_tests_properties (cufoopy_test_suite
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif ()