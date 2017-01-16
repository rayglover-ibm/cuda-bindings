cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}-python"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

add_subdirectory ("thirdparty/pybind11")
set (tgt "binding-python")

configure_file ("${core}/__init__.py" "${core}/__init__.py" COPYONLY)

pybind11_add_module (${tgt} 
    "${core}/binding.cpp"
)
set_target_properties (${tgt} PROPERTIES 
    OUTPUT_NAME "binding"
)
target_link_libraries (${tgt}
    PRIVATE ${core}
)
add_custom_command (TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${tgt}>" "${core}"
)

if (cufoo_WITH_TESTS)
    include (CTest)

    set (TEST_FILES
        "${core}/binding_test.py"
    )
    add_test (
        NAME cufoopy_test_suite
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMAND python "${TEST_FILES}"
    )
    set_tests_properties (cufoopy_test_suite
        PROPERTIES ENVIRONMENT "PYTHONPATH=${cufoopy_BINARY_DIR}"
    )
endif ()