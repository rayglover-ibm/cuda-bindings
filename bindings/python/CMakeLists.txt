cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}py"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

add_subdirectory ("thirdparty/pybind11")
set (tgt "binding-python")

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/src/__init__.py" "${PROJECT_NAME}/__init__.py" COPYONLY)

pybind11_add_module (${tgt} 
    "src/binding.cpp"
)
set_target_properties (${tgt} PROPERTIES 
    OUTPUT_NAME "binding"
)
target_link_libraries (${tgt}
    PRIVATE ${core}
)
add_custom_command (TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${tgt}>" "${PROJECT_NAME}"
)

if (cufoo_WITH_TESTS)
    add_subdirectory (test)
endif ()