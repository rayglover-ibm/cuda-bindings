cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}py"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

set (tgt ${PROJECT_NAME})

add_subdirectory (thirdparty/pybind11)
pybind11_add_module (${tgt} "src/binding.cpp")

target_link_libraries (${tgt}
    PRIVATE ${core}
)

if (cufoo_WITH_TESTS)
    add_subdirectory (test)
endif ()