cmake_minimum_required (VERSION 2.8.12)

set (tgt libsrpy)

add_subdirectory (thirdparty/pybind11)
pybind11_add_module (${tgt} "src/module.cpp")

target_link_libraries (${tgt}
    PRIVATE libsr
)

if (libsr_WITH_TESTS)
    add_subdirectory (test)
endif ()