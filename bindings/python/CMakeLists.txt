cmake_minimum_required (VERSION 3.1)

project ("libsrpy"
    VERSION "${libsr_VERSION_MAJOR}.${libsr_VERSION_MINOR}.${libsr_VERSION_PATCH}"
)

set (tgt libsrpy)

add_subdirectory (thirdparty/pybind11)
pybind11_add_module (${tgt} "src/binding.cpp")

target_link_libraries (${tgt}
    PRIVATE libsr
)

if (libsr_WITH_TESTS)
    add_subdirectory (test)
endif ()