cmake_minimum_required (VERSION 3.2)

include (CTest)
include ("${CMAKE_MODULE_PATH}/DownloadProject.cmake")

# download GoogleTest targets
download_project (
    PROJ            googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         master
    UPDATE_DISCONNECTED 1
)

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# setup unit test libraries
add_subdirectory (${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

# Turn on all warnings when using gcc.
if (CMAKE_COMPILER_IS_GNUCXX)
   add_definitions ("--std=c++14 -W -Wall -pedantic")
endif ()

# main test suite
add_executable (libsr_test
    "libsr/basic.cpp"
)
target_link_libraries (libsr_test
    libsr
    gtest
    gmock_main
)

add_test (
    NAME full_test_suite
    COMMAND libsr_test
)
add_custom_target (copy-test-files ALL
    COMMAND cmake -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data
)