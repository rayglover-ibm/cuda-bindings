cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}java"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

find_package (JNI REQUIRED)
find_package (Java REQUIRED)

include (UseJava)

set (tgt "binding-java")

add_library (${tgt} SHARED 
    "src/binding.cpp"
)
set_target_properties (${tgt} PROPERTIES 
    OUTPUT_NAME "binding"
)
target_include_directories (${tgt} 
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/third_party/jni.hpp/include"
            "${JNI_INCLUDE_DIRS}"
)
target_link_libraries (${tgt}
    ${core}
)

add_jar ("${core}java" "src/cufoo/CuFoo.java"
    OUTPUT_NAME "cufoo"
)

# tests
if (cufoo_WITH_TESTS)
    add_subdirectory (test)
endif ()