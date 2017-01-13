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
    "src/main/com/cufoo/Binding.cpp"
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

add_jar ("${core}java" "src/main/com/cufoo/Binding.java"
    OUTPUT_NAME "cufoo"
)

# tests
if (cufoo_WITH_TESTS)
    set (TEST_FILES
        "src/test/com/cufoo/tests/BindingTest.java"
    )
    set (tgt ${PROJECT_NAME}_test)

    add_jar (${tgt} ${TEST_FILES}
        INCLUDE_JARS "${core}java"
    )
    set (cp
        $<TARGET_PROPERTY:${tgt},JAR_FILE>
        $<TARGET_PROPERTY:${core}java,JAR_FILE>
    )
    add_test (
        NAME ${PROJECT_NAME}_test_suite
        COMMAND java -Xcheck:jni
                     -Djava.library.path=$<TARGET_FILE_DIR:binding-java>
                     -cp "${cp}"
                     "com.cufoo.tests.BindingTest"
    )
endif ()