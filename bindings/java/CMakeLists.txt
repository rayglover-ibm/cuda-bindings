cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")

project ("${core}-java"
    VERSION "${${core}_VERSION_MAJOR}.${${core}_VERSION_MINOR}.${${core}_VERSION_PATCH}"
)

find_package (JNI REQUIRED)
find_package (Java REQUIRED)

include (UseJava)

# a jni library 'binding' and associated .jar
set (binding "binding-java")

add_library (${binding} SHARED 
    "src/main/com/cufoo/Binding.cpp"
)
set_target_properties (${binding} PROPERTIES 
    OUTPUT_NAME "binding"
)
target_include_directories (${binding} 
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/third_party/jni.hpp/include"
            "${JNI_INCLUDE_DIRS}"
)
target_link_libraries (${binding}
    ${core}
)
add_jar ("${core}-java"
    "src/main/com/cufoo/BindingUtil.java"
    "src/main/com/cufoo/Binding.java"

    OUTPUT_NAME "${core}"
)

# tests
if (cufoo_WITH_TESTS)
    include (CTest)
    set (test ${PROJECT_NAME}_test)

    set (TEST_SRC
        "src/test/com/cufoo/tests/BindingTest.java"
    )
    add_jar (${test} ${TEST_SRC}
        INCLUDE_JARS "${core}-java"
    )
    set (cp
        $<TARGET_PROPERTY:${test},JAR_FILE>
        $<TARGET_PROPERTY:${core}-java,JAR_FILE>
    )
    add_test (
        NAME ${PROJECT_NAME}_test_suite
        COMMAND java -enableassertions
                     -Xcheck:jni
                     -Djava.library.path=$<TARGET_FILE_DIR:binding-java>
                     -cp "${cp}"
                     "com.cufoo.tests.BindingTest"
    )
endif ()