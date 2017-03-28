cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (core "${CMAKE_PROJECT_NAME}")
set (binding "java")

project ("${core}-${binding}" VERSION "${PROJECT_VERSION}")

set (src      "src/main/com/mwe/Binding.cpp")
set (src_test "src/test/com/mwe/tests/BindingTest.java")

# android-specific options
if (ANDROID_TOOLCHAIN)
    set (JNI_INCLUDE_DIRS "")
else ()
    find_package (JNI REQUIRED)
endif ()

find_package (Java REQUIRED)
include (UseJava)

# a jni library 'binding' and associated .jar
add_library (${binding} SHARED ${src})
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
    "src/main/com/mwe/BindingUtil.java"
    "src/main/com/mwe/Binding.java"

    OUTPUT_NAME "${core}"
)

# tests
if (mwe_WITH_TESTS)
    include (CTest)
    set (test ${PROJECT_NAME}_test)

    add_jar (${test} ${src_test}
        INCLUDE_JARS "${core}-java"
    )
    set (classpath
        $<TARGET_PROPERTY:${test},JAR_FILE>
        $<TARGET_PROPERTY:${core}-java,JAR_FILE>
    )
    if (NOT WIN32)
        set (classpath "$<JOIN:${classpath},:>")
    endif ()

    add_test (
        NAME ${PROJECT_NAME}_test_suite
        COMMAND java -enableassertions
                     -Xcheck:jni
                     -Djava.library.path=$<TARGET_FILE_DIR:${binding}>
                     -cp "${classpath}"
                     "com.mwe.tests.BindingTest"
    )
endif ()