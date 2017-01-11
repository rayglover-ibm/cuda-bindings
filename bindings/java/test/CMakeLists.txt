cmake_minimum_required (VERSION 3.0)
include (CTest)

find_package (Java REQUIRED)

set (TEST_FILES
    Basic.java
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
                 Basic
)
