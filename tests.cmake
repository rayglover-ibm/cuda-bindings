cmake_minimum_required (VERSION 3.2)
include (CTest)

# main test suite
add_executable (mwe_test
	"src/kernels/add_test.cpp"
)
# note: gtest is a dependency of kernelpp
target_link_libraries (mwe_test
	mwe gtest gmock_main
)

add_test (NAME mwe_test_suite
	COMMAND mwe_test
)