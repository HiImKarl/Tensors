cmake_minimum_required(VERSION 3.0)
project(Example LANGUAGES CXX)

if((CMAKE_CXX_COMPILER_ID MATCHES GNU) OR (CMAKE_CXX_COMPILER_ID MATCHES clang))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

find_package(benchmark QUIET)

if (benchmark_FOUND)
	add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark/)
endif (benchmark_FOUND)

add_subdirectory(${PROJECT_SOURCE_DIR}/test/)

include(CTest)
add_test(NAME run_tests COMMAND tests)

add_custom_target(
	test_memcheck 
	COMMAND ${CMAKE_CTEST_COMMAND}
	--force-new-ctest-process --test-action memcheck
    COMMAND cat "${CMAKE_BINARY_DIR}/Testing/Temporary/MemoryChecker.*.log"
	)

