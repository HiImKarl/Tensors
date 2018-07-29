cmake_minimum_required(VERSION 3.0)
project(Example LANGUAGES CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Wextra -fmax-errors=5")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang") 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Wextra")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std=c++11 /Wall")
endif()

find_package(benchmark QUIET)

if (benchmark_FOUND)
	add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark/)
endif ()

add_subdirectory(${PROJECT_SOURCE_DIR}/test/)

include(CTest)
add_test(NAME run_tests COMMAND tests)

add_custom_target(
	test_memcheck 
	COMMAND ${CMAKE_CTEST_COMMAND}
	--force-new-ctest-process --test-action memcheck
    COMMAND cat "${CMAKE_BINARY_DIR}/Testing/Temporary/MemoryChecker.*.log"
	)

