cmake_minimum_required(VERSION 3.1)
project(Example LANGUAGES CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Wextra -fmax-errors=4")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang") 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Wextra -ferror-limit=4")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std=c++11 /W4")
endif()

find_package(benchmark QUIET)
find_package(OpenCL)

if (benchmark_FOUND)
	add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark/)
endif ()

if (OpenCL_FOUND)
  include_directories(${OpenCL_INCLUDE_DIRS})
  link_directories(${OpenCL_LIBRARY})
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/test/)

include(CTest)
add_test(NAME run_tests COMMAND tests)

add_custom_target(
	test_memcheck 
	COMMAND ${CMAKE_CTEST_COMMAND}
	--force-new-ctest-process --test-action memcheck
    COMMAND cat "${CMAKE_BINARY_DIR}/Testing/Temporary/MemoryChecker.*.log"
	)

if (OpenCL_FOUND)
  target_link_libraries (test ${OpenCL_LIBRARY})
  target_compile_definitions (test ENABLE_CMAKE=1)
endif()

if (OpenCL_FOUND AND benchmark_FOUND)
  target_link_libraries (benchmark ${OpenCL_LIBRARY})
  target_compile_definitions (benchmark ENABLE_CMAKE=1)
endif()
