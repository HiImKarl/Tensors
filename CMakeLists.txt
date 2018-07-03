cmake_minimum_required(VERSION 3.0)
project(Example LANGUAGES CXX)

set(CMAKE_MODULE_PATH 
	${CMAKE_MODULE_DIR}
	${PROJECT_SOURCE_DIR}/cmake
	)

include(CTest)

if((CMAKE_CXX_COMPILER_ID MATCHES GNU) OR (CMAKE_CXX_COMPILER_ID MATCHES clang))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark/)
add_subdirectory(${PROJECT_SOURCE_DIR}/test/)

