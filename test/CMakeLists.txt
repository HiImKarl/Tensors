include_directories(
	${PROJECT_SOURCE_DIR}/external
	${PROJECT_SOURCE_DIR}/include
	)

file(GLOB TEST_CC *.test.cc)
if ((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang")) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_TEST=1")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_TEST=1")
endif()

foreach(ONE_TEST_CC ${TEST_CC})

	get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CC} NAME_WE)

	set(TARGET_NAME test_${ONE_TEST_EXEC})

	add_executable(${TARGET_NAME} ${ONE_TEST_CC} test.cc)
	set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC}) 

endforeach()

add_executable(tests ${TEST_CC} test.cc) 
target_compile_definitions (test _TEST=1)

if (OpenCL_FOUND)
  target_link_libraries (test ${OpenCL_LIBRARY})
  target_compile_definitions (test _ENABLE_OPENCL=1)
endif()


