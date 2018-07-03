include_directories(
	${PROJECT_SOURCE_DIR}/external
	${PROJECT_SOURCE_DIR}/include
	)

file(GLOB TEST_SRC "${PROJECT_SOURCE_DIR}/test/*.cc")
add_executable(tests ${TEST_SRC}) 

add_test(NAME MyTests COMMAND tests)

add_custom_target(
	test_memcheck 
	COMMAND ${CMAKE_CTEST_COMMAND}
	--force-new-ctest-process --test-action memcheck
    COMMAND cat "${CMAKE_BINARY_DIR}/Testing/Temporary/MemoryChecker.*.log"
	)
