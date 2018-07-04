include_directories(
	${PROJECT_SOURCE_DIR}/external
	${PROJECT_SOURCE_DIR}/include
	)

file(GLOB TEST_CC "*.cc")

foreach(ONE_TEST_CC ${TEST_CC})

	get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CC} NAME_WE)

	set(TARGET_NAME test_${ONE_TEST_EXEC})

	add_executable(${TARGET_NAME} ${ONE_TEST_CC})
	set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC}) 

endforeach()

add_executable(tests ${TEST_CC}) 

