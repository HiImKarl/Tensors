find_package(Threads REQUIRED)

file(GLOB_RECURSE ALL_BENCH_CC *.benchmark.cc)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/external
	)

foreach(ONE_BENCH_CC ${ALL_BENCH_CC})

	get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CC} NAME_WE)

   # Avoid name collision 
   set(TARGET_NAME bench_${ONE_BENCH_EXEC})

   add_executable(${TARGET_NAME} ${ONE_BENCH_CC} benchmark.cc)
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
   target_link_libraries(
	   ${TARGET_NAME} benchmark
       ${CMAKE_THREAD_LIBS_INIT}
	   )

endforeach()

add_executable(benchmarks ${ALL_BENCH_CC} benchmark.cc)
target_link_libraries(
   benchmarks benchmark
   ${CMAKE_THREAD_LIBS_INIT}
   )

if (OpenCL_FOUND)
  target_link_libraries (benchmark ${OpenCL_LIBRARY})
  target_compile_definitions (benchmark _ENABLE_OPENCL=1)
endif()
