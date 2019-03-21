file(GLOB_RECURSE ALL_BENCH_CC *.benchmark.cc)

foreach(BENCH_CC ${ALL_BENCH_CC})
   get_filename_component(BENCH_EXEC ${BENCH_CC} NAME_WE)

   set(TARGET_NAME benchmark_${BENCH_EXEC})
   add_executable(${TARGET_NAME} ${BENCH_CC} benchmark.cc)
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${BENCH_EXEC}) 
   target_link_libraries(
       ${TARGET_NAME} benchmark
       ${CMAKE_THREAD_LIBS_INIT}
   )
endforeach()

add_executable(benchmark_all ${ALL_BENCH_CC} benchmark.cc)
target_link_libraries(
   benchmark_all benchmark
   ${CMAKE_THREAD_LIBS_INIT}
)

if (OpenCL_FOUND)
  target_link_libraries(benchmark_all ${OpenCL_LIBRARY})
  target_compile_definitions(benchmark_all PUBLIC _ENABLE_OPENCL=1)
endif()

set_target_properties(benchmark_all PROPERTIES OUTPUT_NAME all) 

