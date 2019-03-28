file(GLOB TEST_CC *.test.cc)

foreach(ONE_TEST_CC ${TEST_CC})
    get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CC} NAME_WE)

    if (NOT OpenCL_FOUND AND "${ONE_TEST_EXEC}" STREQUAL "opencl")
        continue()
    endif()

    set(TARGET_NAME test_${ONE_TEST_EXEC})
    add_executable(${TARGET_NAME} ${ONE_TEST_CC} test.cc)

    if (OpenCL_FOUND) 
        target_link_libraries(${TARGET_NAME} ${OpenCL_LIBRARY})
        target_compile_definitions(${TARGET_NAME} PUBLIC _ENABLE_OPENCL=1)
    endif()
    
    target_compile_definitions(${TARGET_NAME} PUBLIC _TEST=1)
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC}) 
endforeach()

add_executable(test_all ${TEST_CC} test.cc test.hh) 
target_compile_definitions(test_all PUBLIC _TEST=1)

if (OpenCL_FOUND)
  target_link_libraries(test_all ${OpenCL_LIBRARY})
  target_compile_definitions(test_all PUBLIC _ENABLE_OPENCL=1)
endif()

set_target_properties(test_all PROPERTIES OUTPUT_NAME all) 
