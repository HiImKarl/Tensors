file(GLOB EXAMPLE_CC *.example.cc)

foreach(ONE_EXAMPLE_CC ${EXAMPLE_CC})
    get_filename_component(ONE_EXAMPLE_EXEC ${ONE_EXAMPLE_CC} NAME_WE)

    if (NOT OpenCL_FOUND AND "${ONE_EXAMPLE_EXEC}" STREQUAL "opencl")
        continue()
    endif()

    set(TARGET_NAME example_${ONE_EXAMPLE_EXEC})
    add_executable(${TARGET_NAME} ${ONE_EXAMPLE_CC})

    target_link_libraries(
        ${TARGET_NAME} 
        ${X11_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    target_compile_definitions(${TARGET_NAME} PUBLIC _PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

    if (OpenCL_FOUND)
        target_link_libraries(${TARGET_NAME} ${OpenCL_LIBRARY})
        target_compile_definitions(${TARGET_NAME} PUBLIC _ENABLE_OPENCL=1)
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_EXAMPLE_EXEC}) 
endforeach()

add_executable(example_all ${EXAMPLE_CC}) 
target_link_libraries(
    example_all 
    ${X11_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_definitions(example_all PUBLIC _PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if (OpenCL_FOUND)
  target_link_libraries(example_all ${OpenCL_LIBRARY})
  target_compile_definitions(example_all PUBLIC _ENABLE_OPENCL=1)
endif()
