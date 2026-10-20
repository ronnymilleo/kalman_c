# =============================================================================
# Libraries
# =============================================================================

add_library(kalman-c-angle-lib STATIC
    angle.c
    angle.h
)

target_include_directories(kalman-c-angle-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# fmod() lives in the math library on Unix
if(UNIX)
    target_link_libraries(kalman-c-angle-lib PUBLIC m)
endif()

kalman_c_configure_target(kalman-c-angle-lib)
kalman_c_target_coverage(kalman-c-angle-lib)

add_library(kalman-c-matrix-lib STATIC
    matrix.c
    matrix.h
)

target_include_directories(kalman-c-matrix-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# PUBLIC, so that everything linking the library sees matrix_t with the same size as the library does
target_compile_definitions(kalman-c-matrix-lib PUBLIC MATRIX_MAX_DIM=${KALMAN_C_MATRIX_MAX_DIM})

# fabs() and fmax() live in the math library on Unix
if(UNIX)
    target_link_libraries(kalman-c-matrix-lib PUBLIC m)
endif()

kalman_c_configure_target(kalman-c-matrix-lib)
kalman_c_target_coverage(kalman-c-matrix-lib)

add_library(kalman-c-filter-lib STATIC
    kalman.c
    kalman.h
)

target_link_libraries(kalman-c-filter-lib PUBLIC kalman-c-matrix-lib)

kalman_c_configure_target(kalman-c-filter-lib)
kalman_c_target_coverage(kalman-c-filter-lib)

# Formatting runs before every build, so unformatted code never reaches a commit
add_dependencies(kalman-c-angle-lib format)
add_dependencies(kalman-c-matrix-lib format)
add_dependencies(kalman-c-filter-lib format)
