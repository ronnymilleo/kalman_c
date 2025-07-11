# Simple C test executable
add_executable(test_matrix
    test_matrix.c
    ../matrix_math.c
)

target_include_directories(test_matrix PRIVATE
    ${CMAKE_SOURCE_DIR}
)
