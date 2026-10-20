# =============================================================================
# Unit Tests
# =============================================================================
# Each executable holds the tests of one topic and is registered with CTest.
# =============================================================================

function(kalman_c_add_test name)
    add_executable(${name}
        ${name}.c
        test_support.c
        test_support.h
    )
    target_link_libraries(${name} PRIVATE kalman-c-filter-lib unity)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    kalman_c_configure_target(${name})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES TIMEOUT 30)
endfunction()

kalman_c_add_test(kalman_ekf_tests)
kalman_c_add_test(kalman_tests)
kalman_c_add_test(kalman_tracking_tests)
kalman_c_add_test(matrix_tests)
