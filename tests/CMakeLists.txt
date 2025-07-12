# =============================================================================
# Kalman Filter Implementation - Test Suite
# =============================================================================
# Comprehensive unit tests for the Kalman filter implementation using 
# modern CMake practices for C projects.
#
# This test suite validates:
# - Matrix operations (addition, multiplication, transpose, inversion)
# - Kalman filter state estimation accuracy
# - Extended Kalman Filter (EKF) functionality
# - Memory management and error handling
# =============================================================================

# =============================================================================
# Test Configuration
# =============================================================================

# Test project should not redefine C standard if parent project already set it
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    set(CMAKE_C_EXTENSIONS OFF)
endif()

# =============================================================================
# Test Executables
# =============================================================================

# Matrix operations test suite
add_executable(test_matrix
    test_matrix.c
)

# Link test executable with the main library
target_link_libraries(test_matrix
    PRIVATE
        kalman_lib          # Link with main library target
)

# Configure include directories for tests
target_include_directories(test_matrix
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../  # Access to source headers
)

# Set test executable properties
set_target_properties(test_matrix PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    FOLDER "Tests"
    OUTPUT_NAME "test_matrix"
)

# Apply same warning flags as main project if enabled
if(KALMAN_ENABLE_WARNINGS AND DEFINED KALMAN_WARNING_FLAGS)
    target_compile_options(test_matrix PRIVATE ${KALMAN_WARNING_FLAGS})
endif()

# Extended Kalman Filter test suite
add_executable(test_ekf
    test_ekf.c
)

# Link test executable with the main library
target_link_libraries(test_ekf
    PRIVATE
        kalman_lib          # Link with main library target
)

# Configure include directories for tests
target_include_directories(test_ekf
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../  # Access to source headers
)

# Set test executable properties
set_target_properties(test_ekf PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    FOLDER "Tests"
    OUTPUT_NAME "test_ekf"
)

# Apply same warning flags as main project if enabled
if(KALMAN_ENABLE_WARNINGS AND DEFINED KALMAN_WARNING_FLAGS)
    target_compile_options(test_ekf PRIVATE ${KALMAN_WARNING_FLAGS})
endif()

# =============================================================================
# Custom Test Targets
# =============================================================================

# Create a custom target to run all tests manually
add_custom_target(run_tests
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_matrix
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_ekf
    DEPENDS test_matrix test_ekf
    COMMENT "Running all unit tests"
    VERBATIM
)

# Create individual test run targets
add_custom_target(run_matrix_tests
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_matrix
    DEPENDS test_matrix
    COMMENT "Running matrix math tests"
    VERBATIM
)

add_custom_target(run_ekf_tests
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_ekf
    DEPENDS test_ekf
    COMMENT "Running Extended Kalman Filter tests"
    VERBATIM
)

# Create a custom target for running tests with memory checking (if available)
find_program(VALGRIND_PROGRAM valgrind)
if(VALGRIND_PROGRAM)
    add_custom_target(run_tests_memcheck
        COMMAND ${VALGRIND_PROGRAM} --tool=memcheck --leak-check=full --show-leak-kinds=all 
                --track-origins=yes --error-exitcode=1
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_matrix
        COMMAND ${VALGRIND_PROGRAM} --tool=memcheck --leak-check=full --show-leak-kinds=all 
                --track-origins=yes --error-exitcode=1
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_ekf
        DEPENDS test_matrix test_ekf
        COMMENT "Running tests with memory checking using Valgrind"
        VERBATIM
    )
    message(STATUS "Memory checking target 'run_tests_memcheck' available")
endif()

# =============================================================================
# Test Coverage Configuration (Optional)
# =============================================================================

option(KALMAN_ENABLE_COVERAGE "Enable code coverage analysis" OFF)

if(KALMAN_ENABLE_COVERAGE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # Add coverage flags to test executables
        target_compile_options(test_matrix PRIVATE --coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(test_matrix PRIVATE --coverage)
        
        target_compile_options(test_ekf PRIVATE --coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(test_ekf PRIVATE --coverage)
        
        # Find gcov or llvm-cov
        find_program(GCOV_PROGRAM gcov)
        find_program(LLVM_COV_PROGRAM llvm-cov)
        
        if(GCOV_PROGRAM OR LLVM_COV_PROGRAM)
            add_custom_target(coverage
                COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_matrix
                COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_ekf
                COMMAND echo "Generating coverage report..."
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                DEPENDS test_matrix test_ekf
                COMMENT "Generating code coverage report"
                VERBATIM
            )
            message(STATUS "Code coverage target 'coverage' available")
        endif()
    else()
        message(WARNING "Code coverage is only supported with GCC or Clang")
    endif()
endif()

# =============================================================================
# Test Configuration Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Test Configuration Summary ===")
message(STATUS "Test executables:     test_matrix, test_ekf")
message(STATUS "Coverage enabled:     ${KALMAN_ENABLE_COVERAGE}")
if(VALGRIND_PROGRAM)
    message(STATUS "Valgrind available:   Yes (${VALGRIND_PROGRAM})")
else()
    message(STATUS "Valgrind available:   No")
endif()
message(STATUS "Custom targets:       run_tests, run_matrix_tests, run_ekf_tests")
if(VALGRIND_PROGRAM)
    message(STATUS "                      run_tests_memcheck")
endif()
if(KALMAN_ENABLE_COVERAGE AND (GCOV_PROGRAM OR LLVM_COV_PROGRAM))
    message(STATUS "                      coverage")
endif()
message(STATUS "===================================")
message(STATUS "")
