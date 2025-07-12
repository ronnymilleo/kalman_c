#==============================================================================
# CMake configuration for the test suite
#
# This file defines the build targets for the unit tests, including tests
# for the matrix math library and the Extended Kalman Filter (EKF).
#==============================================================================

#------------------------------------------------------------------------------
# Test Executable for Matrix Math
#------------------------------------------------------------------------------

# Define the executable for the matrix math tests
add_executable(test_matrix
    test_matrix.c
    ../kalman.c
    ../matrix_math.c
)

# Add the parent directory to the include path for the matrix test
target_include_directories(test_matrix PRIVATE ${CMAKE_SOURCE_DIR})

#------------------------------------------------------------------------------
# Test Executable for Extended Kalman Filter (EKF)
#------------------------------------------------------------------------------

# Define the executable for the EKF tests
add_executable(test_ekf
    test_ekf.c
    ../kalman.c
    ../matrix_math.c
)

# Add the parent directory to the include path for the EKF test
target_include_directories(test_ekf PRIVATE ${CMAKE_SOURCE_DIR})

#------------------------------------------------------------------------------
# CTest Configuration
#------------------------------------------------------------------------------

# Enable testing with CTest
if(CMAKE_PROJECT_NAME STREQUAL "KalmanFilter")
    include(CTest)
endif()

# Add the matrix test to the CTest suite
add_test(NAME test_matrix COMMAND test_matrix)

# Add the EKF test to the CTest suite
add_test(NAME test_ekf COMMAND test_ekf)