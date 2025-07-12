#==============================================================================
# CMake configuration for the Kalman Filter project
#
# This file sets up the project, defines build targets, and configures
# the build environment for the Kalman filter library and its example
# application.
#==============================================================================

# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

#------------------------------------------------------------------------------
# Project Definition
#------------------------------------------------------------------------------

# Define the project name and language
project(KalmanFilter C)

#------------------------------------------------------------------------------
# Build Configuration
#------------------------------------------------------------------------------

# Set the C standard to C99 and enforce it
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set the output directory for all executables to 'bin'
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#------------------------------------------------------------------------------
# Main Executable Target
#------------------------------------------------------------------------------

# Define the main executable for the Kalman filter demonstration
add_executable(kalman_demo
    main.c
    kalman.c
    matrix_math.c
)

# Add the project's root directory to the include path for the demo
target_include_directories(kalman_demo PRIVATE ${CMAKE_SOURCE_DIR})

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------

# Find the math library (libm) for functions like sqrt()
find_library(MATH_LIBRARY m)

# Link the math library to the main executable if it was found
if(MATH_LIBRARY)
    target_link_libraries(kalman_demo ${MATH_LIBRARY})
endif()

#------------------------------------------------------------------------------
# Subdirectories
#------------------------------------------------------------------------------

# Include the 'tests' subdirectory, which contains its own CMakeLists.txt
add_subdirectory(tests)