# =============================================================================
# Kalman Filter Implementation - Modern C Project
# =============================================================================
# A professional implementation of a Kalman filter for real-time state 
# estimation with comprehensive matrix operations and robust design.
#
# Author: Ronny Milleo
# License: See LICENSE file in project root
# =============================================================================

cmake_minimum_required(VERSION 3.28)

# Project definition with version and metadata
project(KalmanFilter
    VERSION 1.0.0
    DESCRIPTION "Modern C Kalman Filter Implementation"
    LANGUAGES C
)

# =============================================================================
# Project Configuration
# =============================================================================

# Only allow in-source builds if explicitly requested
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT DEFINED ENV{KALMAN_ALLOW_IN_SOURCE})
    message(FATAL_ERROR 
        "In-source builds are not allowed. Please create a separate build directory.\n"
        "Usage: mkdir build && cd build && cmake ..\n"
        "To override this behavior, set KALMAN_ALLOW_IN_SOURCE environment variable.")
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# =============================================================================
# C Standard and Compiler Requirements
# =============================================================================

# Require C99 for the implementation
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)  # Disable compiler-specific extensions

# =============================================================================
# Build Configuration Options
# =============================================================================

# Project options
option(KALMAN_BUILD_TESTS "Build unit tests" ON)
option(KALMAN_BUILD_EXAMPLES "Build example programs" ON)
option(KALMAN_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(KALMAN_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# =============================================================================
# Compiler Flags and Warnings
# =============================================================================

# Enable compile commands export for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set output directories for better organization
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/lib")

# Compiler-specific warning flags
if(KALMAN_ENABLE_WARNINGS)
    if(MSVC)
        set(KALMAN_WARNING_FLAGS /W4)
        if(KALMAN_WARNINGS_AS_ERRORS)
            list(APPEND KALMAN_WARNING_FLAGS /WX)
        endif()
    else()
        set(KALMAN_WARNING_FLAGS 
            -Wall -Wextra -Wpedantic
            -Wcast-align -Wcast-qual -Wformat=2 
            -Winit-self -Wmissing-declarations 
            -Wmissing-include-dirs -Wredundant-decls
            -Wshadow -Wstrict-overflow=5 
            -Wswitch-default -Wundef
        )
        if(KALMAN_WARNINGS_AS_ERRORS)
            list(APPEND KALMAN_WARNING_FLAGS -Werror)
        endif()
    endif()
endif()

# =============================================================================
# Main Library Target
# =============================================================================

# Create a library target for the Kalman filter implementation
add_library(kalman_lib STATIC
    kalman.c
    kalman.h
    matrix_math.c
    matrix_math.h
)

# Set target properties
set_target_properties(kalman_lib PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Apply compiler warnings to library
if(KALMAN_ENABLE_WARNINGS)
    target_compile_options(kalman_lib PRIVATE ${KALMAN_WARNING_FLAGS})
endif()

# Configure include directories
target_include_directories(kalman_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link math library to the Kalman library
target_link_libraries(kalman_lib PUBLIC m)

# =============================================================================
# Main Executable Target
# =============================================================================

if(KALMAN_BUILD_EXAMPLES)
    # Main demonstration executable
    add_executable(kalman_demo
        main.c
    )

    # Link with the Kalman library
    target_link_libraries(kalman_demo
        PRIVATE
            kalman_lib
    )

    # Apply compiler warnings to executable
    if(KALMAN_ENABLE_WARNINGS)
        target_compile_options(kalman_demo PRIVATE ${KALMAN_WARNING_FLAGS})
    endif()

    # Set target properties
    set_target_properties(kalman_demo PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
        OUTPUT_NAME "kalman"
    )

    # Comprehensive Kalman filter demonstration executable  
    add_executable(kalman_comprehensive_demo
        demos/kalman_demo.c
    )

    # Link with the Kalman library
    target_link_libraries(kalman_comprehensive_demo
        PRIVATE
            kalman_lib
    )

    # Apply compiler warnings to executable
    if(KALMAN_ENABLE_WARNINGS)
        target_compile_options(kalman_comprehensive_demo PRIVATE ${KALMAN_WARNING_FLAGS})
    endif()

    # Set target properties
    set_target_properties(kalman_comprehensive_demo PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
        OUTPUT_NAME "kalman_comprehensive_demo"
    )
endif()

# =============================================================================
# Testing Configuration
# =============================================================================

if(KALMAN_BUILD_TESTS)
    enable_testing()
    message(STATUS "Building tests enabled")
    add_subdirectory(tests)
endif()

# =============================================================================
# Installation Configuration
# =============================================================================

# Install library and headers
install(TARGETS kalman_lib
    EXPORT KalmanFilterTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install header files
install(FILES
    kalman.h
    matrix_math.h
    DESTINATION include
)

# Install executable if built
if(KALMAN_BUILD_EXAMPLES)
    install(TARGETS kalman_demo kalman_comprehensive_demo
        DESTINATION bin
    )
endif()

# =============================================================================
# Project Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Kalman Filter Project Configuration ===")
message(STATUS "Project:              ${PROJECT_NAME}")
message(STATUS "Version:              ${PROJECT_VERSION}")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "C standard:           ${CMAKE_C_STANDARD}")
message(STATUS "Compiler:             ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "Build tests:          ${KALMAN_BUILD_TESTS}")
message(STATUS "Build examples:       ${KALMAN_BUILD_EXAMPLES}")
message(STATUS "Enable warnings:      ${KALMAN_ENABLE_WARNINGS}")
message(STATUS "Warnings as errors:   ${KALMAN_WARNINGS_AS_ERRORS}")
message(STATUS "Install prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "============================================")
message(STATUS "")