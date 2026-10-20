# =============================================================================
# KalmanC
# =============================================================================
# Kalman and Extended Kalman filters in C, without dynamic memory allocation
#
# Author: Ronny Milleo
# License: See LICENSE file in project root
# =============================================================================

cmake_minimum_required(VERSION 3.28)

project(KalmanC
    VERSION 2.0.0
    DESCRIPTION "Kalman and Extended Kalman filters in C"
    LANGUAGES C
)

# Refuse in-source builds, so a stray `cmake .` never spreads build files among the sources
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Configure in a separate build folder: cmake -S . -B build")
endif()

# =============================================================================
# C Standard and Compiler Requirements
# =============================================================================

# C17 is the oldest standard the code needs (designated initializers, compound literals and <stdbool.h>), and the one
# embedded toolchains support best
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# =============================================================================
# Default Build Type
# =============================================================================

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# =============================================================================
# Project Options
# =============================================================================

option(KALMAN_C_BUILD_TESTS "Build unit tests" OFF)
option(KALMAN_C_ENABLE_COVERAGE "Instrument the libraries for coverage reports (GCC or Clang)" OFF)
option(KALMAN_C_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(KALMAN_C_WARNINGS_AS_ERRORS "Treat warnings in the project's own code as errors" OFF)

# Matrices, and so filters, have a fixed capacity; this is the largest number of rows or columns
set(KALMAN_C_MATRIX_MAX_DIM 8 CACHE STRING "Largest matrix dimension, and so largest number of states")

# =============================================================================
# Compiler Configuration
# =============================================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/${BUILD_TYPE_LOWER}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/${BUILD_TYPE_LOWER}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/${BUILD_TYPE_LOWER}/lib")

# Strict warnings, sanitizers and coverage for the project's own targets; vendored code is left alone
function(kalman_c_configure_target target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(KALMAN_C_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
        if(KALMAN_C_ENABLE_SANITIZERS)
            target_compile_options(${target} PRIVATE /fsanitize=address)
        endif()
        return()
    endif()

    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wswitch-enum -Wswitch-default
    )
    if(KALMAN_C_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(KALMAN_C_ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined
        )
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

# Coverage is measured on the libraries only, so neither the tests nor Unity count; the link option is PUBLIC because
# whatever links a static library built with --coverage needs the runtime that writes the data
function(kalman_c_target_coverage target)
    if(KALMAN_C_ENABLE_COVERAGE AND CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang)$")
        target_compile_options(${target} PRIVATE --coverage)
        target_link_options(${target} PUBLIC --coverage)
    endif()
endfunction()

# =============================================================================
# Formatting Target (clang-format)
# =============================================================================

set(KALMAN_C_FORMAT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/kalman.c
    ${CMAKE_SOURCE_DIR}/src/kalman.h
    ${CMAKE_SOURCE_DIR}/src/matrix.c
    ${CMAKE_SOURCE_DIR}/src/matrix.h
    ${CMAKE_SOURCE_DIR}/tests/kalman_ekf_tests.c
    ${CMAKE_SOURCE_DIR}/tests/kalman_tests.c
    ${CMAKE_SOURCE_DIR}/tests/kalman_tracking_tests.c
    ${CMAKE_SOURCE_DIR}/tests/matrix_tests.c
    ${CMAKE_SOURCE_DIR}/tests/test_support.c
    ${CMAKE_SOURCE_DIR}/tests/test_support.h
)

find_program(CLANG_FORMAT NAMES clang-format)
if(CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${KALMAN_C_FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatting source code with clang-format"
        VERBATIM
    )
else()
    message(STATUS "clang-format not found; the format target does nothing")
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E echo "clang-format not found, skipping formatting"
    )
endif()

# =============================================================================
# Libraries
# =============================================================================

add_subdirectory(src)

# =============================================================================
# Dependencies
# =============================================================================

if(KALMAN_C_BUILD_TESTS)
    add_subdirectory(vendor)
endif()

# =============================================================================
# Testing
# =============================================================================

if(KALMAN_C_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# =============================================================================
# Post-Build Commands
# =============================================================================

# Lets clangd and clang-tidy find the compile commands from the project root
add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
    COMMENT "Copying compile_commands.json to the project root"
    VERBATIM
)

# =============================================================================
# Project Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== KalmanC Configuration ===")
message(STATUS "Version:              ${PROJECT_VERSION}")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "C standard:           ${CMAKE_C_STANDARD}")
message(STATUS "Compiler:             ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "Matrix max dim:       ${KALMAN_C_MATRIX_MAX_DIM}")
message(STATUS "Build tests:          ${KALMAN_C_BUILD_TESTS}")
message(STATUS "Sanitizers:           ${KALMAN_C_ENABLE_SANITIZERS}")
message(STATUS "Coverage:             ${KALMAN_C_ENABLE_COVERAGE}")
message(STATUS "Warnings as errors:   ${KALMAN_C_WARNINGS_AS_ERRORS}")
message(STATUS "=============================")
message(STATUS "")
