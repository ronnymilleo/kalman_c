cmake_minimum_required(VERSION 3.14)
project(Kalman C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Enable math library
find_library(MATH_LIBRARY m)

# Main Kalman executable
add_executable(Kalman
        main.c
        kalman.c
        matrix_math.c
)

target_include_directories(Kalman PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link math library if found
if(MATH_LIBRARY)
    target_link_libraries(Kalman ${MATH_LIBRARY})
endif()

# Add tests subdirectory (will be converted to C later)
add_subdirectory(tests)

# Documentation generation with Doxygen
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    # Set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # Request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # Add a custom target to run Doxygen
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    
    message(STATUS "Doxygen found: documentation can be built with 'make docs' or 'cmake --build . --target docs'")
else()
    message(STATUS "Doxygen not found: documentation will not be available")
endif()