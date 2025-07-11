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
