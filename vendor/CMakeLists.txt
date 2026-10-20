# =============================================================================
# Third-party dependencies
# =============================================================================

# Unity is a git submodule: git submodule add https://github.com/ThrowTheSwitch/Unity.git vendor/unity
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unity/CMakeLists.txt)
    message(FATAL_ERROR "Unity is missing. Run: git submodule update --init vendor/unity")
endif()

# SYSTEM keeps its headers out of the warnings and clang-tidy checks, which are for our own code
add_subdirectory(unity SYSTEM)

# Unity leaves double comparisons out unless asked, and the filter tests compare doubles
target_compile_definitions(unity PUBLIC UNITY_INCLUDE_DOUBLE)
