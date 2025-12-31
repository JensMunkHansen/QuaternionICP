cmake_minimum_required(VERSION 3.22...3.28)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(QuaternionICP LANGUAGES CXX)

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

include(spsCompileCommands)
# Set output directories for all binaries
include(spsInstallDirs)

# Create interface library for compiler flags (must be before spsCompilerPlatformFlags)
add_library(build INTERFACE)

# Configure compiler/platform-specific flags (uses the 'build' target)
option(SPS_DISABLE_AVX512 "Disable AVX-512 instructions" OFF)
option(SPS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
include(spsCompilerPlatformFlags)
include(spsEigen)

# Optional TBB support
option(USE_TBB "Enable TBB for parallel algorithms" ON)
if(USE_TBB)
    include(spsTBB)
endif()

# Enable testing
option(BUILD_TESTING "Build tests" ON)
add_subdirectory(ICP)
add_subdirectory(SingleICP)
add_subdirectory(MultiICP)

if(BUILD_TESTING)
    include(spsTesting)
    add_subdirectory(ICPTest)
endif()

# Documentation
option(BUILD_DOCUMENTATION "Build API documentation with Doxygen" OFF)
if(BUILD_DOCUMENTATION)
    include(spsDoxygen)
    sps_setup_doxygen(
        TARGET doc
        INPUT_DIRS
            ${PROJECT_SOURCE_DIR}/README.md
            ${PROJECT_SOURCE_DIR}/Doc/SE3_FullChartJacobian.md
            ${PROJECT_SOURCE_DIR}/ICP
        OUTPUT_DIR ${PROJECT_BINARY_DIR}/doc
        PROJECT_NAME "QuaternionICP"
        PROJECT_BRIEF "Multi-view point cloud registration using ray-projected ICP"
    )
endif()
