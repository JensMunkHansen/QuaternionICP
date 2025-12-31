cmake_minimum_required(VERSION 3.22)
project(EigenMKLTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../CMake")

# Options
option(USE_MKL "Enable Intel MKL support for Eigen" OFF)
option(USE_TBB "Enable TBB threading" OFF)

# Enable testing
enable_testing()

# Include spsEigen to find/fetch Eigen3
include(spsEigen)

# Create Eigen test executable
add_executable(eigen_test eigen_test.cpp)
target_link_libraries(eigen_test PRIVATE Eigen3::Eigen)

# Create Ceres test executable
find_package(Ceres QUIET)
if(Ceres_FOUND)
  add_executable(ceres_test ceres_test.cpp)
  target_link_libraries(ceres_test PRIVATE Ceres::ceres Eigen3::Eigen)
  message(STATUS "Ceres found: ${Ceres_DIR}")
else()
  message(STATUS "Ceres not found - skipping ceres_test")
endif()

# MKL configuration
if(USE_MKL)
  include(spsMKL)
  sps_target_link_mkl(eigen_test)
  if(TARGET ceres_test)
    sps_target_link_mkl(ceres_test)
  endif()
endif()

# TBB configuration
if(USE_TBB)
  include(spsTBB)
  if(TARGET ceres_test AND TBB_FOUND)
    target_link_libraries(ceres_test PRIVATE TBB::tbb)
    target_compile_definitions(ceres_test PRIVATE USE_TBB)
  endif()
endif()

# Register CTest tests with expected MKL state
if(USE_MKL)
  add_test(NAME eigen_test COMMAND eigen_test --expect-mkl)
else()
  add_test(NAME eigen_test COMMAND eigen_test --expect-no-mkl)
endif()

if(TARGET ceres_test)
  add_test(NAME ceres_test COMMAND ceres_test)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Eigen MKL Test Configuration ===")
message(STATUS "  USE_MKL: ${USE_MKL}")
message(STATUS "  USE_TBB: ${USE_TBB}")
message(STATUS "  Eigen3_DIR: ${Eigen3_DIR}")
message(STATUS "")
