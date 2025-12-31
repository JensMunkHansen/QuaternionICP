cmake_minimum_required(VERSION 3.22...3.28)

# Enable CUDA language if requested (must be before project())
option(USE_CUDA "Enable CUDA support in Ceres" OFF)
if(USE_CUDA)
  project(Dependencies LANGUAGES C CXX CUDA)
  message(STATUS "CUDA language enabled")
else()
  project(Dependencies LANGUAGES C CXX)
endif()

# Require MULTI_REGISTRATION_ROOT environment variable (for reference in messages)
# CMAKE_INSTALL_PREFIX should be passed from command line (e.g., by dependencies.sh)
if(DEFINED ENV{MULTI_REGISTRATION_ROOT})
  set(MULTI_REGISTRATION_ROOT "$ENV{MULTI_REGISTRATION_ROOT}")
  message(STATUS "MULTI_REGISTRATION_ROOT: ${MULTI_REGISTRATION_ROOT}")
  message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
else()
  message(FATAL_ERROR "MULTI_REGISTRATION_ROOT environment variable not set. "
                      "Set it to the directory where dependencies should be built/installed.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add CMake module path for SBOM functionality
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../CMake")

include(ExternalProject)
include(CMakeDependentOption)
include(spsSBOM)

# Control verbosity of ExternalProject builds
option(DEPS_VERBOSE "Enable verbose output for dependency builds" OFF)
if(NOT DEPS_VERBOSE)
  set(EP_LOG_LEVEL OFF)
else()
  set(EP_LOG_LEVEL ON)
endif()

# Version definitions (from versions.txt or fallback)
sps_get_version(EIGEN3_VERSION "3.4.0")
sps_get_version(ABSEIL_VERSION "20230125.3")
sps_get_version(CERES_VERSION "2.2.0")
sps_get_version(SOPHUS_VERSION "1.22.10")
sps_get_version(ARGS_VERSION "6.4.7")
sps_get_version(MINIZ_VERSION "3.0.2")
sps_get_version(TINYEXR_VERSION "v1.0.9")
sps_get_version(GRIDSEARCH_VERSION "v1.0.0")
sps_get_version(SIMDMATH_VERSION "v1.2.0")
sps_get_version(CATCH2_VERSION "3.5.2")
sps_get_version(SUITESPARSE_VERSION "7.6.0")

# Build options
option(BUILD_CATCH2 "Build and deploy Catch2 test framework" ON)
option(BUILD_EIGEN3 "Build Eigen3 library" ON)
option(BUILD_ABSEIL "Build Abseil library" ON)
option(BUILD_CERES "Build Ceres Solver" ON)
option(BUILD_SOPHUS "Build Sophus library" OFF)
option(BUILD_ARGS "Build args command line parser (with bash completion)" OFF)
option(BUILD_MINIZ "Build miniz compression library" OFF)
option(BUILD_SIMDMATH "Build SIMDMath" ON)
cmake_dependent_option(BUILD_TINYEXR "Build tinyexr OpenEXR loader" OFF
  "BUILD_MINIZ" OFF)
cmake_dependent_option(BUILD_GRIDSEARCH "Build GridSearch" OFF
  "BUILD_SIMDMATH" OFF)

# Optional accelerator support (disabled by default for simple builds)
# Note: USE_CUDA option is declared at top of file before project()
option(USE_MKL "Enable Intel MKL (LAPACK) in Ceres" OFF)
option(USE_TBB "Enable TBB threading in Ceres" OFF)
option(USE_SUITESPARSE "Enable SuiteSparse in Ceres" OFF)

# CUDA configuration
# Auto-detect CUDA toolkit root from nvcc location, fallback to /usr/local/cuda
find_program(NVCC_EXECUTABLE nvcc)
if(NVCC_EXECUTABLE)
  get_filename_component(_nvcc_dir "${NVCC_EXECUTABLE}" DIRECTORY)
  get_filename_component(_cuda_root "${_nvcc_dir}" DIRECTORY)
  set(CUDA_TOOLKIT_ROOT "${_cuda_root}" CACHE PATH "CUDA toolkit root")
else()
  set(CUDA_TOOLKIT_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
endif()
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

#------------------------------------------------------------------------------
# MKL detection (for Ceres LAPACK) - uses spsMKL for detection
#------------------------------------------------------------------------------
if(USE_MKL)
  include(spsMKL)
  sps_find_mkl()

  if(NOT SPS_MKL_FOUND)
    message(FATAL_ERROR "USE_MKL=ON but MKL not found. Set MKLROOT in .bashrc")
  endif()

  # Determine BLA_VENDOR based on whether iomp5 was found
  if(TARGET MKL::iomp5)
    set(MKL_BLA_VENDOR "Intel10_64lp")  # Threaded
  else()
    set(MKL_BLA_VENDOR "Intel10_64lp_seq")  # Sequential
  endif()

  set(MKLROOT "$ENV{MKLROOT}")
endif()

#------------------------------------------------------------------------------
# TBB detection (for Ceres threading)
#------------------------------------------------------------------------------
if(USE_TBB)
  include(spsTBB)
  if(NOT TBB_FOUND)
    message(FATAL_ERROR "USE_TBB=ON but TBB not found")
  endif()
  message(STATUS "TBB_DIR: ${TBB_DIR}")
endif()

# Common CMake arguments for all ExternalProject_Add
set(COMMON_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_CXX_STANDARD=17
)

#------------------------------------------------------------------------------
# NOTE: MKL integration happens at application level, not here.
# Use CMake/spsMKL.cmake in your application:
#   include(spsMKL)
#   sps_target_link_mkl(myapp)
#------------------------------------------------------------------------------

if (BUILD_CATCH2)
  message("Catch2 will be build")
  # Google's test framework
  ExternalProject_Add(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v${CATCH2_VERSION}
    GIT_SHALLOW TRUE
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
      ${ASAN_CMAKE_ARGS}
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL})
endif()


#------------------------------------------------------------------------------
# Eigen3 - Header-only linear algebra library (no dependencies)
#------------------------------------------------------------------------------
if(BUILD_EIGEN3)
  message(STATUS "Will build Eigen3 ${EIGEN3_VERSION}")
  ExternalProject_Add(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        ${EIGEN3_VERSION}
    GIT_SHALLOW    TRUE
    UPDATE_COMMAND ""
    CMAKE_ARGS
      ${COMMON_CMAKE_ARGS}
      -DCMAKE_Fortran_COMPILER:FILEPATH=
      -DEIGEN_BUILD_DOC=OFF
      -DEIGEN_BUILD_TESTING=OFF
      -DBUILD_TESTING=OFF
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

#------------------------------------------------------------------------------
# Abseil - Google's C++ common libraries (no dependencies)
#------------------------------------------------------------------------------
if(BUILD_ABSEIL)
  message(STATUS "Will build Abseil ${ABSEIL_VERSION}")
  ExternalProject_Add(Abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        ${ABSEIL_VERSION}
    GIT_SHALLOW    TRUE
    UPDATE_COMMAND ""
    CMAKE_ARGS
      ${COMMON_CMAKE_ARGS}
      -DABSL_BUILD_TESTING=OFF
      -DABSL_USE_GOOGLETEST_HEAD=OFF
      -DABSL_ENABLE_INSTALL=ON
      -DABSL_PROPAGATE_CXX_STD=ON
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

#------------------------------------------------------------------------------
# SuiteSparse - Sparse matrix factorization with optional GPU acceleration
# (depends on Eigen3, optionally CUDA)
#------------------------------------------------------------------------------
if(USE_SUITESPARSE)
  message(STATUS "Will build SuiteSparse ${SUITESPARSE_VERSION} (GPU: ${USE_CUDA})")

  set(SUITESPARSE_DEPENDS "")
  if(BUILD_EIGEN3)
    list(APPEND SUITESPARSE_DEPENDS Eigen3)
  endif()

  set(SUITESPARSE_CMAKE_ARGS
    ${COMMON_CMAKE_ARGS}
    -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
    -DSUITESPARSE_ENABLE_PROJECTS=suitesparse_config$<SEMICOLON>amd$<SEMICOLON>colamd$<SEMICOLON>camd$<SEMICOLON>ccolamd$<SEMICOLON>cholmod$<SEMICOLON>spqr
    -DSUITESPARSE_USE_FORTRAN=OFF
    -DBUILD_TESTING=OFF
    -DBLA_VENDOR=Generic
  )

  # Enable CUDA for CHOLMOD if requested
  if(USE_CUDA)
    list(APPEND SUITESPARSE_CMAKE_ARGS
      -DSUITESPARSE_USE_CUDA=ON
      -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
      -DCMAKE_CUDA_COMPILER=${CUDA_TOOLKIT_ROOT}/bin/nvcc
    )
  else()
    list(APPEND SUITESPARSE_CMAKE_ARGS -DSUITESPARSE_USE_CUDA=OFF)
  endif()

  ExternalProject_Add(SuiteSparse
    GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    GIT_TAG        v${SUITESPARSE_VERSION}
    GIT_SHALLOW    TRUE
    UPDATE_COMMAND ""
    DEPENDS        ${SUITESPARSE_DEPENDS}
    CMAKE_ARGS     ${SUITESPARSE_CMAKE_ARGS}
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

#------------------------------------------------------------------------------
# Ceres Solver - Nonlinear least squares (depends on Eigen3, Abseil)
#------------------------------------------------------------------------------
if(BUILD_CERES)
  message(STATUS "Will build Ceres ${CERES_VERSION}")

  set(CERES_DEPENDS "")
  if(BUILD_EIGEN3)
    list(APPEND CERES_DEPENDS Eigen3)
  else()
    # When not building Eigen, explicitly tell Ceres where to find it
    # Assumes Eigen was previously installed to CMAKE_INSTALL_PREFIX
    set(EIGEN3_DIR_HINT "${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake")
    if(EXISTS "${EIGEN3_DIR_HINT}")
      message(STATUS "  Using pre-installed Eigen3: ${EIGEN3_DIR_HINT}")
    else()
      message(WARNING "BUILD_EIGEN3=OFF but Eigen3 not found at ${EIGEN3_DIR_HINT}")
    endif()
  endif()
  if(BUILD_ABSEIL)
    list(APPEND CERES_DEPENDS Abseil)
  endif()
  if(USE_SUITESPARSE)
    list(APPEND CERES_DEPENDS SuiteSparse)
  endif()

  # Base Ceres arguments - minimal build
  set(CERES_CMAKE_ARGS
    ${COMMON_CMAKE_ARGS}
    -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
    -DBUILD_TESTING=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_BENCHMARKS=OFF
    -DBUILD_DOCUMENTATION=OFF
    -DMINIGLOG=ON
    -DGFLAGS=OFF
    -DSUITESPARSE=OFF
    -DCXSPARSE=OFF
    -DACCELERATESPARSE=OFF
    -DEIGENSPARSE=ON
    -DUSE_CUDA=OFF
  )

  # CUDA support
  if(USE_CUDA)
    message(STATUS "  Ceres: CUDA enabled")
    list(APPEND CERES_CMAKE_ARGS
      -DUSE_CUDA=ON
      -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
      -DCMAKE_CUDA_COMPILER=${CUDA_TOOLKIT_ROOT}/bin/nvcc
      -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT}
    )
  endif()

  # MKL/LAPACK support
  if(USE_MKL)
    message(STATUS "  Ceres: MKL/LAPACK enabled (${MKL_BLA_VENDOR})")
    # Pass MKL location so FindBLAS can find it without setvars.sh
    list(APPEND CERES_CMAKE_ARGS
      -DLAPACK=ON
      -DBLA_VENDOR=${MKL_BLA_VENDOR}
      -DMKLROOT=${MKLROOT}
      "-DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX};${MKLROOT}/lib/cmake"
    )
  else()
    list(APPEND CERES_CMAKE_ARGS -DLAPACK=OFF)
  endif()

  # TBB threading support
  if(USE_TBB)
    message(STATUS "  Ceres: TBB threading enabled")
    list(APPEND CERES_CMAKE_ARGS
      -DCERES_THREADING_MODEL=TBB
      -DTBB_DIR=${TBB_DIR}
    )
  else()
    list(APPEND CERES_CMAKE_ARGS
      -DCERES_THREADING_MODEL=CXX_THREADS
    )
  endif()

  # SuiteSparse support
  if(USE_SUITESPARSE)
    message(STATUS "  Ceres: SuiteSparse enabled")
    list(APPEND CERES_CMAKE_ARGS
      -DSUITESPARSE=ON
      -DSuiteSparse_DIR=${CMAKE_INSTALL_PREFIX}/lib/cmake/suitesparse-${SUITESPARSE_VERSION}
    )
  endif()

  ExternalProject_Add(Ceres
    GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
    GIT_TAG        ${CERES_VERSION}
    GIT_SHALLOW    TRUE
    UPDATE_COMMAND ""
    DEPENDS        ${CERES_DEPENDS}
    CMAKE_ARGS     ${CERES_CMAKE_ARGS}
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

#------------------------------------------------------------------------------
# Sophus - Lie groups library (depends on Eigen3, optionally Ceres)
#------------------------------------------------------------------------------
if(BUILD_SOPHUS)
  message(STATUS "Will build Sophus ${SOPHUS_VERSION}")

  set(SOPHUS_DEPENDS "")
  if(BUILD_EIGEN3)
    list(APPEND SOPHUS_DEPENDS Eigen3)
  endif()
  if(BUILD_CERES)
    list(APPEND SOPHUS_DEPENDS Ceres)
  endif()

  ExternalProject_Add(Sophus
    GIT_REPOSITORY https://github.com/strasdat/Sophus.git
    GIT_TAG        ${SOPHUS_VERSION}
    GIT_SHALLOW    TRUE
    UPDATE_COMMAND ""
    DEPENDS        ${SOPHUS_DEPENDS}
    CMAKE_ARGS
      ${COMMON_CMAKE_ARGS}
      -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
      -DBUILD_SOPHUS_TESTS=OFF
      -DBUILD_SOPHUS_EXAMPLES=OFF
      -DSOPHUS_USE_BASIC_LOGGING=ON
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

#------------------------------------------------------------------------------
# args - Command line parser with bash completion (header-only, no dependencies)
# https://github.com/Taywee/args
#------------------------------------------------------------------------------
if(BUILD_ARGS)
  message(STATUS "Will build args ${ARGS_VERSION}")
  ExternalProject_Add(args
    GIT_REPOSITORY https://github.com/Taywee/args.git
    GIT_TAG        ${ARGS_VERSION}
    GIT_SHALLOW    TRUE
    UPDATE_COMMAND ""
    CMAKE_ARGS
      ${COMMON_CMAKE_ARGS}
      -DARGS_BUILD_EXAMPLE=OFF
      -DARGS_BUILD_UNITTESTS=OFF
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

#------------------------------------------------------------------------------
# miniz - Compression library (no dependencies)
#------------------------------------------------------------------------------
if(BUILD_MINIZ)
  message(STATUS "Will build miniz ${MINIZ_VERSION}")
  ExternalProject_Add(miniz
    GIT_REPOSITORY https://github.com/richgel999/miniz.git
    GIT_TAG        ${MINIZ_VERSION}
    GIT_SHALLOW    TRUE
    UPDATE_COMMAND ""
    CMAKE_ARGS
      ${COMMON_CMAKE_ARGS}
      -DBUILD_EXAMPLES=OFF
      -DBUILD_FUZZERS=OFF
      -DBUILD_HEADER_ONLY=OFF
      -DBUILD_SHARED_LIBS=OFF
      -DINSTALL_PROJECT=ON
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

#------------------------------------------------------------------------------
# tinyexr - OpenEXR loader (header-only, depends on miniz)
# Note: tinyexr doesn't have a proper CMake install, so we just copy the header
#------------------------------------------------------------------------------
if(BUILD_TINYEXR)
  message(STATUS "Will build tinyexr ${TINYEXR_VERSION}")

  set(TINYEXR_DEPENDS "")
  if(BUILD_MINIZ)
    list(APPEND TINYEXR_DEPENDS miniz)
  endif()

  # Generate tinyexrConfig.cmake from template
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/tinyexrConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/tinyexrConfig.cmake
    @ONLY
  )

  ExternalProject_Add(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        ${TINYEXR_VERSION}
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES ""
    UPDATE_COMMAND ""
    DEPENDS        ${TINYEXR_DEPENDS}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/include/tinyexr
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/lib/cmake/tinyexr
            COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/tinyexr.h ${CMAKE_INSTALL_PREFIX}/include/tinyexr/
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/tinyexrConfig.cmake ${CMAKE_INSTALL_PREFIX}/lib/cmake/tinyexr/
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

if (BUILD_SIMDMATH)
  ExternalProject_Add(simdmath
    GIT_REPOSITORY git@github.com:JensMunkHansen/SIMDMath.git
    GIT_TAG        ${SIMDMATH_VERSION}
    GIT_SHALLOW    FALSE
    UPDATE_COMMAND ""
    CMAKE_ARGS
      ${COMMON_CMAKE_ARGS}
      -DBUILD_TESTING=OFF
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()

if (BUILD_GRIDSEARCH)
  ExternalProject_Add(gridsearch
    GIT_REPOSITORY git@github.com:JensMunkHansen/GridSearch.git
    GIT_TAG        main
    GIT_SHALLOW    FALSE
    UPDATE_COMMAND ""
    DEPENDS        simdmath
    CMAKE_ARGS
      ${COMMON_CMAKE_ARGS}
      -DBUILD_TESTING=OFF
      -DSIMDMath_DIR=${CMAKE_INSTALL_PREFIX}
      -DGRIDSEARCH_EXHAUSTIVE_SEARCH=OFF
    LOG_BUILD ${EP_LOG_LEVEL}
    LOG_CONFIGURE ${EP_LOG_LEVEL}
    LOG_INSTALL ${EP_LOG_LEVEL}
  )
endif()
