find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
include(spsArgs)
find_package(tinyexr REQUIRED)
find_package(GridSearch REQUIRED)

# Detect Ceres features and generate Config.h
include(spsCeres)
sps_detect_ceres_features()

# Set additional config variables
set(ICP_HAS_MKL ${USE_MKL} CACHE BOOL "Intel MKL available" FORCE)
set(ICP_USE_TBB ${USE_TBB} CACHE BOOL "TBB enabled" FORCE)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    @ONLY
)

set(TARGET_NAME ICP)

# ICP library with Grid and shared utilities
add_library(${TARGET_NAME} STATIC
    Grid.cpp
    GridFactory.cpp
    GridLoader.cpp
    MultiViewICP.cpp
    CommonOptions.cpp
)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
)
target_link_libraries(${TARGET_NAME} PUBLIC
    build
    Eigen3::Eigen
    GridSearch::GridSearch
    Ceres::ceres
    Sophus::Sophus
    tinyexr::tinyexr
    taywee::args
)

if(USE_TBB AND TBB_FOUND)
    target_link_libraries(${TARGET_NAME} PUBLIC TBB::tbb)
    target_compile_definitions(${TARGET_NAME} PUBLIC ICP_USE_TBB)
endif()

if(USE_MKL AND TARGET MKL::MKL)
    sps_target_link_mkl(${TARGET_NAME})
endif()
