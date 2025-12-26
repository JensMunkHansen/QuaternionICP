find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
include(spsArgs)
find_package(tinyexr REQUIRED)
find_package(GridSearch REQUIRED)

set(TARGET_NAME ICP)

# ICP library with Grid and shared utilities
add_library(${TARGET_NAME} STATIC
    Grid.cpp
    GridFactory.cpp
    CommonOptions.cpp
)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(${TARGET_NAME} PUBLIC
    Eigen3::Eigen
    Ceres::ceres
    GridSearch::GridSearch
    tinyexr::tinyexr
    taywee::args
)
