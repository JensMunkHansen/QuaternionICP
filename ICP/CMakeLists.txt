find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
include(spsArgs)
find_package(tinyexr REQUIRED)

# Intersection backend options (at least one must be enabled)
option(USE_GRIDSEARCH "Use GridSearch for ray-grid intersection" ON)
option(USE_EMBREE "Use Intel Embree for ray tracing" OFF)

if(NOT USE_GRIDSEARCH AND NOT USE_EMBREE)
    message(FATAL_ERROR "At least one intersection backend must be enabled (USE_GRIDSEARCH or USE_EMBREE)")
endif()

if(USE_GRIDSEARCH)
    find_package(GridSearch REQUIRED)
endif()

if(USE_EMBREE)
    find_package(embree 4 REQUIRED)
endif()

# Detect Ceres features and generate Config.h
include(spsCeres)
sps_detect_ceres_features()

# Set additional config variables
set(ICP_HAS_MKL ${USE_MKL} CACHE BOOL "Intel MKL available" FORCE)
set(ICP_USE_TBB ${USE_TBB} CACHE BOOL "TBB enabled" FORCE)
set(ICP_USE_GRIDSEARCH ${USE_GRIDSEARCH} CACHE BOOL "GridSearch backend" FORCE)
set(ICP_USE_EMBREE ${USE_EMBREE} CACHE BOOL "Embree backend" FORCE)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    @ONLY
)

set(TARGET_NAME ICP)

# ICP library with Grid and shared utilities
set(ICP_HEADERS
    CommonOptions.h
    Correspondences.h
    EigenTypes.h
    EigenUtils.h
    Grid.h
    GridFactory.h
    GridLoader.h
    ICPCeresSolver.h
    ICPCeresTwoPose.h
    ICPHelpers.h
    ICPLM.h
    ICPParams.h
    ICPSolver.h
    IntersectionBackend.h
    JacobiansAmbient.h
    JacobiansAmbientTwoPose.h
    MultiViewICP.h
    SE3.h
    TestGridUtils.h
    TriangulationMarks.h
)

if(USE_GRIDSEARCH)
    list(APPEND ICP_HEADERS GridSearchBackend.h)
endif()

if(USE_EMBREE)
    list(APPEND ICP_HEADERS EmbreeBackend.h)
endif()

add_library(${TARGET_NAME} STATIC
    BackendFactory.cpp
    CommonOptions.cpp
    Grid.cpp
    GridFactory.cpp
    GridLoader.cpp
    MultiViewICP.cpp
    ${ICP_HEADERS}
)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
)
target_link_libraries(${TARGET_NAME} PUBLIC
    build
    Eigen3::Eigen
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

if(USE_GRIDSEARCH)
    target_link_libraries(${TARGET_NAME} PRIVATE GridSearch::GridSearch)
endif()

if(USE_EMBREE)
    target_link_libraries(${TARGET_NAME} PRIVATE embree)
endif()
