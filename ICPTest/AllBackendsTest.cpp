// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <ICP/CeresCapabilities.h>
#include <ICP/Config.h>
#include <ICP/GridFactory.h>
#include <ICP/ICPCeresSolver.h>
#include <ICP/IntersectionBackend.h>

using namespace ICP;
using Catch::Matchers::WithinAbs;

namespace
{

// Create simple test grids for solver testing
std::pair<Grid, Grid> createTestGrids()
{
    // Create source grid with two hemispheres (constrains all 6 DOF)
    Grid source = createTwoHemispheresGrid(32, 32, 0.2f, 0.2f, 2.0f, 6.0f);

    // Create identical target and perturb its pose
    Grid target = createTwoHemispheresGrid(32, 32, 0.2f, 0.2f, 2.0f, 6.0f);
    target.perturbPose(0.5, 0.05, 42);

    return {std::move(source), std::move(target)};
}

// Run ICP with specified linear solver and verify convergence
bool runICPWithSolver(LinearSolverType solverType, const std::string& solverName)
{
    auto [source, target] = createTestGrids();

    OuterParams outer;
    outer.maxIterations = 10;
    outer.maxDist = 100.0f;

    InnerParams inner;
    inner.linearSolverType = solverType;
    inner.maxIterations = 20;

    // Initial pose (identity - source and target start aligned before perturbation)
    Pose7 initialPose = Pose7::Zero();
    initialPose[3] = 1.0;  // qw = 1 for identity quaternion

    // Ray direction (shooting down in -Z)
    Vector3 rayDir(0, 0, -1);

    auto result = solveICPCeres(source, target, initialPose, rayDir,
                                GeometryWeighting(), inner, outer);

    INFO("Solver: " << solverName);
    INFO("Converged: " << result.converged);
    INFO("RMS: " << result.rms);
    INFO("Outer iterations: " << result.outer_iterations);
    INFO("Total inner iterations: " << result.total_inner_iterations);

    // Should converge with reasonable RMS
    return result.rms < 1e-3;
}

}  // namespace

// =============================================================================
// Intersection Backend Tests
// =============================================================================

#if ICP_USE_GRIDSEARCH
TEST_CASE("GridSearch intersection backend", "[backends][gridsearch]")
{
    auto backend = createIntersectionBackend(IntersectionBackendType::GridSearch);
    REQUIRE(backend != nullptr);

    auto [source, target] = createTestGrids();
    backend->build(target);

    // Get ray origins from source
    auto origins = source.getTriangleVertices(4, 4);
    int numRays = static_cast<int>(origins.size() / 3);
    REQUIRE(numRays > 0);

    // Shoot rays
    Eigen::Vector3f rayDir(0, 0, -1);
    Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();

    auto hits = backend->intersectParallel(origins.data(), numRays, rayDir, identity, 100.0f);

    INFO("GridSearch hits: " << hits.size() << " / " << numRays);
    CHECK(hits.size() > 0);
}
#endif

#if ICP_USE_EMBREE
TEST_CASE("Embree intersection backend", "[backends][embree]")
{
    auto backend = createIntersectionBackend(IntersectionBackendType::Embree);
    REQUIRE(backend != nullptr);

    auto [source, target] = createTestGrids();
    backend->build(target);

    // Get ray origins from source
    auto origins = source.getTriangleVertices(4, 4);
    int numRays = static_cast<int>(origins.size() / 3);
    REQUIRE(numRays > 0);

    // Shoot rays
    Eigen::Vector3f rayDir(0, 0, -1);
    Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();

    auto hits = backend->intersectParallel(origins.data(), numRays, rayDir, identity, 100.0f);

    INFO("Embree hits: " << hits.size() << " / " << numRays);
    CHECK(hits.size() > 0);
}
#endif

#if ICP_USE_GRIDSEARCH && ICP_USE_EMBREE
TEST_CASE("GridSearch vs Embree consistency", "[backends][consistency]")
{
    auto gsBackend = createIntersectionBackend(IntersectionBackendType::GridSearch);
    auto emBackend = createIntersectionBackend(IntersectionBackendType::Embree);

    auto [source, target] = createTestGrids();
    gsBackend->build(target);
    emBackend->build(target);

    auto origins = source.getTriangleVertices(4, 4);
    int numRays = static_cast<int>(origins.size() / 3);

    Eigen::Vector3f rayDir(0, 0, -1);
    Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();

    auto gsHits = gsBackend->intersectParallel(origins.data(), numRays, rayDir, identity, 100.0f);
    auto emHits = emBackend->intersectParallel(origins.data(), numRays, rayDir, identity, 100.0f);

    INFO("GridSearch hits: " << gsHits.size());
    INFO("Embree hits: " << emHits.size());

    // Hit counts should be similar (may differ slightly due to edge cases)
    CHECK(std::abs(static_cast<int>(gsHits.size()) - static_cast<int>(emHits.size())) < 5);
}
#endif

// =============================================================================
// Linear Solver Tests (CPU)
// =============================================================================

TEST_CASE("DenseQR linear solver", "[backends][solver][cpu]")
{
    CHECK(runICPWithSolver(LinearSolverType::DenseQR, "DenseQR"));
}

TEST_CASE("DenseSchur linear solver", "[backends][solver][cpu]")
{
    CHECK(runICPWithSolver(LinearSolverType::DenseSchur, "DenseSchur"));
}

TEST_CASE("IterativeSchur linear solver", "[backends][solver][cpu]")
{
    CHECK(runICPWithSolver(LinearSolverType::IterativeSchur, "IterativeSchur"));
}

// SparseSchur requires SuiteSparse or at least EigenSparse
TEST_CASE("SparseSchur linear solver", "[backends][solver][cpu]")
{
    CHECK(runICPWithSolver(LinearSolverType::SparseSchur, "SparseSchur"));
}

// =============================================================================
// CUDA Solver Tests (conditional)
// =============================================================================

TEST_CASE("CUDA Dense Cholesky solver", "[backends][solver][cuda]")
{
    if (!hasCudaSupport())
    {
        WARN("Skipping: Ceres not built with CUDA support");
        return;
    }

    CHECK(runICPWithSolver(LinearSolverType::CudaDenseCholesky, "CudaDenseCholesky"));
}

TEST_CASE("CUDA Sparse Cholesky solver", "[backends][solver][cuda][suitesparse]")
{
    if (!hasCudaSupport())
    {
        WARN("Skipping: Ceres not built with CUDA support");
        return;
    }
    if (!hasSuiteSparseSupport())
    {
        WARN("Skipping: Ceres not built with SuiteSparse support");
        return;
    }

    CHECK(runICPWithSolver(LinearSolverType::CudaSparseCholesky, "CudaSparseCholesky"));
}

// =============================================================================
// Capability Reporting
// =============================================================================

TEST_CASE("Report available backends", "[backends][info]")
{
    INFO("=== Compiled Intersection Backends ===");
#if ICP_USE_GRIDSEARCH
    INFO("  GridSearch: YES");
#else
    INFO("  GridSearch: NO");
#endif
#if ICP_USE_EMBREE
    INFO("  Embree: YES");
#else
    INFO("  Embree: NO");
#endif

    INFO("=== Ceres Capabilities ===");
    INFO("  SuiteSparse: " << (hasSuiteSparseSupport() ? "YES" : "NO"));
    INFO("  CUDA: " << (hasCudaSupport() ? "YES" : "NO"));

    INFO("=== Available Linear Solvers ===");
    INFO("  dense-qr: YES");
    INFO("  dense-schur: YES");
    INFO("  sparse-schur: YES");
    INFO("  iterative-schur: YES");
    INFO("  cuda-dense: " << (isLinearSolverAvailable("cuda-dense") ? "YES" : "NO"));
    INFO("  cuda-sparse: " << (isLinearSolverAvailable("cuda-sparse") ? "YES" : "NO"));

    // This test always passes - it's just for reporting
    CHECK(true);
}
