// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once
/**
 * Shared test utilities for ICP tests.
 *
 * Provides common fixtures, solver presets, and result reporting.
 */

// Standard C++ headers
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Catch2 headers
#include <catch2/catch_test_macros.hpp>

// Internal headers
#include <ICP/Correspondences.h>
#include <ICP/EigenUtils.h>
#include <ICP/GridFactory.h>
#include <ICP/ICPCeresSolver.h>
#include <ICP/ICPSolver.h>

namespace TestUtils
{

using namespace ICP;

// ============================================================================
// Test Fixture
// ============================================================================

struct SolverTestFixture
{
    Grid source;
    Grid target;
    GeometryWeighting weighting;
    Vector3 rayDir{ 0, 0, -1 };
    float maxDist = 100.0f;

    SolverTestFixture()
    {
        float spacing = 2.0f / 44.0f;
        source = createHeightfieldGrid(45, 45, spacing, spacing);
        target = source;

        weighting.enable_weight = false;
        weighting.enable_gate = false;
    }

    BidirectionalCorrs computeCorrs(const Pose7& pose) const
    {
        Quaternion q(pose[3], pose[0], pose[1], pose[2]);
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        srcToTgt.linear() = q.toRotationMatrix();
        srcToTgt.translation() = pose.tail<3>();
        return computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);
    }

    static Pose7 createPose(const Vector3& axisAngle, double tx = 0, double ty = 0, double tz = 0)
    {
        Pose7 pose = rotationPose(axisAngle);
        pose[4] = tx;
        pose[5] = ty;
        pose[6] = tz;
        return pose;
    }
};

// ============================================================================
// Solver Presets
// ============================================================================

namespace Presets
{

inline InnerParams gaussNewton(int maxIter = 20, bool verbose = true)
{
    InnerParams p;
    p.solverType = SolverType::GaussNewton;
    p.maxIterations = maxIter;
    p.translationThreshold = 1e-9;
    p.rotationThreshold = 1e-9;
    p.damping = 0.0;
    p.verbose = verbose;
    return p;
}

inline InnerParams gaussNewtonWithLineSearch(int maxIter = 20, bool verbose = true)
{
    InnerParams p = gaussNewton(maxIter, verbose);
    p.lineSearch.enabled = true;
    p.lineSearch.maxIterations = 10;
    p.lineSearch.alpha = 1.0;
    p.lineSearch.beta = 0.5;
    return p;
}

inline InnerParams lmFixed(double lambda = 1e-6, int maxIter = 20, bool verbose = true)
{
    InnerParams p;
    p.solverType = SolverType::LevenbergMarquardt;
    p.maxIterations = maxIter;
    p.translationThreshold = 1e-9;
    p.rotationThreshold = 1e-9;
    p.verbose = verbose;
    p.lm.lambda = lambda;
    p.lm.fixedLambda = true;
    return p;
}

inline InnerParams lmAdaptive(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
{
    InnerParams p;
    p.solverType = SolverType::LevenbergMarquardt;
    p.maxIterations = maxIter;
    p.translationThreshold = 1e-9;
    p.rotationThreshold = 1e-9;
    p.verbose = verbose;
    p.lm.lambda = lambda;
    p.lm.fixedLambda = false;
    p.lm.lambdaUp = 10.0;
    p.lm.lambdaDown = 0.1;
    p.lm.lambdaMin = 1e-10;
    p.lm.lambdaMax = 1e10;
    return p;
}

inline InnerParams lmAdaptiveWithLineSearch(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
{
    InnerParams p = lmAdaptive(lambda, maxIter, verbose);
    p.lineSearch.enabled = true;
    p.lineSearch.maxIterations = 10;
    p.lineSearch.alpha = 1.0;
    p.lineSearch.beta = 0.5;
    return p;
}

inline InnerParams lmGainRatio(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
{
    InnerParams p;
    p.solverType = SolverType::LevenbergMarquardt;
    p.maxIterations = maxIter;
    p.translationThreshold = 1e-9;
    p.rotationThreshold = 1e-9;
    p.verbose = verbose;
    p.lm.strategy = LMStrategy::GainRatio;
    p.lm.lambda = lambda;
    p.lm.lambdaMin = 1e-10;
    p.lm.lambdaMax = 1e10;
    p.lm.minRelativeDecrease = 1e-3;
    return p;
}

inline InnerParams ceresLM(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
{
    InnerParams p;
    p.solverType = SolverType::LevenbergMarquardt;
    p.maxIterations = maxIter;
    p.translationThreshold = 1e-9;
    p.rotationThreshold = 1e-9;
    p.lm.lambda = lambda;
    p.verbose = verbose;
    return p;
}

inline InnerParams ceresGN(int maxIter = 20, bool verbose = true)
{
    InnerParams p;
    p.solverType = SolverType::GaussNewton;
    p.maxIterations = maxIter;
    p.translationThreshold = 1e-9;
    p.rotationThreshold = 1e-9;
    p.verbose = verbose;
    return p;
}

} // namespace Presets

// ============================================================================
// Result Reporting
// ============================================================================

struct SolverResult
{
    std::string name;
    int iterations;
    double rms;
};

inline void printResults(const std::vector<SolverResult>& results)
{
    WARN("\n=== Summary ===");
    for (const auto& r : results)
    {
        WARN(std::left << std::setw(20) << r.name << ": iters=" << r.iterations
                       << ", rms=" << std::scientific << std::setprecision(6) << r.rms);
    }
}

} // namespace TestUtils
