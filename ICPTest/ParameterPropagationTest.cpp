// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * Tests for parameter propagation from CommonOptions to solver structs.
 *
 * Verifies that command-line options are correctly converted to
 * SessionParams, OuterParams, and InnerParams.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <ICP/CommonOptions.h>
#include <ICP/ICPParams.h>

using namespace ICP;
using Catch::Matchers::WithinAbs;

TEST_CASE("InnerParams receives all CommonOptions fields", "[params]")
{
    CommonOptions opts;

    // Set non-default values
    opts.solver = CommonOptions::Solver::GaussNewton;
    opts.innerIterations = 42;
    opts.translationThreshold = 1e-8;
    opts.rotationThreshold = 2e-8;
    opts.verbose = true;

    // Line search
    opts.lineSearch.enabled = true;
    opts.lineSearch.maxIterations = 15;
    opts.lineSearch.alpha = 0.9;
    opts.lineSearch.beta = 0.4;

    // LM params
    opts.lm.lambda = 0.01;
    opts.lm.fixedLambda = false;
    opts.lm.lambdaUp = 5.0;
    opts.lm.lambdaDown = 0.2;
    opts.lm.lambdaMin = 1e-12;
    opts.lm.lambdaMax = 1e8;

    InnerParams inner = commonOptionsToInnerParams(opts);

    // Verify all fields
    REQUIRE(inner.solverType == SolverType::GaussNewton);
    REQUIRE(inner.maxIterations == 42);
    REQUIRE_THAT(inner.translationThreshold, WithinAbs(1e-8, 1e-15));
    REQUIRE_THAT(inner.rotationThreshold, WithinAbs(2e-8, 1e-15));
    REQUIRE(inner.verbose == true);

    // Line search
    REQUIRE(inner.lineSearch.enabled == true);
    REQUIRE(inner.lineSearch.maxIterations == 15);
    REQUIRE_THAT(inner.lineSearch.alpha, WithinAbs(0.9, 1e-10));
    REQUIRE_THAT(inner.lineSearch.beta, WithinAbs(0.4, 1e-10));

    // LM params
    REQUIRE_THAT(inner.lm.lambda, WithinAbs(0.01, 1e-10));
    REQUIRE(inner.lm.fixedLambda == false);
    REQUIRE_THAT(inner.lm.lambdaUp, WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(inner.lm.lambdaDown, WithinAbs(0.2, 1e-10));
    REQUIRE_THAT(inner.lm.lambdaMin, WithinAbs(1e-12, 1e-20));
    REQUIRE_THAT(inner.lm.lambdaMax, WithinAbs(1e8, 1e-2));
}

TEST_CASE("OuterParams receives all CommonOptions fields", "[params]")
{
    CommonOptions opts;

    // Set non-default values
    opts.outerIterations = 25;
    opts.rmsTol = 1e-12;
    opts.subsampleX = 8;
    opts.subsampleY = 16;
    opts.verbose = true;

    // Geometry weighting
    opts.enableIncidenceWeight = false;
    opts.enableGrazingGate = false;
    opts.incidenceTau = 0.5;

    // Multi-view options
    opts.maxCorrespondences = 500;
    opts.maxNeighbors = 10;

    OuterParams outer = commonOptionsToOuterParams(opts);

    // Verify all fields
    REQUIRE(outer.maxIterations == 25);
    REQUIRE_THAT(outer.convergenceTol, WithinAbs(1e-12, 1e-20));
    REQUIRE(outer.subsampleX == 8);
    REQUIRE(outer.subsampleY == 16);
    REQUIRE(outer.verbose == true);

    // Geometry weighting
    REQUIRE(outer.weighting.enable_weight == false);
    REQUIRE(outer.weighting.enable_gate == false);
    REQUIRE_THAT(outer.weighting.tau, WithinAbs(0.5, 1e-10));

    // Multi-view options
    REQUIRE(outer.maxCorrespondences == 500);
    REQUIRE(outer.maxNeighbors == 10);
}

TEST_CASE("SessionParams receives all CommonOptions fields", "[params]")
{
    CommonOptions opts;

    // Set non-default values
    opts.backend = CommonOptions::Backend::Ceres7;
    opts.useGridPoses = false;
    opts.fixFirstPose = false;
    opts.verbose = true;

    SessionParams session = commonOptionsToSessionParams(opts);

    // Verify all fields
    REQUIRE(session.backend == SolverBackend::Ceres);
    REQUIRE(session.useGridPoses == false);
    REQUIRE(session.fixFirstPose == false);
    REQUIRE(session.verbose == true);
}

TEST_CASE("LM solver type propagates correctly", "[params]")
{
    CommonOptions opts;
    opts.solver = CommonOptions::Solver::LevenbergMarquardt;

    InnerParams inner = commonOptionsToInnerParams(opts);
    REQUIRE(inner.solverType == SolverType::LevenbergMarquardt);
}

TEST_CASE("HandRolled backend propagates correctly", "[params]")
{
    CommonOptions opts;
    opts.backend = CommonOptions::Backend::HandRolled7D;

    SessionParams session = commonOptionsToSessionParams(opts);
    REQUIRE(session.backend == SolverBackend::HandRolled);
}

TEST_CASE("Ceres options derived from InnerParams", "[params]")
{
    InnerParams inner;
    inner.solverType = SolverType::LevenbergMarquardt;
    inner.maxIterations = 50;
    inner.translationThreshold = 1e-10;
    inner.lm.lambda = 0.1;

    ceres::Solver::Options ceresOpts = toCeresSolverOptions(inner);

    REQUIRE(ceresOpts.max_num_iterations == 50);
    REQUIRE_THAT(ceresOpts.function_tolerance, WithinAbs(1e-10, 1e-18));
    REQUIRE(ceresOpts.minimizer_type == ceres::TRUST_REGION);
    REQUIRE(ceresOpts.trust_region_strategy_type == ceres::LEVENBERG_MARQUARDT);
    // Trust region radius = 1/lambda = 1/0.1 = 10
    REQUIRE_THAT(ceresOpts.initial_trust_region_radius, WithinAbs(10.0, 1e-6));
}

TEST_CASE("Default CommonOptions converts to consistent defaults", "[params]")
{
    // Default CommonOptions should produce the same defaults as the param structs
    CommonOptions opts;

    InnerParams inner = commonOptionsToInnerParams(opts);
    OuterParams outer = commonOptionsToOuterParams(opts);
    SessionParams session = commonOptionsToSessionParams(opts);

    // Verify defaults match ICPParams.h defaults
    InnerParams innerDefault;
    OuterParams outerDefault;
    SessionParams sessionDefault;

    // InnerParams defaults
    REQUIRE(inner.solverType == innerDefault.solverType);
    REQUIRE(inner.maxIterations == opts.innerIterations);  // CLI overrides default
    REQUIRE_THAT(inner.translationThreshold, WithinAbs(innerDefault.translationThreshold, 1e-15));
    REQUIRE_THAT(inner.rotationThreshold, WithinAbs(innerDefault.rotationThreshold, 1e-15));
    REQUIRE(inner.lineSearch.enabled == innerDefault.lineSearch.enabled);
    REQUIRE_THAT(inner.lm.lambda, WithinAbs(innerDefault.lm.lambda, 1e-15));

    // OuterParams defaults
    REQUIRE(outer.maxIterations == opts.outerIterations);  // CLI overrides default
    REQUIRE_THAT(outer.convergenceTol, WithinAbs(outerDefault.convergenceTol, 1e-15));
    REQUIRE(outer.subsampleX == outerDefault.subsampleX);
    REQUIRE(outer.subsampleY == outerDefault.subsampleY);
    // maxDist not exposed in CLI, uses OuterParams default
    REQUIRE_THAT(outer.maxDist, WithinAbs(outerDefault.maxDist, 1e-6));
    // minMatch not exposed in CLI, uses OuterParams default
    REQUIRE(outer.minMatch == outerDefault.minMatch);

    // SessionParams defaults
    REQUIRE(session.backend == sessionDefault.backend);
    REQUIRE(session.useGridPoses == sessionDefault.useGridPoses);
    REQUIRE(session.fixFirstPose == sessionDefault.fixFirstPose);
}
