// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#include <ICP/MultiViewICP.h>
#include <ICP/ICPCeresSolver.h>
#include <ICP/JacobiansAmbientTwoPose.h>

#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>

#include <algorithm>
#include <execution>
#include <iomanip>
#include <iostream>
#include <limits>

namespace ICP
{

std::vector<Edge> buildEdges(
    const std::vector<Grid>& grids,
    const std::vector<Pose7>& poses,
    const OuterParams& outer,
    bool verbose)
{
    std::vector<Edge> edges;
    const size_t numGrids = grids.size();

    if (numGrids < 2)
        return edges;

    // Compute world AABBs using current poses
    std::vector<Grid::AABB> aabbs;
    aabbs.reserve(numGrids);
    for (size_t i = 0; i < numGrids; i++)
    {
        Eigen::Isometry3d worldPose = pose7ToIsometry(poses[i]);
        aabbs.push_back(grids[i].computeWorldAABB(worldPose));
    }

    Eigen::Vector3f rayDirF = outer.rayDir.cast<float>();

    // Generate candidate pairs (i, j) where i < j, respecting maxNeighbors limit
    struct PairInfo
    {
        size_t i;
        size_t j;
    };
    std::vector<PairInfo> pairs;
    std::vector<int> neighborCount(numGrids, 0);

    for (size_t i = 0; i < numGrids; i++)
    {
        for (size_t j = i + 1; j < numGrids; j++)
        {
            // Skip if either grid has reached neighbor limit
            if (outer.maxNeighbors > 0 &&
                (neighborCount[i] >= outer.maxNeighbors || neighborCount[j] >= outer.maxNeighbors))
            {
                continue;
            }

            if (aabbs[i].overlaps(aabbs[j], outer.maxDist))
            {
                pairs.push_back({i, j});
                neighborCount[i]++;
                neighborCount[j]++;
            }
        }
    }

    // Pre-allocate storage for parallel results (one per pair)
    std::vector<BidirectionalCorrs> pairCorrs(pairs.size());

    // Sort correspondences by quality: |n·d| descending (higher = more perpendicular = better)
    auto sortByQuality = [&rayDirF](std::vector<Correspondence>& corrs)
    {
        std::sort(corrs.begin(), corrs.end(),
            [&rayDirF](const Correspondence& a, const Correspondence& b) {
                return std::abs(a.tgtNormal.dot(rayDirF)) >
                       std::abs(b.tgtNormal.dot(rayDirF));
            });
    };

    // Compute correspondences in parallel - each thread writes to its own slot
    std::for_each(std::execution::par, pairs.begin(), pairs.end(),
        [&](const PairInfo& pair)
        {
            size_t idx = static_cast<size_t>(&pair - pairs.data());

            // Compute relative pose: srcToTgt = poses[j]^{-1} * poses[i]
            Eigen::Isometry3d poseI = pose7ToIsometry(poses[pair.i]);
            Eigen::Isometry3d poseJ = pose7ToIsometry(poses[pair.j]);
            Eigen::Isometry3d srcToTgt = poseJ.inverse() * poseI;

            pairCorrs[idx] = computeBidirectionalCorrs(
                grids[pair.i], grids[pair.j], rayDirF, srcToTgt, outer.maxDist,
                outer.subsampleX, outer.subsampleY);

            // Sort by quality (best correspondences first)
            sortByQuality(pairCorrs[idx].forward);
            sortByQuality(pairCorrs[idx].reverse);
        });

    // Collect edges sequentially
    for (size_t idx = 0; idx < pairs.size(); idx++)
    {
        const auto& pair = pairs[idx];
        auto& corr = pairCorrs[idx];

        // Apply maxCorrespondences limit (now keeps best ones due to sorting)
        if (outer.maxCorrespondences > 0 &&
            corr.forward.size() > static_cast<size_t>(outer.maxCorrespondences))
            corr.forward.resize(outer.maxCorrespondences);
        if (outer.maxCorrespondences > 0 &&
            corr.reverse.size() > static_cast<size_t>(outer.maxCorrespondences))
            corr.reverse.resize(outer.maxCorrespondences);

        if (static_cast<int>(corr.forward.size()) >= outer.minMatch)
        {
            edges.push_back({static_cast<int>(pair.i), static_cast<int>(pair.j),
                             std::move(corr.forward)});
        }

        if (static_cast<int>(corr.reverse.size()) >= outer.minMatch)
        {
            edges.push_back({static_cast<int>(pair.j), static_cast<int>(pair.i),
                             std::move(corr.reverse)});
        }

        if (verbose && (!corr.forward.empty() || !corr.reverse.empty()))
        {
            std::cout << "  Pair (" << pair.i << "," << pair.j << "): "
                      << corr.forward.size() << " fwd, "
                      << corr.reverse.size() << " rev\n";
        }
    }

    return edges;
}

MultiViewICPResult runMultiViewICP(
    std::vector<Grid>& grids,
    const SessionParams& session,
    const OuterParams& outer,
    const InnerParams& inner)
{
    MultiViewICPResult result;
    const size_t numGrids = grids.size();
    const bool verbose = session.verbose || outer.verbose;

    if (numGrids < 2)
    {
        std::cerr << "runMultiViewICP: need at least 2 grids\n";
        return result;
    }

    // Convert grid.pose (Pose7) to Sophus::SE3d for Ceres optimization
    std::vector<Sophus::SE3d> poses(numGrids);
    for (size_t i = 0; i < numGrids; i++)
    {
        const auto& p = grids[i].pose;
        Quaternion q(p[3], p[0], p[1], p[2]);
        Vector3 t(p[4], p[5], p[6]);
        poses[i] = Sophus::SE3d(q, t);
    }

    double cosAngleThresh = outer.weighting.enable_gate ? outer.weighting.tau : 0.0;
    double prevRms = std::numeric_limits<double>::max();

    // Outer loop
    for (int outerIter = 0; outerIter < outer.maxIterations; outerIter++)
    {
        result.outerIterations++;

        // Convert current Sophus poses to Pose7 for edge building
        std::vector<Pose7> currentPoses(numGrids);
        for (size_t i = 0; i < numGrids; i++)
        {
            Quaternion q = poses[i].unit_quaternion();
            Vector3 t = poses[i].translation();
            currentPoses[i] << q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z();
        }

        // Build edges using current poses
        std::vector<Edge> edges = buildEdges(grids, currentPoses, outer, verbose);

        size_t totalCorr = 0;
        for (const auto& e : edges)
            totalCorr += e.correspondences.size();

        if (verbose)
        {
            std::cout << "Outer " << outerIter << ": " << edges.size() << " edges, "
                      << totalCorr << " correspondences\n";

            // Print connectivity matrix on first iteration if small enough
            if (outerIter == 0 && numGrids <= 20)
            {
                printConnectivityMatrix(buildCorrespondenceMatrix(edges, numGrids));
            }
        }

        if (edges.empty())
        {
            std::cerr << "No edges found!\n";
            break;
        }

        // Build Ceres problem
        ceres::Problem problem;

        // Add pose parameter blocks
        std::vector<double*> poseData(numGrids);
        for (size_t i = 0; i < numGrids; i++)
        {
            poseData[i] = poses[i].data();
            problem.AddParameterBlock(poseData[i], Sophus::SE3d::num_parameters,
                                      new SophusSE3Manifold());
        }

        // Fix first pose (gauge freedom)
        if (session.fixFirstPose)
        {
            problem.SetParameterBlockConstant(poseData[0]);
        }

        // Add residuals for each edge
        for (const auto& edge : edges)
        {
            // Determine if this is a forward edge (srcIdx < dstIdx) or reverse edge
            bool isForward = edge.srcIdx < edge.dstIdx;
            int poseIdxA = isForward ? edge.srcIdx : edge.dstIdx;  // Canonical order: lower index
            int poseIdxB = isForward ? edge.dstIdx : edge.srcIdx;  // Canonical order: higher index

            for (const auto& c : edge.correspondences)
            {
                double nDotD = c.tgtNormal.cast<double>().dot(outer.rayDir);
                if (std::abs(nDotD) < cosAngleThresh)
                    continue;

                ceres::CostFunction* costFn = nullptr;
                if (isForward)
                {
                    // Forward: ray from A to B, srcPoint in A, tgtPoint/tgtNormal in B
                    if (inner.jacobianPolicy == JacobianPolicy::Consistent)
                    {
                        costFn = ForwardRayCostTwoPose<RayJacobianConsistent>::Create(
                            c.srcPoint.cast<double>(),
                            c.tgtPoint.cast<double>(),
                            c.tgtNormal.cast<double>(),
                            outer.rayDir,
                            outer.weighting);
                    }
                    else
                    {
                        costFn = ForwardRayCostTwoPose<RayJacobianSimplified>::Create(
                            c.srcPoint.cast<double>(),
                            c.tgtPoint.cast<double>(),
                            c.tgtNormal.cast<double>(),
                            outer.rayDir,
                            outer.weighting);
                    }
                }
                else
                {
                    // Reverse: ray from B to A, srcPoint in B (pB), tgtPoint/tgtNormal in A (qA, nA)
                    if (inner.jacobianPolicy == JacobianPolicy::Consistent)
                    {
                        costFn = ReverseRayCostTwoPose<RayJacobianConsistent>::Create(
                            c.srcPoint.cast<double>(),   // pB
                            c.tgtPoint.cast<double>(),   // qA
                            c.tgtNormal.cast<double>(),  // nA
                            outer.rayDir,
                            outer.weighting);
                    }
                    else
                    {
                        costFn = ReverseRayCostTwoPose<RayJacobianSimplified>::Create(
                            c.srcPoint.cast<double>(),   // pB
                            c.tgtPoint.cast<double>(),   // qA
                            c.tgtNormal.cast<double>(),  // nA
                            outer.rayDir,
                            outer.weighting);
                    }
                }
                problem.AddResidualBlock(costFn, nullptr, poseData[poseIdxA], poseData[poseIdxB]);
            }
        }

        int validCount = problem.NumResidualBlocks();
        if (validCount == 0)
        {
            std::cerr << "No valid residuals!\n";
            break;
        }

        // Configure and solve using canonical params
        ceres::Solver::Options options = toCeresSolverOptions(inner);

        // Override linear solver for large multi-view problems (>2 grids)
        // DenseQR is fine for 2 grids, but ITERATIVE_SCHUR scales better
        if (inner.linearSolverType == LinearSolverType::DenseQR && numGrids > 2)
        {
            options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            // Preconditioner is already set by toCeresSolverOptions from inner.preconditionerType
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        result.totalInnerIterations += static_cast<int>(summary.iterations.size());

        // Compute RMS
        double rms = std::sqrt(2.0 * summary.final_cost / validCount);
        result.rms = rms;

        if (verbose)
        {
            std::cout << "  RMS: " << std::scientific << std::setprecision(6) << rms
                      << ", valid: " << validCount
                      << ", ceres iters: " << summary.iterations.size() << "\n";
        }

        // Check convergence
        double rmsChange = std::abs(prevRms - rms);
        if (rms < outer.convergenceTol ||
            rmsChange < outer.convergenceTol * rms)
        {
            result.converged = true;
            break;
        }
        prevRms = rms;

        // Early exit if Ceres converged in 1 iteration
        if (summary.iterations.size() <= 1)
        {
            result.converged = true;
            break;
        }
    }

    // Write final poses back to grids
    for (size_t i = 0; i < numGrids; i++)
    {
        Quaternion q = poses[i].unit_quaternion();
        Vector3 t = poses[i].translation();
        grids[i].pose << q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z();
    }

    return result;
}

std::vector<std::vector<int>> buildCorrespondenceMatrix(
    const std::vector<Edge>& edges,
    size_t numGrids)
{
    std::vector<std::vector<int>> matrix(numGrids, std::vector<int>(numGrids, 0));
    for (const auto& edge : edges)
        matrix[edge.srcIdx][edge.dstIdx] = static_cast<int>(edge.correspondences.size());
    return matrix;
}

void printConnectivityMatrix(const std::vector<std::vector<int>>& counts)
{
    size_t n = counts.size();
    if (n == 0)
        return;

    std::cout << "\nCorrespondence matrix (src -> dst):\n     ";
    for (size_t j = 0; j < n; j++)
        std::cout << std::setw(6) << j;
    std::cout << "\n";

    for (size_t i = 0; i < n; i++)
    {
        std::cout << std::setw(4) << i << " ";
        for (size_t j = 0; j < n; j++)
        {
            if (i == j)
                std::cout << "     -";
            else if (counts[i][j] > 0)
                std::cout << std::setw(6) << counts[i][j];
            else
                std::cout << "     .";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void printResultsTable(
    const std::vector<Pose7>& finalPoses,
    const std::vector<Pose7>& initialPoses,
    const std::vector<Pose7>& groundTruth)
{
    constexpr double RAD_TO_DEG = 180.0 / M_PI;
    const size_t n = finalPoses.size();

    std::cout << "\nPer-grid errors (vs ground truth):\n";
    std::cout << "Grid | Init trans | Final trans | Init rot (deg) | Final rot (deg)\n";
    std::cout << "-----|------------|-------------|----------------|----------------\n";

    double initTotalTrans = 0.0, finalTotalTrans = 0.0;
    double initTotalRot = 0.0, finalTotalRot = 0.0;

    for (size_t i = 1; i < n; i++)
    {
        // Extract translations
        Vector3 initT(initialPoses[i][4], initialPoses[i][5], initialPoses[i][6]);
        Vector3 finalT(finalPoses[i][4], finalPoses[i][5], finalPoses[i][6]);
        Vector3 gtT(groundTruth[i][4], groundTruth[i][5], groundTruth[i][6]);

        double initTransErr = (initT - gtT).norm();
        double finalTransErr = (finalT - gtT).norm();

        // Extract quaternions and compute rotation errors
        Quaternion initQ(initialPoses[i][3], initialPoses[i][0],
                         initialPoses[i][1], initialPoses[i][2]);
        Quaternion finalQ(finalPoses[i][3], finalPoses[i][0],
                          finalPoses[i][1], finalPoses[i][2]);
        Quaternion gtQ(groundTruth[i][3], groundTruth[i][0],
                       groundTruth[i][1], groundTruth[i][2]);

        // Rotation error = angle of (q_est^-1 * q_gt)
        Quaternion initRotDiff = initQ.conjugate() * gtQ;
        Quaternion finalRotDiff = finalQ.conjugate() * gtQ;
        double initRotErr = Eigen::AngleAxisd(initRotDiff).angle() * RAD_TO_DEG;
        double finalRotErr = Eigen::AngleAxisd(finalRotDiff).angle() * RAD_TO_DEG;

        std::cout << std::setw(4) << i << " | "
                  << std::setw(10) << std::fixed << std::setprecision(6) << initTransErr << " | "
                  << std::setw(11) << finalTransErr << " | "
                  << std::setw(14) << std::setprecision(4) << initRotErr << " | "
                  << std::setw(14) << finalRotErr << "\n";

        initTotalTrans += initTransErr;
        finalTotalTrans += finalTransErr;
        initTotalRot += initRotErr;
        finalTotalRot += finalRotErr;
    }

    std::cout << "-----|------------|-------------|----------------|----------------\n";
    std::cout << " SUM | "
              << std::setw(10) << std::fixed << std::setprecision(6) << initTotalTrans << " | "
              << std::setw(11) << finalTotalTrans << " | "
              << std::setw(14) << std::setprecision(4) << initTotalRot << " | "
              << std::setw(14) << finalTotalRot << "\n";
}

} // namespace ICP
