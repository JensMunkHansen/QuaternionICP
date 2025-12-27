#include <ICP/MultiViewICP.h>
#include <ICP/ICPCeres.h>
#include <ICP/JacobiansAmbientTwoPose.h>

#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>

#include <iomanip>
#include <iostream>
#include <limits>

namespace ICP
{

std::vector<Edge> buildEdges(
    const std::vector<Grid>& grids,
    const std::vector<Pose7>& poses,
    const Vector3& rayDir,
    float maxDistance,
    int minMatch,
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

    Eigen::Vector3f rayDirF = rayDir.cast<float>();

    // Check all pairs
    for (size_t i = 0; i < numGrids; i++)
    {
        for (size_t j = i + 1; j < numGrids; j++)
        {
            // AABB overlap check
            if (!aabbs[i].overlaps(aabbs[j], maxDistance))
                continue;

            // Compute relative pose: srcToTgt = poses[j]^{-1} * poses[i]
            Eigen::Isometry3d poseI = pose7ToIsometry(poses[i]);
            Eigen::Isometry3d poseJ = pose7ToIsometry(poses[j]);
            Eigen::Isometry3d srcToTgt = poseJ.inverse() * poseI;

            // Forward edge: i -> j
            auto corr_ij = computeBidirectionalCorrs(
                grids[i], grids[j], rayDirF, srcToTgt, maxDistance);

            if (static_cast<int>(corr_ij.forward.size()) >= minMatch)
            {
                edges.push_back({static_cast<int>(i), static_cast<int>(j),
                                 std::move(corr_ij.forward)});
            }

            // Reverse edge: j -> i
            if (static_cast<int>(corr_ij.reverse.size()) >= minMatch)
            {
                edges.push_back({static_cast<int>(j), static_cast<int>(i),
                                 std::move(corr_ij.reverse)});
            }

            if (verbose && (!corr_ij.forward.empty() || !corr_ij.reverse.empty()))
            {
                std::cout << "  Pair (" << i << "," << j << "): "
                          << corr_ij.forward.size() << " fwd, "
                          << corr_ij.reverse.size() << " rev\n";
            }
        }
    }

    return edges;
}

MultiViewICPResult runMultiViewICP(
    const std::vector<Grid>& grids,
    const std::vector<Pose7>& initialPoses,
    const MultiViewICPParams& params)
{
    MultiViewICPResult result;
    const size_t numGrids = grids.size();

    if (numGrids < 2)
    {
        std::cerr << "runMultiViewICP: need at least 2 grids\n";
        return result;
    }

    if (initialPoses.size() != numGrids)
    {
        std::cerr << "runMultiViewICP: poses size mismatch\n";
        return result;
    }

    // Convert Pose7 to Sophus::SE3d for Ceres optimization
    std::vector<Sophus::SE3d> poses(numGrids);
    for (size_t i = 0; i < numGrids; i++)
    {
        Quaternion q(initialPoses[i][3], initialPoses[i][0],
                     initialPoses[i][1], initialPoses[i][2]);
        Vector3 t(initialPoses[i][4], initialPoses[i][5], initialPoses[i][6]);
        poses[i] = Sophus::SE3d(q, t);
    }

    double cosAngleThresh = params.weighting.enable_gate ? params.weighting.tau : 0.0;
    double prevRms = std::numeric_limits<double>::max();

    // Outer loop
    for (int outer = 0; outer < params.maxOuterIterations; outer++)
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
        std::vector<Edge> edges = buildEdges(
            grids, currentPoses, params.rayDir,
            params.maxDistance, params.minMatch, params.verbose);

        size_t totalCorr = 0;
        for (const auto& e : edges)
            totalCorr += e.correspondences.size();

        if (params.verbose)
        {
            std::cout << "Outer " << outer << ": " << edges.size() << " edges, "
                      << totalCorr << " correspondences\n";
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
        if (params.fixFirstPose)
        {
            problem.SetParameterBlockConstant(poseData[0]);
        }

        // Add residuals for each edge
        for (const auto& edge : edges)
        {
            for (const auto& c : edge.correspondences)
            {
                double nDotD = c.tgtNormal.cast<double>().dot(params.rayDir);
                if (std::abs(nDotD) < cosAngleThresh)
                    continue;

                // ForwardRayCostTwoPose: srcPoint in src frame, tgtPoint/tgtNormal in dst frame
                problem.AddResidualBlock(
                    ForwardRayCostTwoPose<RayJacobianSimplified>::Create(
                        c.srcPoint.cast<double>(),
                        c.tgtPoint.cast<double>(),
                        c.tgtNormal.cast<double>(),
                        params.rayDir,
                        params.weighting),
                    nullptr,
                    poseData[edge.srcIdx],
                    poseData[edge.dstIdx]);
            }
        }

        int validCount = problem.NumResidualBlocks();
        if (validCount == 0)
        {
            std::cerr << "No valid residuals!\n";
            break;
        }

        // Configure and solve
        ceres::Solver::Options options;
        configureCeresOptions(options, params.ceresOptions);

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        result.totalInnerIterations += static_cast<int>(summary.iterations.size());

        // Compute RMS
        double rms = std::sqrt(2.0 * summary.final_cost / validCount);
        result.rms = rms;

        if (params.verbose)
        {
            std::cout << "  RMS: " << std::scientific << std::setprecision(6) << rms
                      << ", valid: " << validCount
                      << ", ceres iters: " << summary.iterations.size() << "\n";
        }

        // Check convergence
        double rmsChange = std::abs(prevRms - rms);
        if (rms < params.convergenceTol ||
            rmsChange < params.convergenceTol * rms)
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

    // Convert final poses back to Pose7
    result.poses.resize(numGrids);
    for (size_t i = 0; i < numGrids; i++)
    {
        Quaternion q = poses[i].unit_quaternion();
        Vector3 t = poses[i].translation();
        result.poses[i] << q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z();
    }

    return result;
}

} // namespace ICP
