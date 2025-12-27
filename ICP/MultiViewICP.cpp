#include <ICP/MultiViewICP.h>
#include <ICP/ICPCeres.h>
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
    const Vector3& rayDir,
    float maxDistance,
    int minMatch,
    int subsampleX,
    int subsampleY,
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

    // Generate all candidate pairs (i, j) where i < j
    struct PairInfo
    {
        size_t i;
        size_t j;
    };
    std::vector<PairInfo> pairs;
    for (size_t i = 0; i < numGrids; i++)
    {
        for (size_t j = i + 1; j < numGrids; j++)
        {
            if (aabbs[i].overlaps(aabbs[j], maxDistance))
            {
                pairs.push_back({i, j});
            }
        }
    }

    // Pre-allocate storage for parallel results (one per pair)
    std::vector<BidirectionalCorrs> pairCorrs(pairs.size());

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
                grids[pair.i], grids[pair.j], rayDirF, srcToTgt, maxDistance,
                subsampleX, subsampleY);
        });

    // Collect edges sequentially
    for (size_t idx = 0; idx < pairs.size(); idx++)
    {
        const auto& pair = pairs[idx];
        auto& corr = pairCorrs[idx];

        if (static_cast<int>(corr.forward.size()) >= minMatch)
        {
            edges.push_back({static_cast<int>(pair.i), static_cast<int>(pair.j),
                             std::move(corr.forward)});
        }

        if (static_cast<int>(corr.reverse.size()) >= minMatch)
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
            params.maxDistance, params.minMatch,
            params.subsampleX, params.subsampleY, params.verbose);

        size_t totalCorr = 0;
        for (const auto& e : edges)
            totalCorr += e.correspondences.size();

        if (params.verbose)
        {
            std::cout << "Outer " << outer << ": " << edges.size() << " edges, "
                      << totalCorr << " correspondences\n";

            // Print connectivity matrix on first iteration if small enough
            if (outer == 0 && numGrids <= 20)
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
                ceres::CostFunction* costFn = nullptr;
                if (params.ceresOptions.jacobianPolicy == JacobianPolicy::Consistent)
                {
                    costFn = ForwardRayCostTwoPose<RayJacobianConsistent>::Create(
                        c.srcPoint.cast<double>(),
                        c.tgtPoint.cast<double>(),
                        c.tgtNormal.cast<double>(),
                        params.rayDir,
                        params.weighting);
                }
                else
                {
                    costFn = ForwardRayCostTwoPose<RayJacobianSimplified>::Create(
                        c.srcPoint.cast<double>(),
                        c.tgtPoint.cast<double>(),
                        c.tgtNormal.cast<double>(),
                        params.rayDir,
                        params.weighting);
                }
                problem.AddResidualBlock(costFn, nullptr,
                    poseData[edge.srcIdx], poseData[edge.dstIdx]);
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

} // namespace ICP
