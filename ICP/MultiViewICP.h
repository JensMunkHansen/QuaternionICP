#pragma once
/**
 * Multi-view ICP for N grids.
 *
 * Edge-based approach:
 * 1. Use AABB overlap to find candidate grid pairs
 * 2. Compute bidirectional correspondences for overlapping pairs
 * 3. Build Ceres problem with all edges using ForwardRayCostTwoPose/ReverseRayCostTwoPose
 */

// Standard C++ headers
#include <vector>

// Internal headers
#include <ICP/Correspondences.h>
#include <ICP/EigenTypes.h>
#include <ICP/EigenUtils.h>
#include <ICP/Grid.h>
#include <ICP/ICPParams.h>
#include <ICP/SE3.h>

namespace ICP
{

/**
 * Edge between two grids with correspondences.
 */
struct Edge
{
    int srcIdx;   // Source grid index
    int dstIdx;   // Destination grid index
    std::vector<Correspondence> correspondences;
};

/**
 * Result of multi-view ICP.
 * Note: Final poses are written directly to grids[i].pose.
 */
struct MultiViewICPResult
{
    double rms = 0.0;
    int outerIterations = 0;
    int totalInnerIterations = 0;
    bool converged = false;
};

/**
 * Build edges between overlapping grid pairs using current poses.
 *
 * For each pair (i, j) where i < j:
 * 1. Check AABB overlap (using current poses)
 * 2. Compute forward correspondences (i → j)
 * 3. Compute reverse correspondences (j → i)
 * 4. Add edges that meet minMatch threshold
 *
 * @param grids   Vector of grids
 * @param poses   Current Pose7 for each grid
 * @param outer   Outer loop parameters (rayDir, maxDist, subsampling, etc.)
 * @param verbose Print edge info
 * @return        Vector of edges
 */
std::vector<Edge> buildEdges(
    const std::vector<Grid>& grids,
    const std::vector<Pose7>& poses,
    const OuterParams& outer,
    bool verbose = false);

/**
 * Run multi-view ICP on a set of grids.
 *
 * Optimizes grid.pose for each grid in place.
 *
 * @param grids   Grids with initial poses in grid.pose (modified in place)
 * @param session Session parameters (fixFirstPose, verbose)
 * @param outer   Outer loop parameters (correspondences, weighting)
 * @param inner   Inner loop parameters (solver type, iterations)
 * @return        Result with statistics (final poses also in grid.pose)
 */
MultiViewICPResult runMultiViewICP(
    std::vector<Grid>& grids,
    const SessionParams& session,
    const OuterParams& outer,
    const InnerParams& inner);

/**
 * Build correspondence count matrix from edges.
 *
 * @param edges     Vector of edges with correspondences
 * @param numGrids  Number of grids
 * @return          NxN matrix where [i][j] = count of correspondences from i to j
 */
std::vector<std::vector<int>> buildCorrespondenceMatrix(
    const std::vector<Edge>& edges,
    size_t numGrids);

/**
 * Print correspondence connectivity matrix.
 *
 * Shows how many correspondences exist between each grid pair.
 * Diagonal shows '-', zero entries show '.'.
 *
 * @param counts NxN matrix of correspondence counts
 */
void printConnectivityMatrix(const std::vector<std::vector<int>>& counts);

/**
 * Print per-grid results table comparing initial and final errors.
 *
 * Shows translation and rotation error for each grid vs ground truth,
 * comparing initial (perturbed) and final (optimized) poses.
 *
 * @param finalPoses     Final optimized poses
 * @param initialPoses   Initial (perturbed) poses before ICP
 * @param groundTruth    Ground truth poses
 */
void printResultsTable(
    const std::vector<Pose7>& finalPoses,
    const std::vector<Pose7>& initialPoses,
    const std::vector<Pose7>& groundTruth);

} // namespace ICP
