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
 * Parameters for multi-view ICP.
 */
struct MultiViewICPParams
{
    // Ray direction (local frame)
    Vector3 rayDir{0.0, 0.0, -1.0};

    // Correspondence finding
    float maxDistance = 10.0f;    // Max ray distance and AABB margin
    int minMatch = 50;            // Minimum correspondences per edge
    int subsampleX = 1;           // X subsampling stride
    int subsampleY = 1;           // Y subsampling stride

    // Geometry weighting
    GeometryWeighting weighting;

    // Outer loop
    int maxOuterIterations = 10;
    double convergenceTol = 1e-9;  // RMS change threshold

    // Inner loop (Ceres) - default to ITERATIVE_SCHUR for multi-view efficiency
    CeresICPOptions ceresOptions{
        .linearSolverType = ceres::ITERATIVE_SCHUR,
        .preconditionerType = ceres::SCHUR_JACOBI
    };

    // First pose fixed (gauge freedom)
    bool fixFirstPose = true;

    bool verbose = false;
};

/**
 * Result of multi-view ICP.
 */
struct MultiViewICPResult
{
    std::vector<Pose7> poses;     // Final poses for all grids
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
 * @param grids       Vector of grids
 * @param poses       Current Pose7 for each grid
 * @param rayDir      Ray direction in local frame
 * @param maxDistance Max ray distance and AABB margin
 * @param minMatch    Minimum correspondences per edge
 * @param subsampleX  X subsampling stride
 * @param subsampleY  Y subsampling stride
 * @param verbose     Print edge info
 * @return            Vector of edges
 */
std::vector<Edge> buildEdges(
    const std::vector<Grid>& grids,
    const std::vector<Pose7>& poses,
    const Vector3& rayDir,
    float maxDistance,
    int minMatch,
    int subsampleX = 1,
    int subsampleY = 1,
    bool verbose = false);

/**
 * Run multi-view ICP on a set of grids.
 *
 * @param grids         Input grids
 * @param initialPoses  Initial poses for each grid
 * @param params        ICP parameters
 * @return              Result with final poses and statistics
 */
MultiViewICPResult runMultiViewICP(
    const std::vector<Grid>& grids,
    const std::vector<Pose7>& initialPoses,
    const MultiViewICPParams& params);

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

} // namespace ICP
