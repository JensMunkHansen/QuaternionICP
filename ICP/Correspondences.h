#pragma once

// Standard C++ headers
#include <vector>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// Internal headers
#include <ICP/Grid.h>
#include <ICP/IntersectionBackend.h>

namespace ICP
{

struct Correspondence
{
    Eigen::Vector3f srcPoint;
    Eigen::Vector3f tgtPoint;
    Eigen::Vector3f tgtNormal;
    float weight = 1.0f;
    int srcVertexIdx = -1;
    int tgtFacetId = -1;
};

/**
 * Compute ray-grid correspondences using the grid's intersection backend.
 *
 * Shoots rays from source grid vertices along a given direction and finds
 * intersections with the target grid mesh.
 *
 * @param srcGrid Source grid (ray origins)
 * @param tgtGrid Target grid (ray target mesh)
 * @param rayDir Ray direction in source-local coordinates (normalized)
 * @param srcToTgt Transform from source-local to target-local coordinates
 * @param maxDist Maximum ray distance
 * @param subsampleX Grid subsampling in X (1=all)
 * @param subsampleY Grid subsampling in Y (1=all)
 * @return Vector of correspondences
 */
inline std::vector<Correspondence> computeRayCorrespondences(
    const Grid& srcGrid,
    const Grid& tgtGrid,
    const Eigen::Vector3f& rayDir,
    const Eigen::Isometry3d& srcToTgt,
    float maxDist,
    int subsampleX = 1,
    int subsampleY = 1)
{
    // Get ray origins from source grid (valid triangle vertices)
    std::vector<int> srcIndices = srcGrid.getTriangleVertexIndices(subsampleX, subsampleY);
    std::vector<float> origins = srcGrid.getTriangleVertices(subsampleX, subsampleY);
    int nRays = static_cast<int>(origins.size() / 3);

    if (nRays == 0)
    {
        return {};
    }

    // Use target grid's backend for intersection
    auto hits = tgtGrid.getBackend().intersectParallel(
        origins.data(),
        nRays,
        rayDir,
        srcToTgt,
        maxDist);

    // Convert hits to correspondences
    std::vector<Correspondence> correspondences;
    correspondences.reserve(hits.size());

    for (const auto& hit : hits)
    {
        Correspondence corr;
        corr.srcVertexIdx = srcIndices[hit.rayIndex];
        corr.srcPoint = srcGrid.getVertex(corr.srcVertexIdx);
        corr.tgtPoint = hit.point;
        corr.tgtNormal = hit.normal;
        corr.tgtFacetId = hit.facetId;
        corr.weight = 1.0f;

        correspondences.push_back(corr);
    }

    return correspondences;
}

/// Bidirectional correspondences for ICP
struct BidirectionalCorrs
{
    std::vector<Correspondence> forward;   // source → target
    std::vector<Correspondence> reverse;   // target → source
};

/**
 * Compute bidirectional ray correspondences.
 *
 * @param source   Source grid (rays originate here for forward)
 * @param target   Target grid (rays originate here for reverse)
 * @param rayDir   Ray direction in local frame (typically [0,0,-1])
 * @param srcToTgt Transform from source to target coordinates
 * @param maxDist  Maximum ray distance
 * @param subsampleX Grid subsampling in X (1=all)
 * @param subsampleY Grid subsampling in Y (1=all)
 * @return Forward and reverse correspondences
 */
inline BidirectionalCorrs computeBidirectionalCorrs(
    const Grid& source,
    const Grid& target,
    const Eigen::Vector3f& rayDir,
    const Eigen::Isometry3d& srcToTgt,
    float maxDist,
    int subsampleX = 1,
    int subsampleY = 1)
{
    BidirectionalCorrs result;

    // Forward: rays from source, intersect target
    result.forward = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist,
                                                subsampleX, subsampleY);

    // Reverse: rays from target, intersect source
    Eigen::Isometry3d tgtToSrc = srcToTgt.inverse();
    result.reverse = computeRayCorrespondences(target, source, rayDir, tgtToSrc, maxDist,
                                                subsampleX, subsampleY);

    return result;
}

} // namespace ICP
