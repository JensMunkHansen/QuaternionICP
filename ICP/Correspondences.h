#pragma once

#include <ICP/Grid.h>

#include <GridSearch/GridSearchC.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>

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
 * Compute ray-grid correspondences using GridSearch.
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
    std::vector<Correspondence> correspondences;

    // Create projection for target grid
    GridSearchProjection* projection = GridSearch_CreateLinearProjectionWithOffset(
        tgtGrid.dx(), tgtGrid.dy(),
        tgtGrid.offsetX(), tgtGrid.offsetY(), tgtGrid.offsetZ()
    );

    // Get ray origins from source grid (valid triangle vertices)
    std::vector<int> srcIndices = srcGrid.getTriangleVertexIndices(subsampleX, subsampleY);
    std::vector<float> origins = srcGrid.getTriangleVertices(subsampleX, subsampleY);
    int nRays = static_cast<int>(origins.size() / 3);

    if (nRays == 0)
    {
        GridSearch_DestroyProjection(projection);
        return correspondences;
    }

    // Row-major 4x4 transform for GridSearch
    using RowMajor4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
    RowMajor4d worldToGrid = srcToTgt.matrix();

    // Call GridSearch
    GridSearchHit* results = nullptr;
    int hitCount = 0;

    GridSearch_ComputeIntersectionsParallel(
        origins.data(),
        rayDir.data(),
        nRays,
        tgtGrid.verticesData(),
        tgtGrid.nRows(),
        tgtGrid.nCols(),
        worldToGrid.data(),
        tgtGrid.marksData(),
        projection,
        &results,
        &hitCount,
        maxDist
    );

    // Convert hits to correspondences
    correspondences.reserve(hitCount);
    for (int h = 0; h < hitCount; h++)
    {
        const auto& hit = results[h];

        Correspondence corr;
        corr.srcVertexIdx = srcIndices[hit.rayIndex];
        corr.srcPoint = srcGrid.getVertex(corr.srcVertexIdx);
        corr.tgtPoint = Eigen::Map<const Eigen::Vector3f>(hit.point);
        corr.tgtFacetId = hit.facetId;
        corr.tgtNormal = tgtGrid.computeFacetNormalFromId(hit.facetId);
        corr.weight = 1.0f;

        correspondences.push_back(corr);
    }

    // Cleanup
    GridSearch_FreeResults(results);
    GridSearch_DestroyProjection(projection);

    return correspondences;
}

} // namespace ICP
