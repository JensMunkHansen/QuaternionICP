// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once

// Internal headers
#include <ICP/Grid.h>
#include <ICP/IntersectionBackend.h>

// GridSearch headers
#include <GridSearch/GridSearchC.h>

namespace ICP
{

/**
 * GridSearch-based intersection backend.
 *
 * Uses linear projection for structured grids. This is very efficient for
 * grids where vertices follow a regular row/column pattern:
 * - O(1) build time (just stores projection configuration)
 * - No triangle index extraction needed
 * - Direct pointer access to grid data
 */
class GridSearchBackend : public IntersectionBackend
{
public:
    ~GridSearchBackend() override
    {
        if (projection_)
        {
            GridSearch_DestroyProjection(projection_);
        }
    }

    void build(const Grid& grid) override
    {
        grid_ = &grid;

        // Store pointers to grid data (no copies)
        vertices_ = grid.verticesData();
        marks_ = grid.marksData();
        nRows_ = grid.nRows();
        nCols_ = grid.nCols();

        // Create projection configuration
        if (projection_)
        {
            GridSearch_DestroyProjection(projection_);
        }
        projection_ = GridSearch_CreateLinearProjectionWithOffset(
            grid.dx(), grid.dy(),
            grid.offsetX(), grid.offsetY(), grid.offsetZ());
    }

    std::vector<RayHit> intersectParallel(
        const float* rayOrigins,
        int numRays,
        const Eigen::Vector3f& rayDir,
        const Eigen::Isometry3d& rayToGrid,
        float maxDist) const override
    {
        std::vector<RayHit> hits;

        if (numRays == 0 || !projection_)
        {
            return hits;
        }

        // Row-major 4x4 transform for GridSearch
        using RowMajor4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
        RowMajor4d worldToGrid = rayToGrid.matrix();

        // Call GridSearch
        GridSearchHit* results = nullptr;
        int hitCount = 0;

        GridSearch_ComputeIntersectionsParallel(
            rayOrigins,
            rayDir.data(),
            numRays,
            vertices_,
            nRows_,
            nCols_,
            worldToGrid.data(),
            marks_,
            projection_,
            &results,
            &hitCount,
            maxDist);

        // Convert to RayHit with normals
        hits.reserve(hitCount);
        for (int h = 0; h < hitCount; h++)
        {
            const auto& gsHit = results[h];

            RayHit hit;
            hit.rayIndex = gsHit.rayIndex;
            hit.facetId = gsHit.facetId;
            hit.point = Eigen::Map<const Eigen::Vector3f>(gsHit.point);
            hit.normal = grid_->computeFacetNormalFromId(gsHit.facetId);

            hits.push_back(hit);
        }

        // Cleanup GridSearch results
        GridSearch_FreeResults(results);

        return hits;
    }

private:
    GridSearchProjection* projection_ = nullptr;

    // Pointers to Grid data (no copies, Grid must outlive backend)
    const Grid* grid_ = nullptr;
    const float* vertices_ = nullptr;
    const uint8_t* marks_ = nullptr;
    int nRows_ = 0;
    int nCols_ = 0;
};

}  // namespace ICP
