// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * @file BackendComparisonTest.cpp
 * @brief Compare Embree and GridSearch intersection backends.
 *
 * Both backends should find matching intersection points for shared hits.
 * Small differences at grid boundaries are expected due to different
 * intersection algorithms (Embree: true 3D, GridSearch: 2D projection).
 */

// Standard C++ headers
#include <unordered_map>

// Catch2 headers
#include <catch2/catch_test_macros.hpp>

// Internal headers
#include <ICP/Config.h>
#include <ICP/GridFactory.h>

#if ICP_USE_EMBREE && ICP_USE_GRIDSEARCH
#include <ICP/EmbreeBackend.h>
#include <ICP/GridSearchBackend.h>

using namespace ICP;

TEST_CASE("Embree vs GridSearch backend", "[backend]")
{
    Grid grid = createTwoHemispheresGrid();

    auto embree = std::make_unique<EmbreeBackend>();
    auto gridsearch = std::make_unique<GridSearchBackend>();
    embree->build(grid);
    gridsearch->build(grid);

    auto origins = grid.getTriangleVertices();
    int nRays = static_cast<int>(origins.size() / 3);
    Eigen::Vector3f rayDir(0, 0, -1);
    float maxDist = 100.0f;

    // Small rotation creates forward/backward hits
    Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
    srcToTgt.linear() = Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitX()).toRotationMatrix();

    auto eHits = embree->intersectParallel(origins.data(), nRays, rayDir, srcToTgt, maxDist);
    auto gHits = gridsearch->intersectParallel(origins.data(), nRays, rayDir, srcToTgt, maxDist);

    // Index Embree hits by rayIndex
    std::unordered_map<int, const RayHit*> embreeByRay;
    for (const auto& h : eHits) embreeByRay[h.rayIndex] = &h;

    // Check if two facetIds are adjacent (same quad or neighboring cell)
    auto isAdjacent = [&grid](int f1, int f2) {
        int idx1 = f1 & 0xFFFF, idx2 = f2 & 0xFFFF;
        int cfg1 = f1 & 0x30000, cfg2 = f2 & 0x30000;
        // Same cell, different triangle config
        if (idx1 == idx2 && cfg1 != cfg2) return true;
        // Neighboring cell (±1 or ±width)
        int diff = std::abs(idx1 - idx2);
        return diff == 1 || diff == grid.width || diff == grid.width + 1 || diff == grid.width - 1;
    };

    // Verify GridSearch hits are found by Embree with matching points/normals
    int found = 0, pointMismatch = 0, normalMismatch = 0, normalErrWithSameFacet = 0, notAdjacent = 0;
    for (const auto& gh : gHits)
    {
        auto it = embreeByRay.find(gh.rayIndex);
        if (it == embreeByRay.end()) continue;

        found++;
        const auto& eh = *it->second;
        if ((eh.point - gh.point).norm() > 1e-4f) pointMismatch++;
        if ((eh.normal - gh.normal).norm() > 1e-4f)
        {
            normalMismatch++;
            if (eh.facetId == gh.facetId) normalErrWithSameFacet++;
            else if (!isAdjacent(eh.facetId, gh.facetId)) notAdjacent++;
        }
    }

    int missed = static_cast<int>(gHits.size()) - found;
    std::printf("GS: %zu, found: %d, missed: %d, pointErr: %d, normalErr: %d (sameFacet: %d, notAdjacent: %d)\n",
        gHits.size(), found, missed, pointMismatch, normalMismatch, normalErrWithSameFacet, notAdjacent);

    CHECK(found > gHits.size() * 0.99);
    CHECK(pointMismatch == 0);
    CHECK(normalErrWithSameFacet == 0);
    CHECK(notAdjacent == 0);
}

#endif
