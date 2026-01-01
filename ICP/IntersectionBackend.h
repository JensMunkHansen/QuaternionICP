// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once

// Standard C++ headers
#include <memory>
#include <vector>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// Forward declarations
struct Grid;

namespace ICP
{

/**
 * Result of a ray-mesh intersection.
 */
struct RayHit
{
    Eigen::Vector3f point;   // Intersection point in grid-local coordinates
    Eigen::Vector3f normal;  // Surface normal at intersection
    int rayIndex;            // Index of the ray that hit
    int facetId;             // Triangle/facet identifier for debugging
};

/**
 * Abstract interface for ray-mesh intersection backends.
 *
 * This allows swapping between different intersection implementations:
 * - GridSearchBackend: Uses linear projection for structured grids (fast, no BVH)
 * - EmbreeBackend: Uses Intel Embree BVH for general meshes (requires triangle extraction)
 *
 * The backend is built once per grid and cached for repeated intersection queries.
 */
class IntersectionBackend
{
public:
    virtual ~IntersectionBackend() = default;

    /**
     * Build/prepare acceleration structure from grid.
     *
     * For GridSearch: Just caches projection configuration (O(1)).
     * For Embree: Extracts triangles and builds BVH (O(n)).
     *
     * @param grid The grid to build the backend for
     */
    virtual void build(const Grid& grid) = 0;

    /**
     * Shoot parallel rays and find intersections.
     *
     * All rays share the same direction (common in ICP ray-casting).
     *
     * @param rayOrigins   Ray origin positions (N×3 floats, AOS layout)
     * @param numRays      Number of rays
     * @param rayDir       Shared ray direction (normalized)
     * @param rayToGrid    Transform from ray coordinate system to grid-local coordinates
     * @param maxDist      Maximum ray travel distance
     * @return Vector of hits (only rays that intersected the mesh)
     */
    virtual std::vector<RayHit> intersectParallel(
        const float* rayOrigins,
        int numRays,
        const Eigen::Vector3f& rayDir,
        const Eigen::Isometry3d& rayToGrid,
        float maxDist) const = 0;
};

/**
 * Factory function to create the appropriate backend.
 *
 * Returns GridSearchBackend by default.
 * When USE_EMBREE is defined at compile time, returns EmbreeBackend.
 */
std::unique_ptr<IntersectionBackend> createIntersectionBackend();

}  // namespace ICP
