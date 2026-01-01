#pragma once

#include <embree4/rtcore.h>

#include <ICP/Grid.h>
#include <ICP/IntersectionBackend.h>

namespace ICP
{

/**
 * Embree-based intersection backend.
 *
 * Uses Intel Embree BVH for ray tracing. Requires triangle extraction
 * from the grid during build(), but provides fast BVH-accelerated queries.
 *
 * TODO: Implement this backend
 */
class EmbreeBackend : public IntersectionBackend
{
public:
    EmbreeBackend()
    {
        device_ = rtcNewDevice(nullptr);
    }

    ~EmbreeBackend() override
    {
        if (scene_)
            rtcReleaseScene(scene_);
        if (device_)
            rtcReleaseDevice(device_);
    }

    void build(const Grid& grid) override
    {
        grid_ = &grid;

        // TODO: Extract triangles from grid and build Embree scene
        // - Get triangle vertices and indices
        // - Create RTCGeometry with RTC_GEOMETRY_TYPE_TRIANGLE
        // - Build scene

        scene_ = rtcNewScene(device_);
        // ... triangle extraction and geometry creation ...
        rtcCommitScene(scene_);
    }

    std::vector<RayHit> intersectParallel(
        const float* rayOrigins,
        int numRays,
        const Eigen::Vector3f& rayDir,
        const Eigen::Isometry3d& rayToGrid,
        float maxDist) const override
    {
        std::vector<RayHit> hits;

        // TODO: Implement Embree ray tracing
        // - Transform rays to grid coordinates
        // - Use rtcIntersect1 or rtcIntersect4/8/16 for SIMD
        // - Convert hits to RayHit with normals

        return hits;
    }

private:
    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;
    const Grid* grid_ = nullptr;

    // Map from Embree geometry primID to Grid facetId
    std::vector<int> primIdToFacetId_;
};

}  // namespace ICP
