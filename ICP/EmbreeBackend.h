#pragma once

// Embree headers
#include <embree4/rtcore.h>

// Internal headers
#include <ICP/Grid.h>
#include <ICP/IntersectionBackend.h>

namespace ICP
{

/**
 * Embree-based intersection backend.
 *
 * Uses Intel Embree BVH for ray tracing. Requires triangle extraction
 * from the grid during build(), but provides fast BVH-accelerated queries.
 */
class EmbreeBackend : public IntersectionBackend
{
public:
    EmbreeBackend()
    {
        device_ = rtcNewDevice(nullptr);
        if (!device_)
        {
            throw std::runtime_error("Failed to create Embree device");
        }
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

        // Clean up any existing scene
        if (scene_)
        {
            rtcReleaseScene(scene_);
            scene_ = nullptr;
        }
        primIdToFacetId_.clear();

        scene_ = rtcNewScene(device_);

        // Extract triangles from grid
        // Iterate over cells, each cell (quad) can have up to 2 triangles
        const int nRows = grid.height;
        const int nCols = grid.width;
        const uint8_t* marks = grid.marksData();

        // First pass: count triangles
        int triCount = 0;
        for (int j = 0; j < nRows - 1; j++)
        {
            for (int i = 0; i < nCols - 1; i++)
            {
                uint8_t m = marks[j * nCols + i];
                if (m & 0x08) triCount++;  // Upper triangle
                if (m & 0x10) triCount++;  // Lower triangle
            }
        }

        if (triCount == 0)
        {
            rtcCommitScene(scene_);
            return;
        }

        // Create geometry
        RTCGeometry geom = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);

        // Allocate vertex buffer - we'll store unique vertices per cell corner
        // For simplicity, reuse the grid's vertex layout (nRows * nCols vertices)
        float* vertices = (float*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            3 * sizeof(float), nRows * nCols);

        // Copy vertices from grid
        std::memcpy(vertices, grid.verticesData(), nRows * nCols * 3 * sizeof(float));

        // Allocate index buffer
        unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            3 * sizeof(unsigned), triCount);

        // Reserve space for facetId mapping
        primIdToFacetId_.reserve(triCount);

        // Second pass: fill triangle indices
        // Cell vertex layout (matching GridSearch):
        //   i0----i1      (row j)
        //   |      |
        //   i3----i2      (row j+1)
        //
        // Mark bit 0x20 (MarkQuadDiagonal) determines diagonal:
        // - Not set (diag v1-v3): Upper=UL(i0,i1,i3), Lower=LR(i1,i2,i3)
        // - Set (diag v0-v2): Upper=UR(i0,i1,i2), Lower=LL(i0,i2,i3)
        //
        // FacetId encoding (for normal computation):
        // - 0x30000 (UL): v0, v1, v3
        // - 0x20000 (LR): v1, v2, v3
        // - 0x10000 (UR): v0, v1, v2
        // - 0x00000 (LL): v0, v2, v3
        int primId = 0;
        for (int j = 0; j < nRows - 1; j++)
        {
            for (int i = 0; i < nCols - 1; i++)
            {
                int i0 = j * nCols + i;           // upper-left
                int i1 = j * nCols + i + 1;       // upper-right
                int i2 = (j + 1) * nCols + i + 1; // lower-right
                int i3 = (j + 1) * nCols + i;     // lower-left

                uint8_t m = marks[i0];
                bool hasUpper = (m & 0x08) != 0;
                bool hasLower = (m & 0x10) != 0;
                bool diag02 = (m & 0x20) != 0;

                if (hasUpper)
                {
                    if (diag02)
                    {
                        // UR: i0, i1, i2
                        indices[primId * 3 + 0] = i0;
                        indices[primId * 3 + 1] = i1;
                        indices[primId * 3 + 2] = i2;
                        primIdToFacetId_.push_back(i0 | 0x10000);  // UR config
                    }
                    else
                    {
                        // UL: i0, i1, i3
                        indices[primId * 3 + 0] = i0;
                        indices[primId * 3 + 1] = i1;
                        indices[primId * 3 + 2] = i3;
                        primIdToFacetId_.push_back(i0 | 0x30000);  // UL config
                    }
                    primId++;
                }

                if (hasLower)
                {
                    if (diag02)
                    {
                        // LL: i0, i2, i3
                        indices[primId * 3 + 0] = i0;
                        indices[primId * 3 + 1] = i2;
                        indices[primId * 3 + 2] = i3;
                        primIdToFacetId_.push_back(i0 | 0x00000);  // LL config
                    }
                    else
                    {
                        // LR: i1, i2, i3
                        indices[primId * 3 + 0] = i1;
                        indices[primId * 3 + 1] = i2;
                        indices[primId * 3 + 2] = i3;
                        primIdToFacetId_.push_back(i0 | 0x20000);  // LR config
                    }
                    primId++;
                }
            }
        }

        rtcCommitGeometry(geom);
        rtcAttachGeometry(scene_, geom);
        rtcReleaseGeometry(geom);  // Scene holds reference now

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

        if (numRays == 0 || !scene_ || primIdToFacetId_.empty())
        {
            return hits;
        }

        hits.reserve(numRays / 2);  // Assume ~50% hit rate

        // Precompute transform components
        Eigen::Matrix3f R = rayToGrid.rotation().cast<float>();
        Eigen::Vector3f t = rayToGrid.translation().cast<float>();

        // Transform ray direction once (direction is shared)
        Eigen::Vector3f gridRayDir = R * rayDir;

        // Process rays
        for (int i = 0; i < numRays; i++)
        {
            // Get ray origin in source coordinates
            Eigen::Vector3f srcOrigin(
                rayOrigins[i * 3 + 0],
                rayOrigins[i * 3 + 1],
                rayOrigins[i * 3 + 2]);

            // Transform to grid-local coordinates
            Eigen::Vector3f gridOrigin = R * srcOrigin + t;

            // Embree doesn't support negative tnear (intersections behind origin are ignored).
            // For bidirectional search (matching GridSearch), shoot two rays in opposite directions
            // and take the closest hit. This is the recommended approach per Intel:
            // https://community.intel.com/t5/Intel-Embree-Ray-Tracing-Kernels/Bidirectional-Ray-Tracing/m-p/1564491

            // Forward ray (along rayDir)
            // Use small negative tnear to catch surface-on-surface hits at t≈0
            constexpr float epsilon = 1e-4f;
            RTCRayHit fwdRay;
            fwdRay.ray.org_x = gridOrigin.x();
            fwdRay.ray.org_y = gridOrigin.y();
            fwdRay.ray.org_z = gridOrigin.z();
            fwdRay.ray.dir_x = gridRayDir.x();
            fwdRay.ray.dir_y = gridRayDir.y();
            fwdRay.ray.dir_z = gridRayDir.z();
            fwdRay.ray.tnear = -epsilon;  // Allow hits slightly behind origin
            fwdRay.ray.tfar = maxDist;
            fwdRay.ray.mask = -1;
            fwdRay.ray.flags = 0;
            fwdRay.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            fwdRay.hit.primID = RTC_INVALID_GEOMETRY_ID;

            // Backward ray (opposite direction)
            RTCRayHit bwdRay;
            bwdRay.ray.org_x = gridOrigin.x();
            bwdRay.ray.org_y = gridOrigin.y();
            bwdRay.ray.org_z = gridOrigin.z();
            bwdRay.ray.dir_x = -gridRayDir.x();
            bwdRay.ray.dir_y = -gridRayDir.y();
            bwdRay.ray.dir_z = -gridRayDir.z();
            bwdRay.ray.tnear = 0.0f;
            bwdRay.ray.tfar = maxDist;
            bwdRay.ray.mask = -1;
            bwdRay.ray.flags = 0;
            bwdRay.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            bwdRay.hit.primID = RTC_INVALID_GEOMETRY_ID;

            // Trace both rays
            rtcIntersect1(scene_, &fwdRay);
            rtcIntersect1(scene_, &bwdRay);

            // Take the closest hit from either direction
            bool fwdHit = (fwdRay.hit.geomID != RTC_INVALID_GEOMETRY_ID);
            bool bwdHit = (bwdRay.hit.geomID != RTC_INVALID_GEOMETRY_ID);

            RTCRayHit* bestRay = nullptr;
            Eigen::Vector3f hitDir = gridRayDir;

            if (fwdHit && bwdHit)
            {
                // Both hit - take closer one
                bestRay = (fwdRay.ray.tfar <= bwdRay.ray.tfar) ? &fwdRay : &bwdRay;
                if (bestRay == &bwdRay) hitDir = -gridRayDir;
            }
            else if (fwdHit)
            {
                bestRay = &fwdRay;
            }
            else if (bwdHit)
            {
                bestRay = &bwdRay;
                hitDir = -gridRayDir;
            }

            // Check for hit
            if (bestRay)
            {
                RayHit hit;
                hit.rayIndex = i;

                // Compute hit point
                hit.point = gridOrigin + hitDir * bestRay->ray.tfar;

                // Look up facetId and compute normal using Grid's method
                unsigned primID = bestRay->hit.primID;
                if (primID < primIdToFacetId_.size())
                {
                    hit.facetId = primIdToFacetId_[primID];
                    hit.normal = grid_->computeFacetNormalFromId(hit.facetId);
                }
                else
                {
                    // Fallback: use Embree's geometric normal
                    hit.facetId = -1;
                    hit.normal = Eigen::Vector3f(bestRay->hit.Ng_x, bestRay->hit.Ng_y, bestRay->hit.Ng_z).normalized();
                }

                hits.push_back(hit);
            }
        }

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
