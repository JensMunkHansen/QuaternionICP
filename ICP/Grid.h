// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once

// Standard C++ headers
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// Forward declarations
namespace ICP
{
class IntersectionBackend;
}

// Pose7: [qx, qy, qz, qw, tx, ty, tz]
using GridPose7 = Eigen::Matrix<double, 7, 1>;

struct Grid
{
    // All special member functions must be defined in Grid.cpp
    // where IntersectionBackend is complete (due to unique_ptr member)
    Grid();
    ~Grid();
    Grid(Grid&&) noexcept;
    Grid& operator=(Grid&&) noexcept;
    Grid(const Grid& other);
    Grid& operator=(const Grid& other);

    // Full grid data in AOS format (width * height * 3 floats) for GridSearch
    std::vector<float> verticesAOS;
    std::vector<uint8_t> marks;
    std::vector<uint8_t> colorsRGB;  // Colors in AOS format (width * height * 3 bytes: R, G, B)

    // Poses: initialPose is loaded from EXR, pose is the current/optimized pose
    Eigen::Isometry3d initialPose = Eigen::Isometry3d::Identity();
    GridPose7 pose = (GridPose7() << 0, 0, 0, 1, 0, 0, 0).finished();  // Identity: quat(0,0,0,1), t(0,0,0)

    int width = 0;
    int height = 0;
    std::string filename;

    bool loadFromExr(const std::string& filepath, bool loadColors = true);

    // Raw pointer access for C interface (GridSearch uses float)
    const float* verticesData() const { return verticesAOS.empty() ? nullptr : verticesAOS.data(); }
    const uint8_t* marksData() const { return marks.empty() ? nullptr : marks.data(); }

    // Get vertex position at index
    Eigen::Vector3f getVertex(int idx) const
    {
        return Eigen::Vector3f(verticesAOS[idx * 3], verticesAOS[idx * 3 + 1], verticesAOS[idx * 3 + 2]);
    }


    /**
     * Compute facet normal on-the-fly from GridSearch facetId.
     *
     * The facetId encodes both vertex index and triangle configuration:
     * - Lower 16 bits: vertex index (upper-left of quad)
     * - Upper bits: FacetConfig (LL=0x00000, UR=0x10000, LR=0x20000, UL=0x30000)
     *
     * Vertex layout (matching GridSearch):
     *   v0 +---+ v1      (row j)
     *      |   |
     *   v3 +---+ v2      (row j+1)
     *
     * Default diagonal (DIAGONAL=0): edge v3-v1 (lower-left to upper-right)
     * - UL (0x30000): v0, v1, v3
     * - LR (0x20000): v1, v2, v3
     *
     * Alternate diagonal (DIAGONAL=1): edge v0-v2 (upper-left to lower-right)
     * - UR (0x10000): v0, v1, v2
     * - LL (0x00000): v0, v2, v3
     */
    Eigen::Vector3f computeFacetNormalFromId(int facetId) const
    {
        int i0 = facetId & 0xFFFF;
        int config = facetId & 0x30000;

        int i1 = i0 + 1;
        int i2 = i0 + width + 1;
        int i3 = i0 + width;

        Eigen::Vector3f v0 = getVertex(i0);
        Eigen::Vector3f v1 = getVertex(i1);
        Eigen::Vector3f v2 = getVertex(i2);
        Eigen::Vector3f v3 = getVertex(i3);

        // Normal formulas from spsDecodeTriangulation25D (CCW winding, +Z for flat surface)
        // v0=upper-left, v1=upper-right, v2=lower-right, v3=lower-left
        Eigen::Vector3f normal;
        switch (config)
        {
            case 0x30000:  // UL (default diagonal, upper)
                normal = (v3 - v1).cross(v0 - v1);
                break;
            case 0x20000:  // LR (default diagonal, lower)
                normal = (v3 - v2).cross(v1 - v2);
                break;
            case 0x10000:  // UR (alternate diagonal, upper)
                normal = (v2 - v1).cross(v0 - v1);
                break;
            default:       // LL (alternate diagonal, lower, 0x00000)
                normal = (v3 - v2).cross(v0 - v2);
                break;
        }
        return normal.normalized();
    }

    // Test which side of diagonal the point is on (inline for performance)
    // Returns true if point is in upper triangle
    static inline bool isUpperTriangle(const Eigen::Vector3f& p,
        const Eigen::Vector3f& v0, const Eigen::Vector3f& v1,
        const Eigen::Vector3f& v2, const Eigen::Vector3f& v3, bool diag02)
    {
        if (diag02)
        {
            // Diagonal v0 -> v2: upper if point on left side
            float sign = (p.x() - v2.x()) * (v0.y() - v2.y())
                       - (v0.x() - v2.x()) * (p.y() - v2.y());
            return sign >= 0;
        }
        else
        {
            // Diagonal v1 -> v3: upper if point on left side
            float sign = (p.x() - v3.x()) * (v1.y() - v3.y())
                       - (v1.x() - v3.x()) * (p.y() - v3.y());
            return sign >= 0;
        }
    }

    // Get color at index (RGB as 0-255)
    Eigen::Vector3i getColor(int idx) const
    {
        if (colorsRGB.empty())
            return Eigen::Vector3i(255, 255, 255);
        return Eigen::Vector3i(colorsRGB[idx * 3], colorsRGB[idx * 3 + 1], colorsRGB[idx * 3 + 2]);
    }


    // Grid dimensions for RaySearch (nRows = height, nCols = width)
    int nRows() const { return height; }
    int nCols() const { return width; }

    // Direction is the z-axis of the initial pose (third column of rotation matrix)
    Eigen::Vector3f direction() const { return initialPose.rotation().col(2).cast<float>(); }

    // Grid spacing (assumes rectilinear grid)
    float dx() const;
    float dy() const;

    // Projection offset: first vertex position (origin of the grid)
    float offsetX() const { return verticesAOS.empty() ? 0.0f : verticesAOS[0]; }
    float offsetY() const { return verticesAOS.empty() ? 0.0f : verticesAOS[1]; }
    float offsetZ() const { return verticesAOS.empty() ? 0.0f : verticesAOS[2]; }

    // Get indices of triangle vertices (Valid + Vertex bits set)
    // subsampleX/Y: stride for subsampling (1=all, 4=every 4th cell, etc.)
    std::vector<int> getTriangleVertexIndices(int subsampleX = 1, int subsampleY = 1) const;

    // Get triangle vertices as flat array (for ray origins)
    // subsampleX/Y: stride for subsampling (1=all, 4=every 4th cell, etc.)
    std::vector<float> getTriangleVertices(int subsampleX = 1, int subsampleY = 1) const;

    // Debug output
    void showInfo() const;
    void showMarksInfo() const;

    /**
     * Axis-aligned bounding box in world coordinates.
     * Used for fast overlap detection between grids.
     */
    struct AABB
    {
        Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f max = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());

        void expand(const Eigen::Vector3f& point)
        {
            min = min.cwiseMin(point);
            max = max.cwiseMax(point);
        }

        bool overlaps(const AABB& other, float margin = 0.0f) const
        {
            return (min.x() - margin <= other.max.x()) && (max.x() + margin >= other.min.x()) &&
                   (min.y() - margin <= other.max.y()) && (max.y() + margin >= other.min.y()) &&
                   (min.z() - margin <= other.max.z()) && (max.z() + margin >= other.min.z());
        }

        Eigen::Vector3f center() const { return (min + max) * 0.5f; }
        Eigen::Vector3f extents() const { return max - min; }
    };

    /**
     * Compute world-space AABB for valid vertices.
     * Uses the current pose to transform vertices to world coordinates.
     */
    AABB computeWorldAABB() const;

    /**
     * Compute world-space AABB using an explicit world pose.
     * Use this during optimization when the current Pose7 differs from grid.pose.
     */
    AABB computeWorldAABB(const Eigen::Isometry3d& worldPose) const;


    /**
     * Perturb the grid pose with random rotation and translation.
     *
     * Simulates initialization error - starting from an imperfect pose estimate.
     *
     * @param rotationDeg Maximum rotation angle in degrees (applied around random axis)
     * @param translationUnits Maximum translation magnitude in grid units
     * @param seed Random seed for reproducibility (0 = random)
     */
    void perturbPose(double rotationDeg, double translationUnits, unsigned int seed = 0);

    /**
     * Get the intersection backend for this grid.
     *
     * The backend is lazily created on first access and cached for subsequent calls.
     * The grid data must not change after first access, or call invalidateBackend().
     *
     * @return Reference to the intersection backend
     */
    ICP::IntersectionBackend& getBackend() const;

    /**
     * Invalidate the cached backend.
     *
     * Call this if grid vertices or marks are modified after backend creation.
     */
    void invalidateBackend();

private:
    // Cached intersection backend (lazy-initialized, thread-safe)
    mutable std::unique_ptr<ICP::IntersectionBackend> backend_;
    mutable std::mutex backendMutex_;
};
