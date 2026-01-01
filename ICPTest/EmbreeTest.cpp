// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * @file EmbreeTest.cpp
 * @brief Basic tests to understand and validate Embree 4 API usage.
 */

// Standard C++ headers
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

// Catch2 headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Embree headers
#include <embree4/rtcore.h>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// Internal headers
#include <ICP/Config.h>
#include <ICP/EmbreeBackend.h>
#include <ICP/Grid.h>
#include <ICP/GridFactory.h>
#if ICP_USE_GRIDSEARCH
#include <ICP/GridSearchBackend.h>
#endif

using Catch::Matchers::WithinAbs;

// Simple triangle: flat on XY plane at Z=0
// Vertices: (0,0,0), (1,0,0), (0,1,0)
// Normal should point in +Z direction

TEST_CASE("Embree device and scene creation", "[embree]")
{
    RTCDevice device = rtcNewDevice(nullptr);
    REQUIRE(device != nullptr);

    RTCScene scene = rtcNewScene(device);
    REQUIRE(scene != nullptr);

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

TEST_CASE("Single triangle intersection", "[embree]")
{
    RTCDevice device = rtcNewDevice(nullptr);
    REQUIRE(device != nullptr);

    RTCScene scene = rtcNewScene(device);

    // Create triangle geometry
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Set vertices: triangle in XY plane at Z=0
    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), 3);

    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;  // v0
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;  // v1
    vertices[6] = 0.0f; vertices[7] = 1.0f; vertices[8] = 0.0f;  // v2

    // Set indices
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), 1);

    indices[0] = 0; indices[1] = 1; indices[2] = 2;

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    rtcCommitScene(scene);

    SECTION("Ray from above hits triangle")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.25f;
        rayhit.ray.org_y = 0.25f;
        rayhit.ray.org_z = 1.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        CHECK(rayhit.hit.primID == 0);
        CHECK_THAT(rayhit.ray.tfar, WithinAbs(1.0f, 1e-5f));

        // Embree provides geometric normal (unnormalized)
        Eigen::Vector3f Ng(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
        Ng.normalize();
        INFO("Geometric normal: " << Ng.transpose());
        // For CCW winding (0,1,2), normal should point in +Z or -Z
        CHECK(std::abs(Ng.z()) > 0.99f);
    }

    SECTION("Ray from below hits triangle")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.25f;
        rayhit.ray.org_y = 0.25f;
        rayhit.ray.org_z = -1.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = 1.0f;  // Shooting up
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        // Embree intersects both sides by default
        REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        CHECK_THAT(rayhit.ray.tfar, WithinAbs(1.0f, 1e-5f));
    }

    SECTION("Ray misses triangle")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 2.0f;  // Outside triangle
        rayhit.ray.org_y = 2.0f;
        rayhit.ray.org_z = 1.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        CHECK(rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID);
    }

    SECTION("Hit point computation")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.3f;
        rayhit.ray.org_y = 0.2f;
        rayhit.ray.org_z = 5.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);

        // Compute hit point: origin + t * direction
        Eigen::Vector3f origin(rayhit.ray.org_x, rayhit.ray.org_y, rayhit.ray.org_z);
        Eigen::Vector3f dir(rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z);
        Eigen::Vector3f hitPoint = origin + rayhit.ray.tfar * dir;

        INFO("Hit point: " << hitPoint.transpose());
        CHECK_THAT(hitPoint.x(), WithinAbs(0.3f, 1e-5f));
        CHECK_THAT(hitPoint.y(), WithinAbs(0.2f, 1e-5f));
        CHECK_THAT(hitPoint.z(), WithinAbs(0.0f, 1e-5f));

        // Check barycentric coordinates (u, v)
        // For triangle (v0, v1, v2), hit = (1-u-v)*v0 + u*v1 + v*v2
        float u = rayhit.hit.u;
        float v = rayhit.hit.v;
        INFO("Barycentric: u=" << u << ", v=" << v);
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

TEST_CASE("Two triangles forming a quad", "[embree]")
{
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Quad vertices:
    //   v0(0,0,0) --- v1(1,0,0)
    //      |            |
    //   v3(0,1,0) --- v2(1,1,0)
    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), 4);

    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;   // v0
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;   // v1
    vertices[6] = 1.0f; vertices[7] = 1.0f; vertices[8] = 0.0f;   // v2
    vertices[9] = 0.0f; vertices[10] = 1.0f; vertices[11] = 0.0f; // v3

    // Two triangles: (v0,v1,v3) and (v1,v2,v3)
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), 2);

    indices[0] = 0; indices[1] = 1; indices[2] = 3;  // Upper triangle
    indices[3] = 1; indices[4] = 2; indices[5] = 3;  // Lower triangle

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    SECTION("Hit upper triangle")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.2f;
        rayhit.ray.org_y = 0.2f;
        rayhit.ray.org_z = 1.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        CHECK(rayhit.hit.primID == 0);  // First triangle
    }

    SECTION("Hit lower triangle")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.8f;
        rayhit.ray.org_y = 0.8f;
        rayhit.ray.org_z = 1.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        CHECK(rayhit.hit.primID == 1);  // Second triangle
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

TEST_CASE("Tilted triangle - normal computation", "[embree]")
{
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Triangle tilted 45 degrees around X axis
    // v0 at origin, v1 along X, v2 tilted up in Y-Z plane
    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), 3);

    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;  // v0
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;  // v1
    float angle = M_PI / 4.0f;  // 45 degrees
    vertices[6] = 0.0f;
    vertices[7] = std::cos(angle);
    vertices[8] = std::sin(angle);  // v2

    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), 1);
    indices[0] = 0; indices[1] = 1; indices[2] = 2;

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    // Shoot ray from above
    RTCRayHit rayhit;
    rayhit.ray.org_x = 0.25f;
    rayhit.ray.org_y = 0.25f;
    rayhit.ray.org_z = 2.0f;
    rayhit.ray.dir_x = 0.0f;
    rayhit.ray.dir_y = 0.0f;
    rayhit.ray.dir_z = -1.0f;
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene, &rayhit);

    REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);

    // Compute expected normal: cross product of edges
    Eigen::Vector3f v0(0, 0, 0);
    Eigen::Vector3f v1(1, 0, 0);
    Eigen::Vector3f v2(0, std::cos(angle), std::sin(angle));
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;
    Eigen::Vector3f expectedNormal = edge1.cross(edge2).normalized();

    // Embree's geometric normal
    Eigen::Vector3f Ng(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    Ng.normalize();

    INFO("Expected normal: " << expectedNormal.transpose());
    INFO("Embree Ng: " << Ng.transpose());

    // Normals should match (or be opposite, depending on winding)
    float dot = std::abs(expectedNormal.dot(Ng));
    CHECK_THAT(dot, WithinAbs(1.0f, 1e-5f));

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

TEST_CASE("Rays from grid vertices - self intersection scenario", "[embree]")
{
    // Simulate shooting rays from grid vertices at the same grid
    // This is what happens in ICP when source == target with identity transform

    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Create a simple 3x3 heightfield grid (4 quads, 8 triangles)
    // Grid layout:
    //   v0 -- v1 -- v2
    //   |  \  |  \  |
    //   v3 -- v4 -- v5
    //   |  \  |  \  |
    //   v6 -- v7 -- v8
    //
    // With some height variation to make it interesting

    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), 9);

    // Row 0: y=0
    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;   // v0
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;   // v1
    vertices[6] = 2.0f; vertices[7] = 0.0f; vertices[8] = 0.0f;   // v2
    // Row 1: y=1
    vertices[9] = 0.0f;  vertices[10] = 1.0f; vertices[11] = 0.0f;  // v3
    vertices[12] = 1.0f; vertices[13] = 1.0f; vertices[14] = 0.5f;  // v4 - raised!
    vertices[15] = 2.0f; vertices[16] = 1.0f; vertices[17] = 0.0f;  // v5
    // Row 2: y=2
    vertices[18] = 0.0f; vertices[19] = 2.0f; vertices[20] = 0.0f;  // v6
    vertices[21] = 1.0f; vertices[22] = 2.0f; vertices[23] = 0.0f;  // v7
    vertices[24] = 2.0f; vertices[25] = 2.0f; vertices[26] = 0.0f;  // v8

    // 8 triangles (2 per quad)
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), 8);

    // Quad (0,1,4,3): triangles (0,1,3) and (1,4,3)
    indices[0] = 0; indices[1] = 1; indices[2] = 3;
    indices[3] = 1; indices[4] = 4; indices[5] = 3;
    // Quad (1,2,5,4): triangles (1,2,4) and (2,5,4)
    indices[6] = 1; indices[7] = 2; indices[8] = 4;
    indices[9] = 2; indices[10] = 5; indices[11] = 4;
    // Quad (3,4,7,6): triangles (3,4,6) and (4,7,6)
    indices[12] = 3; indices[13] = 4; indices[14] = 6;
    indices[15] = 4; indices[16] = 7; indices[17] = 6;
    // Quad (4,5,8,7): triangles (4,5,7) and (5,8,7)
    indices[18] = 4; indices[19] = 5; indices[20] = 7;
    indices[21] = 5; indices[22] = 8; indices[23] = 7;

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    SECTION("Ray from raised vertex (v4) shooting down")
    {
        // v4 is at (1, 1, 0.5), shooting down
        // v4 is a vertex of triangles that include (1,4,3), (1,2,4), (3,4,6), (4,5,7), (4,7,6), (2,5,4)
        // Since v4 is raised, shooting straight down should NOT hit any triangle
        // because v4 is the highest point and triangles slope down from it
        RTCRayHit rayhit;
        rayhit.ray.org_x = 1.0f;
        rayhit.ray.org_y = 1.0f;
        rayhit.ray.org_z = 0.5f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = 100.0f;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        // v4 is at the tip, triangles slope away, so ray should miss
        // (unless there's self-intersection at t=0)
        bool hit = (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        CAPTURE(hit, rayhit.ray.tfar, rayhit.hit.primID);

        // Check: if hit, is it a self-intersection (t ≈ 0)?
        if (hit)
        {
            CHECK(rayhit.ray.tfar > 0.01f);  // Should not be self-intersection
        }
    }

    SECTION("Ray from flat vertex (v0) shooting down - expect miss")
    {
        // v0 is at (0, 0, 0), shooting down - should miss (nothing below z=0)
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.0f;
        rayhit.ray.org_y = 0.0f;
        rayhit.ray.org_z = 0.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = 100.0f;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        bool hit = (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        CAPTURE(hit, rayhit.ray.tfar, rayhit.hit.primID);

        // Should miss - no geometry below z=0
        // But might get self-intersection at t=0
        if (hit)
        {
            // If hit, check it's not self-intersection
            CHECK(rayhit.ray.tfar > 0.01f);
        }
    }

    SECTION("Ray from above v4 hitting the raised surface")
    {
        // Ray from above should definitely hit
        RTCRayHit rayhit;
        rayhit.ray.org_x = 1.0f;
        rayhit.ray.org_y = 1.0f;
        rayhit.ray.org_z = 2.0f;  // Above the grid
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = 100.0f;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        // v4 is at z=0.5, so from z=2, distance should be 1.5
        CHECK_THAT(rayhit.ray.tfar, WithinAbs(1.5f, 0.01f));
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

#if ICP_USE_GRIDSEARCH
TEST_CASE("Compare backends with Z translation", "[embree][comparison]")
{
    Grid grid = ICP::createTwoHemispheresGrid();

    auto vertices = grid.getTriangleVertices();
    int nRays = static_cast<int>(vertices.size() / 3);
    Eigen::Vector3f rayDir(0, 0, -1);
    float maxDist = 100.0f;

    auto embreeBackend = std::make_unique<ICP::EmbreeBackend>();
    embreeBackend->build(grid);

    auto gridsearchBackend = std::make_unique<ICP::GridSearchBackend>();
    gridsearchBackend->build(grid);

    // Test with Z translation (matches the failing test case)
    Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
    srcToTgt.translation() = Eigen::Vector3d(0, 0, -1.0);  // Source shifted down relative to target

    auto embreeHits = embreeBackend->intersectParallel(
        vertices.data(), nRays, rayDir, srcToTgt, maxDist);
    auto gridsearchHits = gridsearchBackend->intersectParallel(
        vertices.data(), nRays, rayDir, srcToTgt, maxDist);

    std::printf("\n=== With Z translation (srcToTgt.translation = (0,0,-1)): %d rays ===\n", nRays);
    std::printf("GridSearch hits: %zu\n", gridsearchHits.size());
    std::printf("Embree hits: %zu\n", embreeHits.size());

    // Show sample rays and their transforms
    std::printf("\nSample ray transforms:\n");
    Eigen::Matrix3f R = srcToTgt.rotation().cast<float>();
    Eigen::Vector3f t = srcToTgt.translation().cast<float>();
    for (int i = 0; i < std::min(3, nRays); i++)
    {
        Eigen::Vector3f srcOrigin(vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
        Eigen::Vector3f gridOrigin = R * srcOrigin + t;
        std::printf("  ray %d: src=(%.2f,%.2f,%.2f) -> grid=(%.2f,%.2f,%.2f)\n",
            i, srcOrigin.x(), srcOrigin.y(), srcOrigin.z(),
            gridOrigin.x(), gridOrigin.y(), gridOrigin.z());
    }

    // Show first few hits
    std::printf("\nFirst 3 GridSearch hits:\n");
    for (size_t i = 0; i < std::min(size_t(3), gridsearchHits.size()); i++)
    {
        const auto& h = gridsearchHits[i];
        std::printf("  ray %d: point=(%.3f, %.3f, %.3f)\n",
            h.rayIndex, h.point.x(), h.point.y(), h.point.z());
    }

    std::printf("\nFirst 3 Embree hits:\n");
    for (size_t i = 0; i < std::min(size_t(3), embreeHits.size()); i++)
    {
        const auto& h = embreeHits[i];
        std::printf("  ray %d: point=(%.3f, %.3f, %.3f)\n",
            h.rayIndex, h.point.x(), h.point.y(), h.point.z());
    }

    CHECK(embreeHits.size() == gridsearchHits.size());
}

TEST_CASE("Compare Embree vs GridSearch backends", "[embree][comparison]")
{
    // Create identical grids and compare intersection results from both backends
    Grid grid = ICP::createTwoHemispheresGrid();

    // Get ray origins and direction
    auto vertices = grid.getTriangleVertices();
    int nRays = static_cast<int>(vertices.size() / 3);
    Eigen::Vector3f rayDir(0, 0, -1);
    Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    float maxDist = 100.0f;

    // Create both backends explicitly
    auto embreeBackend = std::make_unique<ICP::EmbreeBackend>();
    embreeBackend->build(grid);

    auto gridsearchBackend = std::make_unique<ICP::GridSearchBackend>();
    gridsearchBackend->build(grid);

    // Run intersection with both
    auto embreeHits = embreeBackend->intersectParallel(
        vertices.data(), nRays, rayDir, identity, maxDist);
    auto gridsearchHits = gridsearchBackend->intersectParallel(
        vertices.data(), nRays, rayDir, identity, maxDist);

    CAPTURE(nRays);
    CAPTURE(embreeHits.size());
    CAPTURE(gridsearchHits.size());

    // Print some sample hits for debugging
    std::printf("\n=== Comparison (identity transform): %d rays ===\n", nRays);
    std::printf("GridSearch hits: %zu\n", gridsearchHits.size());
    std::printf("Embree hits: %zu\n", embreeHits.size());

    // Show first few GridSearch hits
    std::printf("\nFirst 5 GridSearch hits:\n");
    for (size_t i = 0; i < std::min(size_t(5), gridsearchHits.size()); i++)
    {
        const auto& h = gridsearchHits[i];
        std::printf("  ray %d: point=(%.3f, %.3f, %.3f) facetId=%d\n",
            h.rayIndex, h.point.x(), h.point.y(), h.point.z(), h.facetId);
    }

    // Show first few Embree hits
    std::printf("\nFirst 5 Embree hits:\n");
    for (size_t i = 0; i < std::min(size_t(5), embreeHits.size()); i++)
    {
        const auto& h = embreeHits[i];
        std::printf("  ray %d: point=(%.3f, %.3f, %.3f) facetId=%d\n",
            h.rayIndex, h.point.x(), h.point.y(), h.point.z(), h.facetId);
    }

    // With bidirectional search (tnear < 0), both backends should find similar hit counts
    // Note: facetIds may differ slightly at vertices (shared by multiple triangles)
    CHECK(embreeHits.size() == gridsearchHits.size());
}

TEST_CASE("Embree with offset rays (simulate real ICP scenario)", "[embree][comparison]")
{
    // For Embree to find correspondences, rays need to originate ABOVE the target
    // This simulates a real ICP scenario where source and target are at different positions

    Grid grid = ICP::createTwoHemispheresGrid();

    auto vertices = grid.getTriangleVertices();
    int nRays = static_cast<int>(vertices.size() / 3);

    // Offset rays 1 unit above the surface (in +Z)
    std::vector<float> offsetVertices(vertices.size());
    for (int i = 0; i < nRays; i++)
    {
        offsetVertices[i*3 + 0] = vertices[i*3 + 0];
        offsetVertices[i*3 + 1] = vertices[i*3 + 1];
        offsetVertices[i*3 + 2] = vertices[i*3 + 2] + 1.0f;  // Offset in Z
    }

    Eigen::Vector3f rayDir(0, 0, -1);
    Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    float maxDist = 100.0f;

    auto embreeBackend = std::make_unique<ICP::EmbreeBackend>();
    embreeBackend->build(grid);

    auto embreeHits = embreeBackend->intersectParallel(
        offsetVertices.data(), nRays, rayDir, identity, maxDist);

    std::printf("\n=== Embree with offset rays (+1Z): %d rays ===\n", nRays);
    std::printf("Embree hits: %zu\n", embreeHits.size());

    // Show first few hits
    std::printf("\nFirst 5 Embree hits:\n");
    for (size_t i = 0; i < std::min(size_t(5), embreeHits.size()); i++)
    {
        const auto& h = embreeHits[i];
        std::printf("  ray %d: point=(%.3f, %.3f, %.3f)\n",
            h.rayIndex, h.point.x(), h.point.y(), h.point.z());
    }

    // With offset, all rays should now hit the surface
    CAPTURE(embreeHits.size());
    CHECK(embreeHits.size() > 3000);  // Should find most rays
}

TEST_CASE("Embree bidirectional search with negative tnear", "[embree][comparison]")
{
    // Test if Embree supports negative tnear for bidirectional search
    // GridSearch supports t < 0, meaning rays can find surfaces "behind" the origin

    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Simple triangle at z=0
    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), 3);
    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;
    vertices[3] = 2.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;
    vertices[6] = 1.0f; vertices[7] = 2.0f; vertices[8] = 0.0f;

    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), 1);
    indices[0] = 0; indices[1] = 1; indices[2] = 2;

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    SECTION("Ray from ON surface with tnear=0")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.5f;
        rayhit.ray.org_y = 0.5f;
        rayhit.ray.org_z = 0.0f;  // ON the surface
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = 100.0f;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        bool hit = (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        std::printf("\ntnear=0, on surface: hit=%d, tfar=%.6f\n", hit, rayhit.ray.tfar);
    }

    SECTION("Ray from ON surface with negative tnear")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.5f;
        rayhit.ray.org_y = 0.5f;
        rayhit.ray.org_z = 0.0f;  // ON the surface
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = -1.0f;  // Negative tnear!
        rayhit.ray.tfar = 100.0f;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        bool hit = (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        std::printf("tnear=-1, on surface: hit=%d, tfar=%.6f\n", hit, rayhit.ray.tfar);

        // If hit, check the t value
        if (hit)
        {
            std::printf("  -> Found hit at t=%.6f (negative t means behind origin)\n",
                rayhit.ray.tfar);
        }
    }

    SECTION("Ray from BELOW surface shooting UP with negative tnear")
    {
        // Surface at z=0, ray from z=-0.5 shooting up (+z)
        // Should hit at t=0.5
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.5f;
        rayhit.ray.org_y = 0.5f;
        rayhit.ray.org_z = -0.5f;  // Below surface
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = 1.0f;   // Shooting up
        rayhit.ray.tnear = -100.0f;
        rayhit.ray.tfar = 100.0f;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        bool hit = (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        std::printf("tnear=-100, below surface shooting up: hit=%d, tfar=%.6f\n",
            hit, rayhit.ray.tfar);

        REQUIRE(hit);
        CHECK_THAT(rayhit.ray.tfar, WithinAbs(0.5f, 0.01f));
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

/**
 * Systematic bidirectional comparison test.
 *
 * Creates a flat grid and shoots rays from:
 *   - Above the grid (forward hit, t > 0)
 *   - On the grid surface (t ≈ 0)
 *   - Below the grid (backward hit, t < 0 for GridSearch)
 *
 * Both backends should find the same intersections for ICP to work correctly.
 */
TEST_CASE("Systematic bidirectional backend comparison", "[embree][comparison]")
{
    // Create a simple flat grid: 5x5 vertices at z=0
    Grid grid;
    grid.width = 5;
    grid.height = 5;
    grid.verticesAOS.resize(5 * 5 * 3);
    grid.marks.resize(5 * 5);

    // Fill vertices: flat grid on XY plane at z=0
    for (int j = 0; j < 5; j++)
    {
        for (int i = 0; i < 5; i++)
        {
            int idx = j * 5 + i;
            grid.verticesAOS[idx * 3 + 0] = static_cast<float>(i);  // x
            grid.verticesAOS[idx * 3 + 1] = static_cast<float>(j);  // y
            grid.verticesAOS[idx * 3 + 2] = 0.0f;                   // z = 0
            // Mark all cells as having both triangles (except last row/col)
            if (i < 4 && j < 4)
            {
                grid.marks[idx] = 0x03 | 0x08 | 0x10;  // Valid + Vertex + Upper + Lower
            }
            else
            {
                grid.marks[idx] = 0x03;  // Valid + Vertex only (boundary)
            }
        }
    }

    // Create both backends explicitly
    auto embreeBackend = std::make_unique<ICP::EmbreeBackend>();
    embreeBackend->build(grid);

    auto gridsearchBackend = std::make_unique<ICP::GridSearchBackend>();
    gridsearchBackend->build(grid);

    Eigen::Vector3f rayDir(0, 0, -1);  // Shooting down (-Z)
    Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    float maxDist = 100.0f;

    SECTION("Rays from ABOVE grid (t > 0)")
    {
        // Ray origin at center of grid, 2 units above
        float origins[3] = {2.0f, 2.0f, 2.0f};

        auto embreeHits = embreeBackend->intersectParallel(origins, 1, rayDir, identity, maxDist);
        auto gsHits = gridsearchBackend->intersectParallel(origins, 1, rayDir, identity, maxDist);

        std::printf("\n=== ABOVE grid (z=2): ===\n");
        std::printf("GridSearch: %zu hits\n", gsHits.size());
        std::printf("Embree: %zu hits\n", embreeHits.size());

        if (!gsHits.empty())
        {
            std::printf("  GS hit: point=(%.3f,%.3f,%.3f)\n",
                gsHits[0].point.x(), gsHits[0].point.y(), gsHits[0].point.z());
        }
        if (!embreeHits.empty())
        {
            std::printf("  Embree hit: point=(%.3f,%.3f,%.3f)\n",
                embreeHits[0].point.x(), embreeHits[0].point.y(), embreeHits[0].point.z());
        }

        REQUIRE(embreeHits.size() == gsHits.size());
        if (!embreeHits.empty())
        {
            CHECK_THAT(embreeHits[0].point.z(), WithinAbs(0.0f, 0.01f));
        }
    }

    SECTION("Rays from ON grid surface (t ≈ 0)")
    {
        // Ray origin exactly on surface
        float origins[3] = {2.0f, 2.0f, 0.0f};

        auto embreeHits = embreeBackend->intersectParallel(origins, 1, rayDir, identity, maxDist);
        auto gsHits = gridsearchBackend->intersectParallel(origins, 1, rayDir, identity, maxDist);

        std::printf("\n=== ON surface (z=0): ===\n");
        std::printf("GridSearch: %zu hits\n", gsHits.size());
        std::printf("Embree: %zu hits\n", embreeHits.size());

        if (!gsHits.empty())
        {
            std::printf("  GS hit: point=(%.3f,%.3f,%.3f)\n",
                gsHits[0].point.x(), gsHits[0].point.y(), gsHits[0].point.z());
        }
        if (!embreeHits.empty())
        {
            std::printf("  Embree hit: point=(%.3f,%.3f,%.3f)\n",
                embreeHits[0].point.x(), embreeHits[0].point.y(), embreeHits[0].point.z());
        }

        // Both should find the surface
        CHECK(embreeHits.size() == gsHits.size());
    }

    SECTION("Rays from BELOW grid (t < 0 for GridSearch)")
    {
        // Ray origin 2 units below, shooting down
        // For GridSearch with bidirectional search, this should still find surface at t=-2
        float origins[3] = {2.0f, 2.0f, -2.0f};

        auto embreeHits = embreeBackend->intersectParallel(origins, 1, rayDir, identity, maxDist);
        auto gsHits = gridsearchBackend->intersectParallel(origins, 1, rayDir, identity, maxDist);

        std::printf("\n=== BELOW grid (z=-2): ===\n");
        std::printf("GridSearch: %zu hits\n", gsHits.size());
        std::printf("Embree: %zu hits\n", embreeHits.size());

        if (!gsHits.empty())
        {
            std::printf("  GS hit: point=(%.3f,%.3f,%.3f)\n",
                gsHits[0].point.x(), gsHits[0].point.y(), gsHits[0].point.z());
        }
        if (!embreeHits.empty())
        {
            std::printf("  Embree hit: point=(%.3f,%.3f,%.3f)\n",
                embreeHits[0].point.x(), embreeHits[0].point.y(), embreeHits[0].point.z());
        }

        // This is the critical case: GridSearch finds it (t=-2), Embree needs special handling
        CHECK(embreeHits.size() == gsHits.size());
    }

    SECTION("Multiple rays: half above, half below")
    {
        // 4 rays: 2 above (z=1), 2 below (z=-1)
        float origins[] = {
            1.0f, 1.0f, 1.0f,   // above
            2.0f, 2.0f, 1.0f,   // above
            1.0f, 1.0f, -1.0f,  // below
            2.0f, 2.0f, -1.0f   // below
        };

        auto embreeHits = embreeBackend->intersectParallel(origins, 4, rayDir, identity, maxDist);
        auto gsHits = gridsearchBackend->intersectParallel(origins, 4, rayDir, identity, maxDist);

        std::printf("\n=== Mixed (2 above, 2 below): ===\n");
        std::printf("GridSearch: %zu hits\n", gsHits.size());
        std::printf("Embree: %zu hits\n", embreeHits.size());

        for (size_t i = 0; i < gsHits.size(); i++)
        {
            std::printf("  GS ray %d: point=(%.3f,%.3f,%.3f)\n",
                gsHits[i].rayIndex, gsHits[i].point.x(), gsHits[i].point.y(), gsHits[i].point.z());
        }
        for (size_t i = 0; i < embreeHits.size(); i++)
        {
            std::printf("  Embree ray %d: point=(%.3f,%.3f,%.3f)\n",
                embreeHits[i].rayIndex, embreeHits[i].point.x(), embreeHits[i].point.y(), embreeHits[i].point.z());
        }

        CHECK(embreeHits.size() == gsHits.size());
    }
}
#endif  // ICP_USE_GRIDSEARCH

TEST_CASE("Ray with limited tfar", "[embree]")
{
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), 3);
    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;
    vertices[6] = 0.0f; vertices[7] = 1.0f; vertices[8] = 0.0f;

    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), 1);
    indices[0] = 0; indices[1] = 1; indices[2] = 2;

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    SECTION("tfar too short - no hit")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.25f;
        rayhit.ray.org_y = 0.25f;
        rayhit.ray.org_z = 5.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = 2.0f;  // Only 2 units, but triangle is 5 units away
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        CHECK(rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID);
    }

    SECTION("tfar long enough - hit")
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = 0.25f;
        rayhit.ray.org_y = 0.25f;
        rayhit.ray.org_z = 5.0f;
        rayhit.ray.dir_x = 0.0f;
        rayhit.ray.dir_y = 0.0f;
        rayhit.ray.dir_z = -1.0f;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = 10.0f;  // Long enough
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rayhit);

        REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        CHECK_THAT(rayhit.ray.tfar, WithinAbs(5.0f, 1e-5f));
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}
