// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

// Standard C++ headers
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

// tinyexr headers
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

// Internal headers
#include <ICP/Grid.h>
#include <ICP/IntersectionBackend.h>
#include <ICP/TriangulationMarks.h>

namespace fs = std::filesystem;

bool Grid::loadFromExr(const std::string& filepath, bool loadColors)
{
    filename = fs::path(filepath).filename().string();

    EXRVersion version;
    int ret = ParseEXRVersionFromFile(&version, filepath.c_str());
    if (ret != TINYEXR_SUCCESS)
    {
        std::cerr << "Error parsing EXR version: " << filepath << "\n";
        return false;
    }

    EXRHeader header;
    InitEXRHeader(&header);
    const char* err = nullptr;

    ret = ParseEXRHeaderFromFile(&header, &version, filepath.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        std::cerr << "Error parsing EXR header: " << (err ? err : "unknown") << "\n";
        if (err)
            FreeEXRErrorMessage(err);
        return false;
    }

    // Read camera pose from cameraPoseRow0..3 (v4f attributes, 4 floats each)
    const float* poseRows[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < header.num_custom_attributes; i++)
    {
        const auto& attr = header.custom_attributes[i];
        if (attr.size == 16)  // 4 floats = v4f
        {
            if (std::strcmp(attr.name, "cameraPoseRow0") == 0)
                poseRows[0] = reinterpret_cast<const float*>(attr.value);
            else if (std::strcmp(attr.name, "cameraPoseRow1") == 0)
                poseRows[1] = reinterpret_cast<const float*>(attr.value);
            else if (std::strcmp(attr.name, "cameraPoseRow2") == 0)
                poseRows[2] = reinterpret_cast<const float*>(attr.value);
            else if (std::strcmp(attr.name, "cameraPoseRow3") == 0)
                poseRows[3] = reinterpret_cast<const float*>(attr.value);
        }
    }

    if (poseRows[0] && poseRows[1] && poseRows[2] && poseRows[3])
    {
        Eigen::Matrix4d poseMatrix;
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                poseMatrix(r, c) = static_cast<double>(poseRows[r][c]);
            }
        }
        initialPose = Eigen::Isometry3d(poseMatrix);
    }

    // Request float for all channels
    for (int i = 0; i < header.num_channels; i++)
    {
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    EXRImage image;
    InitEXRImage(&image);

    ret = LoadEXRImageFromFile(&image, &header, filepath.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        std::cerr << "Error loading EXR image: " << (err ? err : "unknown") << "\n";
        if (err)
            FreeEXRErrorMessage(err);
        FreeEXRHeader(&header);
        return false;
    }

    width = image.width;
    height = image.height;

    // Find X, Y, Z, Marks, R, G, B channel indices
    int x_idx = -1, y_idx = -1, z_idx = -1, marks_idx = -1;
    int r_idx = -1, g_idx = -1, b_idx = -1;
    for (int i = 0; i < header.num_channels; i++)
    {
        const char* name = header.channels[i].name;
        if (std::strcmp(name, "X") == 0)
            x_idx = i;
        else if (std::strcmp(name, "Y") == 0)
            y_idx = i;
        else if (std::strcmp(name, "Z") == 0)
            z_idx = i;
        else if (std::strcmp(name, "Marks") == 0)
            marks_idx = i;
        else if (std::strcmp(name, "R") == 0)
            r_idx = i;
        else if (std::strcmp(name, "G") == 0)
            g_idx = i;
        else if (std::strcmp(name, "B") == 0)
            b_idx = i;
    }

    if (x_idx < 0 || y_idx < 0 || z_idx < 0)
    {
        std::cerr << "Missing X/Y/Z channels in " << filepath << "\n";
        FreeEXRImage(&image);
        FreeEXRHeader(&header);
        return false;
    }

    const float* x_data = reinterpret_cast<const float*>(image.images[x_idx]);
    const float* y_data = reinterpret_cast<const float*>(image.images[y_idx]);
    const float* z_data = reinterpret_cast<const float*>(image.images[z_idx]);
    const float* marks_data =
      marks_idx >= 0 ? reinterpret_cast<const float*>(image.images[marks_idx]) : nullptr;
    const float* r_data = (loadColors && r_idx >= 0) ? reinterpret_cast<const float*>(image.images[r_idx]) : nullptr;
    const float* g_data = (loadColors && g_idx >= 0) ? reinterpret_cast<const float*>(image.images[g_idx]) : nullptr;
    const float* b_data = (loadColors && b_idx >= 0) ? reinterpret_cast<const float*>(image.images[b_idx]) : nullptr;
    bool hasColors = loadColors && r_data && g_data && b_data;

    // Store full grid in AOS format (x,y,z, x,y,z, ...)
    // NO column flip: X decreases with column (as stored in EXR).
    // This results in negative dx, which GridSearch LinearProjection handles correctly.
    // The pose stored in EXR is then consistent with the vertex data.
    verticesAOS.resize(width * height * 3);
    marks.resize(width * height);
    if (hasColors)
        colorsRGB.resize(width * height * 3);

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int idx = row * width + col;
            verticesAOS[idx * 3 + 0] = x_data[idx];
            verticesAOS[idx * 3 + 1] = y_data[idx];
            verticesAOS[idx * 3 + 2] = z_data[idx];
            marks[idx] = marks_data ? static_cast<uint8_t>(marks_data[idx]) : 0;
            if (hasColors)
            {
                // EXR colors are typically 0-1 float, convert to 0-255 uint8
                colorsRGB[idx * 3 + 0] =
                  static_cast<uint8_t>(std::clamp(r_data[idx] * 255.0f, 0.0f, 255.0f));
                colorsRGB[idx * 3 + 1] =
                  static_cast<uint8_t>(std::clamp(g_data[idx] * 255.0f, 0.0f, 255.0f));
                colorsRGB[idx * 3 + 2] =
                  static_cast<uint8_t>(std::clamp(b_data[idx] * 255.0f, 0.0f, 255.0f));
            }
        }
    }

    FreeEXRImage(&image);
    FreeEXRHeader(&header);

    // Initialize pose (Pose7) from initialPose (Isometry3d)
    Eigen::Quaterniond q(initialPose.rotation());
    q.normalize();
    pose << q.x(), q.y(), q.z(), q.w(),
            initialPose.translation().x(),
            initialPose.translation().y(),
            initialPose.translation().z();

    return true;
}

float Grid::dx() const
{
    if (width < 2)
        return 1.0f;
    // Compute spacing from first two points in first row
    // X decreases with column (as stored in EXR), so dx is NEGATIVE
    // GridSearch LinearProjection handles negative dx correctly:
    //   u = (x - ox) / dx  where dx < 0
    float x0 = verticesAOS[0];
    float x1 = verticesAOS[3]; // Next point (offset by 3 floats)
    return x1 - x0;
}

float Grid::dy() const
{
    if (height < 2)
        return 1.0f;
    // Compute spacing from first point and point one row down
    // Y increases with row, so dy is positive
    float y0 = verticesAOS[1];
    float y1 = verticesAOS[width * 3 + 1]; // Point one row down
    return y1 - y0;
}

std::vector<int> Grid::getTriangleVertexIndices(int subsampleX, int subsampleY) const
{
    std::vector<int> indices;
    int nRows = height;
    int nCols = width;
    bool isSubsampled = (subsampleX > 1 || subsampleY > 1);

    if (isSubsampled)
    {
        // For subsampled grids, iterate with stride and check MarkPointVertex
        for (int j = 0; j < nRows; j += subsampleY)
        {
            for (int i = 0; i < nCols; i += subsampleX)
            {
                int idx = j * nCols + i;
                if (TriangulationMarks::IsValidVertex(marks[idx]))
                {
                    indices.push_back(idx);
                }
            }
        }
    }
    else
    {
        // For full resolution, check all vertices for MarkPointVertex
        for (int i = 0; i < nRows * nCols; i++)
        {
            if (TriangulationMarks::IsValidVertex(marks[i]))
            {
                indices.push_back(i);
            }
        }
    }
    return indices;
}

std::vector<float> Grid::getTriangleVertices(int subsampleX, int subsampleY) const
{
    auto indices = getTriangleVertexIndices(subsampleX, subsampleY);
    std::vector<float> vertices;
    vertices.reserve(indices.size() * 3);
    for (int i : indices)
    {
        vertices.push_back(verticesAOS[i * 3 + 0]);
        vertices.push_back(verticesAOS[i * 3 + 1]);
        vertices.push_back(verticesAOS[i * 3 + 2]);
    }
    return vertices;
}

void Grid::showInfo() const
{
    std::printf("  === %s ===\n", filename.c_str());
    std::printf("  Dimensions: %dx%d (%d points)\n", width, height, width * height);
    std::printf("  Spacing: dx=%.6f, dy=%.6f\n", dx(), dy());
    std::printf("  First vertex[0,0]: (%.6f, %.6f, %.6f)\n", offsetX(), offsetY(), offsetZ());

    if (width > 1)
    {
        std::printf(
          "  Vertex[0,1]: (%.6f, %.6f, %.6f)\n", verticesAOS[3], verticesAOS[4], verticesAOS[5]);
    }
    if (height > 1)
    {
        std::printf("  Vertex[1,0]: (%.6f, %.6f, %.6f)\n", verticesAOS[width * 3],
          verticesAOS[width * 3 + 1], verticesAOS[width * 3 + 2]);
    }

    Eigen::Matrix4d m = initialPose.matrix();
    std::printf("  Initial pose matrix:\n");
    for (int r = 0; r < 4; r++)
    {
        std::printf("    [%.6f, %.6f, %.6f, %.6f]\n", m(r, 0), m(r, 1), m(r, 2), m(r, 3));
    }
}

void Grid::showMarksInfo() const
{
    int nRows = height;
    int nCols = width;

    // Count triangles from cell marks (upper-left corner of each cell)
    int triangleCount = 0;
    int upperCount = 0, lowerCount = 0;
    for (int j = 0; j < nRows - 1; j++)
    {
        for (int i = 0; i < nCols - 1; i++)
        {
            uint8_t m = marks[j * nCols + i];
            if (m & 0x08)
            {
                triangleCount++;
                upperCount++;
            } // Upper triangle
            if (m & 0x10)
            {
                triangleCount++;
                lowerCount++;
            } // Lower triangle
        }
    }

    int triangleVertices = static_cast<int>(getTriangleVertexIndices().size());

    std::printf("  %s: %d triangles (upper=%d, lower=%d), %d triangle vertices\n", filename.c_str(),
      triangleCount, upperCount, lowerCount, triangleVertices);
}

void Grid::perturbPose(double rotationDeg, double translationUnits, unsigned int seed)
{
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);

    // Random rotation axis (normalized)
    Eigen::Vector3d axis(uniform(gen), uniform(gen), uniform(gen));
    axis.normalize();

    // Random rotation angle
    double angle = rotationDeg * M_PI / 180.0;
    Eigen::AngleAxisd rotation(angle, axis);

    // Random translation direction (normalized) scaled by magnitude
    Eigen::Vector3d transDir(uniform(gen), uniform(gen), uniform(gen));
    transDir.normalize();
    Eigen::Vector3d translation = transDir * translationUnits;

    // Apply perturbation: new_pose = perturbation * original_pose
    Eigen::Isometry3d perturbation = Eigen::Isometry3d::Identity();
    perturbation.linear() = rotation.toRotationMatrix();
    perturbation.translation() = translation;

    // Convert Pose7 to Isometry3d, perturb, convert back
    Eigen::Quaterniond q(pose[3], pose[0], pose[1], pose[2]);
    Eigen::Vector3d t(pose[4], pose[5], pose[6]);
    Eigen::Isometry3d currentPose = Eigen::Isometry3d::Identity();
    currentPose.linear() = q.toRotationMatrix();
    currentPose.translation() = t;

    Eigen::Isometry3d newPose = perturbation * currentPose;
    Eigen::Quaterniond qNew(newPose.rotation());
    qNew.normalize();
    pose << qNew.x(), qNew.y(), qNew.z(), qNew.w(),
            newPose.translation().x(), newPose.translation().y(), newPose.translation().z();
}

Grid::AABB Grid::computeWorldAABB() const
{
    // Convert Pose7 to Isometry3d
    Eigen::Quaterniond q(pose[3], pose[0], pose[1], pose[2]);
    Eigen::Vector3d t(pose[4], pose[5], pose[6]);
    Eigen::Isometry3d worldPose = Eigen::Isometry3d::Identity();
    worldPose.linear() = q.toRotationMatrix();
    worldPose.translation() = t;
    return computeWorldAABB(worldPose);
}

Grid::AABB Grid::computeWorldAABB(const Eigen::Isometry3d& worldPose) const
{
    AABB aabb;

    // Iterate over valid vertices only
    const int nVerts = width * height;
    for (int i = 0; i < nVerts; i++)
    {
        // Check if vertex is valid (Valid + Vertex bits)
        if ((marks[i] & 0x03) != 0x03)
            continue;

        // Get vertex in local coordinates
        Eigen::Vector3f localPt = getVertex(i);

        // Transform to world coordinates
        Eigen::Vector3f worldPt = (worldPose * localPt.cast<double>()).cast<float>();

        aabb.expand(worldPt);
    }

    return aabb;
}

ICP::IntersectionBackend& Grid::getBackend() const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    if (!backend_)
    {
        backend_ = ICP::createIntersectionBackend();
        backend_->build(*this);
    }
    return *backend_;
}

void Grid::invalidateBackend()
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    backend_.reset();
}

// Special member functions - must be defined here where IntersectionBackend is complete
Grid::Grid() = default;
Grid::~Grid() = default;

Grid::Grid(Grid&& other) noexcept
    : verticesAOS(std::move(other.verticesAOS))
    , marks(std::move(other.marks))
    , colorsRGB(std::move(other.colorsRGB))
    , initialPose(std::move(other.initialPose))
    , pose(std::move(other.pose))
    , width(other.width)
    , height(other.height)
    , filename(std::move(other.filename))
    , backend_(std::move(other.backend_))
    // backendMutex_ is default-constructed (not moved)
{
}

Grid& Grid::operator=(Grid&& other) noexcept
{
    if (this != &other)
    {
        verticesAOS = std::move(other.verticesAOS);
        marks = std::move(other.marks);
        colorsRGB = std::move(other.colorsRGB);
        initialPose = std::move(other.initialPose);
        pose = std::move(other.pose);
        width = other.width;
        height = other.height;
        filename = std::move(other.filename);
        backend_ = std::move(other.backend_);
        // backendMutex_ stays as-is (new mutex for this object)
    }
    return *this;
}

Grid::Grid(const Grid& other)
    : verticesAOS(other.verticesAOS)
    , marks(other.marks)
    , colorsRGB(other.colorsRGB)
    , initialPose(other.initialPose)
    , pose(other.pose)
    , width(other.width)
    , height(other.height)
    , filename(other.filename)
    , backend_(nullptr)  // Don't copy backend - will be lazily rebuilt
{
}

Grid& Grid::operator=(const Grid& other)
{
    if (this != &other)
    {
        verticesAOS = other.verticesAOS;
        marks = other.marks;
        colorsRGB = other.colorsRGB;
        initialPose = other.initialPose;
        pose = other.pose;
        width = other.width;
        height = other.height;
        filename = other.filename;
        invalidateBackend();  // Thread-safe backend invalidation
    }
    return *this;
}
