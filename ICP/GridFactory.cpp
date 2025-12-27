// Standard C++ headers
#include <cmath>

// Internal headers
#include <ICP/GridFactory.h>
#include <ICP/TriangulationMarks.h>

namespace ICP
{

Grid createHeightfieldGrid(int nx, int ny, float dx, float dy, float amp, float freq)
{
    Grid grid;
    grid.width = nx;
    grid.height = ny;

    // Allocate vertices and marks
    int nVerts = nx * ny;
    grid.verticesAOS.resize(nVerts * 3);
    grid.marks.resize(nVerts, 0);

    // Generate vertices: first vertex at (0,0), Z = amp * sin(freq*X) * sin(freq*Y)
    // Phase is relative to grid center so it matches Python (centered at origin)
    float centerX = (nx - 1) * dx * 0.5f;
    float centerY = (ny - 1) * dy * 0.5f;

    for (int row = 0; row < ny; row++)
    {
        for (int col = 0; col < nx; col++)
        {
            int idx = row * nx + col;
            float x = col * dx;
            float y = row * dy;
            float z = amp * std::sin(freq * (x - centerX)) * std::sin(freq * (y - centerY));

            grid.verticesAOS[idx * 3 + 0] = x;
            grid.verticesAOS[idx * 3 + 1] = y;
            grid.verticesAOS[idx * 3 + 2] = z;

            // Mark all vertices as valid
            grid.marks[idx] = TriangulationMarks::MarkValidVertex;
        }
    }

    // Set triangle marks for each quad cell
    // Cell layout:
    //   v0 (row, col)     --- v1 (row, col+1)
    //   |                      |
    //   v3 (row+1, col)   --- v2 (row+1, col+1)
    //
    // Diagonal options:
    //   bit5=0: diagonal v1-v3, creates UL (v0,v1,v3) and LR (v1,v2,v3)
    //   bit5=1: diagonal v0-v2, creates UR (v0,v1,v2) and LL (v0,v2,v3)

    for (int row = 0; row < ny - 1; row++)
    {
        for (int col = 0; col < nx - 1; col++)
        {
            int i0 = row * nx + col;
            int i1 = row * nx + col + 1;
            int i2 = (row + 1) * nx + col + 1;
            int i3 = (row + 1) * nx + col;

            Eigen::Vector3f v0 = grid.getVertex(i0);
            Eigen::Vector3f v1 = grid.getVertex(i1);
            Eigen::Vector3f v2 = grid.getVertex(i2);
            Eigen::Vector3f v3 = grid.getVertex(i3);

            // Compute diagonal lengths
            float diag02 = (v2 - v0).norm();  // v0-v2 diagonal
            float diag13 = (v3 - v1).norm();  // v1-v3 diagonal

            // Choose shorter diagonal
            uint8_t diagBit = (diag02 < diag13) ? TriangulationMarks::MarkQuadDiagonal : 0;

            // Set upper and lower triangle bits on the upper-left vertex of the quad
            grid.marks[i0] |= TriangulationMarks::MarkQuadUpper
                           |  TriangulationMarks::MarkQuadLower
                           |  diagBit;
        }
    }

    return grid;
}

Grid createTwoHemispheresGrid(int nx, int ny, float dx, float dy, float radius, float separation)
{
    Grid grid;
    grid.width = nx;
    grid.height = ny;

    int nVerts = nx * ny;
    grid.verticesAOS.resize(nVerts * 3);
    grid.marks.resize(nVerts, 0);

    // Grid centered at ((nx-1)*dx/2, (ny-1)*dy/2)
    float centerX = (nx - 1) * dx * 0.5f;
    float centerY = (ny - 1) * dy * 0.5f;

    // Two hemisphere centers (along X axis, symmetric around grid center)
    float cx1 = centerX - separation * 0.5f;
    float cx2 = centerX + separation * 0.5f;
    float cy = centerY;

    for (int row = 0; row < ny; row++)
    {
        for (int col = 0; col < nx; col++)
        {
            int idx = row * nx + col;
            float x = col * dx;
            float y = row * dy;

            // Distance squared to each hemisphere center
            float d1_sq = (x - cx1) * (x - cx1) + (y - cy) * (y - cy);
            float d2_sq = (x - cx2) * (x - cx2) + (y - cy) * (y - cy);

            float z = 0.0f;
            if (d1_sq < radius * radius)
                z = std::sqrt(radius * radius - d1_sq);
            else if (d2_sq < radius * radius)
                z = std::sqrt(radius * radius - d2_sq);

            grid.verticesAOS[idx * 3 + 0] = x;
            grid.verticesAOS[idx * 3 + 1] = y;
            grid.verticesAOS[idx * 3 + 2] = z;

            grid.marks[idx] = TriangulationMarks::MarkValidVertex;
        }
    }

    // Set triangle marks - choose shorter diagonal for each quad
    for (int row = 0; row < ny - 1; row++)
    {
        for (int col = 0; col < nx - 1; col++)
        {
            int i0 = row * nx + col;
            int i1 = row * nx + col + 1;
            int i2 = (row + 1) * nx + col + 1;
            int i3 = (row + 1) * nx + col;

            Eigen::Vector3f v0 = grid.getVertex(i0);
            Eigen::Vector3f v1 = grid.getVertex(i1);
            Eigen::Vector3f v2 = grid.getVertex(i2);
            Eigen::Vector3f v3 = grid.getVertex(i3);

            float diag02 = (v2 - v0).norm();
            float diag13 = (v3 - v1).norm();

            uint8_t diagBit = (diag02 < diag13) ? TriangulationMarks::MarkQuadDiagonal : 0;

            grid.marks[i0] |= TriangulationMarks::MarkQuadUpper
                           |  TriangulationMarks::MarkQuadLower
                           |  diagBit;
        }
    }

    return grid;
}

} // namespace ICP
