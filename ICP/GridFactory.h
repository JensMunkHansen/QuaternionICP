#pragma once

// Internal headers
#include <ICP/Grid.h>

namespace ICP
{

/**
 * Create a sinusoidal heightfield grid for testing.
 *
 * First vertex is at (0, 0). Grid extends to ((nx-1)*dx, (ny-1)*dy).
 *
 * @param nx Number of vertices in X direction
 * @param ny Number of vertices in Y direction
 * @param dx Grid spacing in X direction
 * @param dy Grid spacing in Y direction
 * @param amp Amplitude of sinusoidal height variation
 * @param freq Frequency of sinusoidal pattern
 * @return Grid with vertices, marks set (shorter diagonal chosen per quad)
 */
Grid createHeightfieldGrid(int nx = 45, int ny = 45, float dx = 0.05f, float dy = 0.05f,
                           float amp = 0.10f, float freq = 3.0f);

/**
 * Create a grid with two hemispheres (bumps) for testing.
 *
 * This geometry constrains all 6 DOF - no rotational symmetry.
 * First vertex is at (0, 0). Grid is centered around ((nx-1)*dx/2, (ny-1)*dy/2).
 *
 * @param nx Number of vertices in X direction
 * @param ny Number of vertices in Y direction
 * @param dx Grid spacing in X direction
 * @param dy Grid spacing in Y direction
 * @param radius Radius of each hemisphere
 * @param separation Distance between hemisphere centers
 * @return Grid with vertices, marks set (shorter diagonal chosen per quad)
 */
Grid createTwoHemispheresGrid(int nx = 64, int ny = 64, float dx = 0.2f, float dy = 0.2f,
                              float radius = 2.0f, float separation = 6.0f);

} // namespace ICP
