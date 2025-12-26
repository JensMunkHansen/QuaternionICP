#pragma once
/**
 * SE(3) and SO(3) utilities for quaternion-based pose representation.
 *
 * Quaternion convention: Eigen stores [x, y, z, w] internally (coeffs()).
 * SE(3) pose: 7D vector [qx, qy, qz, qw, tx, ty, tz]
 * Tangent space: 6D vector [v_x, v_y, v_z, w_x, w_y, w_z] (translation, rotation)
 *
 * Uses right-multiplication (body/moving frame) convention:
 *   T_new = T * Exp(delta^)
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace ICP
{

// -----------------------------
// SO(3) utilities
// -----------------------------

/**
 * Skew-symmetric matrix from 3-vector: [w]_x such that [w]_x * v = w x v
 */
inline Eigen::Matrix3d skew(const Eigen::Vector3d& w)
{
    Eigen::Matrix3d S;
    S << 0, -w.z(), w.y(),
         w.z(), 0, -w.x(),
        -w.y(), w.x(), 0;
    return S;
}

} // namespace ICP
