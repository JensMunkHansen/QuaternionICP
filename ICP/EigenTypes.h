/**
 * @file EigenTypes.h
 * @brief Eigen type aliases for the ICP namespace.
 *
 * Provides convenient type aliases for commonly used Eigen vector,
 * matrix, and geometry types throughout the ICP library.
 */

#pragma once

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ICP
{

/// @name Vector Types
/// @{
using Vector2 = Eigen::Vector2d;    ///< 2D vector (double)
using Vector3 = Eigen::Vector3d;    ///< 3D vector (double)
using Vector4 = Eigen::Vector4d;    ///< 4D vector (double)
using VectorX = Eigen::VectorXd;    ///< Dynamic-size vector (double)
using Vector6 = Eigen::Matrix<double, 6, 1>;  ///< 6D vector (tangent space)
/// @}

/// @brief SE(3) pose as 7D vector: [qx, qy, qz, qw, tx, ty, tz]
using Pose7 = Eigen::Matrix<double, 7, 1>;

/// @name Matrix Types
/// @{
using Matrix2 = Eigen::Matrix2d;    ///< 2x2 matrix (double)
using Matrix3 = Eigen::Matrix3d;    ///< 3x3 matrix (double)
using Matrix4 = Eigen::Matrix4d;    ///< 4x4 matrix (double)
using MatrixX = Eigen::MatrixXd;    ///< Dynamic-size matrix (double)
using Matrix6 = Eigen::Matrix<double, 6, 6>;    ///< 6x6 matrix
using Matrix3x4 = Eigen::Matrix<double, 3, 4>;  ///< 3x4 matrix
using Matrix4x3 = Eigen::Matrix<double, 4, 3>;  ///< 4x3 matrix
using Matrix7x6 = Eigen::Matrix<double, 7, 6>;  ///< 7x6 matrix (Plus Jacobian)
/// @}

/// @name Geometry Types
/// @{
using Quaternion = Eigen::Quaterniond;  ///< Unit quaternion (double)
using AngleAxis = Eigen::AngleAxisd;    ///< Axis-angle rotation (double)
/// @}

} // namespace ICP
