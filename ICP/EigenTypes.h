#pragma once
/**
 * Eigen type aliases for the ICP namespace.
 */

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ICP
{

// Vectors
using Vector2 = Eigen::Vector2d;
using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;
using VectorX = Eigen::VectorXd;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Vector7 = Eigen::Matrix<double, 7, 1>;

// Matrices
using Matrix2 = Eigen::Matrix2d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;
using MatrixX = Eigen::MatrixXd;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Matrix3x4 = Eigen::Matrix<double, 3, 4>;
using Matrix4x3 = Eigen::Matrix<double, 4, 3>;
using Matrix7x6 = Eigen::Matrix<double, 7, 6>;

// Geometry
using Quaternion = Eigen::Quaterniond;
using AngleAxis = Eigen::AngleAxisd;

} // namespace ICP
