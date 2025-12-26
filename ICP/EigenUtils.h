#pragma once
/**
 * Generic Eigen utilities for random generation and common operations.
 */

#include <ICP/EigenTypes.h>
#include <random>
#include <cmath>

namespace ICP
{

/// Generate random unit quaternion
inline Quaternion randomQuaternion(std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    Quaternion q(dist(rng), dist(rng), dist(rng), dist(rng));
    q.normalize();
    return q;
}

/// Generate random unit vector in R^N
template <int N>
Eigen::Matrix<double, N, 1> randomUnitVector(std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::Matrix<double, N, 1> v;
    for (int i = 0; i < N; ++i)
        v[i] = dist(rng);
    return v.normalized();
}

/// Generate random vector in R^3 with uniform distribution in [-scale, scale]
inline Vector3 randomVector3(std::mt19937& rng, double scale = 1.0)
{
    std::uniform_real_distribution<double> dist(-scale, scale);
    return Vector3(dist(rng), dist(rng), dist(rng));
}

/// Identity pose: [qx=0, qy=0, qz=0, qw=1, tx=0, ty=0, tz=0]
inline Vector7 identityPose()
{
    Vector7 pose;
    pose << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    return pose;
}

/// Create pose from translation only (identity rotation)
inline Vector7 translationPose(double tx, double ty, double tz)
{
    Vector7 pose;
    pose << 0.0, 0.0, 0.0, 1.0, tx, ty, tz;
    return pose;
}

/// Create pose from axis-angle rotation (no translation)
inline Vector7 rotationPose(const Vector3& axisAngle)
{
    double theta = axisAngle.norm();
    Vector7 pose;
    if (theta < 1e-12)
    {
        pose << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    }
    else
    {
        Vector3 axis = axisAngle / theta;
        double half = 0.5 * theta;
        double s = std::sin(half);
        pose << axis.x() * s, axis.y() * s, axis.z() * s, std::cos(half), 0.0, 0.0, 0.0;
    }
    return pose;
}

} // namespace ICP
