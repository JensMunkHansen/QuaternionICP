#pragma once
/**
 * Generic Eigen utilities for random generation and common operations.
 */

#include <ICP/EigenTypes.h>
#include <random>

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

} // namespace ICP
