#pragma once
/**
 * Generic Eigen utilities for random generation and common operations.
 */

#include <ICP/EigenTypes.h>
#include <random>
#include <cmath>
#include <numbers>

namespace ICP
{

/**
 * Deterministic seeds for reproducible random perturbations.
 * Using different seeds ensures independence between noise sources.
 */
namespace RandomSeeds
{
    constexpr unsigned int ROTATION_PERTURBATION = 42;
    constexpr unsigned int TRANSLATION_PERTURBATION = 43;
    constexpr unsigned int DEPTH_NOISE = 44;
    constexpr unsigned int GENERAL_PURPOSE = 1337;

    // Test-specific seeds
    constexpr unsigned int JACOBIAN_SIMPLIFIED_TEST = 123;
    constexpr unsigned int JACOBIAN_SANITY_TEST = 999;
    constexpr unsigned int JACOBIAN_EPSILON_TEST = 777;
    constexpr unsigned int JACOBIAN_POLICY_TEST = 888;
    constexpr unsigned int JACOBIAN_REVERSE_TEST = 555;
    constexpr unsigned int JACOBIAN_MULTI_POSE_BASE = 1000;
}

/**
 * @brief Container for random number generators used in pose perturbation.
 *
 * Maintains separate RNG instances for each perturbation type to ensure
 * independence and reproducibility. Each RNG is initialized with a fixed seed.
 */
struct PerturbationRNGs
{
    std::mt19937 rotation;
    std::mt19937 translation;
    std::mt19937 depth;

    /// Initialize with default seeds
    PerturbationRNGs()
        : rotation(RandomSeeds::ROTATION_PERTURBATION)
        , translation(RandomSeeds::TRANSLATION_PERTURBATION)
        , depth(RandomSeeds::DEPTH_NOISE)
    {}

    /// Initialize with custom seeds (for testing different realizations)
    PerturbationRNGs(unsigned int rot_seed, unsigned int trans_seed, unsigned int depth_seed)
        : rotation(rot_seed)
        , translation(trans_seed)
        , depth(depth_seed)
    {}
};

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

/**
 * @brief Add random rotation noise to a quaternion.
 *
 * Applies a random rotation around a random axis with angle drawn from
 * a Gaussian distribution.
 *
 * @param q Original quaternion (unit)
 * @param stddevDeg Standard deviation of rotation angle in degrees
 * @param gen Random number generator
 * @return Perturbed quaternion (normalized)
 */
inline Eigen::Quaterniond perturbRotation(const Eigen::Quaterniond& q, double stddevDeg, std::mt19937& gen)
{
    if (stddevDeg <= 0.0)
        return q;
    std::normal_distribution<double> dist(0.0, 1.0);
    double angle = dist(gen) * stddevDeg * std::numbers::pi / 180.0;
    Eigen::Vector3d axis(dist(gen), dist(gen), dist(gen));
    axis.normalize();
    Eigen::Quaterniond perturbation(Eigen::AngleAxisd(angle, axis));
    return (perturbation * q).normalized();
}

/**
 * @brief Perturb the rotation component of a Pose7 in-place.
 *
 * @param pose Pose to modify [qx, qy, qz, qw, tx, ty, tz]
 * @param stddevDeg Standard deviation of rotation noise in degrees
 * @param gen Random number generator
 */
inline void perturbPoseRotation(Vector7& pose, double stddevDeg, std::mt19937& gen)
{
    if (stddevDeg <= 0.0)
        return;
    Quaternion q(pose[3], pose[0], pose[1], pose[2]);
    Quaternion q_perturbed = perturbRotation(q, stddevDeg, gen);
    pose[0] = q_perturbed.x();
    pose[1] = q_perturbed.y();
    pose[2] = q_perturbed.z();
    pose[3] = q_perturbed.w();
}

/**
 * @brief Perturb the translation component of a Pose7 in-place.
 *
 * @param pose Pose to modify [qx, qy, qz, qw, tx, ty, tz]
 * @param stddev Standard deviation of translation noise
 * @param gen Random number generator
 */
inline void perturbPoseTranslation(Vector7& pose, double stddev, std::mt19937& gen)
{
    if (stddev <= 0.0)
        return;
    std::normal_distribution<double> dist(0.0, stddev);
    pose[4] += dist(gen);
    pose[5] += dist(gen);
    pose[6] += dist(gen);
}

} // namespace ICP
