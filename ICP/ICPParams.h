#pragma once
/**
 * ICP parameters and configuration structures.
 */

#include <cmath>

namespace ICP
{

/**
 * Incidence weighting mode for grazing angle handling.
 */
enum class WeightingMode
{
    Abs,      // weight = |c|
    SqrtAbs   // weight = sqrt(|c|)
};

/**
 * Geometry weighting parameters for incidence-based weighting.
 *
 * Controls how grazing angles are handled in ray-projection ICP.
 * When the ray direction is nearly parallel to the surface (small |n^T d|),
 * the correspondence is either down-weighted or rejected entirely.
 */
struct GeometryWeighting
{
    bool enable_weight = true;   // Apply incidence-based weighting
    bool enable_gate = true;     // Reject correspondences below tau threshold
    double tau = 0.2;            // Threshold for gating (0.1-0.4 typical)
    WeightingMode mode = WeightingMode::SqrtAbs;

    /**
     * Compute incidence weight from denominator c = n^T d.
     *
     * @param c  The denominator (dot product of normal and ray direction)
     * @return   Weight in [0, 1], or 0 if gated out
     */
    double weight(double c) const
    {
        double ac = std::abs(c);
        if (enable_gate && ac < tau)
        {
            return 0.0;
        }
        if (!enable_weight)
        {
            return 1.0;
        }
        ac = std::max(tau, std::min(1.0, ac));
        switch (mode)
        {
            case WeightingMode::Abs:
                return ac;
            case WeightingMode::SqrtAbs:
                return std::sqrt(ac);
        }
        return 1.0;  // fallback
    }
};

} // namespace ICP
