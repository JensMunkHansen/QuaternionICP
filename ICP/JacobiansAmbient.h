#pragma once
/**
 * Ambient (7D) Jacobians for ray-projection ICP.
 *
 * Ceres 2.2-compatible cost functions using 7D ambient parameterization.
 * Policy-based design via template specialization.
 *
 * Residual: r = a/b where
 *   a = n^T (x_transformed - q_surface)
 *   b = n^T d  (ray direction dot normal)
 */

// Ceres headers
#include <ceres/sized_cost_function.h>

// Internal headers
#include <ICP/ICPParams.h>
#include <ICP/SE3.h>

namespace ICP
{

// ============================================================================
// Jacobian policy tags
// ============================================================================

struct RayJacobianSimplified
{
    static constexpr const char* name = "Simplified";
};

struct RayJacobianConsistent
{
    static constexpr const char* name = "Consistent";
};

// ============================================================================
// Forward ray-projection cost - primary template (not defined)
// ============================================================================

template<typename JacobianPolicy>
class ForwardRayCost;

// ============================================================================
// ForwardRayCost<RayJacobianSimplified>
// ============================================================================

template<>
class ForwardRayCost<RayJacobianSimplified> : public ceres::SizedCostFunction<1, 7>
{
public:
    using policy_tag = RayJacobianSimplified;

    ForwardRayCost(const Vector3& pS, const Vector3& qT, const Vector3& nT, const Vector3& dS0,
                   const GeometryWeighting& weighting = GeometryWeighting())
        : pS_(pS), qT_(qT), nT_(nT), dS0_(dS0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pS, const Vector3& qT,
                                       const Vector3& nT, const Vector3& dS0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ForwardRayCost(pS, qT, nT, dS0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* x = parameters[0];

        Quaternion q(x[3], x[0], x[1], x[2]);
        q.normalize();
        Vector3 t(x[4], x[5], x[6]);
        Matrix3 R = q.toRotationMatrix();

        Vector3 xT = R * pS_ + t;
        Vector3 d = R * dS0_;

        double a = nT_.dot(xT - qT_);
        double b = nT_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians && jacobians[0])
        {
            Vector3 dr_dt = (w * nT_) / b;

            // Simplified: dr/dq = da_dq / b (ignores db_dq)
            Matrix3x4 dRp_dq = dRv_dq(q, pS_);
            Eigen::RowVector4d dr_dq = (w * nT_.transpose() * dRp_dq) / b;

            jacobians[0][0] = dr_dq(0);
            jacobians[0][1] = dr_dq(1);
            jacobians[0][2] = dr_dq(2);
            jacobians[0][3] = dr_dq(3);
            jacobians[0][4] = dr_dt(0);
            jacobians[0][5] = dr_dt(1);
            jacobians[0][6] = dr_dt(2);
        }

        return true;
    }

    // Backward compatibility wrapper
    bool operator()(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const
    {
        return Evaluate(parameters, residuals, jacobians);
    }

private:
    Vector3 pS_, qT_, nT_, dS0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// ForwardRayCost<RayJacobianConsistent>
// ============================================================================

template<>
class ForwardRayCost<RayJacobianConsistent> : public ceres::SizedCostFunction<1, 7>
{
public:
    using policy_tag = RayJacobianConsistent;

    ForwardRayCost(const Vector3& pS, const Vector3& qT, const Vector3& nT, const Vector3& dS0,
                   const GeometryWeighting& weighting = GeometryWeighting())
        : pS_(pS), qT_(qT), nT_(nT), dS0_(dS0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pS, const Vector3& qT,
                                       const Vector3& nT, const Vector3& dS0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ForwardRayCost(pS, qT, nT, dS0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* x = parameters[0];

        Quaternion q(x[3], x[0], x[1], x[2]);
        q.normalize();
        Vector3 t(x[4], x[5], x[6]);
        Matrix3 R = q.toRotationMatrix();

        Vector3 xT = R * pS_ + t;
        Vector3 d = R * dS0_;

        double a = nT_.dot(xT - qT_);
        double b = nT_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians && jacobians[0])
        {
            Vector3 dr_dt = (w * nT_) / b;

            // Consistent: full quotient rule
            Matrix3x4 dRp_dq = dRv_dq(q, pS_);
            Matrix3x4 dRd_dq = dRv_dq(q, dS0_);
            Eigen::RowVector4d da_dq = nT_.transpose() * dRp_dq;
            Eigen::RowVector4d db_dq = nT_.transpose() * dRd_dq;
            Eigen::RowVector4d dr_dq = w * (da_dq * b - a * db_dq) / (b * b);

            jacobians[0][0] = dr_dq(0);
            jacobians[0][1] = dr_dq(1);
            jacobians[0][2] = dr_dq(2);
            jacobians[0][3] = dr_dq(3);
            jacobians[0][4] = dr_dt(0);
            jacobians[0][5] = dr_dt(1);
            jacobians[0][6] = dr_dt(2);
        }

        return true;
    }

    // Backward compatibility wrapper
    bool operator()(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const
    {
        return Evaluate(parameters, residuals, jacobians);
    }

private:
    Vector3 pS_, qT_, nT_, dS0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// Reverse ray-projection cost - primary template (not defined)
// ============================================================================

template<typename JacobianPolicy>
class ReverseRayCost;

// ============================================================================
// ReverseRayCost<RayJacobianSimplified>
// ============================================================================

template<>
class ReverseRayCost<RayJacobianSimplified> : public ceres::SizedCostFunction<1, 7>
{
public:
    using policy_tag = RayJacobianSimplified;

    ReverseRayCost(const Vector3& pT, const Vector3& qS, const Vector3& nS, const Vector3& dT0,
                   const GeometryWeighting& weighting = GeometryWeighting())
        : pT_(pT), qS_(qS), nS_(nS), dT0_(dT0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pT, const Vector3& qS,
                                       const Vector3& nS, const Vector3& dT0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ReverseRayCost(pT, qS, nS, dT0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* x = parameters[0];

        Quaternion q(x[3], x[0], x[1], x[2]);
        q.normalize();
        Vector3 t(x[4], x[5], x[6]);
        Matrix3 R = q.toRotationMatrix();

        Vector3 u = pT_ - t;
        Vector3 yS = R.transpose() * u;
        Vector3 d = R.transpose() * dT0_;

        double a = nS_.dot(yS - qS_);
        double b = nS_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians && jacobians[0])
        {
            Vector3 dr_dt = (w * (-R * nS_)) / b;

            // Simplified: dr/dq = da_dq / b (ignores db_dq)
            Matrix3x4 dRTu_dq = dRTv_dq(q, u);
            Eigen::RowVector4d dr_dq = (w * nS_.transpose() * dRTu_dq) / b;

            jacobians[0][0] = dr_dq(0);
            jacobians[0][1] = dr_dq(1);
            jacobians[0][2] = dr_dq(2);
            jacobians[0][3] = dr_dq(3);
            jacobians[0][4] = dr_dt(0);
            jacobians[0][5] = dr_dt(1);
            jacobians[0][6] = dr_dt(2);
        }

        return true;
    }

    // Backward compatibility wrapper
    bool operator()(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const
    {
        return Evaluate(parameters, residuals, jacobians);
    }

private:
    Vector3 pT_, qS_, nS_, dT0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// ReverseRayCost<RayJacobianConsistent>
// ============================================================================

template<>
class ReverseRayCost<RayJacobianConsistent> : public ceres::SizedCostFunction<1, 7>
{
public:
    using policy_tag = RayJacobianConsistent;

    ReverseRayCost(const Vector3& pT, const Vector3& qS, const Vector3& nS, const Vector3& dT0,
                   const GeometryWeighting& weighting = GeometryWeighting())
        : pT_(pT), qS_(qS), nS_(nS), dT0_(dT0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pT, const Vector3& qS,
                                       const Vector3& nS, const Vector3& dT0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ReverseRayCost(pT, qS, nS, dT0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* x = parameters[0];

        Quaternion q(x[3], x[0], x[1], x[2]);
        q.normalize();
        Vector3 t(x[4], x[5], x[6]);
        Matrix3 R = q.toRotationMatrix();

        Vector3 u = pT_ - t;
        Vector3 yS = R.transpose() * u;
        Vector3 d = R.transpose() * dT0_;

        double a = nS_.dot(yS - qS_);
        double b = nS_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians && jacobians[0])
        {
            Vector3 dr_dt = (w * (-R * nS_)) / b;

            // Consistent: full quotient rule
            Matrix3x4 dRTu_dq = dRTv_dq(q, u);
            Matrix3x4 dRTd_dq = dRTv_dq(q, dT0_);
            Eigen::RowVector4d da_dq = nS_.transpose() * dRTu_dq;
            Eigen::RowVector4d db_dq = nS_.transpose() * dRTd_dq;
            Eigen::RowVector4d dr_dq = w * (da_dq * b - a * db_dq) / (b * b);

            jacobians[0][0] = dr_dq(0);
            jacobians[0][1] = dr_dq(1);
            jacobians[0][2] = dr_dq(2);
            jacobians[0][3] = dr_dq(3);
            jacobians[0][4] = dr_dt(0);
            jacobians[0][5] = dr_dt(1);
            jacobians[0][6] = dr_dt(2);
        }

        return true;
    }

    // Backward compatibility wrapper
    bool operator()(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const
    {
        return Evaluate(parameters, residuals, jacobians);
    }

private:
    Vector3 pT_, qS_, nS_, dT0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// Type aliases
// ============================================================================

using ForwardRayCostSimplified = ForwardRayCost<RayJacobianSimplified>;
using ForwardRayCostConsistent = ForwardRayCost<RayJacobianConsistent>;
using ReverseRayCostSimplified = ReverseRayCost<RayJacobianSimplified>;
using ReverseRayCostConsistent = ReverseRayCost<RayJacobianConsistent>;

// Backward compatibility
using RayProjectionFwd = ForwardRayCostSimplified;
using RayProjectionFwdConsistent = ForwardRayCostConsistent;
using RayProjectionRev = ReverseRayCostSimplified;
using RayProjectionRevConsistent = ReverseRayCostConsistent;

} // namespace ICP
