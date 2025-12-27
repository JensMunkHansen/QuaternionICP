#pragma once
/**
 * Two-pose ambient (7D) Jacobians for ray-projection ICP.
 *
 * Ceres cost functions with <1, 7, 7> dimensions for optimizing two poses.
 * Policy-based design via template specialization.
 *
 * Forward: ray from frame A hits surface in frame B
 * Reverse: ray from frame B hits surface in frame A
 *
 * Residual: r = a/b where
 *   a = n^T (x_transformed - q_surface)
 *   b = n^T d  (ray direction dot normal)
 */

// Ceres headers
#include <ceres/sized_cost_function.h>

// Internal headers
#include <ICP/ICPParams.h>
#include <ICP/JacobiansAmbient.h>
#include <ICP/SE3.h>

namespace ICP
{

// ============================================================================
// Forward two-pose ray-projection cost - primary template (not defined)
// ============================================================================

template<typename JacobianPolicy>
class ForwardRayCostTwoPose;

// ============================================================================
// ForwardRayCostTwoPose<RayJacobianSimplified>
//
// Ray from frame A -> surface in frame B
// ============================================================================

template<>
class ForwardRayCostTwoPose<RayJacobianSimplified> : public ceres::SizedCostFunction<1, 7, 7>
{
public:
    using policy_tag = RayJacobianSimplified;

    /**
     * @param pA    Point in frame A (ray origin)
     * @param qB    Target surface point in frame B
     * @param nB    Target surface normal in frame B
     * @param dA0   Ray direction in frame A (typically [0,0,-1])
     * @param weighting Geometry weighting parameters
     */
    ForwardRayCostTwoPose(const Vector3& pA, const Vector3& qB, const Vector3& nB,
                          const Vector3& dA0,
                          const GeometryWeighting& weighting = GeometryWeighting())
        : pA_(pA), qB_(qB), nB_(nB), dA0_(dA0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pA, const Vector3& qB,
                                       const Vector3& nB, const Vector3& dA0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ForwardRayCostTwoPose(pA, qB, nB, dA0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* xA = parameters[0];
        const double* xB = parameters[1];

        // Extract pose A
        Quaternion qA(xA[3], xA[0], xA[1], xA[2]);
        qA.normalize();
        Vector3 tA(xA[4], xA[5], xA[6]);
        Matrix3 RA = qA.toRotationMatrix();

        // Extract pose B
        Quaternion qBpose(xB[3], xB[0], xB[1], xB[2]);
        qBpose.normalize();
        Vector3 tB(xB[4], xB[5], xB[6]);
        Matrix3 RB = qBpose.toRotationMatrix();

        // Transform point from A to world, then to B
        Vector3 U = RA * pA_ + (tA - tB);    // pA in world, relative to tB
        Vector3 xBv = RB.transpose() * U;     // pA in frame B

        // Transform ray direction from A to B
        Vector3 u_d = RA * dA0_;              // ray in world
        Vector3 d = RB.transpose() * u_d;     // ray in frame B

        double a = nB_.dot(xBv - qB_);
        double b = nB_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians)
        {
            // Jacobian w.r.t. pose A
            if (jacobians[0])
            {
                // dr/dtA = (nB^T @ RB^T) / b
                Vector3 dr_dtA = (w * RB.transpose().transpose() * nB_) / b;
                // Note: RB^T^T = RB, and nB is already in B frame
                // Actually: dr/dtA = nB^T @ RB^T / b, as 1x3
                // In vector form for our storage: (RB * nB) / b... wait let me check
                // da/dtA = nB^T @ d(RB^T @ U)/dtA = nB^T @ RB^T @ dU/dtA = nB^T @ RB^T @ I
                // So dr/dtA = nB^T @ RB^T / b (row vector 1x3)
                // Transposed for storage: RB @ nB / b... no wait
                // Let me be careful: we store jacobians as [dr/dx0, dr/dx1, ...]
                // dr/dtA is a 1x3, we need to store elements 4,5,6
                Eigen::RowVector3d dr_dtA_row = (w * nB_.transpose() * RB.transpose()) / b;

                // dr/dqA: simplified (ignore db_dq)
                // dx_dqA = RB^T @ dR(qA)*pA_dq
                Matrix3x4 dRApA_dqA = dRv_dq(qA, pA_);
                Matrix3x4 dx_dqA = RB.transpose() * dRApA_dqA;
                Eigen::RowVector4d da_dqA = nB_.transpose() * dx_dqA;
                Eigen::RowVector4d dr_dqA = (w * da_dqA) / b;

                jacobians[0][0] = dr_dqA(0);
                jacobians[0][1] = dr_dqA(1);
                jacobians[0][2] = dr_dqA(2);
                jacobians[0][3] = dr_dqA(3);
                jacobians[0][4] = dr_dtA_row(0);
                jacobians[0][5] = dr_dtA_row(1);
                jacobians[0][6] = dr_dtA_row(2);
            }

            // Jacobian w.r.t. pose B
            if (jacobians[1])
            {
                // dr/dtB = nB^T @ (-RB^T) / b
                Eigen::RowVector3d dr_dtB_row = (w * nB_.transpose() * (-RB.transpose())) / b;

                // dr/dqB: simplified
                // dx_dqB = d(RB^T @ U)/dqB = dRT_dq(qB, U)
                Matrix3x4 dx_dqB = dRTv_dq(qBpose, U);
                Eigen::RowVector4d da_dqB = nB_.transpose() * dx_dqB;
                Eigen::RowVector4d dr_dqB = (w * da_dqB) / b;

                jacobians[1][0] = dr_dqB(0);
                jacobians[1][1] = dr_dqB(1);
                jacobians[1][2] = dr_dqB(2);
                jacobians[1][3] = dr_dqB(3);
                jacobians[1][4] = dr_dtB_row(0);
                jacobians[1][5] = dr_dtB_row(1);
                jacobians[1][6] = dr_dtB_row(2);
            }
        }

        return true;
    }

private:
    Vector3 pA_, qB_, nB_, dA0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// ForwardRayCostTwoPose<RayJacobianConsistent>
//
// Ray from frame A -> surface in frame B
// Full quotient-rule Jacobians: dr = (b*da - a*db) / b^2
// ============================================================================

template<>
class ForwardRayCostTwoPose<RayJacobianConsistent> : public ceres::SizedCostFunction<1, 7, 7>
{
public:
    using policy_tag = RayJacobianConsistent;

    /**
     * @param pA    Point in frame A (ray origin)
     * @param qB    Target surface point in frame B
     * @param nB    Target surface normal in frame B
     * @param dA0   Ray direction in frame A (typically [0,0,-1])
     * @param weighting Geometry weighting parameters
     */
    ForwardRayCostTwoPose(const Vector3& pA, const Vector3& qB, const Vector3& nB,
                          const Vector3& dA0,
                          const GeometryWeighting& weighting = GeometryWeighting())
        : pA_(pA), qB_(qB), nB_(nB), dA0_(dA0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pA, const Vector3& qB,
                                       const Vector3& nB, const Vector3& dA0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ForwardRayCostTwoPose(pA, qB, nB, dA0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* xA = parameters[0];
        const double* xB = parameters[1];

        // Extract pose A
        Quaternion qA(xA[3], xA[0], xA[1], xA[2]);
        qA.normalize();
        Vector3 tA(xA[4], xA[5], xA[6]);
        Matrix3 RA = qA.toRotationMatrix();

        // Extract pose B
        Quaternion qBpose(xB[3], xB[0], xB[1], xB[2]);
        qBpose.normalize();
        Vector3 tB(xB[4], xB[5], xB[6]);
        Matrix3 RB = qBpose.toRotationMatrix();

        // Transform point from A to world, then to B
        Vector3 U = RA * pA_ + (tA - tB);    // pA in world, relative to tB
        Vector3 xBv = RB.transpose() * U;     // pA in frame B

        // Transform ray direction from A to B
        Vector3 u_d = RA * dA0_;              // ray in world
        Vector3 d = RB.transpose() * u_d;     // ray in frame B

        double a = nB_.dot(xBv - qB_);
        double b = nB_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians)
        {
            double b2 = b * b;

            // Jacobian w.r.t. pose A
            if (jacobians[0])
            {
                // da/dtA = nB^T @ RB^T, db/dtA = 0
                Eigen::RowVector3d da_dtA = nB_.transpose() * RB.transpose();
                Eigen::RowVector3d dr_dtA = (w * da_dtA) / b;

                // Consistent: full quotient rule for dq
                Matrix3x4 dRApA_dqA = dRv_dq(qA, pA_);
                Matrix3x4 dx_dqA = RB.transpose() * dRApA_dqA;
                Eigen::RowVector4d da_dqA = nB_.transpose() * dx_dqA;

                Matrix3x4 dRAd_dqA = dRv_dq(qA, dA0_);
                Matrix3x4 dd_dqA = RB.transpose() * dRAd_dqA;
                Eigen::RowVector4d db_dqA = nB_.transpose() * dd_dqA;

                Eigen::RowVector4d dr_dqA = w * (da_dqA * b - a * db_dqA) / b2;

                jacobians[0][0] = dr_dqA(0);
                jacobians[0][1] = dr_dqA(1);
                jacobians[0][2] = dr_dqA(2);
                jacobians[0][3] = dr_dqA(3);
                jacobians[0][4] = dr_dtA(0);
                jacobians[0][5] = dr_dtA(1);
                jacobians[0][6] = dr_dtA(2);
            }

            // Jacobian w.r.t. pose B
            if (jacobians[1])
            {
                // da/dtB = nB^T @ (-RB^T), db/dtB = 0
                Eigen::RowVector3d da_dtB = nB_.transpose() * (-RB.transpose());
                Eigen::RowVector3d dr_dtB = (w * da_dtB) / b;

                // Consistent: full quotient rule for dq
                Matrix3x4 dx_dqB = dRTv_dq(qBpose, U);
                Eigen::RowVector4d da_dqB = nB_.transpose() * dx_dqB;

                Matrix3x4 dd_dqB = dRTv_dq(qBpose, u_d);
                Eigen::RowVector4d db_dqB = nB_.transpose() * dd_dqB;

                Eigen::RowVector4d dr_dqB = w * (da_dqB * b - a * db_dqB) / b2;

                jacobians[1][0] = dr_dqB(0);
                jacobians[1][1] = dr_dqB(1);
                jacobians[1][2] = dr_dqB(2);
                jacobians[1][3] = dr_dqB(3);
                jacobians[1][4] = dr_dtB(0);
                jacobians[1][5] = dr_dtB(1);
                jacobians[1][6] = dr_dtB(2);
            }
        }

        return true;
    }

private:
    Vector3 pA_, qB_, nB_, dA0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// Reverse two-pose ray-projection cost - primary template (not defined)
// ============================================================================

template<typename JacobianPolicy>
class ReverseRayCostTwoPose;

// ============================================================================
// ReverseRayCostTwoPose<RayJacobianSimplified>
//
// Ray from frame B -> surface in frame A
// ============================================================================

template<>
class ReverseRayCostTwoPose<RayJacobianSimplified> : public ceres::SizedCostFunction<1, 7, 7>
{
public:
    using policy_tag = RayJacobianSimplified;

    /**
     * @param pB    Point in frame B (ray origin)
     * @param qA    Target surface point in frame A
     * @param nA    Target surface normal in frame A
     * @param dB0   Ray direction in frame B (typically [0,0,-1])
     * @param weighting Geometry weighting parameters
     */
    ReverseRayCostTwoPose(const Vector3& pB, const Vector3& qA, const Vector3& nA,
                          const Vector3& dB0,
                          const GeometryWeighting& weighting = GeometryWeighting())
        : pB_(pB), qA_(qA), nA_(nA), dB0_(dB0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pB, const Vector3& qA,
                                       const Vector3& nA, const Vector3& dB0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ReverseRayCostTwoPose(pB, qA, nA, dB0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* xA = parameters[0];
        const double* xB = parameters[1];

        // Extract pose A
        Quaternion qApose(xA[3], xA[0], xA[1], xA[2]);
        qApose.normalize();
        Vector3 tA(xA[4], xA[5], xA[6]);
        Matrix3 RA = qApose.toRotationMatrix();

        // Extract pose B
        Quaternion qBpose(xB[3], xB[0], xB[1], xB[2]);
        qBpose.normalize();
        Vector3 tB(xB[4], xB[5], xB[6]);
        Matrix3 RB = qBpose.toRotationMatrix();

        // Transform point from B to world, then to A
        Vector3 V = RB * pB_ + (tB - tA);     // pB in world, relative to tA
        Vector3 xAv = RA.transpose() * V;      // pB in frame A

        // Transform ray direction from B to A
        Vector3 v_d = RB * dB0_;               // ray in world
        Vector3 d = RA.transpose() * v_d;      // ray in frame A

        double a = nA_.dot(xAv - qA_);
        double b = nA_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians)
        {
            // Jacobian w.r.t. pose A
            if (jacobians[0])
            {
                // dr/dtA = nA^T @ (-RA^T) / b
                Eigen::RowVector3d dr_dtA_row = (w * nA_.transpose() * (-RA.transpose())) / b;

                // dr/dqA: simplified
                // dx_dqA = d(RA^T @ V)/dqA = dRT_dq(qA, V)
                Matrix3x4 dx_dqA = dRTv_dq(qApose, V);
                Eigen::RowVector4d da_dqA = nA_.transpose() * dx_dqA;
                Eigen::RowVector4d dr_dqA = (w * da_dqA) / b;

                jacobians[0][0] = dr_dqA(0);
                jacobians[0][1] = dr_dqA(1);
                jacobians[0][2] = dr_dqA(2);
                jacobians[0][3] = dr_dqA(3);
                jacobians[0][4] = dr_dtA_row(0);
                jacobians[0][5] = dr_dtA_row(1);
                jacobians[0][6] = dr_dtA_row(2);
            }

            // Jacobian w.r.t. pose B
            if (jacobians[1])
            {
                // dr/dtB = nA^T @ RA^T / b
                Eigen::RowVector3d dr_dtB_row = (w * nA_.transpose() * RA.transpose()) / b;

                // dr/dqB: simplified
                // dx_dqB = RA^T @ dR(qB)*pB_dq
                Matrix3x4 dRBpB_dqB = dRv_dq(qBpose, pB_);
                Matrix3x4 dx_dqB = RA.transpose() * dRBpB_dqB;
                Eigen::RowVector4d da_dqB = nA_.transpose() * dx_dqB;
                Eigen::RowVector4d dr_dqB = (w * da_dqB) / b;

                jacobians[1][0] = dr_dqB(0);
                jacobians[1][1] = dr_dqB(1);
                jacobians[1][2] = dr_dqB(2);
                jacobians[1][3] = dr_dqB(3);
                jacobians[1][4] = dr_dtB_row(0);
                jacobians[1][5] = dr_dtB_row(1);
                jacobians[1][6] = dr_dtB_row(2);
            }
        }

        return true;
    }

private:
    Vector3 pB_, qA_, nA_, dB0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// ReverseRayCostTwoPose<RayJacobianConsistent>
//
// Ray from frame B -> surface in frame A
// Full quotient-rule Jacobians: dr = (b*da - a*db) / b^2
// ============================================================================

template<>
class ReverseRayCostTwoPose<RayJacobianConsistent> : public ceres::SizedCostFunction<1, 7, 7>
{
public:
    using policy_tag = RayJacobianConsistent;

    /**
     * @param pB    Point in frame B (ray origin)
     * @param qA    Target surface point in frame A
     * @param nA    Target surface normal in frame A
     * @param dB0   Ray direction in frame B (typically [0,0,-1])
     * @param weighting Geometry weighting parameters
     */
    ReverseRayCostTwoPose(const Vector3& pB, const Vector3& qA, const Vector3& nA,
                          const Vector3& dB0,
                          const GeometryWeighting& weighting = GeometryWeighting())
        : pB_(pB), qA_(qA), nA_(nA), dB0_(dB0), weighting_(weighting)
    {
    }

    static ceres::CostFunction* Create(const Vector3& pB, const Vector3& qA,
                                       const Vector3& nA, const Vector3& dB0,
                                       const GeometryWeighting& weighting = GeometryWeighting())
    {
        return new ReverseRayCostTwoPose(pB, qA, nA, dB0, weighting);
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const double* xA = parameters[0];
        const double* xB = parameters[1];

        // Extract pose A
        Quaternion qApose(xA[3], xA[0], xA[1], xA[2]);
        qApose.normalize();
        Vector3 tA(xA[4], xA[5], xA[6]);
        Matrix3 RA = qApose.toRotationMatrix();

        // Extract pose B
        Quaternion qBpose(xB[3], xB[0], xB[1], xB[2]);
        qBpose.normalize();
        Vector3 tB(xB[4], xB[5], xB[6]);
        Matrix3 RB = qBpose.toRotationMatrix();

        // Transform point from B to world, then to A
        Vector3 V = RB * pB_ + (tB - tA);     // pB in world, relative to tA
        Vector3 xAv = RA.transpose() * V;      // pB in frame A

        // Transform ray direction from B to A
        Vector3 v_d = RB * dB0_;               // ray in world
        Vector3 d = RA.transpose() * v_d;      // ray in frame A

        double a = nA_.dot(xAv - qA_);
        double b = nA_.dot(d);
        double r = a / b;

        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians)
        {
            double b2 = b * b;

            // Jacobian w.r.t. pose A
            if (jacobians[0])
            {
                // da/dtA = nA^T @ (-RA^T), db/dtA = 0
                Eigen::RowVector3d da_dtA = nA_.transpose() * (-RA.transpose());
                Eigen::RowVector3d dr_dtA = (w * da_dtA) / b;

                // Consistent: full quotient rule for dq
                Matrix3x4 dx_dqA = dRTv_dq(qApose, V);
                Eigen::RowVector4d da_dqA = nA_.transpose() * dx_dqA;

                Matrix3x4 dd_dqA = dRTv_dq(qApose, v_d);
                Eigen::RowVector4d db_dqA = nA_.transpose() * dd_dqA;

                Eigen::RowVector4d dr_dqA = w * (da_dqA * b - a * db_dqA) / b2;

                jacobians[0][0] = dr_dqA(0);
                jacobians[0][1] = dr_dqA(1);
                jacobians[0][2] = dr_dqA(2);
                jacobians[0][3] = dr_dqA(3);
                jacobians[0][4] = dr_dtA(0);
                jacobians[0][5] = dr_dtA(1);
                jacobians[0][6] = dr_dtA(2);
            }

            // Jacobian w.r.t. pose B
            if (jacobians[1])
            {
                // da/dtB = nA^T @ RA^T, db/dtB = 0
                Eigen::RowVector3d da_dtB = nA_.transpose() * RA.transpose();
                Eigen::RowVector3d dr_dtB = (w * da_dtB) / b;

                // Consistent: full quotient rule for dq
                Matrix3x4 dRBpB_dqB = dRv_dq(qBpose, pB_);
                Matrix3x4 dx_dqB = RA.transpose() * dRBpB_dqB;
                Eigen::RowVector4d da_dqB = nA_.transpose() * dx_dqB;

                Matrix3x4 dRBd_dqB = dRv_dq(qBpose, dB0_);
                Matrix3x4 dd_dqB = RA.transpose() * dRBd_dqB;
                Eigen::RowVector4d db_dqB = nA_.transpose() * dd_dqB;

                Eigen::RowVector4d dr_dqB = w * (da_dqB * b - a * db_dqB) / b2;

                jacobians[1][0] = dr_dqB(0);
                jacobians[1][1] = dr_dqB(1);
                jacobians[1][2] = dr_dqB(2);
                jacobians[1][3] = dr_dqB(3);
                jacobians[1][4] = dr_dtB(0);
                jacobians[1][5] = dr_dtB(1);
                jacobians[1][6] = dr_dtB(2);
            }
        }

        return true;
    }

private:
    Vector3 pB_, qA_, nA_, dB0_;
    GeometryWeighting weighting_;
};

// ============================================================================
// Type aliases
// ============================================================================

using ForwardRayCostTwoPoseSimplified = ForwardRayCostTwoPose<RayJacobianSimplified>;
using ForwardRayCostTwoPoseConsistent = ForwardRayCostTwoPose<RayJacobianConsistent>;
using ReverseRayCostTwoPoseSimplified = ReverseRayCostTwoPose<RayJacobianSimplified>;
using ReverseRayCostTwoPoseConsistent = ReverseRayCostTwoPose<RayJacobianConsistent>;

} // namespace ICP
