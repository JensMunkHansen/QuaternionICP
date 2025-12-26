#pragma once
/**
 * Ambient (7D) Jacobians for ray-projection ICP.
 *
 * Ceres 2.2-compatible cost functions using 7D ambient parameterization.
 * Simplified Jacobians: treats denominator b = n^T d as constant for dq.
 *
 * Residual: r = a/b where
 *   a = n^T (x_transformed - q_surface)
 *   b = n^T d  (ray direction dot normal)
 */

#include <ICP/SE3.h>
#include <ICP/ICPParams.h>

namespace ICP
{

/**
 * Forward ray-projection residual (source -> target).
 *
 * Transforms source point pS by pose x, computes signed distance
 * to target surface point qT along normal nT.
 */
class RayProjectionFwd
{
public:
    RayProjectionFwd(const Vector3& pS, const Vector3& qT, const Vector3& nT, const Vector3& dS0,
                     const GeometryWeighting& weighting = GeometryWeighting())
        : pS_(pS), qT_(qT), nT_(nT), dS0_(dS0), weighting_(weighting)
    {
    }

    /**
     * Ceres-style Evaluate.
     *
     * @param parameters  parameters[0] points to 7D pose [qx,qy,qz,qw, tx,ty,tz]
     * @param residuals   residuals[0] = r = a/b (ray projection residual)
     * @param jacobians   if non-null, jacobians[0][0..6] receives dr/dx (simplified)
     * @return            always true
     */
    bool operator()(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const
    {
        const double* x = parameters[0];

        // Extract and normalize quaternion
        Quaternion q(x[3], x[0], x[1], x[2]);  // w, x, y, z
        q.normalize();
        Vector3 t(x[4], x[5], x[6]);
        Matrix3 R = q.toRotationMatrix();

        // Transform source point and ray direction
        Vector3 xT = R * pS_ + t;
        Vector3 d = R * dS0_;

        // Residual: r = a/b
        double a = nT_.dot(xT - qT_);
        double b = nT_.dot(d);
        double r = a / b;

        // Compute incidence weight
        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians && jacobians[0])
        {
            // dr/dt = n / b
            Vector3 dr_dt = (w * nT_) / b;

            // dr/dq = da_dq / b  (simplified: ignores db_dq term)
            Matrix3x4 dRp_dq = dRv_dq(q, pS_);
            Eigen::RowVector4d dr_dq = (w * nT_.transpose() * dRp_dq) / b;

            // Pack: [dr/dqx, dr/dqy, dr/dqz, dr/dqw, dr/dtx, dr/dty, dr/dtz]
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

private:
    Vector3 pS_;   // Source point
    Vector3 qT_;   // Target surface point
    Vector3 nT_;   // Target surface normal
    Vector3 dS0_;  // Ray direction in source frame
    GeometryWeighting weighting_;
};


/**
 * Reverse ray-projection residual (target -> source).
 *
 * Transforms target point pT by inverse of pose x, computes signed distance
 * to source surface point qS along normal nS.
 */
class RayProjectionRev
{
public:
    RayProjectionRev(const Vector3& pT, const Vector3& qS, const Vector3& nS, const Vector3& dT0,
                     const GeometryWeighting& weighting = GeometryWeighting())
        : pT_(pT), qS_(qS), nS_(nS), dT0_(dT0), weighting_(weighting)
    {
    }

    /**
     * Ceres-style Evaluate.
     *
     * @param parameters  parameters[0] points to 7D pose [qx,qy,qz,qw, tx,ty,tz]
     * @param residuals   residuals[0] = r = a/b (ray projection residual)
     * @param jacobians   if non-null, jacobians[0][0..6] receives dr/dx (simplified)
     * @return            always true
     */
    bool operator()(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const
    {
        const double* x = parameters[0];

        // Extract and normalize quaternion
        Quaternion q(x[3], x[0], x[1], x[2]);  // w, x, y, z
        q.normalize();
        Vector3 t(x[4], x[5], x[6]);
        Matrix3 R = q.toRotationMatrix();

        // Transform target point to source frame: yS = R^T (pT - t)
        Vector3 u = pT_ - t;
        Vector3 yS = R.transpose() * u;
        Vector3 d = R.transpose() * dT0_;

        // Residual: r = a/b
        double a = nS_.dot(yS - qS_);
        double b = nS_.dot(d);
        double r = a / b;

        // Compute incidence weight
        double w = weighting_.weight(b);
        residuals[0] = w * r;

        if (jacobians && jacobians[0])
        {
            // dr/dt = n^T (-R^T) / b
            Vector3 dr_dt = (w * (-R * nS_)) / b;

            // dr/dq = da_dq / b  (simplified: ignores db_dq term)
            Matrix3x4 dRTu_dq = dRTv_dq(q, u);
            Eigen::RowVector4d dr_dq = (w * nS_.transpose() * dRTu_dq) / b;

            // Pack: [dr/dqx, dr/dqy, dr/dqz, dr/dqw, dr/dtx, dr/dty, dr/dtz]
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

private:
    Vector3 pT_;   // Target point
    Vector3 qS_;   // Source surface point
    Vector3 nS_;   // Source surface normal
    Vector3 dT0_;  // Ray direction in target frame
    GeometryWeighting weighting_;
};

} // namespace ICP
