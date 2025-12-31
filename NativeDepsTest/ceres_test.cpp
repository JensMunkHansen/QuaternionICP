#include <ceres/ceres.h>
#include <chrono>
#include <iostream>

// Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2
// Minimum at (a, a^2), typically a=1, b=100
struct RosenbrockCost
{
    template <typename T>
    bool operator()(const T* const x, T* residual) const
    {
        residual[0] = T(1.0) - x[0];
        residual[1] = T(10.0) * (x[1] - x[0] * x[0]);
        return true;
    }
};

// Large bundle-adjustment-like problem to stress linear solver
struct BundleCost
{
    BundleCost(double observed_x, double observed_y)
        : observed_x_(observed_x)
        , observed_y_(observed_y)
    {
    }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // Simple projection model
        T p[3];
        p[0] = point[0] - camera[0];
        p[1] = point[1] - camera[1];
        p[2] = point[2] - camera[2];

        T predicted_x = p[0] / p[2];
        T predicted_y = p[1] / p[2];

        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);
        return true;
    }

    double observed_x_;
    double observed_y_;
};

int main()
{
    std::cout << "Ceres MKL Test\n";
    std::cout << "==============\n\n";

#ifdef EIGEN_USE_MKL_ALL
    std::cout << "MKL: ENABLED\n\n";
#else
    std::cout << "MKL: DISABLED\n\n";
#endif

    // Simple Rosenbrock test
    {
        double x[2] = {-1.0, 1.0};

        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RosenbrockCost, 2, 2>(new RosenbrockCost),
            nullptr,
            x);

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.linear_solver_type = ceres::DENSE_QR;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Rosenbrock: x=" << x[0] << ", y=" << x[1];
        std::cout << " (expected: 1, 1)\n\n";
    }

    // Larger problem to benchmark linear solver
    {
        const int num_cameras = 50;
        const int num_points = 500;
        const int observations_per_point = 5;

        std::vector<double> cameras(num_cameras * 3, 0.0);
        std::vector<double> points(num_points * 3);

        // Initialize cameras in a circle
        for (int i = 0; i < num_cameras; ++i)
        {
            double angle = 2.0 * M_PI * i / num_cameras;
            cameras[i * 3 + 0] = 10.0 * cos(angle);
            cameras[i * 3 + 1] = 10.0 * sin(angle);
            cameras[i * 3 + 2] = 0.0;
        }

        // Initialize points randomly
        srand(42);
        for (int i = 0; i < num_points * 3; ++i)
        {
            points[i] = (rand() / (double)RAND_MAX - 0.5) * 2.0;
        }

        ceres::Problem problem;

        // Add observations
        for (int i = 0; i < num_points; ++i)
        {
            for (int j = 0; j < observations_per_point; ++j)
            {
                int cam_idx = (i * observations_per_point + j) % num_cameras;
                double obs_x = 0.1 * (rand() / (double)RAND_MAX - 0.5);
                double obs_y = 0.1 * (rand() / (double)RAND_MAX - 0.5);

                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<BundleCost, 2, 3, 3>(new BundleCost(obs_x, obs_y)),
                    nullptr,
                    &cameras[cam_idx * 3],
                    &points[i * 3]);
            }
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 50;

        std::cout << "Bundle adjustment: " << num_cameras << " cameras, " << num_points << " points\n";

        auto start = std::chrono::high_resolution_clock::now();
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Solve time: " << duration.count() << " ms\n";
        std::cout << "Iterations: " << summary.iterations.size() << "\n";
        std::cout << "Final cost: " << summary.final_cost << "\n";
    }

    return 0;
}
