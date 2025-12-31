#include <Eigen/Dense>
#include <chrono>
#include <cstring>
#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "Eigen MKL Test\n";
    std::cout << "==============\n\n";

#ifdef EIGEN_USE_MKL_ALL
    constexpr bool mkl_enabled = true;
    std::cout << "MKL: ENABLED\n\n";
#else
    constexpr bool mkl_enabled = false;
    std::cout << "MKL: DISABLED\n\n";
#endif

    // Check command-line arguments for expected state
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect-mkl") == 0) {
            if (!mkl_enabled) {
                std::cerr << "FAIL: Expected MKL to be enabled but it is not\n";
                return 1;
            }
            std::cout << "OK: MKL is enabled as expected\n\n";
        } else if (std::strcmp(argv[i], "--expect-no-mkl") == 0) {
            if (mkl_enabled) {
                std::cerr << "FAIL: Expected MKL to be disabled but it is enabled\n";
                return 1;
            }
            std::cout << "OK: MKL is disabled as expected\n\n";
        }
    }

    // Matrix size for benchmark
    const int N = 1000;

    std::cout << "Matrix size: " << N << "x" << N << "\n\n";

    // Create random matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);

    // Benchmark matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd C = A * B;
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Matrix multiplication: " << duration.count() << " ms\n";

    // Benchmark LU decomposition
    start = std::chrono::high_resolution_clock::now();
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "LU decomposition:      " << duration.count() << " ms\n";

    // Benchmark linear solve
    Eigen::VectorXd b = Eigen::VectorXd::Random(N);
    start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd x = lu.solve(b);
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Linear solve:          " << duration.count() << " ms\n";

    // Verify solution
    double error = (A * x - b).norm() / b.norm();
    std::cout << "\nRelative error: " << error << "\n";

    return 0;
}
