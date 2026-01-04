// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once

#include <string>

// Ceres headers (for feature detection macros)
#include <ceres/ceres.h>

namespace ICP
{

/**
 * @brief Check if Ceres was built with SuiteSparse support.
 */
inline bool hasSuiteSparseSupport()
{
#ifdef CERES_NO_SUITESPARSE
    return false;
#else
    return true;
#endif
}

/**
 * @brief Check if Ceres was built with CUDA support.
 */
inline bool hasCudaSupport()
{
#ifdef CERES_NO_CUDA
    return false;
#else
    return true;
#endif
}

/**
 * @brief Check if a linear solver type is available.
 * @param solverName The solver name (e.g., "cuda-dense", "cuda-sparse")
 * @return True if the solver is available
 */
inline bool isLinearSolverAvailable(const std::string& solverName)
{
    if (solverName == "cuda-dense")
    {
        return hasCudaSupport();
    }
    else if (solverName == "cuda-sparse")
    {
        return hasCudaSupport() && hasSuiteSparseSupport();
    }
    else if (solverName == "sparse-schur")
    {
        // Sparse Schur works with EigenSparse even without SuiteSparse
        return true;
    }
    // dense-qr, dense-schur, iterative-schur are always available
    return true;
}

/**
 * @brief Get a description of why a solver is unavailable.
 */
inline std::string getLinearSolverUnavailableReason(const std::string& solverName)
{
    if (solverName == "cuda-dense" && !hasCudaSupport())
    {
        return "Ceres was not built with CUDA support";
    }
    else if (solverName == "cuda-sparse")
    {
        if (!hasCudaSupport() && !hasSuiteSparseSupport())
        {
            return "Ceres was not built with CUDA or SuiteSparse support";
        }
        else if (!hasCudaSupport())
        {
            return "Ceres was not built with CUDA support";
        }
        else if (!hasSuiteSparseSupport())
        {
            return "Ceres was not built with SuiteSparse support";
        }
    }
    return "";
}

/**
 * @brief Print available Ceres features to stdout.
 */
inline void printCeresCapabilities()
{
    std::cout << "Ceres capabilities:\n";
    std::cout << "  SuiteSparse: " << (hasSuiteSparseSupport() ? "yes" : "no") << "\n";
    std::cout << "  CUDA: " << (hasCudaSupport() ? "yes" : "no") << "\n";
    std::cout << "Available linear solvers:\n";
    std::cout << "  dense-qr: yes\n";
    std::cout << "  dense-schur: yes\n";
    std::cout << "  sparse-schur: yes\n";
    std::cout << "  iterative-schur: yes\n";
    std::cout << "  cuda-dense: " << (hasCudaSupport() ? "yes" : "no") << "\n";
    std::cout << "  cuda-sparse: " << (hasCudaSupport() && hasSuiteSparseSupport() ? "yes" : "no") << "\n";
}

} // namespace ICP
