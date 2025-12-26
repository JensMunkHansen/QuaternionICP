#include <ICP/CommonOptions.h>
#include <ICP/TestGridUtils.h>
#include <ICP/EigenUtils.h>
#include <ICP/SE3.h>
#include <ICP/ICPSimple.h>
#include <ICP/ICPParams.h>
#include <random>
#include <iostream>

using namespace ICP;

int main(int argc, char** argv)
{
    CommonOptions opts;
    if (!parseArgs(argc, argv, opts, "SingleICP - Ray-projection ICP for two grids"))
    {
        return 1;
    }

    // Create or load grids
    Grid source, target;

    if (opts.useTestGrid)
    {
        auto grids = createTestGrids(opts, opts.verbose);
        source = grids.first;
        target = grids.second;
    }
    else
    {
        // TODO: Load grids from EXR files
        std::cerr << "Error: EXR file loading not yet implemented\n";
        return 1;
    }

    // Create initial pose with perturbations (if requested)
    Pose7 initialPose = identityPose();

    if (opts.rotationNoise > 0.0f || opts.translationNoise > 0.0f)
    {
        // Initialize RNGs with deterministic seeds for reproducibility
        PerturbationRNGs rngs;

        if (opts.verbose)
        {
            std::cout << "\nApplying pose perturbations:\n";
        }

        // Perturb rotation (in-place)
        if (opts.rotationNoise > 0.0f)
        {
            perturbPoseRotation(initialPose, opts.rotationNoise, rngs.rotation);

            if (opts.verbose)
            {
                Quaternion q(initialPose[3], initialPose[0], initialPose[1], initialPose[2]);
                Eigen::AngleAxisd aa(q);
                std::cout << "  Rotation: " << (aa.angle() * 180.0 / M_PI) << " deg around ["
                          << aa.axis().transpose() << "]\n";
            }
        }

        // Perturb translation (in-place)
        if (opts.translationNoise > 0.0f)
        {
            perturbPoseTranslation(initialPose, opts.translationNoise, rngs.translation);

            if (opts.verbose)
            {
                std::cout << "  Translation: [" << initialPose[4] << ", "
                          << initialPose[5] << ", " << initialPose[6] << "]\n";
            }
        }

        if (opts.verbose)
        {
            std::cout << "  Initial pose: q=[" << initialPose[0] << ", " << initialPose[1] << ", "
                      << initialPose[2] << ", " << initialPose[3] << "], t=["
                      << initialPose[4] << ", " << initialPose[5] << ", " << initialPose[6] << "]\n";
        }
    }

    // TODO: Apply depth noise if requested
    if (opts.depthNoise > 0.0f)
    {
        std::cerr << "Warning: Depth noise not yet implemented\n";
    }

    // Set up ICP parameters from command-line options
    InnerParams innerParams;
    innerParams.maxIterations = opts.innerIterations;
    innerParams.stepTol = opts.stepTol;
    innerParams.damping = opts.damping;
    innerParams.verbose = opts.verbose;

    OuterParams outerParams;
    outerParams.maxIterations = opts.outerIterations;
    outerParams.convergenceTol = opts.rmsTol;
    outerParams.verbose = opts.verbose;

    GeometryWeighting weighting;
    weighting.enable_weight = opts.enableIncidenceWeight;
    weighting.enable_gate = opts.enableGrazingGate;
    weighting.tau = opts.incidenceTau;

    // Ray direction (looking down -Z in camera frame)
    Vector3 rayDir(0.0, 0.0, -1.0);

    // Display ICP configuration
    std::cout << "\n=== ICP Configuration ===\n";
    std::cout << "Outer loop:\n";
    std::cout << "\tMax iterations: " << outerParams.maxIterations << "\n";
    std::cout << "\tConvergence tolerance: " << outerParams.convergenceTol << "\n";
    std::cout << "\tMax correspondence distance: " << outerParams.maxDist << "\n";
    std::cout << "\nInner loop:\n";
    std::cout << "\tMax iterations: " << innerParams.maxIterations << "\n";
    std::cout << "\tStep tolerance: " << innerParams.stepTol << "\n";
    std::cout << "\tDamping: " << innerParams.damping << "\n";
    std::cout << "\nGeometry weighting:\n";
    std::cout << "\tIncidence weighting: " << (weighting.enable_weight ? "enabled" : "disabled") << "\n";
    std::cout << "\tGrazing angle gate: " << (weighting.enable_gate ? "enabled" : "disabled") << "\n";
    std::cout << "\tTau threshold: " << weighting.tau << "\n";

    // Run ICP solver
    if (opts.verbose)
    {
        std::cout << "\nRunning ICP solver...\n";
    }
    else
    {
        std::cout << "\nRunning ICP...\n";
    }

    ICPResult icpResult = solveICP(source, target, initialPose, rayDir,
                                   weighting, innerParams, outerParams);

    // Display results
    std::cout << "\n=== ICP Results ===\n";
    std::cout << "\tConverged: " << (icpResult.converged ? "yes" : "no") << "\n";
    std::cout << "\tOuter iterations: " << icpResult.outer_iterations << "\n";
    std::cout << "\tTotal inner iterations: " << icpResult.total_inner_iterations << "\n";
    std::cout << "\tFinal RMS: " << std::scientific << std::setprecision(6)
              << icpResult.rms << "\n";

    if (opts.verbose)
    {
        std::cout << "\n\tFinal pose:\n";
        std::cout << "\t\tq = [" << icpResult.pose[0] << ", " << icpResult.pose[1] << ", "
                  << icpResult.pose[2] << ", " << icpResult.pose[3] << "]\n";
        std::cout << "\t\tt = [" << icpResult.pose[4] << ", " << icpResult.pose[5] << ", "
                  << icpResult.pose[6] << "]\n";

        // Display rotation as axis-angle
        Quaternion q_final(icpResult.pose[3], icpResult.pose[0],
                          icpResult.pose[1], icpResult.pose[2]);
        Eigen::AngleAxisd aa_final(q_final);
        std::cout << "\t\tRotation: " << (aa_final.angle() * 180.0 / M_PI)
                  << " deg around [" << aa_final.axis().transpose() << "]\n";
    }

    return 0;
}
