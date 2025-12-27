#pragma once
/**
 * Utilities for creating and managing test grids.
 */

// Standard C++ headers
#include <iostream>

// Internal headers
#include <ICP/CommonOptions.h>
#include <ICP/Grid.h>
#include <ICP/GridFactory.h>

namespace ICP
{

/**
 * Create a pair of identical test grids based on options.
 *
 * @param opts CommonOptions containing grid type and parameters
 * @param verbose If true, print grid creation details
 * @return Pair of (source, target) grids
 */
inline std::pair<Grid, Grid> createTestGrids(const CommonOptions& opts, bool verbose = false)
{
    Grid source, target;

    if (verbose)
    {
        std::cout << "Creating synthetic test grids:\n";
        std::cout << "\tType: " << (opts.gridType == CommonOptions::GridType::Heightfield ?
                                    "Heightfield" : "TwoHemispheres") << "\n";
        std::cout << "\tSize: " << opts.gridWidth << "x" << opts.gridHeight << "\n";
    }

    // Create synthetic grids based on selected type
    if (opts.gridType == CommonOptions::GridType::Heightfield)
    {
        if (verbose)
        {
            std::cout << "\tAmplitude: " << opts.gridAmplitude << "\n";
            std::cout << "\tFrequency: " << opts.gridFrequency << "\n";
        }

        float dx = 2.0f / (opts.gridWidth - 1);
        float dy = 2.0f / (opts.gridHeight - 1);

        source = createHeightfieldGrid(opts.gridWidth, opts.gridHeight, dx, dy,
                                       opts.gridAmplitude, opts.gridFrequency);
        target = createHeightfieldGrid(opts.gridWidth, opts.gridHeight, dx, dy,
                                       opts.gridAmplitude, opts.gridFrequency);
    }
    else // TwoHemispheres
    {
        if (verbose)
        {
            std::cout << "\tHemisphere radius: " << opts.hemisphereRadius << "\n";
            std::cout << "\tHemisphere separation: " << opts.hemisphereSeparation << "\n";
        }

        float dx = 0.2f;
        float dy = 0.2f;

        source = createTwoHemispheresGrid(opts.gridWidth, opts.gridHeight, dx, dy,
                                          opts.hemisphereRadius, opts.hemisphereSeparation);
        target = createTwoHemispheresGrid(opts.gridWidth, opts.gridHeight, dx, dy,
                                          opts.hemisphereRadius, opts.hemisphereSeparation);
    }

    if (verbose)
    {
        std::cout << "\tSource vertices: " << source.nRows() * source.nCols() << "\n";
        std::cout << "\tTarget vertices: " << target.nRows() * target.nCols() << "\n";
    }

    return {source, target};
}

} // namespace ICP
