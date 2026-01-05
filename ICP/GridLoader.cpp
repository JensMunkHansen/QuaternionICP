// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#include <ICP/GridLoader.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>

namespace fs = std::filesystem;

namespace ICP
{

namespace
{

// Helper: Find and sort EXR files by trailing number
std::vector<std::string> findExrFiles(const std::string& folder)
{
    std::vector<std::string> exr_files;
    for (const auto& entry : fs::directory_iterator(folder))
    {
        if (entry.path().extension() == ".exr")
        {
            exr_files.push_back(entry.path().string());
        }
    }

    // Sort by trailing number in filename
    std::regex num_regex(R"((\d+)\.exr$)");
    std::sort(exr_files.begin(), exr_files.end(),
      [&num_regex](const std::string& a, const std::string& b)
      {
          std::smatch ma, mb;
          int na = std::regex_search(a, ma, num_regex) ? std::stoi(ma[1]) : 0;
          int nb = std::regex_search(b, mb, num_regex) ? std::stoi(mb[1]) : 0;
          return na < nb;
      });

    return exr_files;
}

// Helper: Load a single grid from file (skip colors for ICP - saves memory)
bool loadGrid(const std::string& file, Grid& grid)
{
    if (grid.loadFromExr(file, false))  // loadColors=false
    {
        std::cout << "  " << grid.filename << " - " << grid.width << "x" << grid.height << " ("
                  << grid.validVertexCount() << " valid vertices)"
                  << " dx=" << grid.dx() << " dy=" << grid.dy() << "\n";
        return true;
    }
    return false;
}

} // namespace

std::vector<Grid> loadGrids(const std::string& folder)
{
    std::vector<std::string> exr_files = findExrFiles(folder);
    std::cout << "EXR files found: " << exr_files.size() << "\n";

    std::vector<Grid> grids;
    grids.reserve(exr_files.size());

    for (const auto& file : exr_files)
    {
        Grid grid;
        if (loadGrid(file, grid))
        {
            grids.push_back(std::move(grid));
        }
    }

    return grids;
}

std::vector<Grid> loadGrids(const std::string& folder, const std::vector<int>& indices)
{
    if (indices.empty())
    {
        return loadGrids(folder);
    }

    std::vector<std::string> exr_files = findExrFiles(folder);
    std::cout << "EXR files found: " << exr_files.size() << ", loading indices:";
    for (int idx : indices)
    {
        std::cout << " " << idx;
    }
    std::cout << "\n";

    std::vector<Grid> grids;
    grids.reserve(indices.size());

    for (int idx : indices)
    {
        if (idx < 0 || idx >= static_cast<int>(exr_files.size()))
        {
            std::cerr << "  Warning: index " << idx << " out of range (0-" << exr_files.size() - 1
                      << "), skipping\n";
            continue;
        }

        Grid grid;
        if (loadGrid(exr_files[idx], grid))
        {
            grids.push_back(std::move(grid));
        }
    }

    return grids;
}

} // namespace ICP
