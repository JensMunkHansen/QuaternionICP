#pragma once

#include <ICP/Grid.h>
#include <string>
#include <vector>

namespace ICP
{

// Load all EXR grids from a folder, sorted by trailing number in filename
std::vector<Grid> loadGrids(const std::string& folder);

// Load specific EXR grids by index (0-based, after sorting by trailing number)
// Empty indices vector loads all grids
std::vector<Grid> loadGrids(const std::string& folder, const std::vector<int>& indices);

} // namespace ICP
