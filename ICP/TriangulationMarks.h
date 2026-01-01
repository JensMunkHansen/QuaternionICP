// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once

#include <ICP/Config.h>

#if ICP_USE_GRIDSEARCH
#include <GridSearch/GridSearchTypesC.h>
#else
// Define GridSearch-compatible types and constants when GridSearch is not available
#include <cstdint>
using GridSearchMark = uint8_t;
#define GRIDSEARCH_MARK_VALID (1 << 0)
#define GRIDSEARCH_MARK_VERTEX_POINT (1 << 1)
#define GRIDSEARCH_MARK_QUAD_LOWER (1 << 3)
#define GRIDSEARCH_MARK_QUAD_UPPER (1 << 4)
#define GRIDSEARCH_MARK_QUAD_DIAGONAL (1 << 5)
#define GRIDSEARCH_FACET_CONFIG_LL 0x00000
#define GRIDSEARCH_FACET_CONFIG_UR 0x10000
#define GRIDSEARCH_FACET_CONFIG_LR 0x20000
#define GRIDSEARCH_FACET_CONFIG_UL 0x30000
#define GRIDSEARCH_FACET_CONFIG_MASK 0x30000
#endif

/**
 * @file TriangulationMarks.h
 * @brief C++ wrapper for GridSearch triangulation mark constants.
 *
 * Wraps the C API from GridSearch/GridSearchTypesC.h with C++ conveniences.
 *
 * @section triangulation_grid Grid Triangulation
 *
 * Each grid cell (quad) is split into two triangles. The mark bits encode:
 * - Whether a point is valid
 * - Whether it's a vertex of the triangulation
 * - Which triangles (upper/lower) exist in the quad
 *
 * @section triangulation_facet Facet ID Encoding
 *
 * A facet ID encodes both the vertex index and the triangle configuration:
 * - Lower bits: vertex index in the grid
 * - Upper bits: facet configuration (LL, UR, LR, UL)
 */

namespace TriangulationMarks
{

/// @brief Mark type for triangulation bit flags.
using Mark = GridSearchMark;

/// @name Individual Bit Flags
/// @{

/// @brief Point has valid depth data.
constexpr Mark MarkPointValid = GRIDSEARCH_MARK_VALID;

/// @brief Point is a vertex in the triangulation.
constexpr Mark MarkPointVertex = GRIDSEARCH_MARK_VERTEX_POINT;

/// @brief Quad has a lower triangle.
constexpr Mark MarkQuadLower = GRIDSEARCH_MARK_QUAD_LOWER;

/// @brief Quad has an upper triangle.
constexpr Mark MarkQuadUpper = GRIDSEARCH_MARK_QUAD_UPPER;

/// @brief Quad diagonal orientation flag.
constexpr Mark MarkQuadDiagonal = GRIDSEARCH_MARK_QUAD_DIAGONAL;

/// @}

/// @name Combined Masks
/// @{

/// @brief Combined mask for valid vertex points.
constexpr Mark MarkValidVertex = MarkPointValid | MarkPointVertex;

/// @brief Combined mask for any triangle (upper or lower).
constexpr Mark MarkAnyTriangle = MarkQuadUpper | MarkQuadLower;

/// @}

/// @name Facet Configuration Constants
/// @{

constexpr int FacetConfigLL = GRIDSEARCH_FACET_CONFIG_LL;   ///< Lower-left configuration
constexpr int FacetConfigUR = GRIDSEARCH_FACET_CONFIG_UR;   ///< Upper-right configuration
constexpr int FacetConfigLR = GRIDSEARCH_FACET_CONFIG_LR;   ///< Lower-right configuration
constexpr int FacetConfigUL = GRIDSEARCH_FACET_CONFIG_UL;   ///< Upper-left configuration
constexpr int FacetConfigMask = GRIDSEARCH_FACET_CONFIG_MASK; ///< Mask for extracting config bits

/// @}

/**
 * @brief Check if a mark indicates a valid vertex.
 * @param mark The triangulation mark to check
 * @return True if the point is both valid and a vertex
 */
inline bool IsValidVertex(Mark mark)
{
    return (mark & MarkValidVertex) == MarkValidVertex;
}

/**
 * @brief Check if a quad has an upper triangle.
 * @param mark The triangulation mark to check
 * @return True if the upper triangle exists
 */
inline bool HasUpperTriangle(Mark mark)
{
    return (mark & MarkQuadUpper) != 0;
}

/**
 * @brief Check if a quad has a lower triangle.
 * @param mark The triangulation mark to check
 * @return True if the lower triangle exists
 */
inline bool HasLowerTriangle(Mark mark)
{
    return (mark & MarkQuadLower) != 0;
}

/**
 * @brief Extract vertex index from a facet ID.
 * @param facetId The encoded facet identifier
 * @return The vertex index (configuration bits masked out)
 */
inline int GetVertexIndexFromFacetId(int facetId)
{
    return facetId & ~FacetConfigMask;
}

/**
 * @brief Extract facet configuration from a facet ID.
 * @param facetId The encoded facet identifier
 * @return The facet configuration (LL, UR, LR, or UL)
 */
inline int GetFacetConfigFromFacetId(int facetId)
{
    return facetId & FacetConfigMask;
}

} // namespace TriangulationMarks
