#pragma once
/**
 * @file TriangulationMarks.h
 * @brief C++ wrapper for GridSearch triangulation mark constants.
 *
 * Wraps the C API from GridSearch/GridSearchTypesC.h with C++ conveniences.
 */

// GridSearch headers
#include <GridSearch/GridSearchTypesC.h>

namespace TriangulationMarks
{

// Mark type alias
using Mark = GridSearchMark;

// Individual bit flags (C++ aliases for C macros)
constexpr Mark MarkPointValid = GRIDSEARCH_MARK_VALID;
constexpr Mark MarkPointVertex = GRIDSEARCH_MARK_VERTEX_POINT;
constexpr Mark MarkQuadLower = GRIDSEARCH_MARK_QUAD_LOWER;
constexpr Mark MarkQuadUpper = GRIDSEARCH_MARK_QUAD_UPPER;
constexpr Mark MarkQuadDiagonal = GRIDSEARCH_MARK_QUAD_DIAGONAL;

// Combined masks
constexpr Mark MarkValidVertex = MarkPointValid | MarkPointVertex;
constexpr Mark MarkAnyTriangle = MarkQuadUpper | MarkQuadLower;

// FacetConfig constants
constexpr int FacetConfigLL = GRIDSEARCH_FACET_CONFIG_LL;
constexpr int FacetConfigUR = GRIDSEARCH_FACET_CONFIG_UR;
constexpr int FacetConfigLR = GRIDSEARCH_FACET_CONFIG_LR;
constexpr int FacetConfigUL = GRIDSEARCH_FACET_CONFIG_UL;
constexpr int FacetConfigMask = GRIDSEARCH_FACET_CONFIG_MASK;

inline bool IsValidVertex(Mark mark)
{
    return (mark & MarkValidVertex) == MarkValidVertex;
}

inline bool HasUpperTriangle(Mark mark)
{
    return (mark & MarkQuadUpper) != 0;
}

inline bool HasLowerTriangle(Mark mark)
{
    return (mark & MarkQuadLower) != 0;
}

inline int GetVertexIndexFromFacetId(int facetId)
{
    return facetId & ~FacetConfigMask;
}

inline int GetFacetConfigFromFacetId(int facetId)
{
    return facetId & FacetConfigMask;
}

} // namespace TriangulationMarks
