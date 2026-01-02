#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Jens Munk Hansen
"""
Decode 2.5D triangulation from mark flags.

Converts a structured grid with mark flags into triangulated vtkPolyData.
"""

import numpy as np
from typing import Optional

# VTK imports - specific modules only
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray

# Mark bit flags (from spsTriangulationMarks.h)
MARK_POINT_VALID = 1 << 0      # Point has valid depth
MARK_POINT_VERTEX = 1 << 1     # Point is a triangle vertex
MARK_POINT_BOUNDARY = 1 << 2   # Point is on boundary
MARK_QUAD_LOWER = 1 << 3       # Lower triangle of quad
MARK_QUAD_UPPER = 1 << 4       # Upper triangle of quad
MARK_QUAD_DIAGONAL = 1 << 5    # Diagonal direction (0=UL/LR, 1=UR/LL)

# Combined masks for bit checks (matches C++ logic)
MARK_QUAD_DIAGONAL_LOWER = MARK_QUAD_DIAGONAL | MARK_QUAD_LOWER
MARK_QUAD_DIAGONAL_UPPER = MARK_QUAD_DIAGONAL | MARK_QUAD_UPPER


def decode_triangulation(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                         marks: np.ndarray,
                         R: Optional[np.ndarray] = None,
                         G: Optional[np.ndarray] = None,
                         B: Optional[np.ndarray] = None) -> vtkPolyData:
    """
    Decode mark flags to create triangulated mesh.

    The grid is interpreted as quads, where each quad can have:
    - Lower triangle (bottom-left to top-right or vice versa)
    - Upper triangle (top-left to bottom-right or vice versa)
    - Diagonal direction determines vertex ordering

    Parameters
    ----------
    X, Y, Z : np.ndarray
        Position arrays, shape (height, width).
    marks : np.ndarray
        Mark flags, shape (height, width), uint8.
    R, G, B : np.ndarray, optional
        Color arrays, shape (height, width), values 0-1.

    Returns
    -------
    vtkPolyData
        Triangulated mesh.
    """
    height, width = X.shape

    # Build point index map and collect valid points
    # Point (i, j) -> point index, or -1 if not used
    point_map = np.full((height, width), -1, dtype=np.int64)

    # Collect triangles as list of (i0, j0, i1, j1, i2, j2)
    triangles = []

    # Each cell (i, j) defines the quad from (i, j) to (i+1, j+1)
    # Mark is stored at upper-left corner of quad (v00)
    #
    # Quad layout (C++ indexing):
    #   v00(i0) ---- v01(i1)      In numpy (row,col) = (y,x):
    #      |            |           v00 = (i, j)
    #      |            |           v01 = (i, j+1)
    #   v10(i3) ---- v11(i2)        v10 = (i+1, j)
    #                               v11 = (i+1, j+1)
    #
    # Triangle regions (from C++ comments):
    #   Default (Diagonal=0): UL and LR regions, shared edge v10-v01
    #   Alternate (Diagonal=1): UR and LL regions, shared edge v00-v11

    for i in range(height - 1):
        for j in range(width - 1):
            mark = marks[i, j]

            # C++ bit mask checks (not simple != 0):
            # LR region: (mark & DiagonalLower) == Lower  (Lower set, Diagonal not set)
            # UL region: (mark & DiagonalUpper) == Upper  (Upper set, Diagonal not set)
            # LL region: (mark & DiagonalLower) == DiagonalLower  (both set)
            # UR region: (mark & DiagonalUpper) == DiagonalUpper  (both set)

            # LR (Lower-Right region, default diagonal)
            if (mark & MARK_QUAD_DIAGONAL_LOWER) == MARK_QUAD_LOWER:
                # C++: { i2, i3, i1 } = { v11, v10, v01 }
                triangles.append((i+1, j+1, i+1, j, i, j+1))

            # UL (Upper-Left region, default diagonal)
            if (mark & MARK_QUAD_DIAGONAL_UPPER) == MARK_QUAD_UPPER:
                # C++: { i1, i3, i0 } = { v01, v10, v00 }
                triangles.append((i, j+1, i+1, j, i, j))

            # LL (Lower-Left region, alternate diagonal)
            if (mark & MARK_QUAD_DIAGONAL_LOWER) == MARK_QUAD_DIAGONAL_LOWER:
                # C++: { i2, i3, i0 } = { v11, v10, v00 }
                triangles.append((i+1, j+1, i+1, j, i, j))

            # UR (Upper-Right region, alternate diagonal)
            if (mark & MARK_QUAD_DIAGONAL_UPPER) == MARK_QUAD_DIAGONAL_UPPER:
                # C++: { i1, i2, i0 } = { v01, v11, v00 }
                triangles.append((i, j+1, i+1, j+1, i, j))

    # Mark used points and assign indices
    for tri in triangles:
        for k in range(0, 6, 2):
            pi, pj = tri[k], tri[k+1]
            if point_map[pi, pj] < 0:
                point_map[pi, pj] = 0  # Mark as used

    # Assign sequential indices
    point_idx = 0
    for i in range(height):
        for j in range(width):
            if point_map[i, j] >= 0:
                point_map[i, j] = point_idx
                point_idx += 1

    num_points = point_idx
    num_triangles = len(triangles)

    # Create VTK points
    points = vtkPoints()
    points.SetNumberOfPoints(num_points)

    for i in range(height):
        for j in range(width):
            idx = point_map[i, j]
            if idx >= 0:
                points.SetPoint(idx, X[i, j], Y[i, j], Z[i, j])

    # Create VTK cells (triangles)
    cells = vtkCellArray()
    for tri in triangles:
        idx0 = point_map[tri[0], tri[1]]
        idx1 = point_map[tri[2], tri[3]]
        idx2 = point_map[tri[4], tri[5]]
        cells.InsertNextCell(3, [idx0, idx1, idx2])

    # Create colors if provided
    colors = None
    if R is not None and G is not None and B is not None:
        colors = vtkUnsignedCharArray()
        colors.SetName("Colors")
        colors.SetNumberOfComponents(3)
        colors.SetNumberOfTuples(num_points)

        for i in range(height):
            for j in range(width):
                idx = point_map[i, j]
                if idx >= 0:
                    # Handle float16 inf/nan by converting to float and clamping
                    rv = float(R[i, j])
                    gv = float(G[i, j])
                    bv = float(B[i, j])
                    r = int(min(255, max(0, rv * 255 + 0.5))) if np.isfinite(rv) else 0
                    g = int(min(255, max(0, gv * 255 + 0.5))) if np.isfinite(gv) else 0
                    b = int(min(255, max(0, bv * 255 + 0.5))) if np.isfinite(bv) else 0
                    colors.SetTuple3(idx, r, g, b)

    # Build polydata
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    if colors is not None:
        polydata.GetPointData().SetScalars(colors)

    return polydata


def exr_grid_to_polydata(grid, include_colors: bool = False) -> vtkPolyData:
    """
    Convert EXRGrid to vtkPolyData.

    Parameters
    ----------
    grid : EXRGrid
        Grid data from exr_loader.load_exr().
    include_colors : bool
        If True, include RGB colors from EXR (if valid).
        Default False since many EXR files don't have valid color data.

    Returns
    -------
    vtkPolyData
        Triangulated mesh.
    """
    if grid.marks is None:
        raise ValueError("Grid has no marks array - cannot triangulate")

    R, G, B = None, None, None
    if include_colors and grid.R is not None:
        # Only include if values are in valid [0, 1] range
        r_max = float(np.nanmax(grid.R))
        if r_max <= 1.0:
            R, G, B = grid.R, grid.G, grid.B

    return decode_triangulation(
        grid.X, grid.Y, grid.Z, grid.marks,
        R, G, B
    )
