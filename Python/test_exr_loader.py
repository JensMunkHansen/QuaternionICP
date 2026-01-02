#!/usr/bin/env python3
"""Test the EXR loader and compare with C++ reader."""

import sys
import os
import numpy as np

# Add Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exr_loader import load_exr
from decode_triangulation import exr_grid_to_polydata

# C++ reader for comparison
sys.path.insert(0, os.path.expanduser("~/github/vtkSps"))
import spspython
from spsmodules.spsIOEXR import spsTinyEXRReader
from spsmodules.spsFiltersCore import spsDecodeTriangulation25D

def main():
    exr_path = os.path.expanduser("~/github/QuaternionICP/ExternalData/mesh0000.exr")

    print("=" * 70)
    print("Testing Python EXR Loader vs C++ Reader")
    print("=" * 70)

    # Load with Python
    print("\n[Python Loader]")
    grid = load_exr(exr_path)
    print(f"  Dimensions: {grid.width} x {grid.height}")
    print(f"  X range: [{grid.X.min():.4f}, {grid.X.max():.4f}]")
    print(f"  Y range: [{grid.Y.min():.4f}, {grid.Y.max():.4f}]")
    print(f"  Z range: [{grid.Z.min():.4f}, {grid.Z.max():.4f}]")
    print(f"  Marks range: [{grid.marks.min()}, {grid.marks.max()}]")
    print(f"  Camera pose: {'Yes' if grid.camera_pose is not None else 'No'}")

    # Sample values
    print("\n  Sample values at (0,0):")
    print(f"    X={grid.X[0,0]:.6f}, Y={grid.Y[0,0]:.6f}, Z={grid.Z[0,0]:.6f}")
    print(f"    Marks={grid.marks[0,0]}")

    # C++ reader for comparison
    print("\n[C++ Reader (reference)]")
    reader = spsTinyEXRReader()
    reader.SetFileName(exr_path)
    reader.SetMarksArrayName("Marks")
    reader.ReadPositionsOn()
    reader.Update()
    cpp_grid = reader.GetOutput()

    dims = [0, 0, 0]
    cpp_grid.GetDimensions(dims)
    print(f"  Dimensions: {dims[0]} x {dims[1]}")

    cpp_pt = cpp_grid.GetPoint(0)
    marks_arr = cpp_grid.GetPointData().GetArray("Marks")
    cpp_marks = marks_arr.GetValue(0) if marks_arr else 0
    print(f"\n  Sample values at point 0:")
    print(f"    X={cpp_pt[0]:.6f}, Y={cpp_pt[1]:.6f}, Z={cpp_pt[2]:.6f}")
    print(f"    Marks={cpp_marks}")

    # Compare
    print("\n[Comparison]")
    print(f"  Python X[0,0] = {grid.X[0,0]:.6f}  vs  C++ x = {cpp_pt[0]:.6f}  ->  {'MATCH' if abs(grid.X[0,0] - cpp_pt[0]) < 0.001 else 'DIFFER'}")
    print(f"  Python Y[0,0] = {grid.Y[0,0]:.6f}  vs  C++ y = {cpp_pt[1]:.6f}  ->  {'MATCH' if abs(grid.Y[0,0] - cpp_pt[1]) < 0.001 else 'DIFFER'}")
    print(f"  Python Z[0,0] = {grid.Z[0,0]:.6f}  vs  C++ z = {cpp_pt[2]:.6f}  ->  {'MATCH' if abs(grid.Z[0,0] - cpp_pt[2]) < 0.001 else 'DIFFER'}")
    print(f"  Python Marks[0,0] = {grid.marks[0,0]}  vs  C++ Marks = {cpp_marks}  ->  {'MATCH' if grid.marks[0,0] == int(cpp_marks) else 'DIFFER'}")

    # Triangulate
    print("\n[Triangulation]")
    mesh = exr_grid_to_polydata(grid)
    print(f"  Python mesh: {mesh.GetNumberOfPoints()} points, {mesh.GetNumberOfPolys()} triangles")

    # C++ triangulation
    decoder = spsDecodeTriangulation25D()
    decoder.SetInputData(cpp_grid)
    decoder.SetMarksArrayName("Marks")
    decoder.Update()
    cpp_mesh = decoder.GetOutput()
    print(f"  C++ mesh: {cpp_mesh.GetNumberOfPoints()} points, {cpp_mesh.GetNumberOfPolys()} triangles")

if __name__ == "__main__":
    main()
