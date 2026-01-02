#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Jens Munk Hansen
"""
Lightweight EXR mesh visualization.

Uses pure Python EXR loading and triangulation decoding.
"""

import argparse
import sys
import os
import glob

# Local imports
from exr_loader import load_exr
from decode_triangulation import exr_grid_to_polydata

# VTK imports - specific modules only
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper, vtkRenderer,
    vtkRenderWindow, vtkRenderWindowInteractor
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import vtkmodules.vtkRenderingOpenGL2  # Required for rendering


def load_mesh(filepath, include_colors=True):
    """Load EXR file and decode to mesh with pose."""
    # Load EXR data using pure Python loader
    grid = load_exr(filepath)

    # Decode triangulation to VTK polydata
    mesh = exr_grid_to_polydata(grid, include_colors=include_colors)

    # Compute normals
    normals = vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.ComputeCellNormalsOn()
    normals.Update()
    mesh = normals.GetOutput()

    # Build transform from camera pose
    transform = None
    if grid.camera_pose is not None:
        transform = vtkTransform()
        # Flatten row-major to list for VTK
        matrix = grid.camera_pose.flatten().tolist()
        transform.SetMatrix(matrix)

    return mesh, transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight EXR mesh visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder", nargs="?",
                        default=os.path.expanduser("~/github/QuaternionICP/ExternalData"),
                        help="Folder containing EXR files")
    parser.add_argument("-n", "--max-grids", type=int, default=10,
                        help="Maximum number of grids to display (0=all)")
    parser.add_argument("--indices", type=str, default="",
                        help="Specific grid indices to load, e.g. '0,1,3' or '0-5'")
    return parser.parse_args()


def parse_indices(indices_str):
    """Parse index string like '0,1,3' or '0-5' or '0-5,10,15-20' into list of ints."""
    if not indices_str:
        return None
    result = []
    for part in indices_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def main():
    args = parse_args()
    folder = args.folder

    all_exr_files = sorted(glob.glob(os.path.join(folder, "*.exr")))
    if not all_exr_files:
        print(f"No .exr files in {folder}")
        return 1

    # Filter by indices if specified
    indices = parse_indices(args.indices)
    if indices is not None:
        exr_files = [all_exr_files[i] for i in indices if i < len(all_exr_files)]
    else:
        exr_files = all_exr_files

    # Apply max-grids limit
    if args.max_grids > 0:
        exr_files = exr_files[:args.max_grids]

    print(f"Loading {len(exr_files)} of {len(all_exr_files)} files...")

    # Index-based colors for left renderer
    index_colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
                    (0,1,1), (1,0.5,0), (0.5,0,1), (0,0.5,0), (0.5,0.5,0.5)]

    # Left renderer: index-based coloring
    renderer_left = vtkRenderer()
    renderer_left.SetBackground(0.1, 0.1, 0.1)
    renderer_left.SetViewport(0.0, 0.0, 0.5, 1.0)

    # Right renderer: real colors from EXR
    renderer_right = vtkRenderer()
    renderer_right.SetBackground(0.1, 0.1, 0.1)
    renderer_right.SetViewport(0.5, 0.0, 1.0, 1.0)

    for i, f in enumerate(exr_files):
        print(f"  {os.path.basename(f)}")

        mesh, transform = load_mesh(f, include_colors=True)

        print(f"    triangles={mesh.GetNumberOfPolys()}, points={mesh.GetNumberOfPoints()}")

        # Left actor: index-based color (ignore scalars)
        mapper_left = vtkPolyDataMapper()
        mapper_left.SetInputData(mesh)
        mapper_left.ScalarVisibilityOff()
        actor_left = vtkActor()
        actor_left.SetMapper(mapper_left)
        actor_left.GetProperty().SetColor(*index_colors[i % len(index_colors)])
        if transform:
            actor_left.SetUserTransform(transform)
        renderer_left.AddActor(actor_left)

        # Right actor: real colors from scalars
        mapper_right = vtkPolyDataMapper()
        mapper_right.SetInputData(mesh)
        actor_right = vtkActor()
        actor_right.SetMapper(mapper_right)
        if transform:
            actor_right.SetUserTransform(transform)
        renderer_right.AddActor(actor_right)

        if transform:
            t = transform.GetPosition()
            print(f"    t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")

    # Share camera between renderers
    renderer_right.SetActiveCamera(renderer_left.GetActiveCamera())

    window = vtkRenderWindow()
    window.AddRenderer(renderer_left)
    window.AddRenderer(renderer_right)
    window.SetSize(1600, 900)

    interactor = vtkRenderWindowInteractor()
    window.SetInteractor(interactor)

    renderer_left.ResetCamera()
    window.Render()
    interactor.Start()

    return 0


if __name__ == "__main__":
    sys.exit(main())
