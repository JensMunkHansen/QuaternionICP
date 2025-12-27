#!/usr/bin/env python3
"""
Visualize EXR file alignment using stored camera poses.
"""

import argparse
import sys
import os
import glob

sys.path.insert(0, os.path.expanduser("~/github/vtkSps"))
import spspython

from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper, vtkRenderer,
    vtkRenderWindow, vtkRenderWindowInteractor
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import vtkmodules.vtkRenderingOpenGL2  # Required for rendering

from spsmodules.spsIOEXR import spsTinyEXRReader
from spsmodules.spsFiltersCore import spsDecodeTriangulation25D


def load_exr(filepath):
    """Load EXR file and decode to mesh with pose."""
    reader = spsTinyEXRReader()
    reader.SetFileName(filepath)
    reader.SetMarksArrayName("CellPointMarks")
    reader.SetPoseArrayName("UserTransform")
    reader.ReadPositionsOn()
    reader.Update()

    grid = reader.GetOutput()

    # Decode triangulation
    decoder = spsDecodeTriangulation25D()
    decoder.SetInputData(grid)
    decoder.SetMarksArrayName("CellPointMarks")
    decoder.Update()

    # Compute normals
    normals = vtkPolyDataNormals()
    normals.SetInputData(decoder.GetOutput())
    normals.ComputeCellNormalsOn()
    normals.Update()
    mesh = normals.GetOutput()

    # Extract UserTransform
    user_transform = None
    transform_arr = grid.GetFieldData().GetArray("UserTransform")
    if transform_arr:
        user_transform = vtkTransform()
        matrix = [transform_arr.GetComponent(row, col)
                  for row in range(4) for col in range(4)]
        user_transform.SetMatrix(matrix)

    return mesh, user_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize EXR file alignment using stored camera poses.",
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

    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
              (0,1,1), (1,0.5,0), (0.5,0,1), (0,0.5,0), (0.5,0.5,0.5)]

    renderer = vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)

    for i, f in enumerate(exr_files):
        print(f"  {os.path.basename(f)}")
        mesh, transform = load_exr(f)

        print(f"    triangles={mesh.GetNumberOfPolys()}, points={mesh.GetNumberOfPoints()}")

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*colors[i % len(colors)])
        actor.GetProperty().SetOpacity(0.7)

        if transform:
            actor.SetUserTransform(transform)
            t = transform.GetPosition()
            print(f"    t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")

        renderer.AddActor(actor)

    window = vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1200, 900)

    interactor = vtkRenderWindowInteractor()
    window.SetInteractor(interactor)

    renderer.ResetCamera()
    window.Render()
    interactor.Start()

    return 0


if __name__ == "__main__":
    sys.exit(main())
