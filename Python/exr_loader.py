#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Jens Munk Hansen
"""
Lightweight EXR loader for mesh grids.

Reads EXR files with X, Y, Z position channels, optional RGB, Marks,
and custom attributes (cameraPose, gridSpacing, etc.).
"""

import numpy as np
import OpenEXR
import Imath
import struct
from dataclasses import dataclass
from typing import Optional


@dataclass
class EXRGrid:
    """Container for EXR grid data."""
    width: int
    height: int
    X: np.ndarray  # (height, width) float32
    Y: np.ndarray  # (height, width) float32
    Z: np.ndarray  # (height, width) float32
    marks: Optional[np.ndarray] = None  # (height, width) uint8
    R: Optional[np.ndarray] = None  # (height, width) float16
    G: Optional[np.ndarray] = None  # (height, width) float16
    B: Optional[np.ndarray] = None  # (height, width) float16
    camera_pose: Optional[np.ndarray] = None  # (4, 4) float64
    grid_spacing: Optional[tuple] = None  # (dx, dy)
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None


def _read_v4f_attribute(header_data: bytes, attr_name: str) -> Optional[np.ndarray]:
    """
    Read a v4f (Vector4 float) custom attribute from EXR header data.

    OpenEXR Python bindings don't handle v4f type, so we read it manually.
    """
    name_bytes = attr_name.encode('ascii') + b'\x00'
    pos = header_data.find(name_bytes)
    if pos < 0:
        return None

    type_start = pos + len(name_bytes)

    # Verify it's v4f type
    if header_data[type_start:type_start+4] != b'v4f\x00':
        return None

    # Skip type (4 bytes) and size field (4 bytes)
    data_start = type_start + 4 + 4

    # Read 4 floats (16 bytes)
    try:
        floats = struct.unpack('<4f', header_data[data_start:data_start+16])
        return np.array(floats, dtype=np.float64)
    except struct.error:
        return None


def _read_camera_pose(filepath: str) -> Optional[np.ndarray]:
    """
    Read cameraPose from EXR file.

    The pose is stored as 4 separate v4f attributes: cameraPoseRow0..3
    """
    with open(filepath, 'rb') as f:
        header_data = f.read(8192)

    rows = []
    for i in range(4):
        row = _read_v4f_attribute(header_data, f'cameraPoseRow{i}')
        if row is None:
            return None
        rows.append(row)

    return np.vstack(rows)


def load_exr(filepath: str) -> EXRGrid:
    """
    Load an EXR file and return grid data.

    Parameters
    ----------
    filepath : str
        Path to EXR file.

    Returns
    -------
    EXRGrid
        Container with position data, marks, colors, and metadata.
    """
    exr = OpenEXR.InputFile(filepath)
    header = exr.header()

    # Get dimensions
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Pixel types
    float_pt = Imath.PixelType(Imath.PixelType.FLOAT)
    half_pt = Imath.PixelType(Imath.PixelType.HALF)

    # Read position channels (X, Y, Z are world coordinates)
    channels = header['channels']
    X = np.frombuffer(exr.channel('X', float_pt), dtype=np.float32).reshape(height, width)
    Y = np.frombuffer(exr.channel('Y', float_pt), dtype=np.float32).reshape(height, width)
    Z = np.frombuffer(exr.channel('Z', float_pt), dtype=np.float32).reshape(height, width)

    # Marks are triangulation bit flags stored as FLOAT, convert to uint8
    marks = None
    if 'Marks' in channels:
        marks_channel = np.frombuffer(exr.channel('Marks', float_pt), dtype=np.float32).reshape(height, width)
        marks = marks_channel.astype(np.uint8)

    # Read RGB channels (optional, stored as HALF)
    R = G = B = None
    if 'R' in channels:
        R = np.frombuffer(exr.channel('R', half_pt), dtype=np.float16).reshape(height, width)
    if 'G' in channels:
        G = np.frombuffer(exr.channel('G', half_pt), dtype=np.float16).reshape(height, width)
    if 'B' in channels:
        B = np.frombuffer(exr.channel('B', half_pt), dtype=np.float16).reshape(height, width)

    # Read custom attributes
    # OpenEXR Python doesn't handle v4f type, so read manually
    camera_pose = _read_camera_pose(filepath)

    grid_spacing = None
    spacing_attr = header.get('gridSpacing')
    if spacing_attr is not None:
        # V2f has x and y attributes
        grid_spacing = (spacing_attr.x, spacing_attr.y)

    min_depth = header.get('minDepth')
    max_depth = header.get('maxDepth')

    exr.close()

    return EXRGrid(
        width=width,
        height=height,
        X=X,
        Y=Y,
        Z=Z,
        marks=marks,
        R=R,
        G=G,
        B=B,
        camera_pose=camera_pose,
        grid_spacing=grid_spacing,
        min_depth=min_depth,
        max_depth=max_depth
    )
