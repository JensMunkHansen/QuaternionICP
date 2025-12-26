#!/usr/bin/env python3
"""
Test grids for ICP validation.
"""

import numpy as np
import trimesh


def make_heightfield_mesh(nx=45, ny=45, size=1.0, amp=0.10, freq=3.0, z0=0.0):
    xs = np.linspace(-size, size, nx)
    ys = np.linspace(-size, size, ny)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = z0 + amp * np.sin(freq * X) * np.sin(freq * Y)
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    def vid(i, j): return j * nx + i
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = vid(i, j); v10 = vid(i + 1, j); v01 = vid(i, j + 1); v11 = vid(i + 1, j + 1)
            F.append([v00, v10, v11])
            F.append([v00, v11, v01])
    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(F, dtype=np.int64), process=True)
    return mesh, V
