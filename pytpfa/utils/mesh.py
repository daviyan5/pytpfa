import numpy as np

import sys, petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc

import os
import meshio

from functools import reduce


def mesh_to_dmstag(mesh_path, dofs=None):
    """
    Convert a mesh file to a DMStag object.

    Parameters:
    mesh_path: Path to the mesh file
    dofs: Optional tuple of DOF counts for each dimension (vertex, edge, face, element)

    Returns:
    DMStag object
    """
    mesh = meshio.read(mesh_path)
    points = mesh.points
    dim = points.shape[1]
    coords = [np.sort(np.unique(points[:, d])) for d in range(dim)]
    sizes = tuple(len(c) - 1 for c in coords)

    if not dofs:
        dofs = tuple(1 for _ in range(dim + 1))

    boundary_types = tuple(PETSc.DM.BoundaryType.GHOSTED for _ in range(dim))

    dm = PETSc.DMStag().create(
        dim=dim,
        sizes=sizes,
        dofs=dofs,
        boundary_types=boundary_types,
        stencil_type=PETSc.DMStag.StencilType.BOX,
        stencil_width=1,
        comm=PETSc.COMM_WORLD,
    )

    dm.setCoordinateDMType(PETSc.DM.Type.STAG)
    dm.setUp()
    if dim == 1:
        dm.setUniformCoordinatesExplicit(coords[0][0], coords[0][-1])
    elif dim == 2:
        dm.setUniformCoordinatesExplicit(coords[0][0], coords[0][-1], coords[1][0], coords[1][-1])
    elif dim == 3:
        dm.setUniformCoordinatesExplicit(
            coords[0][0],
            coords[0][-1],
            coords[1][0],
            coords[1][-1],
            coords[2][0],
            coords[2][-1],
        )

    return dm


def create_3d_dmstag(sizes=(2, 2, 2), h=(0.5, 0.5, 0.5), dofs=(1, 1, 1, 1)):
    nx, ny, nz = sizes
    hx, hy, hz = h
    dim = 3
    boundary_types = (
        PETSc.DM.BoundaryType.GHOSTED,
        PETSc.DM.BoundaryType.GHOSTED,
        PETSc.DM.BoundaryType.GHOSTED,
    )
    dm = PETSc.DMStag().create(
        dim=dim,
        sizes=sizes,
        dofs=dofs,
        boundary_types=boundary_types,
        stencil_type=PETSc.DMStag.StencilType.BOX,
        stencil_width=1,
        comm=PETSc.COMM_WORLD,
    )
    dm.setCoordinateDMType(PETSc.DM.Type.STAG)
    dm.setUp()
    dm.setUniformCoordinatesExplicit(0, nx * hx, 0, ny * hy, 0, nz * hz)
    return dm
