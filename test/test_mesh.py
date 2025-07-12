import os
import sys, petsc4py

petsc4py.init(sys.argv)

import pytest
import meshio

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from pytpfa.utils.mesh import mesh_to_dmstag

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshfiles")

if not os.path.isdir(MESH_DIR):
    pytest.skip(f"Mesh directory not found: {MESH_DIR}", allow_module_level=True)

MESH_FILES = [os.path.join(MESH_DIR, f) for f in os.listdir(MESH_DIR) if f.endswith(".msh")]

if not MESH_FILES:
    pytest.skip(f"No .msh files found in {MESH_DIR}", allow_module_level=True)


class TestMeshUtils:
    """
    Test suite for mesh utility functions using pytest.
    """

    @pytest.mark.parametrize("mesh_path", MESH_FILES)
    def test_mesh_to_dmstag(self, mesh_path):
        """
        Test mesh to DMStag conversion for each discovered .msh file.
        Each file is treated as an independent test case.
        """
        print(f"\n==== Testing mesh_to_dmstag with {os.path.basename(mesh_path)} ====")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        try:
            print(f"[Rank {rank + 1}/{size}] Reading mesh: {mesh_path}")
            mesh = meshio.read(mesh_path)
            dim = mesh.points.shape[1]

            print(f"[Rank {rank + 1}/{size}] Converting mesh to DMStag...")
            dm = mesh_to_dmstag(mesh_path)
            print(f"[Rank {rank + 1}/{size}] Mesh dimension: {dim}")

            assert dm.getDimension() == dim

            coords = [np.sort(np.unique(mesh.points[:, d])) for d in range(dim)]
            expected_sizes = [len(c) - 1 for c in coords]
            global_sizes = dm.getGlobalSizes()

            print(
                f"[Rank {rank + 1}/{size}] DMStag size: {global_sizes}, Expected: {expected_sizes}"
            )
            assert (
                list(global_sizes) == expected_sizes
            ), "DMStag global size does not match expected size from mesh"

            try:
                vec = dm.createGlobalVec()
                print(
                    f"[Rank {rank + 1}/{size}] Successfully created global vector with size: {vec.getSize()}"
                )
                vec.destroy()
            except Exception as e:
                pytest.fail(
                    f"Could not create global vector for {os.path.basename(mesh_path)}: {e}"
                )

            print(f"[Rank {rank + 1}/{size}] Test successful for {os.path.basename(mesh_path)}")

        except Exception as e:
            pytest.fail(f"ERROR testing {os.path.basename(mesh_path)}: {e}")
