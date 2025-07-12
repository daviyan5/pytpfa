import os
import pytest
import sys, petsc4py

petsc4py.init(sys.argv)

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from pytpfa.utils.managers.dmstag_manager import DMStagManager
from pytpfa.utils.mesh import mesh_to_dmstag

SL = PETSc.DMStag.StencilLocation


class TestDMStagManagerExtension:
    """
    Test suite for DMStagManager using pytest.
    """

    first_test_executed = False

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """
        A pytest fixture to set up the test environment before each test
        and clean up afterward. This runs automatically for all test methods.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        self.dmstag_manager = DMStagManager(3)
        self.dmstag_manager.add_field("temperature", 1, 0)
        self.dmstag_manager.add_field("scalar_field", 1, 0)
        self.dmstag_manager.add_field("velocity", 3, 1)
        self.dmstag_manager.add_field("edge_scalar", 1, 1)
        self.dmstag_manager.add_field("flux", 3, 2)
        self.dmstag_manager.add_field("face_scalar", 1, 2)
        self.dmstag_manager.add_field("pressure", 1, 3)
        self.dmstag_manager.add_field("element_vector", 3, 3)

        dof_count = self.dmstag_manager.get_dof_count()
        self.dm = mesh_to_dmstag(os.path.join(script_dir, "meshfiles", "hexa3.msh"), dofs=dof_count)
        self.dmstag_manager.set_dm(self.dm)

        self.hx = 0.5 / 16
        self.hy = 0.5 / 16
        self.hz = 0.5 / 16

        self.vertex_count = self.dmstag_manager.get_count(SL.BACK_DOWN_LEFT)
        self.edge_count = self.dmstag_manager.get_count(SL.BACK_DOWN)
        self.face_count = self.dmstag_manager.get_count(SL.BACK)
        self.element_count = self.dmstag_manager.get_count(SL.ELEMENT)

        self.temperatures = np.linspace(0, 100, self.vertex_count)
        self.scalar_fields = np.sin(np.linspace(0, 2 * np.pi, self.vertex_count))
        self.velocities = np.random.rand(self.edge_count, 3) * 10
        self.edge_scalars = np.cos(np.linspace(0, np.pi, self.edge_count))
        self.flux = np.random.rand(self.face_count, 3) * 50
        self.face_scalars = np.exp(-np.linspace(0, 2, self.face_count))
        self.pressures = np.linspace(0, 200, self.element_count)
        self.element_vectors = np.random.rand(self.element_count, 3) * 100

        self.dmstag_manager.set_field("temperature", SL.BACK_DOWN_LEFT, self.temperatures)
        self.dmstag_manager.set_field("scalar_field", SL.BACK_DOWN_LEFT, self.scalar_fields)
        self.dmstag_manager.set_field("velocity", SL.BACK_DOWN, self.velocities)
        self.dmstag_manager.set_field("edge_scalar", SL.BACK_DOWN, self.edge_scalars)
        self.dmstag_manager.set_field("flux", SL.BACK, self.flux)
        self.dmstag_manager.set_field("face_scalar", SL.BACK, self.face_scalars)
        self.dmstag_manager.set_field("pressure", SL.ELEMENT, self.pressures)
        self.dmstag_manager.set_field("element_vector", SL.ELEMENT, self.element_vectors)

        if not TestDMStagManagerExtension.first_test_executed:
            TestDMStagManagerExtension.first_test_executed = True

        yield

        if hasattr(self, "dm") and self.dm:
            self.dm.destroy()

    def test_get_boundary(self):
        """Tests the retrieval of boundary information for different stencil locations."""

        vertex_boundary = self.dmstag_manager.get_boundary(SL.BACK_DOWN_LEFT)
        edge_boundary = self.dmstag_manager.get_boundary(SL.BACK_DOWN)
        face_boundary = self.dmstag_manager.get_boundary(SL.BACK)
        element_boundary = self.dmstag_manager.get_boundary(SL.ELEMENT)

        assert isinstance(vertex_boundary, np.ndarray), "Boundary should be a numpy array"
        assert (
            vertex_boundary.shape[0] == self.vertex_count
        ), "Boundary array should have one entry per vertex"
        assert (
            vertex_boundary.dtype == np.int32
        ), f"Boundary array should be int32, but got: {vertex_boundary.dtype}"

        assert isinstance(edge_boundary, np.ndarray), "Boundary should be a numpy array"
        assert (
            edge_boundary.shape[0] == self.edge_count
        ), "Boundary array should have one entry per edge"

        assert isinstance(face_boundary, np.ndarray), "Boundary should be a numpy array"
        assert (
            face_boundary.shape[0] == self.face_count
        ), "Boundary array should have one entry per face"

        assert isinstance(element_boundary, np.ndarray), "Boundary should be a numpy array"
        assert (
            element_boundary.shape[0] == self.element_count
        ), "Boundary array should have one entry per element"

        valid_boundary_values = {0, 1, 2, 3, 4, 5, 6}
        assert np.all(
            np.isin(vertex_boundary, list(valid_boundary_values))
        ), "All boundary values must be valid direction enums"

    def test_get_connectivity(self):
        """Tests the connectivity information between different grid entities."""

        def check_coordinate_relationship(
            source_coords, target_coords_all, connectivity, expected_offsets, test_name
        ):
            tol = 1e-10
            assert (
                len(expected_offsets) == connectivity.shape[1]
            ), f"Expected {len(expected_offsets)} connections, but got {connectivity.shape[1]}"

            for i, source_coord in enumerate(source_coords):
                connected_indices = connectivity[i]
                # Use unique to handle potential duplicate coordinates from ghost points
                connected_coords = np.unique(target_coords_all[connected_indices], axis=0)
                expected_coords = np.unique(np.array(expected_offsets) + source_coord, axis=0)

                assert np.all(
                    np.isclose(connected_coords, expected_coords, atol=tol)
                ), f"Coordinate relationship failed for {test_name} at source index {i}"
            return True

        # Test Element -> Vertex connectivity
        e_to_v_conn = self.dmstag_manager.get_connectivity(SL.ELEMENT, SL.BACK_DOWN_LEFT)
        element_coords = self.dmstag_manager.get_coordinates(SL.ELEMENT)
        vertex_coords_ghost = self.dmstag_manager.get_coordinates(SL.BACK_DOWN_LEFT, use_ghost=True)

        e_to_v_offsets = [
            [-self.hx, -self.hy, -self.hz],
            [self.hx, -self.hy, -self.hz],
            [-self.hx, self.hy, -self.hz],
            [self.hx, self.hy, -self.hz],
            [-self.hx, -self.hy, self.hz],
            [self.hx, -self.hy, self.hz],
            [-self.hx, self.hy, self.hz],
            [self.hx, self.hy, self.hz],
        ]
        success_e_to_v = check_coordinate_relationship(
            element_coords, vertex_coords_ghost, e_to_v_conn, e_to_v_offsets, "Element to Vertex"
        )

        f_to_v_conn = self.dmstag_manager.get_connectivity(SL.BACK, SL.BACK_DOWN_LEFT)
        face_coords = self.dmstag_manager.get_coordinates(SL.BACK)
        f_to_v_offsets = [
            [-self.hx, -self.hy, 0],
            [self.hx, -self.hy, 0],
            [-self.hx, self.hy, 0],
            [self.hx, self.hy, 0],
        ]
        success_f_to_v = check_coordinate_relationship(
            face_coords, vertex_coords_ghost, f_to_v_conn, f_to_v_offsets, "Face to Vertex"
        )

        assert success_e_to_v and success_f_to_v, "Connectivity coordinate tests failed."

    def test_get_coordinates(self):
        """Tests the retrieval and validity of coordinates for grid entities."""

        vertex_coords = self.dmstag_manager.get_coordinates(SL.BACK_DOWN_LEFT)
        edge_coords = self.dmstag_manager.get_coordinates(SL.BACK_DOWN)
        face_coords = self.dmstag_manager.get_coordinates(SL.BACK)
        element_coords = self.dmstag_manager.get_coordinates(SL.ELEMENT)

        assert isinstance(vertex_coords, np.ndarray)
        assert vertex_coords.shape == (self.vertex_count, 3), "Vertex coordinates shape mismatch"
        assert np.all(vertex_coords >= 0) and np.all(
            vertex_coords <= 1
        ), "Vertex coordinates out of bounds [0,1]"

        assert isinstance(edge_coords, np.ndarray)
        assert edge_coords.shape == (self.edge_count, 3), "Edge coordinates shape mismatch"
        assert np.all(edge_coords >= 0) and np.all(
            edge_coords <= 1
        ), "Edge coordinates out of bounds [0,1]"

        assert isinstance(face_coords, np.ndarray)
        assert face_coords.shape == (self.face_count, 3), "Face coordinates shape mismatch"

        assert isinstance(element_coords, np.ndarray)
        assert element_coords.shape == (self.element_count, 3), "Element coordinates shape mismatch"
        assert np.all(element_coords >= self.hx) and np.all(
            element_coords <= 1 - self.hx
        ), "Element coordinates out of bounds"

        unique_coords, counts = np.unique(element_coords, axis=0, return_counts=True)
        assert np.all(counts == 1), "Element coordinates should be unique"
