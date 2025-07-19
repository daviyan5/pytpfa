# -*- coding: utf-8 -*-
import os
import logging
import numpy as np

import sys, petsc4py

from petsc4py import PETSc
from time import time, sleep

from ..utils.mesh import create_3d_dmstag
from ..utils.managers import DMStagManager, DMDAManager, BoundaryDirection
from ..utils.io import parse_reservoir_config

SL = PETSc.DMStag.StencilLocation
BD = BoundaryDirection

logger = logging.getLogger(__name__)

STENCIL_NAME = {
    SL.LEFT: "LEFT (YZ)",
    SL.BACK: "BACK (XY)",
    SL.DOWN: "DOWN (XZ)",
}


class TPFASolver:
    def __init__(self, name):
        self.name = name

        self.out_info = {
            "name": self.name,
            "rank": 0,
            "n_elements": 0,
            "n_iterations": 0,
            "l1_error": [],
            "l2_error": [],
            "linf_error": [],
            "total_time": 0.0,
            "preprocessing_time": 0.0,
            "solving_time": 0.0,
            "updating_time": 0.0,
        }

    def solve(self, reservoir_path, postprocess=False):
        self.comm = PETSc.COMM_WORLD
        self.rank = PETSc.COMM_WORLD.getRank()
        self.out_info["rank"] = self.rank
        self.b, self.x = None, None
        try:
            self.iteration = 0
            logger.info(
                f"Starting simulation with TPFASolver for {self.name} in rank {self.rank}",
                extra={"context": "Solver SOLVE"},
            )
            self.preprocessing_time = 0.0
            self.solving_time = 0.0
            self.updating_time = 0.0

            starting_time = time()

            preprocessing_start = time()
            self.prepare_fields()
            self.preprocess_problem_data(reservoir_path)
            self.preprocess_mesh()
            self.assemble_transmissibility()
            self.setup_solver()
            preprocessing_end = time()

            self.preprocessing_time = preprocessing_end - preprocessing_start
            self.out_info["preprocessing_time"] = self.preprocessing_time
            logger.info(
                f"Preprocessing time: {self.preprocessing_time:.2f} seconds",
                extra={"context": "Solver SOLVE"},
            )

            times = np.arange(self.TIME_INITIAL, self.TIME_FINAL + self.TIME_STEP, self.TIME_STEP)
            self.out_info["n_iterations"] = len(times) - 1
            self.comm.barrier()
            for t_idx, (t_n, t_np1) in enumerate(zip(times[:-1], times[1:])):

                self.iteration = t_idx
                self.current_time = t_n

                logger.info(
                    f"Starting time {t_n:.2f} to {t_np1:.2f}",
                    extra={"context": f"Solver ITERATION [{self.iteration}]"},
                )
                updating_start = time()
                self.update_system()
                self.set_boundary_conditions()
                self.assemble_system()
                updating_end = time()

                self.updating_time += updating_end - updating_start

                start_solve = time()
                self.solve_system()
                end_solve = time()
                self.check(t_np1)
                self.solving_time += end_solve - start_solve

                iteration_time = end_solve - updating_start
                logger.info(
                    f"Time to execute iteration {iteration_time:.2f} seconds",
                    extra={"context": f"Solver ITERATION [{self.iteration}]"},
                )

                if postprocess:
                    self.postprocess(t_np1)

            end_time = time()
            self.simulation_time = end_time - starting_time

            self.out_info["total_time"] = self.simulation_time
            self.out_info["solving_time"] = self.solving_time
            self.out_info["updating_time"] = self.updating_time
            logger.info(
                f"Total solving time: {self.solving_time:.2f} seconds",
                extra={"context": "Solver SOLVE"},
            )
            logger.info(
                f"Total updating time: {self.updating_time:.2f} seconds",
                extra={"context": "Solver SOLVE"},
            )
            logger.info(
                f"Total simulation time: {self.simulation_time:.2f} seconds",
                extra={"context": "Solver SOLVE"},
            )

        except Exception as e:
            logger.error(
                f"An error occurred during the simulation: {e}", extra={"context": "Solver SOLVE"}
            )
            logger.error("Traceback:", exc_info=True)

    def prepare_fields(self):
        logger.info("Preparing fields for TPFASolver...", extra={"context": "Solver PREPROCESS"})
        self.dmstag_manager = DMStagManager(3)

        # We won't use edges
        self.dmstag_manager.dof_count[1] = 0

        # Element fields
        self.dmstag_manager.add_field("pressure", 1, 3, is_shared=True)
        self.dmstag_manager.add_field("volume", 1, 3)
        self.dmstag_manager.add_field("permeability", 9, 3, is_shared=True)  # K
        self.dmstag_manager.add_field("porosity", 1, 3)  # φ
        self.dmstag_manager.add_field("formation_factor", 1, 3)  # B
        self.dmstag_manager.add_field("accumulation_coefficient", 1, 3)  # Gamma
        self.dmstag_manager.add_field("external_source", 1, 3, is_shared=True)  # S_u
        self.dmstag_manager.add_field("rhs", 1, 3, is_shared=True)  # Right-hand side vector
        self.dmstag_manager.add_field(
            "elem_transmissibility", 1, 3, is_shared=True
        )  # Transmissibility for the element

        # Faces fields
        self.dmstag_manager.add_field("area", 1, 2)
        self.dmstag_manager.add_field("normal_vector", 3, 2)
        self.dmstag_manager.add_field("transmissibility", 1, 2)  # Try, Trz, Trx
        self.dmstag_manager.add_field("fluid_transmissibility", 1, 2)  # Ty, Tz, Tx

    def preprocess_problem_data(self, reservoir_path):
        """
        MODIFIED: This method now uses the new io.py parser.
        """
        logger.info("Preprocessing problem data...", extra={"context": "Solver PREPROCESS"})
        self.dirname = os.path.dirname(reservoir_path)

        config = parse_reservoir_config(reservoir_path, cache=True)

        reservoir_sizes = (config.input.nx, config.input.ny, config.input.nz)
        reservoir_h = (config.input.dx, config.input.dy, config.input.dz)

        self.out_info["n_elements"] = np.prod(reservoir_sizes)
        self.hx, self.hy, self.hz = reservoir_h

        dof_count = self.dmstag_manager.get_dof_count()
        logger.debug(
            f"Reservoir sizes: {reservoir_sizes}, h: {reservoir_h}, Dof count: {dof_count}",
            extra={"context": "Solver PREPROCESS"},
        )
        self.dmstag = create_3d_dmstag(
            reservoir_sizes, reservoir_h, dof_count, name=config.description.name, cache=True
        )
        self.dmstag_manager.set_dm(self.dmstag)

        K_xx = config.input.kx
        K_yy = config.input.ky
        K_zz = config.input.kz

        K = np.zeros((self.dmstag_manager.get_count(SL.ELEMENT), 9))
        K[:] = [K_xx, 0.0, 0.0, 0.0, K_yy, 0.0, 0.0, 0.0, K_zz]
        self.dmstag_manager.set_field("permeability", SL.ELEMENT, K)

        self.ALPHA_C = 5.615
        self.BETA_C = 1.127

        self.FLUID_COMPRESSIBILITY = config.input.cfluid
        self.PORE_COMPRESSIBILITY = config.input.cporo
        self.B_REF_FORMATION_FACTOR = config.input.b
        self.POROSITY_REF = config.input.poro
        self.VISCOSITY = config.input.mu

        self.TIME_INITIAL = config.time_settings.time_initial
        self.current_time = self.TIME_INITIAL

        self.TIME_FINAL = config.time_settings.time_final
        self.TIME_STEP = config.time_settings.time_step

        xe, ye, ze = self.dmstag_manager.get_coordinates(SL.ELEMENT).T
        xeg, yeg, zeg = self.dmstag_manager.get_coordinates(SL.ELEMENT, use_ghost=True).T

        self.INITIAL_PRESSURE = config.initial_condition.pressure(xe, ye, ze)
        self.INITIAL_PRESSURE_GHOST = config.initial_condition.pressure(xeg, yeg, zeg)

        if np.isscalar(self.INITIAL_PRESSURE):
            self.INITIAL_PRESSURE = np.full_like(xe, self.INITIAL_PRESSURE)
        if np.isscalar(self.INITIAL_PRESSURE_GHOST):
            self.INITIAL_PRESSURE_GHOST = np.full_like(xeg, self.INITIAL_PRESSURE_GHOST)

        self.dmstag_manager.set_field("pressure", SL.ELEMENT, self.INITIAL_PRESSURE)

        self.WELLS = {}

        for well_name, well in config.wells.items():
            well_index = (well.block_coord_x, well.block_coord_y, well.block_coord_z)
            well_rate = None
            well_pressure = None
            well_productivity_index = None

            if well.cond == "constant_pressure":
                well_rate = 0.0
                well_pressure = well.value
                Kxy_ratio = K_xx / K_yy
                Kyx_ratio = K_yy / K_xx
                DX = config.input.dx
                DY = config.input.dy

                drainage_radius = np.sqrt(Kyx_ratio**0.5 * DX**2 + Kxy_ratio**0.5 * DY**2)
                drainage_radius *= 0.28 / (Kyx_ratio**0.25 + Kxy_ratio**0.25)

                well_productivity_index = (2.0 * np.pi * well.permeability * well.h) / (
                    (self.VISCOSITY * self.B_REF_FORMATION_FACTOR)
                    * (np.log(drainage_radius / well.radius) + well.skin)
                )
            elif well.cond == "constant_rate":
                well_rate = well.value
                well_pressure = 0.0
                well_productivity_index = 0.0
            else:
                raise ValueError(f"Unrecognized well condition: {well.cond} for well {well_name}")

            self.WELLS[well_name] = {
                "index": well_index,
                "permeability": well.permeability,
                "height": well.h,
                "radius": well.radius,
                "drainage_radius": (
                    drainage_radius if well.cond == "constant_pressure" else well.drainage_radius
                ),
                "skin": well.skin,
                "rate": well_rate,
                "pressure": well_pressure,
                "J": well_productivity_index,
            }

            logger.debug(
                f"Well {well_name} at {well_index} with rate {well_rate:.2f}, pressure {well_pressure:.2f}, J {well_productivity_index:.2f}",
                extra={"context": "Solver PREPROCESS"},
            )

        self.BOUNDARY = {
            BD.LEFT: (
                config.boundaries["left"].type,
                config.boundaries["left"].func,
            ),
            BD.DOWN: (
                config.boundaries["down"].type,
                config.boundaries["down"].func,
            ),
            BD.BACK: (
                config.boundaries["back"].type,
                config.boundaries["back"].func,
            ),
            BD.RIGHT: (
                config.boundaries["right"].type,
                config.boundaries["right"].func,
            ),
            BD.UP: (
                config.boundaries["up"].type,
                config.boundaries["up"].func,
            ),
            BD.FRONT: (
                config.boundaries["front"].type,
                config.boundaries["front"].func,
            ),
        }

        self.SOURCE_TERM = config.source_term
        self.analytical_solution = (
            config.analytical_functions["solution"] if config.analytical_functions else None
        )

    def preprocess_mesh(self):
        """
        This method shall read the mesh file using meshio and PETSc.DMSTAG and preprocess it.
        The prepocessing will include volume, area, centroid and normal_vector calculations
        """
        logger.info("Preprocessing mesh...", extra={"context": "Solver PREPROCESS"})
        self.comm.barrier()
        point_coordinates = self.dmstag_manager.get_coordinates(SL.BACK_DOWN_LEFT, use_ghost=True)

        for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]:
            area = np.zeros(self.dmstag_manager.get_count(face_stencil))

            # Contains the indices, in the local vector, to the points that form the face (may include ghosts)
            face_connectivity = self.dmstag_manager.get_connectivity(
                face_stencil, SL.BACK_DOWN_LEFT
            )

            point_coords_by_face = point_coordinates[face_connectivity]

            face_count = self.dmstag_manager.get_count(face_stencil)
            face_global_count = self.dmstag_manager.get_count(face_stencil, is_global=True)

            centroids = np.mean(point_coords_by_face, axis=1)

            # Sanity check
            face_centroid = self.dmstag_manager.get_coordinates(face_stencil)

            # Centroid MUST match face_centroid
            if not np.allclose(centroids, face_centroid, atol=1e-6):
                logger.error(
                    f"Centroid mismatch for face {STENCIL_NAME[face_stencil]}: "
                    f"Calculated {centroids}, Expected {face_centroid}",
                    extra={"context": "Solver PREPROCESS"},
                )
                raise ValueError("Centroid mismatch for face")

            v1 = point_coords_by_face[:, 1] - point_coords_by_face[:, 0]
            v2 = point_coords_by_face[:, 3] - point_coords_by_face[:, 0]

            cross_product = np.zeros((len(area), 3))
            cross_product[:, 0] = v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1]
            cross_product[:, 1] = v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2]
            cross_product[:, 2] = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

            area = np.linalg.norm(cross_product, axis=1)  # 2 Equal Triangles, so 2 * 0.5 = 1
            normal_vector = cross_product / area[:, np.newaxis]
            self.dmstag_manager.set_field("area", face_stencil, area)
            self.dmstag_manager.set_field("normal_vector", face_stencil, normal_vector)

            logger.debug(
                f"Average area for face: {np.mean(area)}",
                extra={"context": f"Solver {STENCIL_NAME[face_stencil]}"},
            )
            logger.debug(
                f"Average normal vector for face: {np.mean(normal_vector, axis=0)}",
                extra={"context": f"Solver {STENCIL_NAME[face_stencil]}"},
            )

        volume = np.zeros(self.dmstag_manager.get_count(SL.ELEMENT))

        # Contains the indices, in the local vector, to the points that form the element (may include ghosts)
        element_connectivity = self.dmstag_manager.get_connectivity(SL.ELEMENT, SL.BACK_DOWN_LEFT)

        point_coords_by_element = point_coordinates[element_connectivity]

        centroid = np.mean(point_coords_by_element, axis=1)

        # Sanity check
        element_centroid = self.dmstag_manager.get_coordinates(SL.ELEMENT)

        # Centroid MUST match element_centroid
        if not np.allclose(centroid, element_centroid, atol=1e-6):
            logger.error(
                f"Centroid mismatch for element: "
                f"Calculated {centroid}, Expected {element_centroid}",
                extra={"context": "Solver PREPROCESS"},
            )
            raise ValueError("Centroid mismatch for element")

        v1 = point_coords_by_element[:, 1] - point_coords_by_element[:, 0]
        v2 = point_coords_by_element[:, 3] - point_coords_by_element[:, 0]
        v3 = point_coords_by_element[:, 4] - point_coords_by_element[:, 0]

        # Calculate volume using scalar triple product for hexahedron
        # Volume = |v1 · (v2 × v3)| for tetrahedron, sum for hexahedron
        cross_product = np.zeros((len(volume), 3))
        cross_product[:, 0] = v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]
        cross_product[:, 1] = v2[:, 2] * v3[:, 0] - v2[:, 0] * v3[:, 2]
        cross_product[:, 2] = v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0]

        dot_product = np.sum(v1 * cross_product, axis=1)

        volume = np.abs(dot_product)
        self.dmstag_manager.set_field("volume", SL.ELEMENT, volume)
        logger.debug(
            f"Average volume for elements: {np.mean(volume)}",
            extra={"context": "Solver PREPROCESS"},
        )

    def assemble_transmissibility(self):
        """
        Assemble base transmissibility (reservoir properties only)
        Fluid properties will be added during time stepping
        """
        logger.info("Assembling transmissibility...", extra={"context": "Solver ASSEMBLE"})
        self.boundary = {}

        # Swap the boundaries to make sure the first element in the pair is local to the rank
        boundary_correspondence = {
            SL.LEFT: BD.LEFT.value,
            SL.BACK: BD.BACK.value,
            SL.DOWN: BD.DOWN.value,
        }

        self.internal_faces_indices = {
            face_stencil: np.empty(0) for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]
        }
        self.boundary_faces_indices = {
            face_stencil: np.empty(0) for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]
        }
        self.internal_elem_pairs = {
            face_stencil: np.empty((0, 2)) for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]
        }
        self.boundary_elem_pairs = {
            face_stencil: np.empty((0, 2)) for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]
        }

        self.comm.barrier()
        for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]:
            face_area = self.dmstag_manager.get_field("area", face_stencil)
            normal_vector = self.dmstag_manager.get_field("normal_vector", face_stencil)
            face_centroid = self.dmstag_manager.get_coordinates(face_stencil)
            permeability = self.dmstag_manager.get_field("permeability", SL.ELEMENT, use_ghost=True)
            elem_centroids = self.dmstag_manager.get_coordinates(SL.ELEMENT, use_ghost=True)
            elem_pairs = self.dmstag_manager.get_connectivity(face_stencil, SL.ELEMENT)

            self.boundary[face_stencil] = self.dmstag_manager.get_boundary(face_stencil)

            internal_boundary_value = BD.NONE.value

            inverted_indices = np.where(
                np.logical_and(
                    self.boundary[face_stencil] != internal_boundary_value,
                    self.boundary[face_stencil] != boundary_correspondence[face_stencil],
                )
            )[0]

            logger.debug(
                f"Number of inverted indices: {len(inverted_indices)}",
                extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
            )

            elem_pairs[inverted_indices] = elem_pairs[inverted_indices][:, ::-1]

            self.internal_faces_indices[face_stencil] = np.where(
                self.boundary[face_stencil] == internal_boundary_value
            )[0]
            self.boundary_faces_indices[face_stencil] = np.where(
                self.boundary[face_stencil] != internal_boundary_value
            )[0]

            logger.debug(
                f"Number of internal faces: " f"{len(self.internal_faces_indices[face_stencil])}",
                extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
            )
            logger.debug(
                f"Number of boundary faces: " f"{len(self.boundary_faces_indices[face_stencil])}",
                extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
            )

            faces_trans = np.zeros(self.dmstag_manager.get_count(face_stencil))

            cur_internal_faces_indices = self.internal_faces_indices[face_stencil]
            cur_boundary_faces_indices = self.boundary_faces_indices[face_stencil]

            if len(cur_internal_faces_indices) > 0:
                self.internal_elem_pairs[face_stencil] = elem_pairs[cur_internal_faces_indices]
                cur_internal_elem_pairs = self.internal_elem_pairs[face_stencil]

                internal_faces_area = face_area[cur_internal_faces_indices]
                internal_faces_normal = normal_vector[cur_internal_faces_indices]
                internal_faces_centroids = face_centroid[cur_internal_faces_indices]

                K_pairs = permeability[cur_internal_elem_pairs]
                internal_elem_pairs_centroids = elem_centroids[cur_internal_elem_pairs]

                # Distance vectors from cell centers to face center
                h_L = internal_faces_centroids - internal_elem_pairs_centroids[:, 0]
                h_R = internal_elem_pairs_centroids[:, 1] - internal_faces_centroids

                h_L = np.linalg.norm(h_L, axis=1)
                h_R = np.linalg.norm(h_R, axis=1)

                # Sanity check, they must be all equal between thenselves and between left and right
                if (
                    not np.allclose(h_L, h_R, atol=1e-6)
                    or not np.allclose(h_L, h_L[0], atol=1e-6)
                    or not np.allclose(h_R, h_R[0], atol=1e-6)
                ):
                    logger.error(
                        f"Distance vectors h_L and h_R are not equal: " f"h_L: {h_L}, h_R: {h_R}",
                        extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
                    )
                    raise ValueError("Distance vectors h_L and h_R are not equal")

                # Extract permeability tensors
                KL = K_pairs[:, 0].reshape((len(cur_internal_elem_pairs), 3, 3))
                KR = K_pairs[:, 1].reshape((len(cur_internal_elem_pairs), 3, 3))

                # Compute normal permeabilities: K_n = n^T · K · n
                KnL = np.einsum(
                    "ij,ij->i",
                    internal_faces_normal,
                    np.einsum("ij,ikj->ik", internal_faces_normal, KL),
                )
                KnR = np.einsum(
                    "ij,ij->i",
                    internal_faces_normal,
                    np.einsum("ij,ikj->ik", internal_faces_normal, KR),
                )

                # Harmonic mean for equivalent permeability
                Keq = (KnL * KnR) / ((KnL * h_R) + (KnR * h_L))
                faces_trans[cur_internal_faces_indices] = Keq * internal_faces_area
                logger.debug(
                    f"Average transmissibility for internal faces: "
                    f"{np.mean(faces_trans[cur_internal_faces_indices])}",
                    extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
                )

            if len(cur_boundary_faces_indices) > 0:
                self.boundary_elem_pairs[face_stencil] = elem_pairs[cur_boundary_faces_indices]
                cur_boundary_elem_pairs = self.boundary_elem_pairs[face_stencil]

                boundary_faces_area = face_area[cur_boundary_faces_indices]
                boundary_faces_normal = normal_vector[cur_boundary_faces_indices]
                boundary_faces_centroids = face_centroid[cur_boundary_faces_indices]

                K = permeability[cur_boundary_elem_pairs[:, 0]]

                boundary_elem_pairs_centroids = elem_centroids[cur_boundary_elem_pairs[:, 0]]

                # Distance vectors from cell centers to face center
                h = boundary_faces_centroids - boundary_elem_pairs_centroids
                h = np.linalg.norm(h, axis=1)

                # Sanity check, they must all be equal and if face_stencil is LEFT, they must be equal to hx
                if not np.allclose(h, h[0], atol=1e-6):
                    logger.error(
                        f"Distance vector h is not equal: {h}",
                        extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
                    )
                    raise ValueError("Distance vector h is not equal")

                h_to_face_stencil = {
                    SL.LEFT: self.hx / 2,
                    SL.BACK: self.hz / 2,
                    SL.DOWN: self.hy / 2,
                }
                if not np.isclose(h[0], h_to_face_stencil[face_stencil], atol=1e-6):
                    logger.error(
                        f"Distance vector h is not equal to expected value: {h[0]} != "
                        f"{h_to_face_stencil[face_stencil]}",
                        extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
                    )
                    raise ValueError("Distance vector h is not equal to expected value")

                # Extract permeability tensors
                K = K.reshape((len(cur_boundary_elem_pairs), 3, 3))
                Kn = np.einsum(
                    "ij,ij->i",
                    boundary_faces_normal,
                    np.einsum("ij,ikj->ik", boundary_faces_normal, K),
                )
                faces_trans[cur_boundary_faces_indices] = (Kn * boundary_faces_area) / h
                logger.debug(
                    f"Average transmissibility for boundary faces: "
                    f"{np.mean(faces_trans[cur_boundary_faces_indices])}",
                    extra={"context": f"Solver ASSEMBLE {STENCIL_NAME[face_stencil]}"},
                )

            self.dmstag_manager.set_field(
                "transmissibility", face_stencil, faces_trans * self.BETA_C
            )

    def setup_solver(self):
        """
        Setup KSP solver
        """
        logger.info("Setting up the KSP solver...", extra={"context": "Solver SETUP"})

        all_elem_pairs = np.vstack(
            tuple(
                [
                    self.internal_elem_pairs[face_stencil]
                    for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]
                ]
            )
        )

        ghost_nx, ghost_ny, ghost_nz = self.dmstag_manager.get_ghost_sizes(SL.ELEMENT)
        ghost_corners = self.dmstag_manager.get_ghost_corners(SL.ELEMENT)
        i_indices = (all_elem_pairs) % ghost_nx
        j_indices = (all_elem_pairs // ghost_nx) % ghost_ny
        k_indices = all_elem_pairs // (ghost_nx * ghost_ny)

        i_indices += ghost_corners[0]
        j_indices += ghost_corners[1]
        k_indices += ghost_corners[2]

        L_indices = np.stack((i_indices[:, 0], j_indices[:, 0], k_indices[:, 0]), axis=1)
        R_indices = np.stack((i_indices[:, 1], j_indices[:, 1], k_indices[:, 1]), axis=1)

        nx, ny, nz = self.dmstag_manager.get_sizes(SL.ELEMENT, is_global=True)
        global_strides = np.array([1, nx, nx * ny])

        L_indices = np.dot(L_indices, global_strides).astype(np.int32)
        R_indices = np.dot(R_indices, global_strides).astype(np.int32)

        all_elem = self.dmstag_manager.get_global_indices(SL.ELEMENT)

        rows = np.hstack((L_indices, R_indices, all_elem))
        cols = np.hstack((R_indices, L_indices, all_elem))

        logger.debug(
            f"Number of rows: {len(rows)}, Number of cols: {len(cols)}",
            extra={"context": "Solver SETUP"},
        )
        logger.debug(
            f"Range of rows: {np.min(rows)} to {np.max(rows)}",
            extra={"context": "Solver SETUP"},
        )
        logger.debug(
            f"Range of cols: {np.min(cols)} to {np.max(cols)}",
            extra={"context": "Solver SETUP"},
        )

        self.A = PETSc.Mat().create()
        n_elems = self.dmstag_manager.get_count(SL.ELEMENT)
        n_elems_global = self.dmstag_manager.get_count(SL.ELEMENT, is_global=True)

        self.A.setSizes([(n_elems, n_elems_global), (n_elems, n_elems_global)])
        self.A.setFromOptions()

        logger.debug(
            f"Starting preallocation...",
            extra={"context": "Solver SETUP"},
        )
        self.A.setPreallocationCOO(rows, cols)

        logger.debug(
            f"A Mat: {self.A.getInfo()}",
            extra={"context": "Solver SETUP"},
        )

        logger.debug(
            f"Local size of A Mat: {self.A.getLocalSize()}",
            extra={"context": "Solver SETUP"},
        )

        self.ksp = PETSc.KSP().create()
        self.ksp.setType(PETSc.KSP.Type.FGMRES)
        self.ksp.getPC().setType(PETSc.PC.Type.MG)
        self.ksp.setOperators(self.A)
        self.ksp.setFromOptions()

    def update_system(self):
        """
        Variables that change during time stepping:
            Accumulation Coefficient (gamma)
            Fluid Transmissibility (Ty, Tz, Tx)
            Formation Volume Factor (B)

        """
        logger.info(
            "Updating system variables...", extra={"context": f"Solver UPDATE [{self.iteration}]"}
        )
        pressure = self.dmstag_manager.get_field("pressure", SL.ELEMENT)

        B_array = self.B_REF_FORMATION_FACTOR / (
            1.0 + self.FLUID_COMPRESSIBILITY * (pressure - self.INITIAL_PRESSURE)
        )
        self.dmstag_manager.set_field("formation_factor", SL.ELEMENT, B_array)

        phi_array = self.POROSITY_REF / (
            1.0 + self.PORE_COMPRESSIBILITY * (pressure - self.INITIAL_PRESSURE)
        )
        self.dmstag_manager.set_field("porosity", SL.ELEMENT, phi_array)

        volume = self.dmstag_manager.get_field("volume", SL.ELEMENT)
        gamma_array = (volume / self.ALPHA_C) * (
            (self.POROSITY_REF * self.PORE_COMPRESSIBILITY / B_array)
            + (phi_array * self.FLUID_COMPRESSIBILITY / self.B_REF_FORMATION_FACTOR)
        )
        self.dmstag_manager.set_field("accumulation_coefficient", SL.ELEMENT, gamma_array)

        logger.debug(
            f"Setting fluid transmissibility and formation volume factor...",
            extra={"context": f"Solver UPDATE [{self.iteration}]"},
        )
        self.comm.barrier()
        pressure = self.dmstag_manager.get_field("pressure", SL.ELEMENT, use_ghost=True)
        for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]:
            logger.debug(
                f"Updating fluid transmissibility",
                extra={"context": f"Solver UPDATE {STENCIL_NAME[face_stencil]} [{self.iteration}]"},
            )
            transmissibility = self.dmstag_manager.get_field("transmissibility", face_stencil)
            fluid_transmissibility = np.zeros_like(transmissibility)

            cur_internal_faces_indices = self.internal_faces_indices[face_stencil]
            cur_boundary_faces_indices = self.boundary_faces_indices[face_stencil]

            if len(cur_internal_faces_indices) > 0:
                cur_internal_elem_pairs = self.internal_elem_pairs[face_stencil]
                pressure_pairs = pressure[cur_internal_elem_pairs]
                pressure_avg = np.mean(pressure_pairs, axis=1)
                initial_pressure_pairs_avg = np.mean(
                    self.INITIAL_PRESSURE_GHOST[cur_internal_elem_pairs], axis=1
                )
                B_pairs = self.B_REF_FORMATION_FACTOR / (
                    1.0 + self.FLUID_COMPRESSIBILITY * (pressure_avg - initial_pressure_pairs_avg)
                )
                fluid_transmissibility[cur_internal_faces_indices] = transmissibility[
                    cur_internal_faces_indices
                ] / (B_pairs * self.VISCOSITY)

            if len(cur_boundary_faces_indices) > 0:
                cur_boundary_elem_pairs = self.boundary_elem_pairs[face_stencil]
                pressure_elem = pressure[cur_boundary_elem_pairs[:, 0]]
                initial_pressure_elem = self.INITIAL_PRESSURE_GHOST[cur_boundary_elem_pairs[:, 0]]
                B_elem = self.B_REF_FORMATION_FACTOR / (
                    1.0 + self.FLUID_COMPRESSIBILITY * (pressure_elem - initial_pressure_elem)
                )
                fluid_transmissibility[cur_boundary_faces_indices] = transmissibility[
                    cur_boundary_faces_indices
                ] / (B_elem * self.VISCOSITY)

            logger.debug(
                f"Average fluid transmissibility: " f"{np.mean(fluid_transmissibility)}",
                extra={"context": f"Solver UPDATE {STENCIL_NAME[face_stencil]} [{self.iteration}]"},
            )
            self.dmstag_manager.set_field(
                "fluid_transmissibility", face_stencil, fluid_transmissibility
            )

    def set_boundary_conditions(self):
        """
        Set boundary conditions by modifying transmissibility and source terms.
        """
        logger.info(
            "Setting boundary conditions...",
            extra={"context": f"Solver BOUNDARY [{self.iteration}]"},
        )

        S_u = np.zeros(self.dmstag_manager.get_ghost_sizes(SL.ELEMENT)).flatten()

        for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]:
            fluid_transmissibility = self.dmstag_manager.get_field(
                "fluid_transmissibility", face_stencil
            )
            face_area = self.dmstag_manager.get_field("area", face_stencil)
            face_centroid = self.dmstag_manager.get_coordinates(face_stencil)
            normal_vector = self.dmstag_manager.get_field("normal_vector", face_stencil)

            boundary_directions = self.boundary[face_stencil]

            cur_boundary_faces_indices = self.boundary_faces_indices[face_stencil]
            if len(cur_boundary_faces_indices) == 0:
                continue

            cur_boundary_elem_pairs = self.boundary_elem_pairs[face_stencil]
            b_trans = fluid_transmissibility[cur_boundary_faces_indices]
            b_area = face_area[cur_boundary_faces_indices]
            b_dirs = boundary_directions[cur_boundary_faces_indices]
            b_elems = cur_boundary_elem_pairs[:, 0]

            for direction_enum, (bc_type, bc_func) in self.BOUNDARY.items():
                matching_indices = np.where(b_dirs == direction_enum.value)[0]
                logger.debug(
                    f"Boundary conditions with type {bc_type}: {len(matching_indices)}",
                    extra={
                        "context": f"Solver BOUNDARY {STENCIL_NAME[face_stencil]} [{self.iteration}]"
                    },
                )
                if len(matching_indices) == 0:
                    continue

                elements_to_update = b_elems[matching_indices]
                xf, yf, zf = face_centroid[cur_boundary_faces_indices[matching_indices]].T

                if bc_type == "dirichlet":
                    bc_values = bc_func(xf, yf, zf, self.current_time)
                    trans_dir = b_trans[matching_indices]
                    source_contribution = trans_dir * bc_values
                    np.add.at(S_u, elements_to_update, source_contribution)

                elif bc_type == "neumann":
                    # Neumann boundary condition: normal derivative of pressure
                    nx, ny, nz = normal_vector[cur_boundary_faces_indices[matching_indices]].T
                    bc_values = bc_func(xf, yf, zf, self.current_time, nx, ny, nz)
                    area_neu = b_area[matching_indices]
                    source_contribution = area_neu * bc_values
                    np.add.at(S_u, elements_to_update, source_contribution)

                    neu_face_indices = cur_boundary_faces_indices[matching_indices]
                    fluid_transmissibility[neu_face_indices] = 0.0

            self.dmstag_manager.set_field(
                "fluid_transmissibility", face_stencil, fluid_transmissibility
            )

        local_nx, local_ny, local_nz = self.dmstag_manager.get_sizes(SL.ELEMENT)
        start_ix, start_iy, start_iz = self.dmstag_manager.get_corners(SL.ELEMENT)

        ghost_nx, ghost_ny, ghost_nz = self.dmstag_manager.get_ghost_sizes(SL.ELEMENT)

        for well_name, well in self.WELLS.items():
            i, j, k = well["index"]

            # Check if the well is within the local domain of this process
            if (
                (start_ix <= i < start_ix + local_nx)
                and (start_iy <= j < start_iy + local_ny)
                and (start_iz <= k < start_iz + local_nz)
            ):
                logger.debug(
                    f"Adding well source term for well {well_name} at index ({i}, {j}, {k})",
                    extra={"context": f"Solver BOUNDARY [{self.iteration}]"},
                )
                local_i, local_j, local_k = i - start_ix, j - start_iy, k - start_iz

                ghost_i, ghost_j, ghost_k = local_i + 1, local_j + 1, local_k + 1
                ghost_idx = ghost_i + ghost_nx * ghost_j + ghost_nx * ghost_ny * ghost_k

                # Add well source term: q_l_sc + J * p_wf
                S_u[ghost_idx] += well["rate"] + well["J"] * well["pressure"]

        self.dmstag_manager.set_field("external_source", SL.ELEMENT, S_u, use_ghost=True)

    def assemble_system(self):
        """
        Assemble the system matrix and right-hand side vector.
        """
        logger.info(
            "Assembling the system...", extra={"context": f"Solver ASSEMBLE [{self.iteration}]"}
        )

        S_u = self.dmstag_manager.get_field("external_source", SL.ELEMENT)

        xe, ye, ze = self.dmstag_manager.get_coordinates(SL.ELEMENT).T
        S_u += self.SOURCE_TERM(xe, ye, ze, self.current_time)

        gamma = self.dmstag_manager.get_field("accumulation_coefficient", SL.ELEMENT)
        pressure = self.dmstag_manager.get_field("pressure", SL.ELEMENT)

        b = -(S_u + gamma * pressure / self.TIME_STEP)
        self.dmstag_manager.set_field("rhs", SL.ELEMENT, b)

        Tp = np.zeros(self.dmstag_manager.get_ghost_sizes(SL.ELEMENT)).flatten()

        for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]:
            faces_transmissibility = self.dmstag_manager.get_field(
                "fluid_transmissibility", face_stencil
            )
            cur_internal_faces_indices = self.internal_faces_indices[face_stencil]
            cur_boundary_faces_indices = self.boundary_faces_indices[face_stencil]

            if len(cur_internal_faces_indices) > 0:
                cur_internal_elem_pairs = self.internal_elem_pairs[face_stencil]
                np.add.at(
                    Tp,
                    cur_internal_elem_pairs[:, 0],
                    faces_transmissibility[cur_internal_faces_indices],
                )
                np.add.at(
                    Tp,
                    cur_internal_elem_pairs[:, 1],
                    faces_transmissibility[cur_internal_faces_indices],
                )
            if len(cur_boundary_faces_indices) > 0:
                cur_boundary_elem_pairs = self.boundary_elem_pairs[face_stencil]
                np.add.at(
                    Tp,
                    cur_boundary_elem_pairs[:, 0],
                    faces_transmissibility[cur_boundary_faces_indices],
                )

        self.dmstag_manager.set_field("elem_transmissibility", SL.ELEMENT, Tp, use_ghost=True)
        Tp_local = self.dmstag_manager.get_field("elem_transmissibility", SL.ELEMENT)

        all_internal_transmissibility = np.hstack(
            tuple(
                [
                    self.dmstag_manager.get_field("fluid_transmissibility", face_stencil)[
                        self.internal_faces_indices[face_stencil]
                    ]
                    for face_stencil in [SL.LEFT, SL.BACK, SL.DOWN]
                ]
            )
        )

        data = np.hstack(
            (
                all_internal_transmissibility,
                all_internal_transmissibility,
                -(Tp_local + gamma / self.TIME_STEP),
            )
        )

        self.A.setValuesCOO(data, PETSc.InsertMode.INSERT_VALUES)

    def solve_system(self):
        """
        Solve the linear system Ax = b
        """
        logger.info("Solving the system...", extra={"context": f"Solver SOLVE [{self.iteration}]"})
        self.comm.barrier()
        self.b = self.dmstag_manager.get_field_vec("rhs", SL.ELEMENT, self.b)
        if self.x is None:
            self.x = self.dmstag_manager.get_field_vec("pressure", SL.ELEMENT)
        else:
            self.x = self.dmstag_manager.get_field_vec("pressure", SL.ELEMENT, self.x)

        # View A
        viewer = PETSc.Viewer().createBinary(
            f"{self.dirname}/output/A_{int(self.current_time * 1000):06d}.bin", "w", comm=self.comm
        )
        self.A.view(viewer)
        self.ksp.solve(self.b, self.x)

        # self.export_to_npy(self.b, f"b_{self.iteration}")
        # self.export_to_npy(self.x, f"x_{self.iteration}")

        self.dmstag_manager.restore_field_vec("pressure", SL.ELEMENT, self.x)

    def export_to_npy(self, vec, name):
        """
        Export a vector to a .npy file.
        """
        sct, out_vec = PETSc.Scatter.toAll(vec)
        sct.scatter(vec, out_vec)
        if self.rank == 0:
            dirname = self.dirname + "/output"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            index = self.comm.getSize()
            np.save(f"{dirname}/{name}_{index}.npy", out_vec.getArray())
            logger.info(
                f"Exported {name} vector to {dirname}/{name}_{index}.npy",
                extra={"context": "Solver EXPORT"},
            )

    def check(self, time):
        """
        Check the solution against the analytical solution using standard error norms.
        """
        if not self.analytical_solution:
            logger.debug(
                "Problem does not have an analytical solution, skipping check.",
                extra={"context": "Solver CHECK"},
            )
            return

        element_centroid = self.dmstag_manager.get_coordinates(SL.ELEMENT)
        pressure_sim = self.dmstag_manager.get_field("pressure", SL.ELEMENT)

        pressure_truth = self.analytical_solution(
            element_centroid[:, 0],
            element_centroid[:, 1],
            element_centroid[:, 2],
            time,
        )

        error_vector = pressure_sim - pressure_truth

        norm_truth_l1 = np.linalg.norm(pressure_truth, ord=1)
        norm_truth_l2 = np.linalg.norm(pressure_truth, ord=2)
        norm_truth_linf = np.linalg.norm(pressure_truth, ord=np.inf)

        if norm_truth_l2 == 0:
            logger.warning(
                "Analytical solution norm is zero, cannot compute relative error.",
                extra={"context": "Solver CHECK"},
            )
            return

        l1_error = np.linalg.norm(error_vector, ord=1) / norm_truth_l1
        l2_error = np.linalg.norm(error_vector, ord=2) / norm_truth_l2
        linf_error = np.linalg.norm(error_vector, ord=np.inf) / norm_truth_linf

        log_message = f"L1={l1_error:.4e}, L2={l2_error:.4e}, L_inf={linf_error:.4e}"

        if l2_error > 1e-3:
            logger.warning(log_message, extra={"context": "Solver CHECK"})
        else:
            logger.info(log_message, extra={"context": "Solver CHECK"})

        self.out_info["l1_error"].append(l1_error)
        self.out_info["l2_error"].append(l2_error)
        self.out_info["linf_error"].append(linf_error)

    def get_info(self):
        """
        Return the solver information.
        """
        return out_info

    def postprocess(self, time):
        dirname = self.dirname + "/output"
        if self.rank == 0:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        viewer = PETSc.Viewer().createVTK(
            f"{dirname}/output_t{int(time * 1000):06d}.vts", "w", comm=self.comm
        )
        self.dmstag_manager.view_dmda(SL.ELEMENT, viewer)
        logger.debug(
            f"Postprocessing completed, output saved to {dirname}/output_t{int(time * 1000):06d}.vts",
            extra={"context": "Solver POSTPROCESS"},
        )

    def __repr__(self):
        """Return a string representation of the TPFASolver."""
        return "TODO"
