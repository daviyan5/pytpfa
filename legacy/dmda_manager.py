import logging
import numpy as np

from enum import Enum, auto
from petsc4py import PETSc

logger = logging.getLogger(__name__)


class BoundaryDirection(Enum):
    """Enum for boundary directions"""

    LEFT = 1
    DOWN = 2
    BACK = 3
    RIGHT = 4
    UP = 5
    FRONT = 6
    NONE = 0


class Fields:

    def __init__(self, name, index, dmda_name, size, is_shared, cell_info):
        """
        Initialize a field with a name, size, and shared status.
        """
        self.name = name
        self.index = index
        self.dmda_name = dmda_name
        self.size = size
        self.is_shared = is_shared

        count, ghost_count, offset, cur_dof, dof_count = cell_info
        self.dof = cur_dof
        self.scatter = None

        if is_shared:
            field_indexes = np.arange(cur_dof, cur_dof + size, dtype=np.int32)

            # Indices of the field in the local vector
            self.ghost_indices = (np.arange(ghost_count, dtype=np.int32)) * dof_count
            self.ghost_indices = (self.ghost_indices[:, np.newaxis] + field_indexes).flatten()

            # Indices of the field in the global vector
            self.local_indices = (np.arange(count, dtype=np.int32)) * dof_count
            self.local_indices = (
                self.local_indices[:, np.newaxis] + field_indexes
            ).flatten() + offset * dof_count

            self.array = np.zeros(count * size, dtype=np.float64)
            self.garray = np.zeros(ghost_count * size, dtype=np.float64)

            # IS with the local indices of the field
            self.local_indices_is = PETSc.IS().createGeneral(
                self.local_indices, comm=PETSc.COMM_WORLD
            )

            logger.debug(
                f"Field '{name}' sizes: {self.local_indices_is.getSizes()}",
                extra={"context": f"DMDA {self.dmda_name}"},
            )

        else:
            self.array = np.zeros((count, size), dtype=np.float64)


class DMDAManager:
    """Simplified manager for a specific stencil/cell type in a DMDA grid"""

    def __init__(self, dmda, dmda_name, global_vec, cell_dim, stencil_loc, fields):
        """
        Initialize a cell manager for a specific cell dimension and stencil location
        Parameters:
            dmda: DMDA object
            dmda_name: Name of the DMDA
            global_vec: Global vector from the DMDA
            cell_dim: Dimension of the cell (0=vertex, 1=edge, 2=face, 3=element)
            stencil_loc: DMStag.StencilLocation for this cell type
            fields: Description of fields per cell
        """
        self.dmda = dmda
        self.dmda_name = dmda_name
        self.dmda.setName(f"Mesh_{dmda_name}")

        self.global_vec = global_vec
        self.global_vec.setOption(PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)

        self.local_vec = dmda.getLocalVec()
        self.lg_map = dmda.getLGMap()

        self.cell_dim = cell_dim
        self.stencil_loc = stencil_loc

        self.dim = dmda.getDimension()
        self.dof_count = self.dmda.getDof()

        self._initialize_geometry()

        self.fields = {}

        self.used_dof = 0
        for index, (name, size, is_shared) in enumerate(fields):
            cell_info = (
                self.get_count(),
                self.get_ghost_count(),
                self.get_offset(),
                self.used_dof,
                self.dof_count,
            )
            if is_shared:
                if self.used_dof + size > self.dof_count:
                    raise ValueError(
                        f"Field '{name}' exceeds the total number of DOFs ({self.dof_count})."
                    )
                for idx in range(size):
                    self.dmda.setFieldName(self.used_dof + idx, f"{name}_{idx}")
                self.used_dof += size
            field = Fields(name, index, dmda_name, size, is_shared, cell_info)
            self.fields[name] = field

        self.updated_field = np.zeros(len(self.fields), dtype=bool)

        logger.debug(
            f"Initialized DMDAManager for {dmda_name} with {len(self.fields.keys())} fields",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        logger.debug(
            f"Count: {self.get_count()}, Ghost Count: {self.get_ghost_count()}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        logger.debug(
            f"Offset: {self.get_offset()}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        logger.debug(
            f"Used DOFs: {self.used_dof}, Total DOFs: {self.dof_count}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        logger.debug(
            f"Global Vec Range: {self.global_vec.getOwnershipRange()}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )

    def _initialize_geometry(self):
        """Initialize geometric information about the grid"""
        # Get corners and sizes
        corners, sizes = self.dmda.getCorners()
        self.corners = corners
        self.sizes = sizes

        ghost_corners, ghost_sizes = self.dmda.getGhostCorners()
        self.ghost_corners = ghost_corners
        self.ghost_sizes = ghost_sizes

        self.global_sizes = self.dmda.getSizes()

        self.count = np.prod(self.sizes[: self.dim])
        self.ghost_count = np.prod(self.ghost_sizes[: self.dim])
        self.global_count = np.prod(self.global_sizes[: self.dim])

        self.offset = self.global_vec.getOwnershipRange()[0] // self.dof_count

        nx, ny, nz = sizes

        i_indices = np.arange(nx) + corners[0]
        j_indices = np.arange(ny) + corners[1]
        k_indices = np.arange(nz) + corners[2]

        ii, jj, kk = np.meshgrid(i_indices, j_indices, k_indices, indexing="ij")

        # Tuples of the cells that are part of this DMDA
        self.global_tuples = np.stack(
            [ii.ravel(order="F"), jj.ravel(order="F"), kk.ravel(order="F")], axis=1
        )
        nx_global, ny_global, nz_global = self.global_sizes

        # Global indices of the cells in this DMDA
        self.global_indices = self.global_tuples.dot([1, nx_global, nx_global * ny_global])
        self.global_indices = self.global_indices.astype(np.int32)

        self.global_indices_is = PETSc.IS().createGeneral(
            self.global_indices, comm=PETSc.COMM_WORLD
        )

        logger.debug(
            f"DMDA corners: {self.corners}, sizes: {self.sizes}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        logger.debug(
            f"DMDA ghost corners: {self.ghost_corners}, ghost sizes: {self.ghost_sizes}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        logger.debug(
            f"DMDA global sizes: {self.global_sizes}", extra={"context": f"DMDA {self.dmda_name}"}
        )

    def get_count(self):
        """Return the number of local cells of this type"""
        return self.count

    def get_ghost_count(self):
        """Return the number of local cells (with ghosts) of this type"""
        return self.ghost_count

    def get_global_count(self):
        """Return the number of global cells of this type"""
        return self.global_count

    def get_offset(self):
        """Returns the (flatten) index of the first element on the local vec respective to the global vec"""
        return self.offset

    def set_values(self, name, values, use_ghost=False):
        """
        Set the values for specific DOFs at all cells.
        Parameters:
            name: Name of the field
            values: Values to set (shape should match (n_cells, n_dofs))
            use_ghost: If True, use ghost indexing (Only supported for incrementation)
        """
        if name not in self.fields:
            raise ValueError(f"Field '{name}' not found in DMDA.")

        field = self.fields[name]

        if use_ghost and not field.is_shared:
            raise ValueError(
                f"Field '{name}' is not shared across ranks, cannot use ghost indexing."
            )

        size = field.size
        count = self.count if not use_ghost else self.ghost_count

        if not isinstance(values, np.ndarray):
            values = np.squeeze(np.full((count, size), values))

        if np.prod(values.shape) != count * size:
            raise ValueError(
                f"Values shape {values.shape} does not match expected shape ({count}, {size}) for field '{name}'."
            )

        if field.is_shared:
            if use_ghost:
                work_vec = self.local_vec.duplicate()
                work_vec.zeroEntries()

                work_vec.setValues(
                    field.ghost_indices, values.ravel(), PETSc.InsertMode.INSERT_VALUES
                )
                work_vec.assemble()
                self.dmda.localToGlobal(work_vec, self.global_vec, PETSc.InsertMode.ADD_VALUES)
                self.global_vec.assemble()

            else:
                self.global_vec.setValues(
                    field.local_indices, values.ravel(), PETSc.InsertMode.INSERT_VALUES
                )
                self.global_vec.assemble()

            self.updated_field[field.index] = True

        else:
            self.fields[name].array = values

    def get_values(self, name, use_ghost=False):
        """
        Get the values for specific DOFs at all cells.
        Parameters:
            name: Name of the field
            use_ghost: Return the values using local indexing
        Returns:
            Array of values with shape (n_cells, n_dofs)
        """
        if name not in self.fields:
            raise ValueError(f"Field '{name}' not found in DMDA.")

        field = self.fields[name]

        if use_ghost and not field.is_shared:
            raise ValueError(
                f"Field '{name}' is not shared across ranks, cannot use ghost indexing."
            )
        if field.is_shared:
            if use_ghost and self.updated_field[field.index]:
                logger.debug(
                    f"Updating local vector for field '{name}' before getting values.",
                    extra={"context": f"DMDA {self.dmda_name}"},
                )
                self.dmda.globalToLocal(
                    self.global_vec, self.local_vec, PETSc.InsertMode.INSERT_VALUES
                )
                self.updated_field.fill(False)

            vec = self.local_vec if use_ghost else self.global_vec
            arr = field.garray if use_ghost else field.array

            indices = field.ghost_indices if use_ghost else field.local_indices

            size = field.size
            if len(indices) == 0:
                return np.zeros((0, size), dtype=np.float64)

            vec.getValues(indices, arr)
            return np.squeeze(arr.reshape((self.ghost_count if use_ghost else self.count, size)))
        else:
            arr = field.array
            return arr

    def get_vec(self, name, vec=None):
        """
        Get the subvector corresponding to the name for this cell type.
        Parameters:
            name: Name of the field to get the subvector for
            vec: Optional PETSc vector to use instead of creating a new one
        Returns:
            PETSc vector containing the values for the specified DOFs
        """
        if name not in self.fields:
            raise ValueError(f"Field '{name}' not found in DMDA.")

        field = self.fields[name]

        if not field.is_shared:
            raise ValueError(f"Field '{name}' is not shared across ranks.")

        petsc_is = field.local_indices_is
        app_is = self.global_indices_is

        if vec is None:
            vec = PETSc.Vec().create(self.dmda.getComm())
            vec.setSizes(app_is.getSizes())
            vec.setFromOptions()

        if not field.scatter:
            fscatter = PETSc.Scatter().create(self.global_vec, petsc_is, vec, app_is)
            field.fscatter = fscatter
            fscatter.setName(f"{self.dmda.getName()}_{name}_scatter")
        else:
            fscatter = field.fscatter

        fscatter.scatter(
            self.global_vec,
            vec,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

        vec.setName(f"{self.dmda.getName()}_{name}")
        return vec

    def restore_vec(self, name, vec):
        """
        Restore a subvector to the global vector.
        """
        if name not in self.fields:
            raise ValueError(f"Field '{name}' not found in DMDA.")

        field = self.fields[name]

        if not field.is_shared:
            raise ValueError(f"Field '{name}' is not shared across ranks.")

        petsc_is = field.local_indices_is
        app_is = self.global_indices_is

        scatter = PETSc.Scatter().create(vec, app_is, self.global_vec, petsc_is)
        scatter.scatter(
            vec,
            self.global_vec,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

        scatter.destroy()

        self.dmda.globalToLocal(self.global_vec, self.local_vec, PETSc.InsertMode.INSERT_VALUES)

    def get_boundary(self, isFirstRank, isLastRank):
        """
        Get the boundary information for this cell type.
        Returns:
            List of boundary direction for each cell in the local grid
        """

        # If the flag = 1, means the boundary of this direction is in the current rank
        is_left, is_down, is_back = isFirstRank
        is_right, is_up, is_front = isLastRank

        logger.debug(f"Boundary flags: ", extra={"context": f"DMDA {self.dmda_name}"})
        logger.debug(
            f"Left: {is_left}, Down: {is_down}, Back: {is_back}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        logger.debug(
            f"Right: {is_right}, Up: {is_up}, Front: {is_front}",
            extra={"context": f"DMDA {self.dmda_name}"},
        )

        x_left, y_down, z_back = 0, 0, 0
        x_right, y_up, z_front = (
            x_left + self.sizes[0],
            y_down + self.sizes[1],
            z_back + self.sizes[2],
        )

        indices = np.arange(self.get_count()).reshape(self.sizes, order="F")
        boundary = np.zeros_like(indices, dtype=np.int32)

        # Some stencils cannot be at all boundaries directions, so we check.
        # TODO: Add edges
        SL = PETSc.DMStag.StencilLocation

        possibilities = {
            # (LEFT, DOWN, BACK, RIGHT, UP, FRONT)
            SL.LEFT: (1, 0, 0, 1, 0, 0),
            SL.RIGHT: (1, 0, 0, 1, 0, 0),
            SL.DOWN: (0, 1, 0, 0, 1, 0),
            SL.UP: (0, 1, 0, 0, 1, 0),
            SL.BACK: (0, 0, 1, 0, 0, 1),
            SL.FRONT: (0, 0, 1, 0, 0, 1),
        }

        if self.stencil_loc in possibilities:
            is_left = is_left and possibilities[self.stencil_loc][0]
            is_down = is_down and possibilities[self.stencil_loc][1]
            is_back = is_back and possibilities[self.stencil_loc][2]

            is_right = is_right and possibilities[self.stencil_loc][3]
            is_up = is_up and possibilities[self.stencil_loc][4]
            is_front = is_front and possibilities[self.stencil_loc][5]

        total_boundary = 0
        if is_left:
            boundary[x_left, :, :] = BoundaryDirection.LEFT.value
            total_boundary += boundary.shape[1] * boundary.shape[2]
        if is_right:
            boundary[x_right - 1, :, :] = BoundaryDirection.RIGHT.value
            total_boundary += boundary.shape[1] * boundary.shape[2]
        if is_down:
            boundary[:, y_down, :] = BoundaryDirection.DOWN.value
            total_boundary += boundary.shape[0] * boundary.shape[2]
        if is_up:
            boundary[:, y_up - 1, :] = BoundaryDirection.UP.value
            total_boundary += boundary.shape[0] * boundary.shape[2]
        if is_back:
            boundary[:, :, z_back] = BoundaryDirection.BACK.value
            total_boundary += boundary.shape[0] * boundary.shape[1]
        if is_front:
            boundary[:, :, z_front - 1] = BoundaryDirection.FRONT.value
            total_boundary += boundary.shape[0] * boundary.shape[1]

        logger.debug(
            f"Total boundary cells: {total_boundary}", extra={"context": f"DMDA {self.dmda_name}"}
        )

        return boundary.ravel(order="F")

    def get_coordinates(self, use_ghost=False):
        """
        Get the coordinates of the cells in the DMDA.
        Parameters:
            use_ghost: If True, return the coordinates with ghost points
        Returns:
            Array of coordinates with shape (n_cells, dim)
        """
        total_points = self.get_count() if not use_ghost else self.get_ghost_count()
        coordinates_vec = (
            self.dmda.getCoordinates() if not use_ghost else self.dmda.getCoordinatesLocal()
        )
        return coordinates_vec.getArray(readonly=True).reshape(total_points, self.dim)

    def view(self, viewer):
        """
        View the DMDA structure.
        Parameters:
            viewer: Viewer to use for displaying the DMDA
        """
        logger.debug(
            f"Viewing DMDA {self.dmda.getName()} with {len(self.fields)} fields",
            extra={"context": f"DMDA {self.dmda_name}"},
        )
        for dof in range(self.used_dof):
            logger.debug(
                f"Field {dof}: {self.dmda.getFieldName(dof)}",
                extra={"context": f"DMDA {self.dmda_name}"},
            )
        self.dmda.view(viewer)
        self.global_vec.view(viewer)

    def __repr__(self):
        """Return a string representation of the DMDAManager."""
        pass
