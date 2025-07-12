import logging
import numpy as np

from petsc4py import PETSc
from .dmda_manager import DMDAManager

logger = logging.getLogger(__name__)


def _stencil_to_string(stencil_loc):
    """
    Convert a DMStag.StencilLocation to a string representation.
    Parameters:
        stencil_loc: DMStag.StencilLocation
    Returns:
        String representation of the stencil location
    """
    SL = PETSc.DMStag.StencilLocation
    for name, value in SL.__dict__.items():
        if value == stencil_loc:
            return name


class DMStagManager:
    """Manager for DOFs in a DMStag grid"""

    def __init__(self, dim):
        """
        Initialize a DMStagManager for a DMStag object.
        Parameters:
            dim: The dimension of the DMStag object
        """
        self.dim = dim
        self.dof_count = [1] * (self.dim + 1)
        self.fields = [[] for _ in range(self.dim + 1)]
        self.all_fields = set()
        self.dof_map = {}

        self.connectivities = {}
        self.dmda_managers = {}
        self.dmstag = None

        self.local_vec = None
        self.global_vec = None

    def set_dm(self, dmstag):
        """Set the DMStag for this DMStagManager"""
        self.dmstag = dmstag
        self.global_vec = dmstag.createGlobalVec()
        self.local_vec = dmstag.createLocalVec()

    def add_field(self, name, size, cell_dim, is_shared=False):
        """
        Add a field to the DOF manager.
        Parameters:
            name: Name of the field (e.g., 'velocity', 'pressure')
            size: Number of DOFs for this field
            cell_dim: Dimension of the cell (0=vertex, 1=edge, 2=face, 3=element)
        """
        if self.dmstag is not None:
            raise ValueError("Cannot add fields after DMStag is set. The mesh is already setup.")

        if cell_dim > self.dim:
            raise ValueError(f"Cell dimension {cell_dim} exceeds DMStag dimension {self.dim}")

        if name in self.all_fields:
            raise ValueError(f"Field {name} already exists.")

        if is_shared:
            logger.debug(
                f"Adding shared field {name} with size {size} at index {self.dof_count[cell_dim]}",
                extra={"context": "DMSTAG"},
            )
            self.dof_map[name] = {
                "size": size,
                "cell_dim": cell_dim,
                "index": self.dof_count[cell_dim],
            }
            self.dof_count[cell_dim] += size
        else:
            logger.debug(f"Adding local field {name} with size {size}", extra={"context": "DMSTAG"})

        self.fields[cell_dim].append((name, size, is_shared))
        self.all_fields.add(name)

    def get_dof_count(self):
        """Return the current DOF count tuple"""
        return tuple(self.dof_count)

    def _stencil_to_cell_dim(self, stencil_loc):
        """
        Convert a stencil location to a cell dimension.
        Parameters:
            stencil_loc: DMStag.StencilLocation
        Returns:
            Cell dimension (0=vertex, 1=edge, 2=face, 3=element)
        """
        SL = PETSc.DMStag.StencilLocation
        if stencil_loc in [
            SL.BACK_DOWN_LEFT,
            SL.BACK_DOWN_RIGHT,
            SL.BACK_UP_LEFT,
            SL.BACK_UP_RIGHT,
            SL.FRONT_DOWN_LEFT,
            SL.FRONT_DOWN_RIGHT,
            SL.FRONT_UP_LEFT,
            SL.FRONT_UP_RIGHT,
        ]:
            return 0
        elif stencil_loc in [
            SL.BACK_DOWN,
            SL.BACK_UP,
            SL.BACK_LEFT,
            SL.BACK_RIGHT,
            SL.FRONT_DOWN,
            SL.FRONT_UP,
            SL.FRONT_LEFT,
            SL.FRONT_RIGHT,
            SL.DOWN_LEFT,
            SL.DOWN_RIGHT,
            SL.UP_LEFT,
            SL.UP_RIGHT,
        ]:
            return 1
        elif stencil_loc in [SL.BACK, SL.FRONT, SL.LEFT, SL.RIGHT, SL.UP, SL.DOWN]:
            return 2
        elif stencil_loc == SL.ELEMENT:
            return 3

    def _get_dmda_manager(self, stencil_loc):
        """
        Get or create a cell manager for a specific cell dimension and stencil location.
        Parameters:
            cell_dim: Dimension of the cell (0=vertex, 1=edge, 2=face, 3=element)
            stencil_loc: DMStag.StencilLocation for this cell type
        Returns:
            DMDAManager for the specified cell dimension and stencil location
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        cell_dim = self._stencil_to_cell_dim(stencil_loc)
        key = (cell_dim, stencil_loc)
        if key not in self.dmda_managers:
            logger.debug(
                f"Creating DMDAManager for stencil_loc={_stencil_to_string(stencil_loc)} with {len(self.fields[cell_dim])} fields",
                extra={"context": "DMSTAG"},
            )
            dmda, vec_split = self.dmstag.VecSplitToDMDA(
                self.global_vec, stencil_loc, -self.dof_count[cell_dim]
            )
            self.dmda_managers[key] = DMDAManager(
                dmda,
                _stencil_to_string(stencil_loc),
                vec_split,
                cell_dim,
                stencil_loc,
                self.fields[cell_dim],
            )
        return self.dmda_managers[key]

    def set_field(self, name, stencil_loc, values, use_ghost=False):
        """
        Set the local values for a field at a specific stencil location.
        Parameters:
            name: Field name
            stencil_loc: DMStag.StencilLocation to set values at
            values: Values to set (should match the size of the field)
            use_ghost: Whether to use the local vec indexing and set the local vec values.

        Notes:
        Automatically updates the global vector from the local vector after setting values.
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        if name not in self.all_fields:
            raise KeyError(f"Field {name} does not exist.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        dmda_manager.set_values(name, values, use_ghost)

    def get_field(self, name, stencil_loc, use_ghost=False):
        """
        Get the values for a field at a specific stencil location.
        Parameters:
            name: Field name
            stencil_loc: DMStag.StencilLocation to get values from
            use_ghost: Wheter to use the local vec indexing and return the local vec values
        Returns:
            if return_vec is True, returns the PETSc.Vec for the field at the stencil location.
            else, Numpy array of values for the field
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        if name not in self.all_fields:
            raise KeyError(f"Field {name} does not exist.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.get_values(name, use_ghost)

    def get_field_vec(self, name, stencil_loc, vec=None):
        """
        Get the field vector for a specific field and stencil location.
        Parameters:
            name: Field name
            stencil_loc: DMStag.StencilLocation to get the vector for
            vec: Optional PETSc.Vec to use. If None, a new Vec will be created.
        Returns:
            PETSc.Vec for the specified field and stencil location
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        if name not in self.all_fields:
            raise KeyError(f"Field {name} does not exist.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.get_vec(name, vec)

    def restore_field_vec(self, name, stencil_loc, vec):
        """
        Restore the field vector after using it.
        Parameters:
            name: Field name
            stencil_loc: DMStag.StencilLocation to restore the vector for
            vec: The PETSc.Vec to restore
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        if name not in self.all_fields:
            raise KeyError(f"Field {name} does not exist.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        dmda_manager.restore_vec(name, vec)

    def get_coordinates(self, stencil_loc, use_ghost=False):
        """
        Get the coordinates for a specific stencil location.
        Parameters:
            stencil_loc: DMStag.StencilLocation to get coordinates from
        Returns:
            Numpy array of coordinates for the specified stencil location
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)

        return dmda_manager.get_coordinates(use_ghost)

    def get_connectivity(self, source_loc, target_loc):
        """
        Returns the index in the global array of the cells of target_loc that are closer to source_loc in the local vector
        Ex: If source_loc is ELEMENT and target_loc is BACK_DOWN_LEFT (points), it returns the indices of the points,
        in the local point vector (considering ghost points), that are connected to the elements in source_loc.
        With those indices, one can acess the fields of each point around the element in the local vector of DOFs.
        Parameters:
            ource_loc: DMStag.StencilLocation to get boundary information from
        Returns:
            Numpy array of shape (get_count(source_loc), MAX_CONNECTIVITY)
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        if source_loc in self.connectivities and target_loc in self.connectivities[source_loc]:
            return self.connectivities[source_loc][target_loc]

        src_dmda_manager = self._get_dmda_manager(source_loc)

        tgt_dmda_manager = self._get_dmda_manager(target_loc)

        SL = PETSc.DMStag.StencilLocation
        if target_loc not in [SL.BACK_DOWN_LEFT, SL.ELEMENT]:
            raise RuntimeError("Target stencil loc not implemented")

        source_name = _stencil_to_string(source_loc)
        target_name = _stencil_to_string(target_loc)
        logger.debug(
            f"Calculating connectivity for source_loc={source_name} and target_loc={target_name}",
            extra={"context": "DMSTAG"},
        )
        src_count = src_dmda_manager.get_count()

        src_corners = src_dmda_manager.corners
        src_sizes = src_dmda_manager.sizes

        logger.debug(
            f"Source corners: {src_corners}, sizes: {src_sizes}, count: {src_count}",
            extra={"context": "DMSTAG"},
        )

        src_tuples = src_dmda_manager.global_tuples

        # Target -> Source. Ex: Points around faces : diff[points][faces]
        diff = {
            SL.BACK_DOWN_LEFT: {
                SL.LEFT: [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
                SL.DOWN: [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
                SL.BACK: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                SL.ELEMENT: [
                    (0, 0, 0),
                    (0, 0, 1),
                    (1, 0, 1),
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 1, 1),
                    (1, 1, 1),
                    (1, 1, 0),
                ],
            },
            SL.ELEMENT: {
                SL.LEFT: [(0, 0, 0), (-1, 0, 0)],
                SL.DOWN: [(0, 0, 0), (0, -1, 0)],
                SL.BACK: [(0, 0, 0), (0, 0, -1)],
                SL.ELEMENT: [(0, 0, 0)],
            },
        }

        offsets_3d = np.array(diff[target_loc][source_loc])
        num_offsets = len(offsets_3d)

        connectivity_tuples = src_tuples[:, np.newaxis, :] + offsets_3d[np.newaxis, :, :]

        connectivity_tuples -= np.array(tgt_dmda_manager.ghost_corners)  # Adjust for ghost cells
        tgt_ghost_sizes = tgt_dmda_manager.ghost_sizes

        logger.debug(
            f"Target ghost sizes: {tgt_ghost_sizes}, ghost corners: {tgt_dmda_manager.ghost_corners}",
            extra={"context": "DMSTAG"},
        )

        nx_t, ny_t, nz_t = tgt_ghost_sizes
        tgt_strides = np.array([1, nx_t, nx_t * ny_t])

        connectivity_local = np.dot(connectivity_tuples, tgt_strides).astype(np.int32)

        if source_loc not in self.connectivities:
            self.connectivities[source_loc] = {}

        logger.debug(
            f"Shape of connectivity_local: {connectivity_local.shape}", extra={"context": "DMSTAG"}
        )
        logger.debug(
            f"Connectivity of {src_tuples[0]}: {connectivity_local[0].tolist()}",
            extra={"context": "DMSTAG"},
        )

        self.connectivities[source_loc][target_loc] = connectivity_local
        return connectivity_local

    def export_to_npy(self, vec, name):
        """
        Export a vector to a .npy file.
        """
        sct, out_vec = PETSc.Scatter.toAll(vec)
        sct.scatter(vec, out_vec)
        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0:
            index = PETSc.COMM_WORLD.getSize()
            np.save(f"example_1/output/{name}_{index}.npy", out_vec.getArray())

    def get_global_indices(self, stencil_loc):
        """
        Get the global indices for a specific stencil location.
        Parameters:
            stencil_loc: DMStag.StencilLocation to get global indices from
        Returns:
            Numpy array of global indices for the specified stencil location
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.global_indices

    def get_boundary(self, stencil_loc):
        """
        Get the boundary information for a specific stencil location.
        Parameters:
            stencil_loc: DMStag.StencilLocation to get boundary information from
        Returns:
            Numpy array of boundary information for the specified stencil location (dmda_manager.BOUNDARY_DIRECTION)
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)

        isFirstRank = self.dmstag.getIsFirstRank()
        isLastRank = self.dmstag.getIsLastRank()

        return dmda_manager.get_boundary(isFirstRank, isLastRank)

    def create_matrix(self, stencil_loc):
        """
        Create a PETSc matrix for the specified stencil location.
        Parameters:
            stencil_loc: DMStag.StencilLocation to create the matrix for
        Returns:
            PETSc.Mat for the specified stencil location
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.create_matrix()

    def get_count(self, stencil_loc, is_global=False):
        """Returns the count for a particular StencilLocation"""
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.get_global_count() if is_global else dmda_manager.get_count()

    def get_sizes(self, stencil_loc, is_global=False):
        """Returns the sizes for a particular StencilLocation"""
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.global_sizes if is_global else dmda_manager.sizes

    def get_corners(self, stencil_loc):
        """Returns the corners for a particular StencilLocation"""
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.corners

    def get_ghost_sizes(self, stencil_loc):
        """Returns the count for a particular StencilLocation"""
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.ghost_sizes

    def get_ghost_corners(self, stencil_loc):
        """Returns the corners for a particular StencilLocation"""
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.ghost_corners

    def get_offset(self, stencil_loc):
        """
        Returns the (flatten) index of the first element on the local vec respective to the global vec
        Parameters:
            stencil_loc: DMStag.StencilLocation to get offset from
        Returns:
            Numpy array of shape (3,) with the offset for the specified stencil location
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        return dmda_manager.get_offset()

    def view_dmda(self, stencil_loc, viewer):
        """
        View the DMDA for a specific stencil location using the provided viewer.
        """
        if self.dmstag is None:
            raise ValueError("DMStag is not set. Call set_dm() first.")

        dmda_manager = self._get_dmda_manager(stencil_loc)
        dmda_manager.view(viewer)

    def __repr__(self):
        """Return a string representation of the DMStagManager."""
        pass
