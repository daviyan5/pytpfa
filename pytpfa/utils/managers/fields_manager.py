import logging
import numpy as np

from petsc4py import PETSc

logger = logging.getLogger(__name__)


class FieldsManager:
    def __init__(self, name, size, is_shared, cell_info):
        """
        Initialize a field with a name, size, type and shared status.
        """
        self.name = name

        self.size = size
        self.is_shared = is_shared

        count, ghost_count, offset, dmda, global_vec, local_vec = cell_info

        self.count = count
        self.ghost_count = ghost_count
        self.offset = offset

        self.dmda = dmda
        self.dmda_name = dmda.getName()

        self.needs_update_from_global = False

        if is_shared:
            self.gvecs = []
            self.lvecs = []
            for dof in range(size):
                gvec = global_vec.duplicate()
                lvec = local_vec.duplicate()
                gvec.setName(f"{name}_{dof}")

                self.gvecs.append(gvec)
                self.lvecs.append(lvec)

            # Pre-allocated arrays for bulk operations only
            self.larray = np.squeeze(np.zeros((ghost_count, size), dtype=np.float64))
            self.array = np.squeeze(np.zeros((count, size), dtype=np.float64))
        else:
            # Non-shared fields just use numpy arrays
            self.array = np.squeeze(np.zeros((count, size), dtype=np.float64))

    def update_from_global(self):
        """Update local vectors from global vectors (global->local)"""
        if not self.is_shared or not self.needs_update_from_global:
            return

        for dof in range(self.size):
            self.dmda.globalToLocal(self.gvecs[dof], self.lvecs[dof])

        self.needs_update_from_global = False

    def update_from_local(self):
        """Update global vectors from local vectors (local->global with ADD)"""
        if not self.is_shared:
            return

        for dof in range(self.size):
            # Zero global before adding from local
            self.gvecs[dof].zeroEntries()
            self.dmda.localToGlobal(self.lvecs[dof], self.gvecs[dof], PETSc.InsertMode.ADD_VALUES)

    def get_values(self, use_ghost=False, index=None):
        """Get field values, optionally with ghost points or specific indices"""
        if not self.is_shared:
            return self.array if index is None else self.array[index]

        # Update if needed
        if use_ghost and self.needs_update_from_global:
            logger.debug(
                f"Updating local vectors for field '{self.name}'",
                extra={"context": f"DMDA {self.dmda_name}"},
            )
            self.update_from_global()

        source_vecs = self.lvecs if use_ghost else self.gvecs

        if index is None:
            target_array = self.larray if use_ghost else self.array

            for dof in range(self.size):
                if target_array.ndim > 1:
                    target_array[:, dof] = source_vecs[dof].getArray()
                else:
                    target_array[:] = source_vecs[dof].getArray()
            return target_array
        else:
            is_single_item = isinstance(index, int)
            rows_to_get = np.array([index] if is_single_item else index, dtype=np.int32)
            num_rows = len(rows_to_get)

            result = np.empty((num_rows, self.size), dtype=np.float64)
            for dof in range(self.size):
                source_vecs[dof].getValues(
                    rows_to_get, result[:, dof] if result.ndim > 1 else result
                )

            return result[0] if is_single_item else result

    def set_values(self, values, use_ghost=False, index=None):
        """Set field values, optionally with ghost points or specific indices"""

        if np.isscalar(values):
            target_count = self.ghost_count if use_ghost else self.count
            values = np.full((target_count, self.size), values, dtype=np.float64)
            values = np.squeeze(values)

        if not self.is_shared:
            if index is None:
                self.array[:] = values
            else:
                self.array[index] = values
            return

        target_vecs = self.lvecs if use_ghost else self.gvecs

        if index is None:
            # Full array setting
            for dof in range(self.size):
                target_vecs[dof].assemblyBegin()
            for dof in range(self.size):
                target_vecs[dof].setArray(values[:, dof] if values.ndim > 1 else values)
            for dof in range(self.size):
                target_vecs[dof].assemblyEnd()
        else:
            is_single_item = isinstance(index, int)
            rows_to_set = np.array([index] if is_single_item else index, dtype=np.int32)

            if is_single_item and values.ndim == 1:
                values = values.reshape(1, -1)

            for dof in range(self.size):
                target_vecs[dof].setValues(
                    rows_to_set,
                    values[:, dof] if values.ndim > 1 else values,
                    PETSc.InsertMode.INSERT_VALUES,
                )
                target_vecs[dof].assemble()

        if use_ghost:
            self.update_from_local()
        else:
            self.needs_update_from_global = True

    def get_vec(self, component=None):
        """
        Get the vector(s) for this field.
        If component is specified, return that component's vector.
        Otherwise return all vectors.
        """
        if not self.is_shared:
            raise ValueError(f"Field '{self.name}' is not shared across ranks")

        if component is not None:
            if component >= self.size:
                raise ValueError(f"Component {component} out of range for field '{self.name}'")
            return self.gvecs[component]

        return self.gvecs if self.size > 1 else self.gvecs[0]
