import os
import sys
import glob
import logging
import re
import numpy as np
import pyvista as pv
import petsc4py
import pandas as pd
from tabulate import tabulate

petsc4py.init(sys.argv)
from petsc4py import PETSc

from configure import setup_logging

logger = logging.getLogger(__name__)


def parse_command_line():
    opts = PETSc.Options()
    reservoir_path = opts.getString("reservoir", None)
    if not reservoir_path:
        raise ValueError("Missing command-line argument: -reservoir <path_to_reservoir.ini>")
    return reservoir_path


def get_vts_errors(output_grid, truth_grid):
    errors = {}
    field_map = {"Unnamed.rhs_0": "rhs", "Unnamed.pressure_0": "pressure"}
    atol = 1e-1

    for out_field, truth_field in field_map.items():
        if out_field not in output_grid.point_data or truth_field not in truth_grid.point_data:
            logger.warning(
                f"Field pair ({out_field}, {truth_field}) not found.",
                extra={"context": "VTS_CHECK"},
            )
            continue

        out_data = output_grid.point_data[out_field]
        truth_data = truth_grid.point_data[truth_field]

        diff_vector = out_data - truth_data
        l2_error = np.linalg.norm(diff_vector)
        num_diffs = np.count_nonzero(~np.isclose(out_data, truth_data, atol=atol))
        errors[truth_field] = (l2_error, num_diffs)

    return errors


def get_bin_error(output_path, truth_path):
    try:
        out_viewer = PETSc.Viewer().createBinary(output_path, "r")
        truth_viewer = PETSc.Viewer().createBinary(truth_path, "r")
        out_vec, truth_vec = PETSc.Vec().create(), PETSc.Vec().create()
        out_vec.load(out_viewer)
        truth_vec.load(truth_viewer)

        diff_vec = truth_vec.copy()
        diff_vec.axpy(-1, out_vec)
        l2_error = diff_vec.norm(PETSc.NormType.NORM_2)
        num_diffs = np.count_nonzero(~np.isclose(diff_vec.getArray(), 0.0, atol=1e-8))

        return (l2_error, num_diffs)
    except Exception as e:
        logger.error(
            f"Failed to compare BIN Vec files {os.path.basename(output_path)}: {e}",
            extra={"context": "BIN_COMPARE"},
        )
        return (np.nan, np.nan)


def get_matrix_error(output_path, truth_path):
    try:
        # Load both matrices from their binary files
        out_viewer = PETSc.Viewer().createBinary(output_path, "r")
        truth_viewer = PETSc.Viewer().createBinary(truth_path, "r")
        out_mat, truth_mat = PETSc.Mat().create(), PETSc.Mat().create()
        out_mat.load(out_viewer)
        truth_mat.load(truth_viewer)

        # Check for and warn about differing dimensions
        truth_rows, truth_cols = truth_mat.getSize()
        out_rows, out_cols = out_mat.getSize()

        if truth_rows != out_rows or truth_cols != out_cols:
            logger.warning(
                f"Different matrix dimensions: truth ({truth_rows}x{truth_cols}) vs output ({out_rows}x{out_cols})",
                extra={"context": "BIN_COMPARE"},
            )

        # Create a copy of the truth matrix to store the difference
        diff_mat = truth_mat.copy()

        # Subtract the output matrix from the truth matrix (diff = truth - output)
        diff_mat.axpy(-1, out_mat, structure=PETSc.Mat.Structure.DIFFERENT)

        # Calculate the Frobenius norm of the difference matrix
        frob_norm = diff_mat.norm(PETSc.NormType.NORM_FROBENIUS)

        # Get the non-zero values from the difference matrix to count them
        _, _, diff_values = diff_mat.getValuesCSR()

        # Define a tolerance for what counts as a "different" value
        atol = 1e-8
        num_diffs = np.count_nonzero(np.abs(diff_values) > atol)

        return (frob_norm, num_diffs)

    except Exception as e:
        logger.error(
            f"Failed to compare BIN Mat files {os.path.basename(output_path)}: {e}",
            extra={"context": "BIN_COMPARE"},
        )
        return (np.nan, np.nan)


def log_summary_table(results_data):
    """Converts results to a formatted table using tabulate and logs it."""
    if not results_data:
        logger.warning(
            "No results were generated to create a summary.", extra={"context": "SUMMARY"}
        )
        return

    headers = sorted(list(set(key for data in results_data.values() for key in data.keys())))
    table = []

    for timestep, data in sorted(results_data.items(), key=lambda item: int(item[0][1:])):
        row = [timestep]
        for header in headers:
            l2_error, num_diffs = data.get(header, (np.nan, np.nan))
            cell_text = f"{l2_error:.3e} ({int(num_diffs)})" if not np.isnan(l2_error) else "---"
            row.append(cell_text)
        table.append(row)

    full_headers = ["Timestep"] + headers
    summary_string = tabulate(table, headers=full_headers, tablefmt="grid", stralign="center")

    logger.info(
        f"Comparison Summary (L2/Frobenius Error | Number of Differences):\n{summary_string}",
        extra={"context": "SUMMARY"},
    )


def main():
    """Main function to orchestrate the comparison process."""
    try:
        reservoir_path = parse_command_line()
        setup_logging(reservoir_path, log_prefix="compare")
        logger.info("Starting comparison tool.", extra={"context": "MAIN"})

        base_dir = os.path.dirname(os.path.abspath(reservoir_path))
        output_dir = os.path.join(base_dir, "output")
        truth_dir = os.path.join(base_dir, "truth")

        if not os.path.isdir(output_dir) or not os.path.isdir(truth_dir):
            raise FileNotFoundError("Both 'output' and 'truth' directories must exist.")

        results_data = {}

        vts_files = glob.glob(os.path.join(output_dir, "*.vts"))
        bin_files = glob.glob(os.path.join(output_dir, "*.bin"))
        all_output_files = sorted(vts_files + bin_files)

        for output_path in all_output_files:
            filename = os.path.basename(output_path)
            truth_path = os.path.join(truth_dir, filename)

            if not os.path.exists(truth_path):
                logger.warning(
                    f"Truth file not found for {filename}. Skipping.",
                    extra={"context": "FILE_LOOP"},
                )
                continue

            # --- VTS File Processing ---
            if filename.endswith(".vts"):
                match = re.search(r"_t(\d+)", filename)
                if not match:
                    continue

                timestep_key = f"t{int(match.group(1))}"
                results_data.setdefault(timestep_key, {})
                try:
                    output_grid, truth_grid = pv.read(output_path), pv.read(truth_path)
                    vts_errors = get_vts_errors(output_grid, truth_grid)
                    results_data[timestep_key].update(vts_errors)
                except Exception as e:
                    logger.error(
                        f"Failed to process VTS file {filename}: {e}",
                        extra={"context": "VTS_COMPARE"},
                    )

            # --- BIN File Processing ---
            elif filename.endswith(".bin"):
                match = match = re.match(r"([A-Za-z_]+)_(\d+)\.bin$", filename)
                if not match:
                    logger.warning(
                        f"Could not parse BIN filename format: {filename}. Skipping.",
                        extra={"context": "FILE_LOOP"},
                    )
                    continue

                field_name = match.group(1)
                timestep_key = f"t{int(match.group(2))}"
                results_data.setdefault(timestep_key, {})

                if field_name == "A":
                    error_tuple = get_matrix_error(output_path, truth_path)
                else:
                    error_tuple = get_bin_error(output_path, truth_path)

                results_data[timestep_key][field_name] = error_tuple

        log_summary_table(results_data)
        logger.info("Comparison complete.", extra={"context": "MAIN"})

    except Exception as e:
        logger.critical(
            f"A critical error occurred: {e}", extra={"context": "CRITICAL"}, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
