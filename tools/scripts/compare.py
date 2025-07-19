# -*- coding: utf-8 -*-
import os
import sys
import glob
import logging
import re
import numpy as np
import pyvista as pv
import pandas as pd
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc

from configure import setup_logging

logger = logging.getLogger(__name__)


def parse_command_line():
    """Parses command line arguments using PETSc.Options."""
    opts = PETSc.Options()
    reservoir_path = opts.getString("reservoir", None)
    if not reservoir_path:
        raise ValueError("Missing command-line argument: -reservoir <path_to_reservoir.ini>")
    return reservoir_path


def get_vts_errors(output_grid, truth_grid):
    """Compares specified fields in VTS grids and returns a dictionary of L2 errors."""
    errors = {}
    field_map = {"Unnamed.rhs_0": "rhs", "Unnamed.pressure_0": "pressure"}

    for out_field, truth_field in field_map.items():
        if out_field not in output_grid.point_data or truth_field not in truth_grid.point_data:
            logger.warning(
                f"Field pair ({out_field}, {truth_field}) not found.",
                extra={"context": "VTS_CHECK"},
            )
            continue

        out_data = output_grid.point_data[out_field]
        truth_data = truth_grid.point_data[truth_field]

        l2_error = np.linalg.norm(out_data - truth_data)
        errors[truth_field] = l2_error

    return errors


def get_bin_error(output_path, truth_path):
    """Compares two PETSc Vecs from .bin files and returns the L2 error of their difference."""
    try:
        out_viewer = PETSc.Viewer().createBinary(output_path, "r")
        truth_viewer = PETSc.Viewer().createBinary(truth_path, "r")

        out_vec = PETSc.Vec().create()
        out_vec.load(out_viewer)

        truth_vec = PETSc.Vec().create()
        truth_vec.load(truth_viewer)

        diff_vec = truth_vec.copy()
        diff_vec.axpy(-1, out_vec)  # diff_vec = diff_vec - out_vec

        return diff_vec.norm(PETSc.NormType.NORM_2)
    except Exception as e:
        logger.error(
            f"Failed to compare BIN files {os.path.basename(output_path)}: {e}",
            extra={"context": "BIN_COMPARE"},
        )
        return np.nan


def log_summary_table(results_data):
    """Converts the results dictionary to a pandas DataFrame and logs it."""
    if not results_data:
        logger.warning(
            "No results were generated to create a summary.", extra={"context": "SUMMARY"}
        )
        return

    df = pd.DataFrame.from_dict(results_data, orient="index")
    df.index.name = "Timestep"
    df = df.reindex(sorted(df.index), axis=0)
    df = df.reindex(sorted(df.columns), axis=1)

    summary_string = df.to_string(float_format="%.4e")

    logger.info(f"Comparison Summary:\n{summary_string}", extra={"context": "SUMMARY"})


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

        all_output_files = glob.glob(os.path.join(output_dir, "*.*"))

        for output_path in sorted(all_output_files):
            filename = os.path.basename(output_path)
            truth_path = os.path.join(truth_dir, filename)

            if not os.path.exists(truth_path):
                logger.warning(
                    f"Truth file not found for {filename}. Skipping.",
                    extra={"context": "FILE_LOOP"},
                )
                continue

            logger.info(f"Processing file: {filename}", extra={"context": "FILE_LOOP"})

            match = re.search(r"_t?(\d+)", filename)
            if not match:
                logger.warning(
                    f"Could not determine timestep for {filename}. Skipping.",
                    extra={"context": "FILE_LOOP"},
                )
                continue

            timestep_key = f"t{int(match.group(1))}"
            results_data.setdefault(timestep_key, {})

            if filename.endswith(".vts"):
                try:
                    output_grid = pv.read(output_path)
                    truth_grid = pv.read(truth_path)
                    vts_errors = get_vts_errors(output_grid, truth_grid)
                    results_data[timestep_key].update(vts_errors)
                except Exception as e:
                    logger.error(
                        f"Failed to process VTS file {filename}: {e}",
                        extra={"context": "VTS_COMPARE"},
                    )

            elif filename.endswith(".bin"):
                field_name = os.path.splitext(filename)[0].split("_t")[0]
                bin_error = get_bin_error(output_path, truth_path)
                results_data[timestep_key][field_name] = bin_error

        log_summary_table(results_data)
        logger.info("Comparison complete.", extra={"context": "MAIN"})

    except Exception as e:
        logger.critical(
            f"A critical error occurred: {e}", extra={"context": "CRITICAL"}, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
