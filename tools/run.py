import logging
import cProfile
import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from pytpfa.model import TPFASolver

from scripts.configure import (
    setup_logging,
    get_tpfa_methods_to_profile,
    setup_line_profiler,
    save_profiling_results,
)


def parse_command_line_arguments():
    """Parse command line arguments using PETSc Options"""
    OptDB = PETSc.Options()

    name = OptDB.getString("name", "TPFA")
    reservoir_path = OptDB.getString("reservoir", "0")
    log_level = OptDB.getString("log_level", "INFO").upper()

    # New separated flags
    opt_mode = OptDB.getBool("opt", False)
    post_mode = OptDB.getBool("post", False)
    profile_mode = OptDB.getBool("profile", False)

    if reservoir_path == "0":
        raise ValueError(
            "Error: the path to the reservoir file was not provided. Use '-reservoir <path>' to specify it."
        )

    # Pass the raw OptDB for other scripts that might need it
    petsc_opts = {key: OptDB.getString(key) for key in OptDB.getAll().keys()}

    return {
        "name": name,
        "reservoir_path": reservoir_path,
        "log_level": log_level,
        "opt_mode": opt_mode,
        "post_mode": post_mode,
        "profile_mode": profile_mode,
        "petsc_opts": petsc_opts,
    }


def run_simulation(config):
    """Run the TPFA simulation with the given configuration"""
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    # Logging to a file is now controlled by the profile flag
    log_file = setup_logging(config["reservoir_path"], config["profile_mode"])

    main_logger = logging.getLogger(__name__)
    main_logger.info(f"Starting {config['name']} simulation", extra={"context": "MAIN"})
    main_logger.info(f"Reservoir path: {config['reservoir_path']}", extra={"context": "MAIN"})
    main_logger.info(
        f"Running with options: OPT={config['opt_mode']}, POST={config['post_mode']}, PROFILE={config['profile_mode']}",
        extra={"context": "MAIN"},
    )

    cprofile_profiler = None
    line_profiler = None

    try:
        simulation = TPFASolver(config["name"])

        if config["profile_mode"]:
            main_logger.info("Running with profiling enabled", extra={"context": "MAIN"})
            cprofile_profiler = cProfile.Profile()
            cprofile_profiler.enable()
            main_logger.debug("cProfile enabled", extra={"context": "PROFILE"})

            methods_to_profile = get_tpfa_methods_to_profile(PETSc.Options())
            line_profiler = setup_line_profiler(simulation, methods_to_profile)
            line_profiler.enable()
            main_logger.debug(
                f"Line profiler enabled for methods: {methods_to_profile}",
                extra={"context": "PROFILE"},
            )

        # The 'checks' parameter is the inverse of 'opt_mode'
        # The 'postprocess' parameter is controlled by 'post_mode'
        simulation.solve(
            config["reservoir_path"],
            postprocess=config["post_mode"],
            checks=(not config["opt_mode"]),
        )
        main_logger.info("Simulation completed successfully", extra={"context": "MAIN"})

    except Exception as e:
        main_logger.error(f"Simulation failed: {str(e)}", extra={"context": "ERROR"})
        raise

    finally:
        if config["profile_mode"]:
            if cprofile_profiler is not None:
                cprofile_profiler.disable()
                main_logger.debug("cProfile disabled", extra={"context": "PROFILE"})

            if line_profiler is not None:
                line_profiler.disable()
                main_logger.debug("Line profiler disabled", extra={"context": "PROFILE"})

            save_profiling_results(config["reservoir_path"], cprofile_profiler, line_profiler, rank)


def main():
    """Main entry point for the simulation"""
    try:
        config = parse_command_line_arguments()
        run_simulation(config)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
