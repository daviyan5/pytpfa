import os
import re
import logging
import pstats
import cProfile
from line_profiler import LineProfiler
from pathlib import Path
from petsc4py import PETSc
import sys


class RankFilter(logging.Filter):
    """Filter that only allows rank 0 for INFO/DEBUG, but allows all ranks for WARNING+"""

    def __init__(self):
        super().__init__()
        self.comm = PETSc.COMM_WORLD
        self.rank = self.comm.getRank()

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        return self.rank == 0


class ColoredConsoleFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[38;5;75m",
        "INFO": "\033[38;5;114m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;167m",
        "CRITICAL": "\033[48;5;167m",
    }
    CONTEXT_COLORS = [
        "\033[38;5;176m",
        "\033[38;5;111m",
        "\033[38;5;152m",
        "\033[38;5;222m",
        "\033[38;5;151m",
        "\033[38;5;174m",
        "\033[38;5;140m",
        "\033[38;5;109m",
        "\033[38;5;180m",
        "\033[38;5;246m",
    ]
    RESET = "\033[0m"
    WHITE = "\033[38;5;252m"

    _context_color_map = {}
    _color_index = 0
    FILE_INFO_WIDTH = 22

    def format(self, record):
        comm = PETSc.COMM_WORLD
        rank = comm.getRank()
        size = comm.getSize()

        level_color = self.COLORS.get(record.levelname, self.RESET)
        levelname = f"{level_color}{record.levelname:<8}{self.RESET}"

        context = getattr(record, "context", "N/A")

        # Use a base context name for consistent coloring (e.g., "Solver ITERATION" for "Solver ITERATION [1]")
        base_context = re.sub(r"\s*\[.*\]$", "", context)

        if base_context not in self._context_color_map:
            color = self.CONTEXT_COLORS[self._color_index % len(self.CONTEXT_COLORS)]
            self._context_color_map[base_context] = color
            self._color_index += 1

        context_color = self._context_color_map[base_context]
        context_str = f"{context_color}{context:<25}{self.RESET}"

        msg = f"{self.WHITE}{record.msg}{self.RESET}"

        file_line = f"{record.filename}:{record.lineno}"
        record.fileinfo = f"[{file_line:<{self.FILE_INFO_WIDTH}}]"

        # Temporarily replace record attributes for formatting
        old_levelname = record.levelname
        old_context = getattr(record, "context", None)
        old_msg = record.msg
        old_rank = getattr(record, "rank", None)

        record.levelname = levelname
        record.context = context_str
        record.msg = msg
        record.rank = f"{rank+1}/{size}"

        formatted = super().format(record)

        # Restore original attributes
        record.levelname = old_levelname
        record.context = old_context
        record.msg = old_msg
        record.rank = old_rank

        return formatted


class ContextFormatter(logging.Formatter):
    FILE_INFO_WIDTH = 22

    def format(self, record):
        comm = PETSc.COMM_WORLD
        rank = comm.getRank()
        size = comm.getSize()

        context = getattr(record, "context", "N/A")
        record.context = context
        record.rank = f"{rank+1}/{size}"

        file_line = f"{record.filename}:{record.lineno}"
        record.fileinfo = f"[{file_line:<{self.FILE_INFO_WIDTH}}]"

        return super().format(record)


def get_next_index(directory, prefix):
    """Get the next available index for log files in a directory"""
    if not os.path.exists(directory):
        return 0

    files = os.listdir(directory)
    pattern = re.compile(rf"{prefix}_(\d+)\.log")
    indices = []

    for file in files:
        match = pattern.match(file)
        if match:
            indices.append(int(match.group(1)))

    return max(indices) + 1 if indices else 0


def setup_logging(repository_path, optimize_mode=False, log_prefix="simulation"):
    """Configure logging for the simulation or other tools."""
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    repo_dir = os.path.dirname(os.path.abspath(repository_path))
    log_dir = os.path.join(repo_dir, "logs")
    Path(log_dir).mkdir(exist_ok=True)

    rank_dir = os.path.join(log_dir, f"rank_{rank}")
    Path(rank_dir).mkdir(exist_ok=True)

    index = get_next_index(rank_dir, log_prefix)
    log_file = os.path.join(rank_dir, f"{log_prefix}_{index}.log")

    log_format = "[%(asctime)s] %(fileinfo)s [Rank: %(rank)-4s] [%(context)-34s] [%(levelname)-5s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    console_formatter = ColoredConsoleFormatter(log_format, date_format)

    console_handler.setFormatter(console_formatter)

    rank_filter = RankFilter()
    console_handler.addFilter(rank_filter)
    root_logger.addHandler(console_handler)

    if not optimize_mode:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = ContextFormatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return log_file


def get_tpfa_methods_to_profile(OptDB):
    """Get list of TPFA methods to profile from command line options"""
    default_methods = ["solve", "solve_system"]
    methods_str = OptDB.getString("lp_methods", "")

    if methods_str:
        return [m.strip() for m in methods_str.split(",")]
    return default_methods


def setup_line_profiler(simulation, methods_to_profile):
    """Setup line profiler for specified methods"""
    lp = LineProfiler()

    for method_name in methods_to_profile:
        if hasattr(simulation, method_name):
            method = getattr(simulation, method_name)
            lp.add_function(method)

    return lp


def save_profiling_results(repository_path, cprofile_profiler, line_profiler, rank):
    """Save profiling results to files"""
    repo_dir = os.path.dirname(repository_path)
    log_dir = os.path.join(repo_dir, "logs")
    rank_dir = os.path.join(log_dir, f"rank_{rank}")

    logger = logging.getLogger(__name__)

    if cprofile_profiler is not None:
        index = get_next_index(rank_dir, "simulation")
        cprofile_file = os.path.join(rank_dir, f"cProfile_{index}.log")

        with open(cprofile_file, "w") as f:
            ps = pstats.Stats(cprofile_profiler, stream=f)
            ps.sort_stats("cumulative")
            ps.print_stats()

        logger.info(f"cProfile saved to: {cprofile_file}", extra={"context": "PROFILE"})

    if line_profiler is not None:
        index = get_next_index(rank_dir, "simulation")
        line_profile_file = os.path.join(rank_dir, f"line_profile_{index}.log")

        with open(line_profile_file, "w") as f:
            line_profiler.print_stats(stream=f)

        lprof_file = os.path.join(rank_dir, f"line_profile_{index}.lprof")
        line_profiler.dump_stats(lprof_file)

        logger.info(f"Line profile saved to: {line_profile_file}", extra={"context": "PROFILE"})
