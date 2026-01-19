#!/usr/bin/env python3
"""
execute.py - TPFA Solver Test Suite

Available tests:
  - correctness:  Convergence order verification
  - performance:  CPU vs GPU benchmark (same MPI count)
  - strong:       Strong scaling (CPU+GPU, 1-4 processes)
  - weak:         Weak scaling (CPU+GPU, 1-4 processes)
  - bandwidth:    Memory bandwidth and roofline analysis
  - solvers:      Solver comparison (KSP + Preconditioner combinations)
  - all:          All tests (runs solvers FIRST, then uses best config for others)

Usage:
  python execute.py --config jarvis --test performance
  python execute.py --config jarvis --test solvers
  python execute.py --config jarvis --test bandwidth
  python execute.py --config jarvis --test all

Note: When using --test all, the solver comparison runs first and the best
solver configuration is automatically used for performance, scaling, and
bandwidth tests. This ensures empirically optimal solver selection.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import time
import threading
import logging
import re
import numpy as np
import psutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
import configparser
import yaml
from datetime import datetime
import argparse
from dataclasses import dataclass, field


def numpy_to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


def save_yaml(data: Any, filepath: Union[str, Path]) -> None:
    clean_data = numpy_to_python(data)
    with open(filepath, "w") as f:
        yaml.dump(clean_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


class TestLogger:
    def __init__(self, results_dir: Path, test_type: str, timestamp: str):
        self.results_dir = Path(results_dir)
        self.test_type = test_type
        self.timestamp = timestamp
        self.logs_dir = self.results_dir / test_type / timestamp / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._setup_loggers()
    
    def _setup_loggers(self):
        logging.root.handlers = []
        self.master_log = self.logs_dir / "execution.log"
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        master_handler = logging.FileHandler(self.master_log, mode='w')
        master_handler.setFormatter(file_formatter)
        master_handler.setLevel(logging.DEBUG)
        self.logger = logging.getLogger(f"tpfa.{self.test_type}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.addHandler(master_handler)
        self._handlers = [master_handler]
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def section(self, title: str):
        self.logger.info("=" * 70)
        self.logger.info(f"  {title}")
        self.logger.info("=" * 70)
    
    def subsection(self, title: str):
        self.logger.info("-" * 50)
        self.logger.info(f"  {title}")
        self.logger.info("-" * 50)
    
    def metric(self, name: str, value: Any, unit: str = ""):
        self.logger.info(f"  {name:.<40} {value:>12} {unit}")
    
    def close(self):
        for handler in self._handlers:
            handler.close()
        self.logger.handlers.clear()


_logger: Optional[TestLogger] = None


def get_logger() -> TestLogger:
    global _logger
    if _logger is None:
        raise RuntimeError("Logger not initialized.")
    return _logger


def setup_logging(results_dir: Path, test_type: str, timestamp: str) -> TestLogger:
    global _logger
    _logger = TestLogger(results_dir, test_type, timestamp)
    return _logger


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}", flush=True)
    print(f"{msg:^70}", flush=True)
    print(f"{'='*70}{Colors.END}", flush=True)
    if _logger:
        _logger.section(msg)


def print_section(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'-'*70}", flush=True)
    print(f"{msg}", flush=True)
    print(f"{'-'*70}{Colors.END}", flush=True)
    if _logger:
        _logger.subsection(msg)


def print_success(msg: str):
    print(f"{Colors.GREEN}[OK] {msg}{Colors.END}", flush=True)
    if _logger:
        _logger.info(f"[OK] {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.END}", flush=True)
    if _logger:
        _logger.warning(msg)


def print_error(msg: str):
    print(f"{Colors.RED}[ERR] {msg}{Colors.END}", flush=True)
    if _logger:
        _logger.error(msg)


def print_metric(name: str, value: str, unit: str = "", highlight: bool = False):
    if highlight:
        print(f"  {Colors.BOLD}{name:.<40} {Colors.GREEN}{value:>12} {unit}{Colors.END}", flush=True)
    else:
        print(f"  {name:.<40} {value:>12} {unit}", flush=True)
    if _logger:
        _logger.metric(name, value, unit)


@dataclass
class TestCase:
    name: str
    path: str
    description: str
    meshes: List[List[int]]
    tolerance_check: bool = False


@dataclass
class Config:
    run_script: str = ""
    results_dir: str = ""
    temp_dir: str = "/tmp"

    physical_cores: int = 32
    logical_cpus: int = 64
    numa_nodes: int = 2
    num_gpus: int = 4
    vram_per_gpu_gb: int = 16
    total_ram_gb: int = 251
    gpu_model: str = "Quadro RTX 5000"
    gpu_bandwidth_gb_s: float = 448.0

    mpi_processes: List[int] = field(default_factory=lambda: [1, 2, 4])
    bind_to: str = "none"
    map_by: str = "none"
    omp_num_threads: int = 1

    use_mps: bool = False
    vec_type: str = "cuda"
    mat_type: str = "aijcusparse"

    ksp_type: str = "fgmres"
    pc_type: str = "gamg"
    ksp_rtol: float = 1e-16

    correctness_cases: List[TestCase] = field(default_factory=list)
    correctness_timeout: float = 2.0
    correctness_use_gpu: bool = False
    correctness_optimized: bool = False
    correctness_mpi: int = 1
    correctness_postprocess: bool = False

    performance_reservoir: str = ""
    performance_mesh: List[int] = field(default_factory=lambda: [100, 100, 100])
    performance_mpi: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    performance_runs: int = 3
    performance_timeout: float = 2.0
    performance_optimized: bool = True

    strong_reservoir: str = ""
    strong_mesh: List[int] = field(default_factory=lambda: [150, 150, 150])
    strong_mpi: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    strong_runs: int = 3
    strong_timeout: float = 2.0
    strong_optimized: bool = True

    weak_reservoir: str = ""
    weak_base_mesh: List[int] = field(default_factory=lambda: [100, 100, 100])
    weak_mpi: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    weak_decomposition: Dict[int, List[int]] = field(default_factory=dict)
    weak_runs: int = 3
    weak_timeout: float = 2.0
    weak_optimized: bool = True

    bandwidth_reservoir: str = ""
    bandwidth_mesh: List[int] = field(default_factory=lambda: [270, 270, 270])
    bandwidth_mpi: int = 16
    bandwidth_num_gpus: int = 4
    bandwidth_runs: int = 1
    bandwidth_timeout: float = 12.0
    bandwidth_optimized: bool = True
    bandwidth_bytes_per_cell: int = 64
    bandwidth_stencil_size: int = 7
    bandwidth_flops_per_cell: int = 14

    solver_comparison_reservoir: str = ""
    solver_comparison_mesh: List[int] = field(default_factory=lambda: [150, 150, 150])
    solver_comparison_mpi: int = 4
    solver_comparison_runs: int = 2
    solver_comparison_use_gpu: bool = True
    solver_comparison_optimized: bool = True
    solver_comparison_timeout: float = 4.0
    solver_comparison_solvers: List[Dict] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "paths" in data:
            config.run_script = data["paths"].get("run_script", "")
            config.results_dir = data["paths"].get("results_dir", "./results")
            config.temp_dir = data["paths"].get("temp_dir", "/tmp")

        if "hardware" in data:
            config.physical_cores = data["hardware"].get("physical_cores", 32)
            config.logical_cpus = data["hardware"].get("logical_cpus", 64)
            config.numa_nodes = data["hardware"].get("numa_nodes", 2)
            config.num_gpus = data["hardware"].get("num_gpus", 4)
            config.vram_per_gpu_gb = data["hardware"].get("vram_per_gpu_gb", 16)
            config.total_ram_gb = data["hardware"].get("total_ram_gb", 251)
            config.gpu_model = data["hardware"].get("gpu_model", "Quadro RTX 5000")
            config.gpu_bandwidth_gb_s = data["hardware"].get("gpu_bandwidth_gb_s", 448.0)

        if "mpi" in data:
            config.mpi_processes = data["mpi"].get("processes", [1, 2, 4])
            config.bind_to = data["mpi"].get("bind_to", "none")
            config.map_by = data["mpi"].get("map_by", "none")
            config.omp_num_threads = data["mpi"].get("omp_num_threads", 1)

        if "gpu" in data:
            config.use_mps = data["gpu"].get("use_mps", False)
            config.vec_type = data["gpu"].get("vec_type", "cuda")
            config.mat_type = data["gpu"].get("mat_type", "aijcusparse")

        if "solver" in data:
            config.ksp_type = data["solver"].get("ksp_type", "fgmres")
            config.pc_type = data["solver"].get("pc_type", "gamg")
            config.ksp_rtol = data["solver"].get("ksp_rtol", 1e-8)

        if "correctness" in data:
            cases_data = data["correctness"].get("cases", [])
            for case in cases_data:
                tc = TestCase(
                    name=case.get("name", "unknown"),
                    path=case.get("path", ""),
                    description=case.get("description", ""),
                    meshes=case.get("meshes", [[20, 20, 20]]),
                )
                config.correctness_cases.append(tc)
            config.correctness_timeout = data["correctness"].get("timeout_hours", 2.0)
            config.correctness_use_gpu = data["correctness"].get("use_gpu", False)
            config.correctness_optimized = data["correctness"].get("optimized", False)
            config.correctness_mpi = data["correctness"].get("mpi_processes", 1)
            config.correctness_postprocess = data["correctness"].get("postprocess", False)

        if "performance" in data:
            config.performance_reservoir = data["performance"].get("reservoir_base", "")
            config.performance_mesh = data["performance"].get("mesh", [100, 100, 100])
            config.performance_mpi = data["performance"].get("mpi_processes", [1, 2, 4, 8, 16])
            config.performance_runs = data["performance"].get("runs", 3)
            config.performance_timeout = data["performance"].get("timeout_hours", 2.0)
            config.performance_optimized = data["performance"].get("optimized", True)

        if "strong_scaling" in data:
            config.strong_reservoir = data["strong_scaling"].get("reservoir_base", "")
            config.strong_mesh = data["strong_scaling"].get("mesh", [150, 150, 150])
            config.strong_mpi = data["strong_scaling"].get("mpi_processes", [1, 2, 3, 4])
            config.strong_runs = data["strong_scaling"].get("runs", 3)
            config.strong_timeout = data["strong_scaling"].get("timeout_hours", 2.0)
            config.strong_optimized = data["strong_scaling"].get("optimized", True)

        if "weak_scaling" in data:
            config.weak_reservoir = data["weak_scaling"].get("reservoir_base", "")
            config.weak_base_mesh = data["weak_scaling"].get("base_mesh", [100, 100, 100])
            config.weak_mpi = data["weak_scaling"].get("mpi_processes", [1, 2, 3, 4])
            config.weak_runs = data["weak_scaling"].get("runs", 3)
            config.weak_timeout = data["weak_scaling"].get("timeout_hours", 2.0)
            decomp = data["weak_scaling"].get("decomposition", {})
            config.weak_decomposition = {int(k): v for k, v in decomp.items()}
            config.weak_optimized = data["weak_scaling"].get("optimized", True)

        if "bandwidth" in data:
            config.bandwidth_reservoir = data["bandwidth"].get("reservoir_base", "")
            config.bandwidth_mesh = data["bandwidth"].get("mesh", [270, 270, 270])
            config.bandwidth_mpi = data["bandwidth"].get("mpi_processes", 16)
            config.bandwidth_num_gpus = data["bandwidth"].get("num_gpus", 4)
            config.bandwidth_runs = data["bandwidth"].get("runs", 1)
            config.bandwidth_timeout = data["bandwidth"].get("timeout_hours", 12.0)
            config.bandwidth_optimized = data["bandwidth"].get("optimized", True)
            if "analysis" in data["bandwidth"]:
                config.bandwidth_bytes_per_cell = data["bandwidth"]["analysis"].get("bytes_per_cell", 64)
                config.bandwidth_stencil_size = data["bandwidth"]["analysis"].get("stencil_size", 7)
                config.bandwidth_flops_per_cell = data["bandwidth"]["analysis"].get("flops_per_cell", 14)

        if "solver_comparison" in data:
            config.solver_comparison_reservoir = data["solver_comparison"].get("reservoir_base", "")
            config.solver_comparison_mesh = data["solver_comparison"].get("mesh", [150, 150, 150])
            config.solver_comparison_mpi = data["solver_comparison"].get("mpi_processes", 4)
            config.solver_comparison_runs = data["solver_comparison"].get("runs", 2)
            config.solver_comparison_use_gpu = data["solver_comparison"].get("use_gpu", True)
            config.solver_comparison_optimized = data["solver_comparison"].get("optimized", True)
            config.solver_comparison_timeout = data["solver_comparison"].get("timeout_hours", 4.0)
            config.solver_comparison_solvers = data["solver_comparison"].get("solvers", [])

        return config

    def print_summary(self, test_type: str):
        print_header("TEST CONFIGURATION")
        
        print(f"\n{Colors.BOLD}Hardware:{Colors.END}", flush=True)
        print_metric("Physical cores", str(self.physical_cores))
        print_metric("NUMA nodes", str(self.numa_nodes))
        print_metric("GPUs", str(self.num_gpus), f"x {self.vram_per_gpu_gb}GB VRAM")
        print_metric("GPU Model", self.gpu_model)
        print_metric("GPU Peak Bandwidth", f"{self.gpu_bandwidth_gb_s:.0f}", "GB/s")
        print_metric("RAM Total", str(self.total_ram_gb), "GB")

        if test_type == "correctness":
            print(f"\n{Colors.BOLD}Test: CORRECTNESS{Colors.END}", flush=True)
            print_metric("MPI processes", str(self.correctness_mpi))
            print_metric("GPU", "Yes" if self.correctness_use_gpu else "No")

        elif test_type == "performance":
            m = self.performance_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: PERFORMANCE (CPU vs GPU){Colors.END}", flush=True)
            print_metric("Mesh", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.performance_mpi))

        elif test_type == "strong":
            m = self.strong_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: STRONG SCALING (CPU+GPU){Colors.END}", flush=True)
            print_metric("Fixed mesh", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.strong_mpi))

        elif test_type == "weak":
            m = self.weak_base_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: WEAK SCALING (CPU+GPU){Colors.END}", flush=True)
            print_metric("Base/process", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.weak_mpi))

        elif test_type == "bandwidth":
            m = self.bandwidth_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: BANDWIDTH / ROOFLINE ANALYSIS{Colors.END}", flush=True)
            print_metric("Mesh", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.bandwidth_mpi))
            print_metric("GPUs", str(self.bandwidth_num_gpus))
            print_metric("Bytes/cell", str(self.bandwidth_bytes_per_cell))
            print_metric("FLOPs/cell", str(self.bandwidth_flops_per_cell))

        elif test_type == "solvers":
            m = self.solver_comparison_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: SOLVER COMPARISON{Colors.END}", flush=True)
            print_metric("Mesh", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.solver_comparison_mpi))
            print_metric("GPU", "Yes" if self.solver_comparison_use_gpu else "No")
            print_metric("Solvers to test", str(len(self.solver_comparison_solvers)))
            for s in self.solver_comparison_solvers:
                print(f"    - {s['name']}: {s.get('description', '')}", flush=True)


def prepare_gpu_mps(config: Config) -> bool:
    if not config.use_mps:
        return False
    print_section("CHECKING GPU MPS ENVIRONMENT")
    try:
        result = subprocess.run(["pgrep", "-f", "nvidia-cuda-mps"], capture_output=True)
        if result.returncode == 0:
            print_success("MPS daemon detected")
            return True
        else:
            print_warning("MPS not detected")
            return False
    except Exception as e:
        print_error(f"Error checking MPS: {e}")
        return False


def parse_reservoir_ini(reservoir_path: str) -> Dict:
    config = configparser.ConfigParser()
    config.read(reservoir_path)
    return {
        "nx": config.getint("RESERVOIR_INPUT", "NX"),
        "ny": config.getint("RESERVOIR_INPUT", "NY"),
        "nz": config.getint("RESERVOIR_INPUT", "NZ"),
        "lx": config.getfloat("RESERVOIR_INPUT", "LX"),
        "ly": config.getfloat("RESERVOIR_INPUT", "LY"),
        "lz": config.getfloat("RESERVOIR_INPUT", "LZ"),
        "time_step": config.getfloat("TIME_SETTINGS", "TIME_STEP"),
        "time_final": config.getfloat("TIME_SETTINGS", "TIME_FINAL"),
        "time_initial": config.getfloat("TIME_SETTINGS", "TIME_INITIAL"),
    }


def create_reservoir_ini(reservoir_path: str, output_path: str, nx: int, ny: int, nz: int) -> Dict:
    config = configparser.ConfigParser()
    config.read(reservoir_path)

    lx = config.getfloat("RESERVOIR_INPUT", "LX")
    ly = config.getfloat("RESERVOIR_INPUT", "LY")
    lz = config.getfloat("RESERVOIR_INPUT", "LZ")
    base_nx = config.getint("RESERVOIR_INPUT", "NX")
    refinement_ratio = nx / base_nx
    time_step = config.getfloat("TIME_SETTINGS", "TIME_STEP")
    time_final = config.getfloat("TIME_SETTINGS", "TIME_FINAL")
    time_step /= refinement_ratio
    time_final /= refinement_ratio
    config.set("TIME_SETTINGS", "TIME_STEP", str(time_step))
    config.set("TIME_SETTINGS", "TIME_FINAL", str(time_final))
    time_initial = config.getfloat("TIME_SETTINGS", "TIME_INITIAL")
    config.set("RESERVOIR_INPUT", "NX", str(nx))
    config.set("RESERVOIR_INPUT", "NY", str(ny))
    config.set("RESERVOIR_INPUT", "NZ", str(nz))

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_src = Path(reservoir_path).parent / "truth"
    if truth_src.exists():
        truth_dst = output_dir / "truth"
        if truth_dst.exists():
            shutil.rmtree(truth_dst)
        shutil.copytree(truth_src, truth_dst)

    final_output_path = output_dir / "reservoir.ini"
    with open(final_output_path, "w") as f:
        config.write(f)
    
    if _logger:
        _logger.debug(f"Created reservoir.ini at {final_output_path}")

    num_steps = int((time_final - time_initial) / time_step)

    return {
        "nx": nx, "ny": ny, "nz": nz,
        "hx": lx / nx, "hy": ly / ny, "hz": lz / nz,
        "lx": lx, "ly": ly, "lz": lz,
        "timestep": time_step,
        "time_final": time_final,
        "num_steps": num_steps,
        "output_path": str(final_output_path),
    }


class MemoryMonitor:
    def __init__(self, pid: int, use_gpu: bool = False):
        self.pid = pid
        self.use_gpu = use_gpu
        self.running = True
        self.ram_samples = []
        self.vram_samples = []
        self.vram_per_gpu = []
    
    def _get_vram_usage(self) -> Tuple[float, List[Dict]]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return 0.0, []
            total_vram = 0.0
            per_gpu = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_idx = int(parts[0].strip())
                        used_mb = float(parts[1].strip())
                        total_mb = float(parts[2].strip())
                        total_vram += used_mb
                        per_gpu.append({"gpu": gpu_idx, "used_mb": used_mb, "total_mb": total_mb})
            return total_vram, per_gpu
        except Exception:
            return 0.0, []
    
    def monitor(self):
        while self.running:
            try:
                process = psutil.Process(self.pid)
                children = process.children(recursive=True)
                total_mem = process.memory_info().rss
                for child in children:
                    try:
                        total_mem += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.ram_samples.append(total_mem / (1024 * 1024))
                if self.use_gpu:
                    vram_total, vram_per = self._get_vram_usage()
                    self.vram_samples.append(vram_total)
                    self.vram_per_gpu.append(vram_per)
                time.sleep(0.5)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    
    def stop(self):
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "ram": {
                "max_mb": float(max(self.ram_samples)) if self.ram_samples else 0.0,
                "avg_mb": float(np.mean(self.ram_samples)) if self.ram_samples else 0.0,
                "min_mb": float(min(self.ram_samples)) if self.ram_samples else 0.0,
                "samples": len(self.ram_samples),
            }
        }
        if self.use_gpu and self.vram_samples:
            stats["vram"] = {
                "max_mb": float(max(self.vram_samples)),
                "avg_mb": float(np.mean(self.vram_samples)),
                "min_mb": float(min(self.vram_samples)),
                "samples": len(self.vram_samples),
            }
            if self.vram_per_gpu:
                num_gpus = len(self.vram_per_gpu[0]) if self.vram_per_gpu[0] else 0
                if num_gpus > 0:
                    per_gpu_max = []
                    for gpu_idx in range(num_gpus):
                        gpu_samples = [s[gpu_idx]["used_mb"] for s in self.vram_per_gpu if len(s) > gpu_idx]
                        if gpu_samples:
                            per_gpu_max.append({
                                "gpu": gpu_idx,
                                "max_mb": float(max(gpu_samples)),
                                "total_mb": self.vram_per_gpu[0][gpu_idx]["total_mb"]
                            })
                    stats["vram"]["per_gpu"] = per_gpu_max
        return stats


def run_solver(
    config: Config,
    reservoir_path: str,
    mpi_processes: int,
    name: str,
    use_gpu: bool = False,
    opt: bool = True,
    timeout_hours: float = 2.0,
    postprocess: bool = False,
    log_level: str = "INFO"
) -> Tuple[Optional[Dict], Dict]:
    reservoir_path = os.path.abspath(reservoir_path)
    output_dir = Path(reservoir_path).parent / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_py_dir = os.path.dirname(os.path.abspath(config.run_script))
    run_py_name = os.path.basename(config.run_script)

    cmd = [
        "mpiexec", "-n", str(mpi_processes),
        "-env", "OMP_NUM_THREADS", str(config.omp_num_threads),
        "-env", "OMP_PROC_BIND", "false",
        "-env", "OMP_PLACES", "threads",
        "-env", "MKL_NUM_THREADS", "1",
        "-env", "OPENBLAS_NUM_THREADS", "1",
        "python3", run_py_name,
        "-name", name,
        "-reservoir", reservoir_path,
        "-ksp_type", config.ksp_type,
        "-pc_type", config.pc_type,
        "-ksp_rtol", str(config.ksp_rtol),
        "-ksp_reuse_preconditioner", "true",
        "log_level", log_level
    ]

    if opt:
        cmd.append("-opt")
    if use_gpu:
        cmd.extend(["-gpu", "-vec_type", config.vec_type, "-mat_type", config.mat_type])
    if postprocess:
        cmd.append("-post")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config.omp_num_threads)

    if _logger:
        _logger.debug(f"Executing: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd, cwd=run_py_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        monitor = MemoryMonitor(process.pid, use_gpu=use_gpu)
        monitor_thread = threading.Thread(target=monitor.monitor)
        monitor_thread.start()
        timeout_seconds = int(timeout_hours * 3600)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        monitor.stop()
        monitor_thread.join(timeout=2)
        memory_data = monitor.get_stats()

        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""
        
        if process.returncode != 0:
            print_error(f"Non-zero return: {process.returncode}")
            if stderr_str:
                print(f"    {Colors.RED}{stderr_str[:1000]}{Colors.END}", flush=True)
                if _logger:
                    _logger.error(f"stderr: {stderr_str}")
            return None, memory_data

        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            print_error(f"No JSON generated")
            return None, memory_data

        with open(json_files[0], "r") as f:
            results = json.load(f)
        
        return results, memory_data

    except subprocess.TimeoutExpired:
        process.kill()
        print_error(f"Timeout ({timeout_hours}h)")
        return None, {}
    except Exception as e:
        print_error(f"Exception: {e}")
        if _logger:
            _logger.error(f"Exception: {e}")
        return None, {}


def test_correctness_case(config: Config, case: TestCase, temp_dir: Path) -> Dict:
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), case.path
    ))
    
    if not os.path.exists(reservoir_path):
        print_error(f"File not found: {reservoir_path}")
        return {"name": case.name, "description": case.description, "status": "FILE_NOT_FOUND", "results": []}
    
    orig_config = parse_reservoir_ini(reservoir_path)
    print_section(f"Case: {case.name} - {case.description}")    
    results_data = []
    
    for i, mesh in enumerate(case.meshes):
        nx, ny, nz = mesh
        total_size = nx * ny * nz
        h = orig_config['lx'] / nx
        
        print(f"\n  Mesh M{i+1}: {nx}x{ny}x{nz} = {total_size:,} cells (h = {h:.2f})", flush=True)
        
        case_temp_dir = temp_dir / case.name
        case_temp_dir.mkdir(parents=True, exist_ok=True)
        ini_path = case_temp_dir / f"reservoir_{total_size}.ini"
        
        config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
        actual_ini_path = config_info["output_path"]
        
        print(f"    Running...", end=" ", flush=True)
        
        start_time = time.time()
        results, memory = run_solver(
            config, actual_ini_path,
            mpi_processes=config.correctness_mpi,
            name=f"{case.name}_M{i+1}_{total_size}",
            use_gpu=config.correctness_use_gpu,
            opt=config.correctness_optimized,
            timeout_hours=config.correctness_timeout,
            postprocess=config.correctness_postprocess,
            log_level="DEBUG"
        )
        elapsed = time.time() - start_time
        
        if results:
            l2_list = results.get("l2_error") or []
            l2_val = float(l2_list[-1]) if l2_list else -1.0
            print_success(f"Done in {elapsed:.1f}s, L2 = {l2_val:.6e}")
            results_data.append({
                "mesh_id": f"M{i+1}",
                "nx": int(nx), "ny": int(ny), "nz": int(nz),
                "total_cells": int(total_size),
                "h": float(h),
                "l2_error": float(l2_val),
                "times": {
                    "total": float(results.get("total_time", 0)),
                    "preprocessing": float(results.get("preprocessing_time", 0)),
                    "solving": float(results.get("solving_time", 0)),
                    "updating": float(results.get("updating_time", 0)),
                    "wall_clock": float(elapsed),
                },
                "memory": memory,
                "solver_info": {
                    "n_iterations": results.get("n_iterations", 0),
                    "n_ranks": results.get("n_ranks", 1),
                }
            })
        else:
            print_error("FAILED")
            results_data.append({"mesh_id": f"M{i+1}", "status": "FAILED", "memory": memory})
    
    return {"name": case.name, "description": case.description, "results": results_data}


def test_correctness(config: Config, temp_dir: Path) -> List[Dict]:
    print_header("CORRECTNESS TESTS")
    all_results = []
    for case in config.correctness_cases:
        case_result = test_correctness_case(config, case, temp_dir)
        all_results.append(case_result)
    return all_results


def test_performance(config: Config, temp_dir: Path) -> Dict:
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), config.performance_reservoir
    ))
    
    nx, ny, nz = config.performance_mesh
    total_size = nx * ny * nz

    print_header("PERFORMANCE TESTS - CPU vs GPU")
    print_metric("Mesh", f"{nx}x{ny}x{nz}", f"= {total_size:,} cells")

    ini_path = temp_dir / f"reservoir_perf_{total_size}.ini"
    config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
    actual_ini_path = config_info["output_path"]

    results_data = []

    for mpi in config.performance_mpi:
        print_section(f"{mpi} MPI process(es)")
        mpi_result = {"mpi": int(mpi), "total_cells": int(total_size), "cpu": {"runs": []}, "gpu": {"runs": []}}
        
        print(f"\n  {Colors.BOLD}[CPU]{Colors.END}", flush=True)
        for r in range(config.performance_runs):
            print(f"    Run {r+1}/{config.performance_runs}...", end=" ", flush=True)
            results, memory = run_solver(config, actual_ini_path, mpi, f"Perf_CPU_{mpi}p_r{r}",
                                         use_gpu=False, opt=config.performance_optimized,
                                         timeout_hours=config.performance_timeout)
            if results:
                run_data = {
                    "run_id": r,
                    "times": {"total": float(results.get("total_time", 0)),
                              "preprocessing": float(results.get("preprocessing_time", 0)),
                              "solving": float(results.get("solving_time", 0)),
                              "updating": float(results.get("updating_time", 0))},
                    "memory": memory,
                    "solver_info": {"n_iterations": results.get("n_iterations", 0), "n_elements": results.get("n_elements", 0)}
                }
                mpi_result["cpu"]["runs"].append(run_data)
                print(f"{run_data['times']['total']:.2f}s (solve: {run_data['times']['solving']:.2f}s)", flush=True)
            else:
                print_error("FAILED")
                mpi_result["cpu"]["runs"].append({"run_id": r, "status": "FAILED", "memory": memory})

        valid_cpu = [r for r in mpi_result["cpu"]["runs"] if "times" in r]
        if valid_cpu:
            cpu_totals = [r["times"]["total"] for r in valid_cpu]
            mpi_result["cpu"]["summary"] = {
                "avg_total": float(np.mean(cpu_totals)),
                "std_total": float(np.std(cpu_totals)),
                "avg_solving": float(np.mean([r["times"]["solving"] for r in valid_cpu])),
                "avg_preprocessing": float(np.mean([r["times"]["preprocessing"] for r in valid_cpu])),
                "avg_updating": float(np.mean([r["times"]["updating"] for r in valid_cpu])),
                "max_ram_mb": float(max([r["memory"]["ram"]["max_mb"] for r in valid_cpu])),
                "successful_runs": len(valid_cpu),
            }

        print(f"\n  {Colors.BOLD}[GPU]{Colors.END}", flush=True)
        for r in range(config.performance_runs):
            print(f"    Run {r+1}/{config.performance_runs}...", end=" ", flush=True)
            results, memory = run_solver(config, actual_ini_path, mpi, f"Perf_GPU_{mpi}p_r{r}",
                                         use_gpu=True, opt=config.performance_optimized,
                                         timeout_hours=config.performance_timeout)
            if results:
                run_data = {
                    "run_id": r,
                    "times": {"total": float(results.get("total_time", 0)),
                              "preprocessing": float(results.get("preprocessing_time", 0)),
                              "solving": float(results.get("solving_time", 0)),
                              "updating": float(results.get("updating_time", 0))},
                    "memory": memory,
                    "solver_info": {"n_iterations": results.get("n_iterations", 0), "n_elements": results.get("n_elements", 0)}
                }
                mpi_result["gpu"]["runs"].append(run_data)
                vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB" if "vram" in memory else ""
                print(f"{run_data['times']['total']:.2f}s (solve: {run_data['times']['solving']:.2f}s{vram_str})", flush=True)
            else:
                print_error("FAILED")
                mpi_result["gpu"]["runs"].append({"run_id": r, "status": "FAILED", "memory": memory})

        valid_gpu = [r for r in mpi_result["gpu"]["runs"] if "times" in r]
        if valid_gpu:
            gpu_totals = [r["times"]["total"] for r in valid_gpu]
            mpi_result["gpu"]["summary"] = {
                "avg_total": float(np.mean(gpu_totals)),
                "std_total": float(np.std(gpu_totals)),
                "avg_solving": float(np.mean([r["times"]["solving"] for r in valid_gpu])),
                "avg_preprocessing": float(np.mean([r["times"]["preprocessing"] for r in valid_gpu])),
                "avg_updating": float(np.mean([r["times"]["updating"] for r in valid_gpu])),
                "max_ram_mb": float(max([r["memory"]["ram"]["max_mb"] for r in valid_gpu])),
                "successful_runs": len(valid_gpu),
            }
            vram_runs = [r for r in valid_gpu if "vram" in r["memory"]]
            if vram_runs:
                mpi_result["gpu"]["summary"]["max_vram_mb"] = float(max([r["memory"]["vram"]["max_mb"] for r in vram_runs]))

        if mpi_result["cpu"].get("summary") and mpi_result["gpu"].get("summary"):
            cpu_avg = mpi_result["cpu"]["summary"]["avg_total"]
            gpu_avg = mpi_result["gpu"]["summary"]["avg_total"]
            speedup = float(cpu_avg / gpu_avg)
            mpi_result["speedup"] = speedup
            mpi_result["speedup_solving"] = float(
                mpi_result["cpu"]["summary"]["avg_solving"] / mpi_result["gpu"]["summary"]["avg_solving"]
            ) if mpi_result["gpu"]["summary"]["avg_solving"] > 0 else None
            
            if speedup > 1:
                print(f"  {Colors.GREEN}-> GPU {speedup:.2f}x faster{Colors.END}", flush=True)
            else:
                print(f"  {Colors.YELLOW}-> CPU {1/speedup:.2f}x faster{Colors.END}", flush=True)

        results_data.append(mpi_result)

    return {"mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz), "total": int(total_size)}, "results": results_data}


def test_strong_scaling(config: Config, temp_dir: Path) -> Dict:
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), config.strong_reservoir
    ))
    
    nx, ny, nz = config.strong_mesh
    total_size = nx * ny * nz

    print_header("STRONG SCALING - CPU+GPU")
    print_metric("Fixed mesh", f"{nx}x{ny}x{nz}", f"= {total_size:,} cells")

    ini_path = temp_dir / f"reservoir_strong_{total_size}.ini"
    config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
    actual_ini_path = config_info["output_path"]

    results_data = []
    t1 = None

    for mpi in config.strong_mpi:
        print_section(f"{mpi} MPI process(es) + {mpi} GPU(s)")
        mpi_result = {"mpi": int(mpi), "gpus": int(mpi), "total_cells": int(total_size), "runs": []}

        for r in range(config.strong_runs):
            print(f"  Run {r+1}/{config.strong_runs}...", end=" ", flush=True)
            results, memory = run_solver(config, actual_ini_path, mpi, f"Strong_{mpi}p_r{r}",
                                         use_gpu=True, opt=config.strong_optimized,
                                         timeout_hours=config.strong_timeout)
            if results:
                run_data = {
                    "run_id": r,
                    "times": {"total": float(results.get("total_time", 0)),
                              "preprocessing": float(results.get("preprocessing_time", 0)),
                              "solving": float(results.get("solving_time", 0)),
                              "updating": float(results.get("updating_time", 0))},
                    "memory": memory,
                }
                mpi_result["runs"].append(run_data)
                vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB" if "vram" in memory else ""
                print(f"{run_data['times']['total']:.2f}s{vram_str}", flush=True)
            else:
                print_error("FAILED")
                mpi_result["runs"].append({"run_id": r, "status": "FAILED", "memory": memory})

        valid_runs = [r for r in mpi_result["runs"] if "times" in r]
        if valid_runs:
            totals = [r["times"]["total"] for r in valid_runs]
            avg_time = float(np.mean(totals))
            if t1 is None:
                t1 = avg_time
            speedup = float(t1 / avg_time)
            efficiency = float(speedup / mpi * 100)
            mpi_result["summary"] = {
                "avg_total": avg_time,
                "std_total": float(np.std(totals)),
                "avg_solving": float(np.mean([r["times"]["solving"] for r in valid_runs])),
                "avg_preprocessing": float(np.mean([r["times"]["preprocessing"] for r in valid_runs])),
                "avg_updating": float(np.mean([r["times"]["updating"] for r in valid_runs])),
                "max_ram_mb": float(max([r["memory"]["ram"]["max_mb"] for r in valid_runs])),
                "speedup": speedup,
                "efficiency": efficiency,
                "successful_runs": len(valid_runs),
            }
            vram_runs = [r for r in valid_runs if "vram" in r["memory"]]
            if vram_runs:
                mpi_result["summary"]["max_vram_mb"] = float(max([r["memory"]["vram"]["max_mb"] for r in vram_runs]))
            eff_color = Colors.GREEN if efficiency >= 80 else (Colors.YELLOW if efficiency >= 60 else Colors.RED)
            print(f"  Speedup: {speedup:.2f}x, Efficiency: {eff_color}{efficiency:.1f}%{Colors.END}", flush=True)
        
        results_data.append(mpi_result)

    return {"mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz), "total": int(total_size)}, "results": results_data}


def test_weak_scaling(config: Config, temp_dir: Path) -> Dict:
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), config.weak_reservoir
    ))

    print_header("WEAK SCALING - CPU+GPU")
    results_data = []
    t1 = None

    for mpi in config.weak_mpi:
        if mpi not in config.weak_decomposition:
            print_warning(f"No decomposition for {mpi}, skipping")
            continue

        mesh = config.weak_decomposition[mpi]
        nx, ny, nz = mesh
        total_size = nx * ny * nz
        cells_per_proc = total_size // mpi

        print_section(f"{mpi} MPI + {mpi} GPU - {nx}x{ny}x{nz} = {total_size:,} cells")

        ini_path = temp_dir / f"reservoir_weak_{mpi}p_{total_size}.ini"
        config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
        actual_ini_path = config_info["output_path"]

        mpi_result = {
            "mpi": int(mpi), "gpus": int(mpi),
            "mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz)},
            "total_cells": int(total_size), "cells_per_proc": int(cells_per_proc), "runs": [],
        }

        for r in range(config.weak_runs):
            print(f"  Run {r+1}/{config.weak_runs}...", end=" ", flush=True)
            results, memory = run_solver(config, actual_ini_path, mpi, f"Weak_{mpi}p_r{r}",
                                         use_gpu=True, opt=config.weak_optimized,
                                         timeout_hours=config.weak_timeout)
            if results:
                run_data = {
                    "run_id": r,
                    "times": {"total": float(results.get("total_time", 0)),
                              "preprocessing": float(results.get("preprocessing_time", 0)),
                              "solving": float(results.get("solving_time", 0)),
                              "updating": float(results.get("updating_time", 0))},
                    "memory": memory,
                }
                mpi_result["runs"].append(run_data)
                vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB" if "vram" in memory else ""
                print(f"{run_data['times']['total']:.2f}s{vram_str}", flush=True)
            else:
                print_error("FAILED")
                mpi_result["runs"].append({"run_id": r, "status": "FAILED", "memory": memory})

        valid_runs = [r for r in mpi_result["runs"] if "times" in r]
        if valid_runs:
            totals = [r["times"]["total"] for r in valid_runs]
            avg_time = float(np.mean(totals))
            if t1 is None:
                t1 = avg_time
            efficiency = float(t1 / avg_time * 100)
            mpi_result["summary"] = {
                "avg_total": avg_time,
                "std_total": float(np.std(totals)),
                "avg_solving": float(np.mean([r["times"]["solving"] for r in valid_runs])),
                "avg_preprocessing": float(np.mean([r["times"]["preprocessing"] for r in valid_runs])),
                "avg_updating": float(np.mean([r["times"]["updating"] for r in valid_runs])),
                "max_ram_mb": float(max([r["memory"]["ram"]["max_mb"] for r in valid_runs])),
                "efficiency": efficiency,
                "successful_runs": len(valid_runs),
            }
            vram_runs = [r for r in valid_runs if "vram" in r["memory"]]
            if vram_runs:
                mpi_result["summary"]["max_vram_mb"] = float(max([r["memory"]["vram"]["max_mb"] for r in vram_runs]))
            eff_color = Colors.GREEN if efficiency >= 80 else (Colors.YELLOW if efficiency >= 60 else Colors.RED)
            print(f"  Efficiency: {eff_color}{efficiency:.1f}%{Colors.END}", flush=True)
        
        results_data.append(mpi_result)

    return {"base_mesh": list(config.weak_base_mesh), "results": results_data}


def test_bandwidth(config: Config, temp_dir: Path) -> Dict:
    """Bandwidth test: Roofline analysis for memory-bound vs compute-bound classification."""
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), config.bandwidth_reservoir
    ))
    
    nx, ny, nz = config.bandwidth_mesh
    total_cells = nx * ny * nz
    mpi = config.bandwidth_mpi
    num_gpus = config.bandwidth_num_gpus

    print_header("BANDWIDTH / ROOFLINE ANALYSIS")
    print(f"\n{Colors.CYAN}Analyzing memory bandwidth utilization and compute intensity{Colors.END}\n", flush=True)
    
    print_metric("Mesh", f"{nx}x{ny}x{nz}", f"= {total_cells:,} cells")
    print_metric("MPI processes", str(mpi))
    print_metric("GPUs", str(num_gpus))
    print_metric("GPU Model", config.gpu_model)
    print_metric("Peak Bandwidth (per GPU)", f"{config.gpu_bandwidth_gb_s:.0f}", "GB/s")
    print_metric("Total Peak Bandwidth", f"{config.gpu_bandwidth_gb_s * num_gpus:.0f}", "GB/s")

    ini_path = temp_dir / f"reservoir_bandwidth_{total_cells}.ini"
    config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
    actual_ini_path = config_info["output_path"]

    print_section(f"Running bandwidth test ({config.bandwidth_runs} run(s))")

    results_data = {
        "config": {
            "mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz)},
            "total_cells": int(total_cells),
            "mpi_processes": int(mpi),
            "num_gpus": int(num_gpus),
            "gpu_model": config.gpu_model,
            "peak_bandwidth_per_gpu_gb_s": float(config.gpu_bandwidth_gb_s),
            "total_peak_bandwidth_gb_s": float(config.gpu_bandwidth_gb_s * num_gpus),
        },
        "analysis_params": {
            "bytes_per_cell": int(config.bandwidth_bytes_per_cell),
            "stencil_size": int(config.bandwidth_stencil_size),
            "flops_per_cell": int(config.bandwidth_flops_per_cell),
        },
        "runs": [],
    }

    for r in range(config.bandwidth_runs):
        print(f"\n  Run {r+1}/{config.bandwidth_runs}...", end=" ", flush=True)
        
        results, memory = run_solver(
            config, actual_ini_path, mpi,
            f"Bandwidth_r{r}",
            use_gpu=True,
            opt=config.bandwidth_optimized,
            timeout_hours=config.bandwidth_timeout,
        )

        if results:
            total_time = float(results.get("total_time", 0))
            solving_time = float(results.get("solving_time", 0))
            n_iterations = int(results.get("n_iterations", 0))
            
            nnz_per_row = config.bandwidth_stencil_size
            total_nnz = total_cells * nnz_per_row

            bytes_per_spmv = (
                total_cells * 8 +           
                total_cells * 8 +           
                total_nnz * 8 +             
                total_nnz * 4               
            )
            
            
            spmv_per_iteration = 2  
            total_bytes_solver = bytes_per_spmv * spmv_per_iteration * n_iterations
            
            
            flops_per_spmv = 2 * total_nnz
            total_flops_solver = flops_per_spmv * spmv_per_iteration * n_iterations
            
            
            bandwidth_achieved_gb_s = (total_bytes_solver / 1e9) / solving_time if solving_time > 0 else 0
            bandwidth_efficiency = (bandwidth_achieved_gb_s / (config.gpu_bandwidth_gb_s * num_gpus)) * 100
            
            
            arithmetic_intensity = total_flops_solver / total_bytes_solver if total_bytes_solver > 0 else 0
            
            
            gflops_achieved = (total_flops_solver / 1e9) / solving_time if solving_time > 0 else 0
            
            
            ridge_point = 10.0  
            is_memory_bound = arithmetic_intensity < ridge_point
            
            vram_str = ""
            if "vram" in memory:
                vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB"
            
            print_success(f"Done in {total_time:.2f}s (solve: {solving_time:.2f}s{vram_str})")
            
            run_data = {
                "run_id": r,
                "times": {
                    "total": total_time,
                    "solving": solving_time,
                    "preprocessing": float(results.get("preprocessing_time", 0)),
                    "updating": float(results.get("updating_time", 0)),
                },
                "solver_info": {
                    "n_iterations": n_iterations,
                    "n_elements": int(results.get("n_elements", 0)),
                },
                "memory": memory,
                "bandwidth_analysis": {
                    "total_bytes_transferred_gb": float(total_bytes_solver / 1e9),
                    "total_flops_gflop": float(total_flops_solver / 1e9),
                    "bandwidth_achieved_gb_s": float(bandwidth_achieved_gb_s),
                    "bandwidth_efficiency_percent": float(bandwidth_efficiency),
                    "arithmetic_intensity_flop_byte": float(arithmetic_intensity),
                    "gflops_achieved": float(gflops_achieved),
                    "classification": "Memory-Bound" if is_memory_bound else "Compute-Bound",
                    "ridge_point": float(ridge_point),
                },
            }
            results_data["runs"].append(run_data)
            
        else:
            print_error("FAILED")
            results_data["runs"].append({"run_id": r, "status": "FAILED", "memory": memory})

    
    valid_runs = [r for r in results_data["runs"] if "bandwidth_analysis" in r]
    
    if valid_runs:
        print_header("BANDWIDTH ANALYSIS SUMMARY")
        
        avg_bandwidth = np.mean([r["bandwidth_analysis"]["bandwidth_achieved_gb_s"] for r in valid_runs])
        avg_efficiency = np.mean([r["bandwidth_analysis"]["bandwidth_efficiency_percent"] for r in valid_runs])
        avg_ai = np.mean([r["bandwidth_analysis"]["arithmetic_intensity_flop_byte"] for r in valid_runs])
        avg_gflops = np.mean([r["bandwidth_analysis"]["gflops_achieved"] for r in valid_runs])
        classification = valid_runs[0]["bandwidth_analysis"]["classification"]
        
        results_data["summary"] = {
            "avg_bandwidth_gb_s": float(avg_bandwidth),
            "avg_bandwidth_efficiency_percent": float(avg_efficiency),
            "avg_arithmetic_intensity": float(avg_ai),
            "avg_gflops": float(avg_gflops),
            "classification": classification,
            "successful_runs": len(valid_runs),
        }
        
        print(f"\n{Colors.BOLD}Roofline Metrics:{Colors.END}", flush=True)
        print_metric("Effective Bandwidth", f"{avg_bandwidth:.2f}", "GB/s")
        print_metric("Peak Bandwidth (total)", f"{config.gpu_bandwidth_gb_s * num_gpus:.0f}", "GB/s")
        print_metric("Bandwidth Efficiency", f"{avg_efficiency:.1f}", "%")
        print_metric("Arithmetic Intensity", f"{avg_ai:.4f}", "FLOP/Byte")
        print_metric("Ridge Point (approx)", f"{ridge_point:.1f}", "FLOP/Byte")
        print_metric("Achieved GFLOPs", f"{avg_gflops:.2f}", "GFLOP/s")
        
        if classification == "Memory-Bound":
            print(f"\n  {Colors.CYAN}Classification: {Colors.BOLD}MEMORY-BOUND{Colors.END}", flush=True)
            print(f"  {Colors.CYAN}  -> Performance limited by memory bandwidth{Colors.END}", flush=True)
            print(f"  {Colors.CYAN}  -> Arithmetic Intensity ({avg_ai:.4f}) < Ridge Point ({ridge_point}){Colors.END}", flush=True)
        else:
            print(f"\n  {Colors.GREEN}Classification: {Colors.BOLD}COMPUTE-BOUND{Colors.END}", flush=True)
            print(f"  {Colors.GREEN}  -> Performance limited by compute units{Colors.END}", flush=True)
        
        vram_runs = [r for r in valid_runs if "vram" in r.get("memory", {})]
        if vram_runs:
            max_vram = max([r["memory"]["vram"]["max_mb"] for r in vram_runs])
            print(f"\n{Colors.BOLD}Memory Usage:{Colors.END}", flush=True)
            print_metric("Peak VRAM (total)", f"{max_vram:.0f}", "MB")
            print_metric("VRAM per GPU (avg)", f"{max_vram/num_gpus:.0f}", "MB")
            print_metric("VRAM Utilization", f"{(max_vram/1024)/(config.vram_per_gpu_gb*num_gpus)*100:.1f}", "%")

    return results_data


def run_solver_custom(
    config: Config,
    reservoir_path: str,
    mpi_processes: int,
    name: str,
    ksp_type: str,
    pc_type: str,
    use_gpu: bool = False,
    opt: bool = True,
    timeout_hours: float = 2.0,
) -> Tuple[Optional[Dict], Dict]:
    """Run solver with custom KSP/PC configuration."""
    reservoir_path = os.path.abspath(reservoir_path)
    output_dir = Path(reservoir_path).parent / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_py_dir = os.path.dirname(os.path.abspath(config.run_script))
    run_py_name = os.path.basename(config.run_script)

    cmd = [
        "mpiexec", "-n", str(mpi_processes),
        "-env", "OMP_NUM_THREADS", str(config.omp_num_threads),
        "-env", "OMP_PROC_BIND", "false",
        "-env", "OMP_PLACES", "threads",
        "-env", "MKL_NUM_THREADS", "1",
        "-env", "OPENBLAS_NUM_THREADS", "1",
        "python3", run_py_name,
        "-name", name,
        "-reservoir", reservoir_path,
        "-ksp_type", ksp_type,
        "-pc_type", pc_type,
        "-ksp_rtol", str(config.ksp_rtol),
        "-ksp_reuse_preconditioner", "true",
        "log_level", "INFO"
    ]

    if opt:
        cmd.append("-opt")
    if use_gpu:
        cmd.extend(["-gpu", "-vec_type", config.vec_type, "-mat_type", config.mat_type])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config.omp_num_threads)

    if _logger:
        _logger.debug(f"Executing: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd, cwd=run_py_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        monitor = MemoryMonitor(process.pid, use_gpu=use_gpu)
        monitor_thread = threading.Thread(target=monitor.monitor)
        monitor_thread.start()
        timeout_seconds = int(timeout_hours * 3600)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        monitor.stop()
        monitor_thread.join(timeout=2)
        memory_data = monitor.get_stats()

        if process.returncode != 0:
            stderr_str = stderr.decode() if stderr else ""
            if _logger:
                _logger.error(f"Non-zero return: {process.returncode}, stderr: {stderr_str[:500]}")
            return None, memory_data

        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            return None, memory_data

        with open(json_files[0], "r") as f:
            results = json.load(f)
        
        return results, memory_data

    except subprocess.TimeoutExpired:
        process.kill()
        return None, {}
    except Exception as e:
        if _logger:
            _logger.error(f"Exception: {e}")
        return None, {}


def test_solver_comparison(config: Config, temp_dir: Path) -> Dict:
    """Compare different solver configurations (KSP + PC combinations)."""
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), config.solver_comparison_reservoir
    ))
    
    nx, ny, nz = config.solver_comparison_mesh
    total_cells = nx * ny * nz
    mpi = config.solver_comparison_mpi
    use_gpu = config.solver_comparison_use_gpu

    print_header("SOLVER COMPARISON TEST")
    print(f"\n{Colors.CYAN}Comparing different KSP + Preconditioner combinations{Colors.END}\n", flush=True)
    
    print_metric("Mesh", f"{nx}x{ny}x{nz}", f"= {total_cells:,} cells")
    print_metric("MPI processes", str(mpi))
    print_metric("GPU", "Yes" if use_gpu else "No")
    print_metric("Runs per solver", str(config.solver_comparison_runs))

    ini_path = temp_dir / f"reservoir_solvers_{total_cells}.ini"
    config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
    actual_ini_path = config_info["output_path"]

    results_data = {
        "config": {
            "mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz)},
            "total_cells": int(total_cells),
            "mpi_processes": int(mpi),
            "use_gpu": use_gpu,
            "runs_per_solver": config.solver_comparison_runs,
        },
        "solvers": [],
    }

    baseline_time = None

    for solver_config in config.solver_comparison_solvers:
        solver_name = solver_config["name"]
        ksp_type = solver_config["ksp_type"]
        pc_type = solver_config["pc_type"]
        description = solver_config.get("description", "")

        print_section(f"{solver_name} ({ksp_type} + {pc_type})")
        
        solver_result = {
            "name": solver_name,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "description": description,
            "runs": [],
        }

        for r in range(config.solver_comparison_runs):
            print(f"  Run {r+1}/{config.solver_comparison_runs}...", end=" ", flush=True)
            
            results, memory = run_solver_custom(
                config, actual_ini_path, mpi,
                f"Solver_{solver_name.replace('+', '_')}_r{r}",
                ksp_type=ksp_type,
                pc_type=pc_type,
                use_gpu=use_gpu,
                opt=config.solver_comparison_optimized,
                timeout_hours=config.solver_comparison_timeout,
            )

            if results:
                total_time = float(results.get("total_time", 0))
                solving_time = float(results.get("solving_time", 0))
                n_iterations = int(results.get("n_iterations", 0))
                
                run_data = {
                    "run_id": r,
                    "times": {
                        "total": total_time,
                        "solving": solving_time,
                        "preprocessing": float(results.get("preprocessing_time", 0)),
                        "updating": float(results.get("updating_time", 0)),
                    },
                    "solver_info": {
                        "n_iterations": n_iterations,
                        "time_per_iteration_ms": (solving_time / n_iterations * 1000) if n_iterations > 0 else 0,
                    },
                    "memory": memory,
                }
                solver_result["runs"].append(run_data)
                
                vram_str = ""
                if "vram" in memory:
                    vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB"
                
                print_success(f"{total_time:.2f}s (solve: {solving_time:.2f}s, {n_iterations} iters{vram_str})")
            else:
                print_error("FAILED or TIMEOUT")
                solver_result["runs"].append({"run_id": r, "status": "FAILED", "memory": memory})

        
        valid_runs = [r for r in solver_result["runs"] if "times" in r]
        if valid_runs:
            solving_times = [r["times"]["solving"] for r in valid_runs]
            total_times = [r["times"]["total"] for r in valid_runs]
            iterations = [r["solver_info"]["n_iterations"] for r in valid_runs]
            
            avg_solving = float(np.mean(solving_times))
            avg_total = float(np.mean(total_times))
            avg_iters = float(np.mean(iterations))
            
            if baseline_time is None:
                baseline_time = avg_solving
            
            speedup_vs_baseline = float(baseline_time / avg_solving) if avg_solving > 0 else 0
            
            solver_result["summary"] = {
                "avg_solving_time": avg_solving,
                "std_solving_time": float(np.std(solving_times)),
                "avg_total_time": avg_total,
                "avg_iterations": avg_iters,
                "avg_time_per_iteration_ms": float(avg_solving / avg_iters * 1000) if avg_iters > 0 else 0,
                "speedup_vs_first": speedup_vs_baseline,
                "max_ram_mb": float(max([r["memory"]["ram"]["max_mb"] for r in valid_runs])),
                "successful_runs": len(valid_runs),
            }
            
            vram_runs = [r for r in valid_runs if "vram" in r.get("memory", {})]
            if vram_runs:
                solver_result["summary"]["max_vram_mb"] = float(max([r["memory"]["vram"]["max_mb"] for r in vram_runs]))
            
            print(f"  -> Avg: {avg_solving:.2f}s, {avg_iters:.0f} iters, {solver_result['summary']['avg_time_per_iteration_ms']:.2f}ms/iter", flush=True)

        results_data["solvers"].append(solver_result)

    
    valid_solvers = [s for s in results_data["solvers"] if s.get("summary")]
    
    if valid_solvers:
        print_header("SOLVER COMPARISON SUMMARY")
        
        
        
        times = [s["summary"]["avg_solving_time"] for s in valid_solvers]
        vrams = [s["summary"].get("max_vram_mb", 0.0) for s in valid_solvers]
        
        
        min_time, max_time = min(times), max(times)
        min_vram, max_vram = min(vrams), max(vrams)
        
        range_time = max_time - min_time
        range_vram = max_vram - min_vram
        
        
        for s in valid_solvers:
            t = s["summary"]["avg_solving_time"]
            v = s["summary"].get("max_vram_mb", 0.0)
            
            
            norm_time = (t - min_time) / range_time if range_time > 1e-9 else 0.0
            norm_vram = (v - min_vram) / range_vram if range_vram > 1e-9 else 0.0
            
            
            s["summary"]["score"] = (0.3 * norm_time) + (0.7 * norm_vram)
            
        
        best_solver = min(valid_solvers, key=lambda s: s["summary"]["score"])
        baseline_solver = valid_solvers[0]
        
        
        print(f"\n{'Solver':<18} {'Solve (s)':<10} {'Iters':<8} {'ms/iter':<10} {'Speedup':<10} {'VRAM (MB)':<10} {'Score':<6}", flush=True)
        print("-" * 84, flush=True)
        
        for s in valid_solvers:
            summ = s["summary"]
            speedup = baseline_solver["summary"]["avg_solving_time"] / summ["avg_solving_time"] if summ["avg_solving_time"] > 0 else 0
            vram_str = f"{summ.get('max_vram_mb', 0):.0f}" if summ.get('max_vram_mb') else "N/A"
            score_val = summ["score"]
            
            is_best = s["name"] == best_solver["name"]
            color = Colors.GREEN if is_best else ""
            end_color = Colors.END if is_best else ""
            
            print(f"{color}{s['name']:<18} {summ['avg_solving_time']:<10.2f} {summ['avg_iterations']:<8.0f} {summ['avg_time_per_iteration_ms']:<10.2f} {speedup:<10.2f}x {vram_str:<10} {score_val:<6.3f}{end_color}", flush=True)
        
        print(f"\n{Colors.GREEN}Best solver: {best_solver['name']} (Score: {best_solver['summary']['score']:.3f}){Colors.END}", flush=True)
        
        
        results_data["analysis"] = {
            "best_solver": best_solver["name"],
            "best_score": best_solver["summary"]["score"],
            "best_solving_time": best_solver["summary"]["avg_solving_time"],
            "baseline_solver": baseline_solver["name"],
            "baseline_solving_time": baseline_solver["summary"]["avg_solving_time"],
            "by_preconditioner": {},
        }
        
        pc_groups = {}
        for s in valid_solvers:
            pc = s["pc_type"].upper()
            if pc not in pc_groups:
                pc_groups[pc] = []
            pc_groups[pc].append(s)
        
        print(f"\n{Colors.BOLD}Analysis by Preconditioner Type:{Colors.END}", flush=True)
        print("-" * 50, flush=True)
        
        for pc, solvers in sorted(pc_groups.items()):
            best_in_group = min(solvers, key=lambda s: s["summary"]["avg_solving_time"])
            avg_time = np.mean([s["summary"]["avg_solving_time"] for s in solvers])
            avg_iters = np.mean([s["summary"]["avg_iterations"] for s in solvers])
            
            results_data["analysis"]["by_preconditioner"][pc] = {
                "best_solver": best_in_group["name"],
                "best_time": best_in_group["summary"]["avg_solving_time"],
                "avg_time": float(avg_time),
                "avg_iterations": float(avg_iters),
                "count": len(solvers),
            }
            
            print(f"  {pc:<12}: best={best_in_group['name']:<16} time={best_in_group['summary']['avg_solving_time']:.2f}s, avg_iters={avg_iters:.0f}", flush=True)
        
        print(f"\n{Colors.BOLD}Key Comparisons:{Colors.END}", flush=True)
        print("-" * 50, flush=True)
        
        gamg_solvers = [s for s in valid_solvers if s["pc_type"].lower() == "gamg"]
        non_gamg_solvers = [s for s in valid_solvers if s["pc_type"].lower() != "gamg" and s["pc_type"].lower() != "none"]
        
        if gamg_solvers and non_gamg_solvers:
            best_gamg = min(gamg_solvers, key=lambda s: s["summary"]["avg_solving_time"])
            best_non_gamg = min(non_gamg_solvers, key=lambda s: s["summary"]["avg_solving_time"])
            
            ratio = best_non_gamg["summary"]["avg_solving_time"] / best_gamg["summary"]["avg_solving_time"]
            
            results_data["analysis"]["gamg_comparison"] = {
                "best_gamg": best_gamg["name"],
                "best_gamg_time": best_gamg["summary"]["avg_solving_time"],
                "best_gamg_iters": best_gamg["summary"]["avg_iterations"],
                "best_other": best_non_gamg["name"],
                "best_other_time": best_non_gamg["summary"]["avg_solving_time"],
                "best_other_iters": best_non_gamg["summary"]["avg_iterations"],
                "gamg_speedup": float(ratio),
            }
            
            if ratio > 1:
                print(f"  {Colors.GREEN}GAMG ({best_gamg['name']}) is {ratio:.2f}x FASTER than best non-GAMG ({best_non_gamg['name']}){Colors.END}", flush=True)
            else:
                print(f"  {Colors.YELLOW}{best_non_gamg['name']} is {1/ratio:.2f}x faster than GAMG ({best_gamg['name']}){Colors.END}", flush=True)
            
            gamg_iters = best_gamg["summary"]["avg_iterations"]
            other_iters = best_non_gamg["summary"]["avg_iterations"]
            print(f"  GAMG iterations: {gamg_iters:.0f} vs {best_non_gamg['name']}: {other_iters:.0f}", flush=True)
        
        none_solvers = [s for s in valid_solvers if s["pc_type"].lower() == "none"]
        if none_solvers:
            best_none = min(none_solvers, key=lambda s: s["summary"]["avg_solving_time"])
            
            speedup_from_none = best_none["summary"]["avg_solving_time"] / best_solver["summary"]["avg_solving_time"]
            
            results_data["analysis"]["preconditioning_benefit"] = {
                "no_pc_solver": best_none["name"],
                "no_pc_time": best_none["summary"]["avg_solving_time"],
                "no_pc_iters": best_none["summary"]["avg_iterations"],
                "best_pc_speedup": float(speedup_from_none),
            }
            
            print(f"  Preconditioning benefit: {speedup_from_none:.2f}x speedup over no preconditioner", flush=True)
            print(f"  (No PC: {best_none['summary']['avg_iterations']:.0f} iters vs Best: {best_solver['summary']['avg_iterations']:.0f} iters)", flush=True)

    return results_data


def main():
    parser = argparse.ArgumentParser(description="TPFA Solver Test Suite")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument(
        "--test",
        choices=["correctness", "performance", "strong", "weak", "bandwidth", "solvers", "all"],
        default="correctness",
        help="Test type",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / "etc" / f"{args.config}.yaml"
    if not config_path.exists():
        config_path = script_dir / f"{args.config}.yaml"
    if not config_path.exists():
        print_error(f"Config not found: {args.config}.yaml")
        sys.exit(1)

    config = Config.from_yaml(str(config_path))
    
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    if args.test == "all":
        tests_to_run = ["solvers", "performance", "strong", "weak", "bandwidth"]
    else:
        tests_to_run = [args.test]
    
    temp_dir = Path(tempfile.mkdtemp(prefix="TPFA_", dir=config.temp_dir))
    
    
    best_solver_config = {
        "ksp_type": config.ksp_type,
        "pc_type": config.pc_type,
        "name": f"{config.ksp_type}+{config.pc_type}",
        "source": "default",
    }
    
    try:
        for test_type in tests_to_run:
            logger = setup_logging(results_dir, test_type, timestamp)
            logger.info(f"Starting test: {test_type}")
            logger.info(f"Config file: {config_path}")
            logger.info(f"Temp directory: {temp_dir}")
            logger.info(f"Using solver: {best_solver_config['name']} (source: {best_solver_config['source']})")
            
            config.print_summary(test_type)
            
            
            if test_type != "solvers" and best_solver_config["source"] == "solver_comparison":
                print(f"\n{Colors.GREEN}Using best solver from comparison: {best_solver_config['name']}{Colors.END}", flush=True)
            
            prepare_gpu_mps(config)
            
            print(f"\n{Colors.CYAN}Temp directory: {temp_dir}{Colors.END}", flush=True)
            
            results = {}
            
            if test_type == "correctness":
                results["correctness"] = test_correctness(config, temp_dir)
            elif test_type == "performance":
                results["performance"] = test_performance(config, temp_dir)
            elif test_type == "strong":
                results["strong_scaling"] = test_strong_scaling(config, temp_dir)
            elif test_type == "weak":
                results["weak_scaling"] = test_weak_scaling(config, temp_dir)
            elif test_type == "bandwidth":
                results["bandwidth"] = test_bandwidth(config, temp_dir)
            elif test_type == "solvers":
                solver_results = test_solver_comparison(config, temp_dir)
                results["solver_comparison"] = solver_results
                
                
                if solver_results.get("analysis", {}).get("best_solver"):
                    best_name = solver_results["analysis"]["best_solver"]
                    
                    for s in solver_results.get("solvers", []):
                        if s.get("name") == best_name and s.get("summary"):
                            config.ksp_type = s["ksp_type"]
                            config.pc_type = s["pc_type"]
                            best_solver_config = {
                                "ksp_type": s["ksp_type"],
                                "pc_type": s["pc_type"],
                                "name": best_name,
                                "solving_time": s["summary"]["avg_solving_time"],
                                "iterations": s["summary"]["avg_iterations"],
                                "source": "solver_comparison",
                            }
                            print_header("BEST SOLVER SELECTED FOR REMAINING TESTS")
                            print(f"\n{Colors.GREEN}  Solver: {best_name}{Colors.END}", flush=True)
                            print(f"  KSP Type: {s['ksp_type']}", flush=True)
                            print(f"  PC Type: {s['pc_type']}", flush=True)
                            print(f"  Solving Time: {s['summary']['avg_solving_time']:.2f}s", flush=True)
                            print(f"  Iterations: {s['summary']['avg_iterations']:.0f}", flush=True)
                            print(f"\n{Colors.CYAN}  This configuration will be used for: performance, strong, weak, bandwidth{Colors.END}", flush=True)
                            break
            
            output_dir = results_dir / test_type / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "results.yml"
            
            
            metadata = {
                "config": args.config,
                "test_type": test_type,
                "timestamp": timestamp,
                "hostname": os.uname().nodename,
                "hardware": {
                    "physical_cores": config.physical_cores,
                    "num_gpus": config.num_gpus,
                    "vram_per_gpu_gb": config.vram_per_gpu_gb,
                    "total_ram_gb": config.total_ram_gb,
                    "gpu_model": config.gpu_model,
                    "gpu_peak_bandwidth_gb_s": config.gpu_bandwidth_gb_s,
                },
                "solver_used": {
                    "ksp_type": config.ksp_type,
                    "pc_type": config.pc_type,
                    "name": best_solver_config["name"],
                    "source": best_solver_config["source"],
                },
            }
            
            save_yaml({
                "metadata": metadata,
                "results": results,
            }, output_file)
            
            print_header(f"TEST {test_type.upper()} COMPLETED")
            print_success(f"Results: {output_file}")
            print_success(f"Logs: {logger.logs_dir}")
            
            logger.info(f"Results saved to: {output_file}")
            logger.close()
    
    finally:
        print_header("ALL TESTS COMPLETED")
        if best_solver_config["source"] == "solver_comparison":
            print(f"{Colors.GREEN}Best solver used: {best_solver_config['name']}{Colors.END}", flush=True)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"{Colors.CYAN}Temp directory removed{Colors.END}", flush=True)


if __name__ == "__main__":
    main()