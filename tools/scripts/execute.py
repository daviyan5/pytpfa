#!/usr/bin/env python3
"""
execute.py - TPFA Solver Test Suite

Available tests:
  - correctness:  Convergence order verification
  - performance:  CPU vs GPU benchmark (same MPI count)
  - strong:       Strong scaling (CPU+GPU, 1-4 processes)
  - weak:         Weak scaling (CPU+GPU, 1-4 processes)
  - all:          All tests
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
    """Recursively convert numpy types to native Python types."""
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
    """Save data to YAML file, converting numpy types."""
    clean_data = numpy_to_python(data)
    with open(filepath, "w") as f:
        yaml.dump(clean_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


class TestLogger:
    """Manages logging for test execution."""
    
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
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print(f"{msg:^70}")
    print(f"{'='*70}{Colors.END}")
    if _logger:
        _logger.section(msg)


def print_section(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'-'*70}")
    print(f"{msg}")
    print(f"{'-'*70}{Colors.END}")
    if _logger:
        _logger.subsection(msg)


def print_success(msg: str):
    print(f"{Colors.GREEN}[OK] {msg}{Colors.END}")
    if _logger:
        _logger.info(f"[OK] {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.END}")
    if _logger:
        _logger.warning(msg)


def print_error(msg: str):
    print(f"{Colors.RED}[ERR] {msg}{Colors.END}")
    if _logger:
        _logger.error(msg)


def print_metric(name: str, value: str, unit: str = "", highlight: bool = False):
    if highlight:
        print(f"  {Colors.BOLD}{name:.<40} {Colors.GREEN}{value:>12} {unit}{Colors.END}")
    else:
        print(f"  {name:.<40} {value:>12} {unit}")
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

        return config

    def print_summary(self, test_type: str):
        print_header("TEST CONFIGURATION")
        
        print(f"\n{Colors.BOLD}Hardware:{Colors.END}")
        print_metric("Physical cores", str(self.physical_cores))
        print_metric("NUMA nodes", str(self.numa_nodes))
        print_metric("GPUs", str(self.num_gpus), f"x {self.vram_per_gpu_gb}GB VRAM")
        print_metric("RAM Total", str(self.total_ram_gb), "GB")

        if test_type == "correctness":
            print(f"\n{Colors.BOLD}Test: CORRECTNESS{Colors.END}")
            print_metric("MPI processes", str(self.correctness_mpi))
            print_metric("GPU", "Yes" if self.correctness_use_gpu else "No")

        elif test_type == "performance":
            m = self.performance_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: PERFORMANCE (CPU vs GPU){Colors.END}")
            print_metric("Mesh", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.performance_mpi))

        elif test_type == "strong":
            m = self.strong_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: STRONG SCALING (CPU+GPU){Colors.END}")
            print_metric("Fixed mesh", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.strong_mpi))

        elif test_type == "weak":
            m = self.weak_base_mesh
            total = m[0] * m[1] * m[2]
            print(f"\n{Colors.BOLD}Test: WEAK SCALING (CPU+GPU){Colors.END}")
            print_metric("Base/process", f"{m[0]}x{m[1]}x{m[2]}", f"= {total:,} cells")
            print_metric("MPI processes", str(self.weak_mpi))


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


def create_reservoir_ini(
    reservoir_path: str,
    output_path: str,
    nx: int,
    ny: int,
    nz: int,
) -> Dict:
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
    """Monitor RAM and VRAM usage during execution."""
    
    def __init__(self, pid: int, use_gpu: bool = False):
        self.pid = pid
        self.use_gpu = use_gpu
        self.running = True
        self.ram_samples = []
        self.vram_samples = []
        self.vram_per_gpu = []
    
    def _get_vram_usage(self) -> Tuple[float, List[Dict]]:
        """Get VRAM usage from nvidia-smi in MB."""
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
                        per_gpu.append({
                            "gpu": gpu_idx,
                            "used_mb": used_mb,
                            "total_mb": total_mb
                        })
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
        """Return complete memory statistics."""
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
                        gpu_samples = [
                            s[gpu_idx]["used_mb"] 
                            for s in self.vram_per_gpu 
                            if len(s) > gpu_idx
                        ]
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
    """
    Run the solver and return ALL data from the JSON output plus memory stats.
    """
    reservoir_path = os.path.abspath(reservoir_path)
    output_dir = Path(reservoir_path).parent / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_py_dir = os.path.dirname(os.path.abspath(config.run_script))
    run_py_name = os.path.basename(config.run_script)

    cmd = [
        "mpiexec",
        "-n", str(mpi_processes),
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
        process = subprocess.Popen(
            cmd,
            cwd=run_py_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

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
                print(f"    {Colors.RED}{stderr_str[:1000]}{Colors.END}")
                if _logger:
                    _logger.error(f"stderr: {stderr_str}")
            return None, memory_data

        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            print_error(f"No JSON generated")
            if stderr_str:
                error_lines = [l for l in stderr_str.split('\n') if 'ERROR' in l or 'Error' in l]
                if error_lines:
                    for line in error_lines[:5]:
                        print(f"      {Colors.RED}{line.strip()}{Colors.END}")
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
        return {
            "name": case.name,
            "description": case.description,
            "status": "FILE_NOT_FOUND",
            "results": [],
        }
    
    orig_config = parse_reservoir_ini(reservoir_path)
    
    print_section(f"Case: {case.name} - {case.description}")    
    results_data = []
    
    for i, mesh in enumerate(case.meshes):
        nx, ny, nz = mesh
        total_size = nx * ny * nz
        h = orig_config['lx'] / nx
        
        print(f"\n  Mesh M{i+1}: {nx}x{ny}x{nz} = {total_size:,} cells (h = {h:.2f})")
        
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
            results_data.append({
                "mesh_id": f"M{i+1}",
                "status": "FAILED",
                "memory": memory
            })
    
    valid_results = [r for r in results_data if "l2_error" in r and r.get("l2_error", -1) > 0]
    
    
    return {
        "name": case.name,
        "description": case.description,
        "results": results_data,
    }


def test_correctness(config: Config, temp_dir: Path) -> List[Dict]:
    print_header("CORRECTNESS TESTS")
    
    all_results = []
    for case in config.correctness_cases:
        case_result = test_correctness_case(config, case, temp_dir)
        all_results.append(case_result)
    
    return all_results


def test_performance(config: Config, temp_dir: Path) -> Dict:
    """Performance test: CPU vs GPU comparison with same MPI counts."""
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
        
        mpi_result = {
            "mpi": int(mpi),
            "total_cells": int(total_size),
            "cpu": {"runs": []},
            "gpu": {"runs": []},
        }
        
        print(f"\n  {Colors.BOLD}[CPU]{Colors.END}")
        for r in range(config.performance_runs):
            print(f"    Run {r+1}/{config.performance_runs}...", end=" ", flush=True)

            results, memory = run_solver(
                config, actual_ini_path, mpi,
                f"Perf_CPU_{mpi}p_r{r}",
                use_gpu=False,
                opt=config.performance_optimized,
                timeout_hours=config.performance_timeout,
            )

            if results:
                run_data = {
                    "run_id": r,
                    "times": {
                        "total": float(results.get("total_time", 0)),
                        "preprocessing": float(results.get("preprocessing_time", 0)),
                        "solving": float(results.get("solving_time", 0)),
                        "updating": float(results.get("updating_time", 0)),
                    },
                    "memory": memory,
                    "solver_info": {
                        "n_iterations": results.get("n_iterations", 0),
                        "n_elements": results.get("n_elements", 0),
                    }
                }
                mpi_result["cpu"]["runs"].append(run_data)
                print(f"{run_data['times']['total']:.2f}s (solve: {run_data['times']['solving']:.2f}s)")
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
            print(f"    -> CPU avg: {mpi_result['cpu']['summary']['avg_total']:.2f}s")

        print(f"\n  {Colors.BOLD}[GPU]{Colors.END}")
        for r in range(config.performance_runs):
            print(f"    Run {r+1}/{config.performance_runs}...", end=" ", flush=True)

            results, memory = run_solver(
                config, actual_ini_path, mpi,
                f"Perf_GPU_{mpi}p_r{r}",
                use_gpu=True,
                opt=config.performance_optimized,
                timeout_hours=config.performance_timeout,
            )

            if results:
                run_data = {
                    "run_id": r,
                    "times": {
                        "total": float(results.get("total_time", 0)),
                        "preprocessing": float(results.get("preprocessing_time", 0)),
                        "solving": float(results.get("solving_time", 0)),
                        "updating": float(results.get("updating_time", 0)),
                    },
                    "memory": memory,
                    "solver_info": {
                        "n_iterations": results.get("n_iterations", 0),
                        "n_elements": results.get("n_elements", 0),
                    }
                }
                mpi_result["gpu"]["runs"].append(run_data)
                vram_str = ""
                if "vram" in memory:
                    vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB"
                print(f"{run_data['times']['total']:.2f}s (solve: {run_data['times']['solving']:.2f}s{vram_str})")
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
                mpi_result["gpu"]["summary"]["avg_vram_mb"] = float(np.mean([r["memory"]["vram"]["max_mb"] for r in vram_runs]))
            
            print(f"    -> GPU avg: {mpi_result['gpu']['summary']['avg_total']:.2f}s")

        if mpi_result["cpu"].get("summary") and mpi_result["gpu"].get("summary"):
            cpu_avg = mpi_result["cpu"]["summary"]["avg_total"]
            gpu_avg = mpi_result["gpu"]["summary"]["avg_total"]
            speedup = float(cpu_avg / gpu_avg)
            mpi_result["speedup"] = speedup
            
            mpi_result["speedup_solving"] = float(
                mpi_result["cpu"]["summary"]["avg_solving"] / 
                mpi_result["gpu"]["summary"]["avg_solving"]
            ) if mpi_result["gpu"]["summary"]["avg_solving"] > 0 else None
            
            if speedup > 1:
                print(f"  {Colors.GREEN}-> GPU {speedup:.2f}x faster{Colors.END}")
            else:
                print(f"  {Colors.YELLOW}-> CPU {1/speedup:.2f}x faster{Colors.END}")

        results_data.append(mpi_result)

    print_header("SUMMARY - PERFORMANCE CPU vs GPU")
    
    print(f"\n{'MPI':^6} {'CPU (s)':^12} {'GPU (s)':^12} {'Speedup':^12} {'RAM CPU':^12} {'VRAM GPU':^12}")
    print("-" * 70)
    
    for r in results_data:
        cpu_str = f"{r['cpu']['summary']['avg_total']:.2f}" if r['cpu'].get('summary') else "FAILED"
        gpu_str = f"{r['gpu']['summary']['avg_total']:.2f}" if r['gpu'].get('summary') else "FAILED"
        ram_str = f"{r['cpu']['summary']['max_ram_mb']:.0f}MB" if r['cpu'].get('summary') else "N/A"
        vram_str = f"{r['gpu']['summary'].get('max_vram_mb', 0):.0f}MB" if r['gpu'].get('summary') else "N/A"
        
        if r.get('speedup'):
            if r['speedup'] > 1:
                sp_str = f"{Colors.GREEN}{r['speedup']:.2f}x{Colors.END}"
            else:
                sp_str = f"{Colors.YELLOW}{1/r['speedup']:.2f}x CPU{Colors.END}"
        else:
            sp_str = "N/A"
        
        print(f"{r['mpi']:^6} {cpu_str:^12} {gpu_str:^12} {sp_str:^20} {ram_str:^12} {vram_str:^12}")

    return {
        "mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz), "total": int(total_size)},
        "results": results_data,
    }


def test_strong_scaling(config: Config, temp_dir: Path) -> Dict:
    """Strong scaling test using CPU+GPU together."""
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), config.strong_reservoir
    ))
    
    nx, ny, nz = config.strong_mesh
    total_size = nx * ny * nz

    print_header("STRONG SCALING - CPU+GPU")
    print(f"\n{Colors.CYAN}Speedup = T(1)/T(N), Efficiency = Speedup/N{Colors.END}\n")
    print_metric("Fixed mesh", f"{nx}x{ny}x{nz}", f"= {total_size:,} cells")

    ini_path = temp_dir / f"reservoir_strong_{total_size}.ini"
    config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
    actual_ini_path = config_info["output_path"]

    results_data = []
    t1 = None

    for mpi in config.strong_mpi:
        print_section(f"{mpi} MPI process(es) + {mpi} GPU(s)")
        
        mpi_result = {
            "mpi": int(mpi),
            "gpus": int(mpi),
            "total_cells": int(total_size),
            "runs": [],
        }

        for r in range(config.strong_runs):
            print(f"  Run {r+1}/{config.strong_runs}...", end=" ", flush=True)

            results, memory = run_solver(
                config, actual_ini_path, mpi,
                f"Strong_{mpi}p_r{r}",
                use_gpu=True,
                opt=config.strong_optimized,
                timeout_hours=config.strong_timeout,
            )

            if results:
                run_data = {
                    "run_id": r,
                    "times": {
                        "total": float(results.get("total_time", 0)),
                        "preprocessing": float(results.get("preprocessing_time", 0)),
                        "solving": float(results.get("solving_time", 0)),
                        "updating": float(results.get("updating_time", 0)),
                    },
                    "memory": memory,
                }
                mpi_result["runs"].append(run_data)
                vram_str = ""
                if "vram" in memory:
                    vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB"
                print(f"{run_data['times']['total']:.2f}s{vram_str}")
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
            print(f"  Speedup: {speedup:.2f}x, Efficiency: {eff_color}{efficiency:.1f}%{Colors.END}")
        
        results_data.append(mpi_result)

    if results_data:
        print_header("SUMMARY - STRONG SCALING (CPU+GPU)")
        print(f"\n{'MPI/GPU':^10} {'Time (s)':^12} {'Speedup':^10} {'Efficiency':^12} {'RAM (MB)':^12} {'VRAM (MB)':^12}")
        print("-" * 75)
        for r in results_data:
            if r.get("summary"):
                s = r["summary"]
                eff = s["efficiency"]
                eff_color = Colors.GREEN if eff >= 80 else (Colors.YELLOW if eff >= 60 else Colors.RED)
                vram_str = f"{s.get('max_vram_mb', 0):.0f}" if s.get('max_vram_mb') else "N/A"
                print(f"{r['mpi']:^10} {s['avg_total']:^12.2f} {s['speedup']:^10.2f} {eff_color}{eff:^10.1f}%{Colors.END} {s['max_ram_mb']:^12.0f} {vram_str:^12}")

    return {
        "mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz), "total": int(total_size)},
        "results": results_data,
    }


def test_weak_scaling(config: Config, temp_dir: Path) -> Dict:
    """Weak scaling test using CPU+GPU together."""
    reservoir_path = os.path.abspath(os.path.join(
        os.path.dirname(config.run_script), config.weak_reservoir
    ))

    print_header("WEAK SCALING - CPU+GPU")
    print(f"\n{Colors.CYAN}Efficiency = T(1)/T(N) (ideal = 100%){Colors.END}\n")

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

        print_section(f"{mpi} MPI + {mpi} GPU - {nx}x{ny}x{nz} = {total_size:,} cells ({cells_per_proc:,}/proc)")

        ini_path = temp_dir / f"reservoir_weak_{mpi}p_{total_size}.ini"
        config_info = create_reservoir_ini(reservoir_path, str(ini_path), nx, ny, nz)
        actual_ini_path = config_info["output_path"]

        mpi_result = {
            "mpi": int(mpi),
            "gpus": int(mpi),
            "mesh": {"nx": int(nx), "ny": int(ny), "nz": int(nz)},
            "total_cells": int(total_size),
            "cells_per_proc": int(cells_per_proc),
            "runs": [],
        }

        for r in range(config.weak_runs):
            print(f"  Run {r+1}/{config.weak_runs}...", end=" ", flush=True)

            results, memory = run_solver(
                config, actual_ini_path, mpi,
                f"Weak_{mpi}p_r{r}",
                use_gpu=True,
                opt=config.weak_optimized,
                timeout_hours=config.weak_timeout,
            )

            if results:
                run_data = {
                    "run_id": r,
                    "times": {
                        "total": float(results.get("total_time", 0)),
                        "preprocessing": float(results.get("preprocessing_time", 0)),
                        "solving": float(results.get("solving_time", 0)),
                        "updating": float(results.get("updating_time", 0)),
                    },
                    "memory": memory,
                }
                mpi_result["runs"].append(run_data)
                vram_str = ""
                if "vram" in memory:
                    vram_str = f", VRAM: {memory['vram']['max_mb']:.0f}MB"
                print(f"{run_data['times']['total']:.2f}s{vram_str}")
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
            print(f"  Efficiency: {eff_color}{efficiency:.1f}%{Colors.END}")
        
        results_data.append(mpi_result)

    if results_data:
        print_header("SUMMARY - WEAK SCALING (CPU+GPU)")
        print(f"\n{'MPI/GPU':^10} {'Cells':^12} {'Time (s)':^12} {'Efficiency':^12} {'RAM (MB)':^12} {'VRAM (MB)':^12}")
        print("-" * 80)
        for r in results_data:
            if r.get("summary"):
                s = r["summary"]
                eff = s["efficiency"]
                eff_color = Colors.GREEN if eff >= 80 else (Colors.YELLOW if eff >= 60 else Colors.RED)
                vram_str = f"{s.get('max_vram_mb', 0):.0f}" if s.get('max_vram_mb') else "N/A"
                print(f"{r['mpi']:^10} {r['total_cells']:^12,} {s['avg_total']:^12.2f} {eff_color}{eff:^10.1f}%{Colors.END} {s['max_ram_mb']:^12.0f} {vram_str:^12}")

    return {
        "base_mesh": list(config.weak_base_mesh),
        "results": results_data,
    }


def main():
    parser = argparse.ArgumentParser(description="TPFA Solver Test Suite")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument(
        "--test",
        choices=["correctness", "performance", "strong", "weak", "all"],
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
        tests_to_run = ["performance", "strong", "weak"]
    else:
        tests_to_run = [args.test]
    
    temp_dir = Path(tempfile.mkdtemp(prefix="TPFA_", dir=config.temp_dir))
    
    try:
        for test_type in tests_to_run:
            logger = setup_logging(results_dir, test_type, timestamp)
            logger.info(f"Starting test: {test_type}")
            logger.info(f"Config file: {config_path}")
            logger.info(f"Temp directory: {temp_dir}")
            
            config.print_summary(test_type)
            prepare_gpu_mps(config)
            
            print(f"\n{Colors.CYAN}Temp directory: {temp_dir}{Colors.END}")
            
            results = {}
            
            if test_type == "correctness":
                results["correctness"] = test_correctness(config, temp_dir)
            elif test_type == "performance":
                results["performance"] = test_performance(config, temp_dir)
            elif test_type == "strong":
                results["strong_scaling"] = test_strong_scaling(config, temp_dir)
            elif test_type == "weak":
                results["weak_scaling"] = test_weak_scaling(config, temp_dir)
            
            output_dir = results_dir / test_type / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "results.yml"
            
            save_yaml({
                "metadata": {
                    "config": args.config,
                    "test_type": test_type,
                    "timestamp": timestamp,
                    "hostname": os.uname().nodename,
                    "hardware": {
                        "physical_cores": config.physical_cores,
                        "num_gpus": config.num_gpus,
                        "vram_per_gpu_gb": config.vram_per_gpu_gb,
                        "total_ram_gb": config.total_ram_gb,
                    }
                },
                "results": results,
            }, output_file)
            
            print_header(f"TEST {test_type.upper()} COMPLETED")
            print_success(f"Results: {output_file}")
            print_success(f"Logs: {logger.logs_dir}")
            
            logger.info(f"Results saved to: {output_file}")
            logger.close()
    
    finally:
        print_header("ALL TESTS COMPLETED")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"{Colors.CYAN}Temp directory removed{Colors.END}")


if __name__ == "__main__":
    main()