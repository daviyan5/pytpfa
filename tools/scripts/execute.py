#!/usr/bin/env python3
import os
import sys
import json
import shutil
import tempfile
import subprocess
import time
import threading
import numpy as np
import psutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import configparser
import yaml
from datetime import datetime
import argparse
from dataclasses import dataclass, field


@dataclass
class Config:
    reservoir_base: str = ""
    run_script: str = ""
    results_dir: str = ""
    temp_dir: str = "/tmp"

    total_cores: int = 8
    num_gpus: int = 1
    vram_per_gpu_gb: int = 8
    total_ram_gb: int = 16

    mpi_processes: List[int] = field(default_factory=lambda: [1, 2, 4])
    bind_to: str = "core"
    map_by: str = "core"
    omp_num_threads: int = 1

    use_mps: bool = False
    vec_type: str = "cuda"
    mat_type: str = "aijcusparse"

    ksp_type: str = "fgmres"
    pc_type: str = "gamg"
    ksp_rtol: float = 1e-8

    # --- CORRECTNESS ---
    correctness_meshes: List[List[int]] = field(default_factory=list)
    correctness_adjust_timestep: str = "None"
    correctness_timeout: float = 2.0
    correctness_use_gpu: bool = False
    correctness_optimized: bool = False
    correctness_mpi: int = 1

    # --- PERFORMANCE ---
    performance_mesh: List[int] = field(default_factory=lambda: [400, 400, 4])
    performance_mpi: List[int] = field(default_factory=lambda: [1, 2, 4])
    performance_runs: int = 3
    performance_adjust_timestep: str = "None"
    performance_timeout: float = 2.0
    performance_optimized: bool = True

    # --- STRONG SCALING ---
    strong_mesh: List[int] = field(default_factory=lambda: [400, 400, 4])
    strong_mpi: List[int] = field(default_factory=lambda: [1, 2, 4])
    strong_runs: int = 3
    strong_timeout: float = 2.0
    strong_use_gpu: bool = True
    strong_optimized: bool = True

    # --- WEAK SCALING ---
    weak_base_mesh: List[int] = field(default_factory=lambda: [200, 200, 4])
    weak_mpi: List[int] = field(default_factory=lambda: [1, 2, 4])
    weak_decomposition: Dict[int, List[int]] = field(default_factory=dict)
    weak_runs: int = 3
    weak_timeout: float = 2.0
    weak_use_gpu: bool = True
    weak_optimized: bool = True

    domain_lx: float = 2000.0
    domain_ly: float = 2000.0
    domain_lz: float = 1.0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "paths" in data:
            config.reservoir_base = data["paths"].get("reservoir_base", "")
            config.run_script = data["paths"].get("run_script", "")
            config.results_dir = data["paths"].get("results_dir", "./results")
            config.temp_dir = data["paths"].get("temp_dir", "/tmp")

        if "hardware" in data:
            config.total_cores = data["hardware"].get("total_cores", 8)
            config.num_gpus = data["hardware"].get("num_gpus", 1)
            config.vram_per_gpu_gb = data["hardware"].get("vram_per_gpu_gb", 8)
            config.total_ram_gb = data["hardware"].get("total_ram_gb", 16)

        if "mpi" in data:
            config.mpi_processes = data["mpi"].get("processes", [1, 2, 4])
            config.bind_to = data["mpi"].get("bind_to", "core")
            config.map_by = data["mpi"].get("map_by", "core")
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
            config.correctness_meshes = data["correctness"].get(
                "meshes", [[25, 25, 2], [50, 50, 2], [100, 100, 4]]
            )
            config.correctness_adjust_timestep = data["correctness"].get("adjust_timestep", "None")
            config.correctness_timeout = data["correctness"].get("timeout_hours", 2.0)
            config.correctness_use_gpu = data["correctness"].get("use_gpu", False)
            config.correctness_optimized = data["correctness"].get("optimized", False)
            config.correctness_mpi = data["correctness"].get("mpi_processes", 1)

        if "performance" in data:
            config.performance_mesh = data["performance"].get("mesh", [400, 400, 4])
            config.performance_mpi = data["performance"].get("mpi_processes", [1, 2, 4])
            config.performance_runs = data["performance"].get("runs", 3)
            config.performance_adjust_timestep = data["performance"].get("adjust_timestep", "None")
            config.performance_timeout = data["performance"].get("timeout_hours", 2.0)
            config.performance_optimized = data["performance"].get("optimized", True)

        if "strong_scaling" in data:
            config.strong_mesh = data["strong_scaling"].get("mesh", [400, 400, 4])
            config.strong_mpi = data["strong_scaling"].get("mpi_processes", [1, 2, 4])
            config.strong_runs = data["strong_scaling"].get("runs", 3)
            config.strong_timeout = data["strong_scaling"].get("timeout_hours", 2.0)
            config.strong_use_gpu = data["strong_scaling"].get("use_gpu", True)
            config.strong_optimized = data["strong_scaling"].get("optimized", True)

        if "weak_scaling" in data:
            config.weak_base_mesh = data["weak_scaling"].get("base_mesh", [200, 200, 4])
            config.weak_mpi = data["weak_scaling"].get("mpi_processes", [1, 2, 4])
            config.weak_runs = data["weak_scaling"].get("runs", 3)
            config.weak_timeout = data["weak_scaling"].get("timeout_hours", 2.0)
            decomp = data["weak_scaling"].get("decomposition", {})
            config.weak_decomposition = {int(k): v for k, v in decomp.items()}
            config.weak_use_gpu = data["weak_scaling"].get("use_gpu", True)
            config.weak_optimized = data["weak_scaling"].get("optimized", True)

        if "domain" in data:
            config.domain_lx = data["domain"].get("lx", 2000.0)
            config.domain_ly = data["domain"].get("ly", 2000.0)
            config.domain_lz = data["domain"].get("lz", 1.0)

        return config

    def print_summary(self, test_type: str, mode: str = None):
        print("\n" + "=" * 60)
        print("CONFIGURAÇÃO CARREGADA")
        print("=" * 60)
        print(
            f"Hardware: {self.total_cores} cores, {self.num_gpus} GPUs, {self.total_ram_gb} GB RAM"
        )

        if test_type == "correctness":
            print(f"\nTeste: ACURÁCIA")
            print(f"  Malhas 3D:")
            for i, m in enumerate(self.correctness_meshes):
                total = m[0] * m[1] * m[2]
                print(f"    M{i+1}: {m[0]}×{m[1]}×{m[2]} = {total:,} células")
            print(f"  Processos MPI: {self.correctness_mpi}")
            print(f"  GPU: {self.correctness_use_gpu}, Otimizado: {self.correctness_optimized}")
            print(f"  Timeout: {self.correctness_timeout}h")

        elif test_type == "performance":
            m = self.performance_mesh
            total = m[0] * m[1] * m[2]
            print(f"\nTeste: DESEMPENHO ({mode.upper()})")
            print(f"  Malha 3D: {m[0]}×{m[1]}×{m[2]} = {total:,} células")
            print(f"  Processos: {self.performance_mpi}")
            print(f"  Repetições: {self.performance_runs}")
            print(f"  Otimizado: {self.performance_optimized}")

        elif test_type == "strong":
            m = self.strong_mesh
            total = m[0] * m[1] * m[2]
            print(f"\nTeste: ESCALABILIDADE FORTE")
            print(f"  Malha 3D fixa: {m[0]}×{m[1]}×{m[2]} = {total:,} células")
            print(f"  Processos: {self.strong_mpi}")
            print(f"  Repetições: {self.strong_runs}")
            print(f"  GPU: {self.strong_use_gpu}, Otimizado: {self.strong_optimized}")

        elif test_type == "weak":
            m = self.weak_base_mesh
            total = m[0] * m[1] * m[2]
            print(f"\nTeste: ESCALABILIDADE FRACA")
            print(f"  Base 3D: {m[0]}×{m[1]}×{m[2]} = {total:,} células/processo")
            print(f"  Processos: {self.weak_mpi}")
            print(f"  Repetições: {self.weak_runs}")
            print(f"  GPU: {self.weak_use_gpu}, Otimizado: {self.weak_optimized}")

        print("=" * 60)


def prepare_gpu_mps(config: Config) -> bool:
    if not config.use_mps:
        return False

    print("\n" + "=" * 60)
    print("VERIFICANDO AMBIENTE GPU COM MPS")
    print("=" * 60)

    try:
        result = subprocess.run(["pgrep", "-f", "nvidia-cuda-mps"], capture_output=True)

        if result.returncode == 0:
            print("Daemon MPS detectado em execução.")
            return True
        else:
            print("ERRO FATAL: A configuração exige MPS (use_mps: true), mas o processo")
            print("'nvidia-cuda-mps' não foi encontrado.")
            sys.exit(1)

    except FileNotFoundError:
        print("Aviso: comando 'pgrep' não encontrado para verificar MPS.")
        sys.exit(1)
    except Exception as e:
        print(f"Erro inesperado ao verificar MPS: {e}")
        sys.exit(1)


def create_reservoir_ini(
    config: Config,
    base_path: str,
    output_path: str,
    nx: int,
    ny: int,
    nz: int,
    adjust_strategy: str = "None",
) -> dict:
    ini_config = configparser.ConfigParser()
    ini_config.read(base_path)

    original_nx = ini_config["RESERVOIR_INPUT"].getint("NX")
    original_lx = ini_config["RESERVOIR_INPUT"].getfloat("LX")

    original_time_step = ini_config["TIME_SETTINGS"].getfloat("TIME_STEP")
    original_final_time = ini_config["TIME_SETTINGS"].getfloat("TIME_FINAL")

    ini_config["RESERVOIR_INPUT"]["NX"] = str(nx)
    ini_config["RESERVOIR_INPUT"]["NY"] = str(ny)
    ini_config["RESERVOIR_INPUT"]["NZ"] = str(nz)

    ratio_x = (original_nx / nx) ** 2

    new_time_step = original_time_step * ratio_x
    ini_config["TIME_SETTINGS"]["TIME_STEP"] = str(new_time_step)

    if adjust_strategy == "Adjust timestep and final":
        new_final_time = original_final_time * ratio_x
        ini_config["TIME_SETTINGS"]["TIME_FINAL"] = str(new_final_time)

    timestep = ini_config["TIME_SETTINGS"].getfloat("TIME_STEP")
    final_time = ini_config["TIME_SETTINGS"].getfloat("TIME_FINAL")
    num_steps = int(final_time / timestep)

    with open(output_path, "w") as f:
        ini_config.write(f)

    default_ini_path = os.path.join(os.path.dirname(output_path), "reservoir.ini")
    shutil.copy(output_path, default_ini_path)

    base_dir = os.path.dirname(base_path)
    output_dir = os.path.dirname(output_path)
    truth_src = os.path.join(base_dir, "truth")
    if os.path.exists(truth_src):
        truth_dst = os.path.join(output_dir, "truth")
        if not os.path.exists(truth_dst):
            shutil.copytree(truth_src, truth_dst)

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "total_cells": nx * ny * nz,
        "timestep": timestep,
        "final_time": final_time,
        "num_steps": num_steps,
        "hx": config.domain_lx / nx,
        "hy": config.domain_ly / ny,
        "hz": config.domain_lz / nz,
    }


def monitor_memory_thread(pid: int, memory_data: Dict):
    try:
        proc = psutil.Process(pid)
        max_mem = 0
        mem_samples = []

        while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            try:
                mem_mb = proc.memory_info().rss / (1024**2)
                mem_samples.append(mem_mb)
                max_mem = max(max_mem, mem_mb)
                for child in proc.children(recursive=True):
                    try:
                        child_mem = child.memory_info().rss / (1024**2)
                        max_mem = max(max_mem, child_mem)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.1)

        memory_data["max"] = max_mem
        memory_data["samples"] = mem_samples
        memory_data["avg"] = np.mean(mem_samples) if mem_samples else 0
    except psutil.NoSuchProcess:
        memory_data["max"] = 0
        memory_data["samples"] = []
        memory_data["avg"] = 0


def run_solver(
    config: Config,
    reservoir_path: str,
    mpi_processes: int,
    name: str,
    use_gpu: bool = False,
    opt: bool = True,
    timeout_hours: float = 2.0,
) -> Tuple[Optional[Dict], Dict]:
    reservoir_path = os.path.abspath(reservoir_path)
    output_dir = Path(reservoir_path).parent / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_py_dir = os.path.dirname(os.path.abspath(config.run_script))
    run_py_name = os.path.basename(config.run_script)

    cmd = [
        "mpirun",
        "--bind-to",
        config.bind_to,
        "--map-by",
        config.map_by,
        "-n",
        str(mpi_processes),
        "python3",
        run_py_name,
        "-name",
        name,
        "-reservoir",
        reservoir_path,
        "-ksp_type",
        config.ksp_type,
        "-pc_type",
        config.pc_type,
        "-profile",
    ]

    if opt:
        cmd.append("-opt")
    if use_gpu:
        cmd.extend(["-gpu", "-vec_type", config.vec_type, "-mat_type", config.mat_type])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config.omp_num_threads)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"

    if use_gpu:
        num_gpus = min(config.num_gpus, mpi_processes)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    memory_data = {}

    try:
        process = subprocess.Popen(
            cmd,
            cwd=run_py_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        monitor_thread = threading.Thread(
            target=monitor_memory_thread, args=(process.pid, memory_data)
        )
        monitor_thread.start()

        timeout_seconds = int(timeout_hours * 3600)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        monitor_thread.join()

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        log_dir = Path(config.results_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"

        with open(log_file, "w") as f:
            f.write(f"=== CMD: {' '.join(cmd)} ===\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout_str)
            f.write("\n=== STDERR ===\n")
            f.write(stderr_str)

        if process.returncode != 0:
            print(f"Erro (código {process.returncode})")
            print(f"Log salvo em: {log_file}")
            print("-" * 40)
            print("SAÍDA DE ERRO (STDERR):")
            print(stderr_str)
            print("-" * 40)
            return None, memory_data

        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            print(f"Erro: Nenhum JSON gerado. Verifique o log: {log_file}")
            return None, memory_data

        with open(json_files[0], "r") as f:
            results = json.load(f)
        return results, memory_data

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Timeout ({timeout_hours}h)")
        return None, memory_data
    except Exception as e:
        print(f"Erro de Execução: {e}")
        return None, memory_data


def test_correctness(config: Config, temp_dir: Path, reservoir_path: str) -> List[Dict]:
    print("\n" + "=" * 60)
    print("TESTES DE ACURÁCIA (Malhas 3D)")
    print("=" * 60)

    results_data = []

    for i, mesh in enumerate(config.correctness_meshes):
        nx, ny, nz = mesh
        total_size = nx * ny * nz

        print(f"\nM{i+1}: {nx}×{ny}×{nz} = {total_size:,} células...")

        ini_path = temp_dir / f"reservoir_correct_{total_size}.ini"
        config_info = create_reservoir_ini(
            config,
            reservoir_path,
            str(ini_path),
            nx,
            ny,
            nz,
            adjust_strategy=config.correctness_adjust_timestep,
        )

        print(f"  h_xy = {config_info['hx']:.2f} ft, h_z = {config_info['hz']:.4f} ft")
        print(f"  Δt = {config_info['timestep']:.4f} dias")

        results, memory = run_solver(
            config,
            str(ini_path),
            mpi_processes=config.correctness_mpi,
            name=f"Correctness_M{i+1}_{total_size}",
            use_gpu=config.correctness_use_gpu,
            opt=config.correctness_optimized,
            timeout_hours=config.correctness_timeout,
        )

        if results:
            # === CORREÇÃO DE SEGURANÇA ===
            l1_list = results.get("l1_error") or []
            l2_list = results.get("l2_error") or []
            linf_list = results.get("linf_error") or []

            l1_val = float(l1_list[-1]) if l1_list else -1.0
            l2_val = float(l2_list[-1]) if l2_list else -1.0
            linf_val = float(linf_list[-1]) if linf_list else -1.0

            if l1_val == -1.0:
                print("  AVISO: Erro numérico não calculado (lista vazia no JSON).")

            result_entry = {
                "mesh_id": f"M{i+1}",
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "total_cells": total_size,
                "hx": config_info["hx"],
                "hy": config_info["hy"],
                "hz": config_info["hz"],
                "timestep": config_info["timestep"],
                "l1_error": l1_val,
                "l2_error": l2_val,
                "linf_error": linf_val,
                "total_time": float(results.get("total_time", 0)),
                "memory_max_mb": float(memory.get("max", 0)),
            }
            results_data.append(result_entry)
            print(f"  L2 error: {result_entry['l2_error']:.6e} psi")
        else:
            print("  FALHOU")
            results_data.append({"mesh_id": f"M{i+1}", "status": "FAILED"})

    for i in range(1, len(results_data)):
        if (
            "l2_error" in results_data[i]
            and "l2_error" in results_data[i - 1]
            and results_data[i]["l2_error"] > 0
            and results_data[i - 1]["l2_error"] > 0
        ):
            e1, e2 = results_data[i - 1]["l2_error"], results_data[i]["l2_error"]
            h1, h2 = results_data[i - 1]["hx"], results_data[i]["hx"]
            if abs(h1 - h2) > 1e-9:
                results_data[i]["order_l2"] = np.log(e1 / e2) / np.log(h1 / h2)

    return results_data


def test_performance(config: Config, temp_dir: Path, reservoir_path: str, mode: str) -> List[Dict]:
    use_gpu = mode == "gpu"
    mode_name = "GPU" if use_gpu else "CPU"

    nx, ny, nz = config.performance_mesh
    total_size = nx * ny * nz

    print("\n" + "=" * 60)
    print(f"TESTES DE DESEMPENHO - {mode_name}")
    print(f"Malha 3D: {nx}×{ny}×{nz} = {total_size:,} células")
    print(f"Processos: {config.performance_mpi}")
    print("=" * 60)

    ini_path = temp_dir / f"reservoir_perf_{mode}_{total_size}.ini"
    create_reservoir_ini(
        config,
        reservoir_path,
        str(ini_path),
        nx,
        ny,
        nz,
        adjust_strategy=config.performance_adjust_timestep,
    )

    results_data = []

    for mpi in config.performance_mpi:
        print(f"\n{mpi} processo(s)...")
        run_times, run_results = [], []

        for r in range(config.performance_runs):
            print(f"  Run {r+1}/{config.performance_runs}...", end=" ", flush=True)

            results, memory = run_solver(
                config,
                str(ini_path),
                mpi,
                f"Perf_{mode}_{mpi}p_r{r}",
                use_gpu=use_gpu,
                opt=config.performance_optimized,
                timeout_hours=config.performance_timeout,
            )

            if results:
                total_time = (
                    results.get("preprocessing_time", 0)
                    + results.get("updating_time", 0)
                    + results.get("solving_time", 0)
                )
                run_times.append(total_time)
                run_results.append(
                    {
                        "preprocessing": results.get("preprocessing_time", 0),
                        "updating": results.get("updating_time", 0),
                        "solving": results.get("solving_time", 0),
                        "total": total_time,
                        "memory_max_mb": memory.get("max", 0),
                        "iterations": results.get("iterations", []),
                    }
                )
                print(f"{total_time:.2f}s")
            else:
                print("FALHOU")

        if run_results:
            avg_result = {
                "mpi": mpi,
                "mode": mode,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "total_cells": total_size,
                "runs": len(run_results),
                "preprocessing_avg": np.mean([r["preprocessing"] for r in run_results]),
                "updating_avg": np.mean([r["updating"] for r in run_results]),
                "solving_avg": np.mean([r["solving"] for r in run_results]),
                "total_avg": np.mean(run_times),
                "total_std": np.std(run_times),
                "memory_max_mb": np.max([r["memory_max_mb"] for r in run_results]),
                "iterations_avg": np.mean(
                    [np.mean(r["iterations"]) if r["iterations"] else 0 for r in run_results]
                ),
            }
            results_data.append(avg_result)
            print(f"  Média: {avg_result['total_avg']:.2f}s ± {avg_result['total_std']:.2f}s")

    return results_data


def test_strong_scaling(config: Config, temp_dir: Path, reservoir_path: str) -> List[Dict]:
    nx, ny, nz = config.strong_mesh
    total_size = nx * ny * nz

    print("\n" + "=" * 60)
    print("ESCALABILIDADE FORTE (GPU)")
    print(f"Malha 3D fixa: {nx}×{ny}×{nz} = {total_size:,} células")
    print(f"Processos: {config.strong_mpi}")
    print("=" * 60)

    ini_path = temp_dir / f"reservoir_strong_{total_size}.ini"
    create_reservoir_ini(
        config,
        reservoir_path,
        str(ini_path),
        nx,
        ny,
        nz,
        adjust_strategy=config.correctness_adjust_timestep,
    )

    results_data = []
    baseline_time = None

    for mpi in config.strong_mpi:
        print(f"\n{mpi} processo(s)...")
        run_times, run_results = [], []

        for r in range(config.strong_runs):
            print(f"  Run {r+1}/{config.strong_runs}...", end=" ", flush=True)

            results, memory = run_solver(
                config,
                str(ini_path),
                mpi,
                f"Strong_{mpi}p_r{r}",
                use_gpu=config.strong_use_gpu,
                opt=config.strong_optimized,
                timeout_hours=config.strong_timeout,
            )

            if results:
                total_time = (
                    results.get("preprocessing_time", 0)
                    + results.get("updating_time", 0)
                    + results.get("solving_time", 0)
                )
                run_times.append(total_time)
                run_results.append(
                    {
                        "preprocessing": results.get("preprocessing_time", 0),
                        "updating": results.get("updating_time", 0),
                        "solving": results.get("solving_time", 0),
                        "memory_max_mb": memory.get("max", 0),
                    }
                )
                print(f"{total_time:.2f}s")
            else:
                print("FALHOU")

        if run_times:
            avg_time = np.mean(run_times)
            if mpi == config.strong_mpi[0]:
                baseline_time = avg_time

            speedup = baseline_time / avg_time if baseline_time else 1.0
            efficiency = speedup / mpi

            result_entry = {
                "mpi": mpi,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "total_cells": total_size,
                "time_avg": avg_time,
                "time_std": np.std(run_times),
                "speedup": speedup,
                "efficiency": efficiency,
                "preprocessing_avg": np.mean([r["preprocessing"] for r in run_results]),
                "updating_avg": np.mean([r["updating"] for r in run_results]),
                "solving_avg": np.mean([r["solving"] for r in run_results]),
                "memory_max_mb": np.max([r["memory_max_mb"] for r in run_results]),
            }
            results_data.append(result_entry)
            print(f"  Média: {avg_time:.2f}s, Speedup: {speedup:.2f}×, Ef: {efficiency*100:.1f}%")

    return results_data


def test_weak_scaling(config: Config, temp_dir: Path, reservoir_path: str) -> List[Dict]:
    base_nx, base_ny, base_nz = config.weak_base_mesh
    base_cells = base_nx * base_ny * base_nz

    print("\n" + "=" * 60)
    print("ESCALABILIDADE FRACA (GPU)")
    print(f"Base 3D: {base_nx}×{base_ny}×{base_nz} = {base_cells:,} células/processo")
    print(f"Processos: {config.weak_mpi}")
    print("=" * 60)

    results_data = []
    baseline_time = None

    for mpi in config.weak_mpi:
        scale_factor = mpi ** (1 / 3)
        nx = int(base_nx * scale_factor)
        ny = int(base_ny * scale_factor)
        nz = int(base_nz * scale_factor)

        total_size = nx * ny * nz
        cells_per_proc = total_size / mpi

        print(f"\n{mpi}p: {nx}×{ny}×{nz} = {total_size:,} células ({cells_per_proc:,.0f}/proc)...")

        ini_path = temp_dir / f"reservoir_weak_{mpi}p.ini"
        create_reservoir_ini(
            config,
            reservoir_path,
            str(ini_path),
            nx,
            ny,
            nz,
            adjust_strategy=config.correctness_adjust_timestep,
        )

        run_times, run_results = [], []

        for r in range(config.weak_runs):
            print(f"  Run {r+1}/{config.weak_runs}...", end=" ", flush=True)

            results, memory = run_solver(
                config,
                str(ini_path),
                mpi,
                f"Weak_{mpi}p_r{r}",
                use_gpu=config.weak_use_gpu,
                opt=config.weak_optimized,
                timeout_hours=config.weak_timeout,
            )

            if results:
                total_time = (
                    results.get("preprocessing_time", 0)
                    + results.get("updating_time", 0)
                    + results.get("solving_time", 0)
                )
                run_times.append(total_time)
                run_results.append(
                    {
                        "preprocessing": results.get("preprocessing_time", 0),
                        "updating": results.get("updating_time", 0),
                        "solving": results.get("solving_time", 0),
                        "memory_max_mb": memory.get("max", 0),
                    }
                )
                print(f"{total_time:.2f}s")
            else:
                print("FALHOU")

        if run_times:
            avg_time = np.mean(run_times)
            if mpi == config.weak_mpi[0]:
                baseline_time = avg_time

            weak_efficiency = baseline_time / avg_time if baseline_time else 1.0

            result_entry = {
                "mpi": mpi,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "total_cells": total_size,
                "cells_per_proc": cells_per_proc,
                "time_avg": avg_time,
                "time_std": np.std(run_times),
                "weak_efficiency": weak_efficiency,
                "preprocessing_avg": np.mean([r["preprocessing"] for r in run_results]),
                "updating_avg": np.mean([r["updating"] for r in run_results]),
                "solving_avg": np.mean([r["solving"] for r in run_results]),
                "memory_max_mb": np.max([r["memory_max_mb"] for r in run_results]),
            }
            results_data.append(result_entry)
            print(f"  Média: {avg_time:.2f}s, Eficiência fraca: {weak_efficiency*100:.1f}%")

    return results_data


def get_config_dir() -> Path:
    return Path(__file__).parent.resolve() / "etc"


def list_configs():
    config_dir = get_config_dir()
    print(f"\nConfigurações em {config_dir}")
    print("-" * 40)

    if not config_dir.exists():
        print(f"Diretório não encontrado. Crie-o e adicione arquivos .yaml")
        return

    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    if not yaml_files:
        print("Nenhum arquivo .yaml encontrado")
        return

    for f in sorted(yaml_files):
        print(f"  --config {f.stem}")


def main():
    parser = argparse.ArgumentParser(
        description="Testes do simulador TPFA (malhas 3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", "-c", type=str, help="Nome da configuração")
    parser.add_argument(
        "--test", "-t", type=str, choices=["correctness", "performance", "strong", "weak"]
    )
    parser.add_argument("--mode", "-m", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-mps", action="store_true")

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    if not args.config or not args.test:
        parser.error("--config e --test são obrigatórios")

    config_dir = get_config_dir()
    config_path = config_dir / f"{args.config}.yaml"
    if not config_path.exists():
        config_path = config_dir / f"{args.config}.yml"
    if not config_path.exists():
        print(f"Erro: Configuração '{args.config}' não encontrada")
        list_configs()
        sys.exit(1)

    config = Config.from_yaml(str(config_path))

    script_dir = Path(__file__).parent.resolve()
    config.reservoir_base = str(script_dir / config.reservoir_base)
    config.run_script = str(script_dir / config.run_script)
    config.results_dir = str(script_dir / config.results_dir)

    if not os.path.exists(config.reservoir_base):
        print(f"Erro: reservoir não encontrado: {config.reservoir_base}")
        sys.exit(1)

    config.print_summary(args.test, args.mode)

    if args.dry_run:
        print("\n[DRY RUN] Nenhum teste executado.")
        return

    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mps_enabled = False
    if not args.skip_mps and config.use_mps:
        mps_enabled = prepare_gpu_mps(config)

    with tempfile.TemporaryDirectory(prefix="TPFA_", dir=config.temp_dir) as temp_dir:
        temp_path = Path(temp_dir)

        try:
            if args.test == "correctness":
                results = test_correctness(config, temp_path, config.reservoir_base)
                output_file = f"correctness_{args.config}_{timestamp}.yml"
            elif args.test == "performance":
                results = test_performance(config, temp_path, config.reservoir_base, args.mode)
                output_file = f"performance_{args.mode}_{args.config}_{timestamp}.yml"
            elif args.test == "strong":
                results = test_strong_scaling(config, temp_path, config.reservoir_base)
                output_file = f"strong_{args.config}_{timestamp}.yml"
            elif args.test == "weak":
                results = test_weak_scaling(config, temp_path, config.reservoir_base)
                output_file = f"weak_{args.config}_{timestamp}.yml"

            output_path = os.path.join(config.results_dir, output_file)
            with open(output_path, "w") as f:
                yaml.dump(
                    {
                        "metadata": {
                            "config": args.config,
                            "test_type": args.test,
                            "mode": args.mode if args.test == "performance" else "gpu",
                            "timestamp": timestamp,
                        },
                        "results": results,
                    },
                    f,
                    default_flow_style=False,
                )

            print(f"\n{'='*60}\nCONCLUÍDO! Resultados: {output_path}\n{'='*60}")

        except KeyboardInterrupt:
            print("\nInterrompido")
            sys.exit(1)
        except Exception as e:
            import traceback

            print(f"\nErro: {e}")
            traceback.print_exc()
            sys.exit(1)
        finally:
            if mps_enabled:
                cleanup_gpu_mps()


if __name__ == "__main__":
    main()
