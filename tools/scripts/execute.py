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
from typing import List, Tuple, Dict
import configparser
import yaml
from datetime import datetime

OG_RESERVOIR_PATHS = ["../examples/example_Li/reservoir.ini", "../examples/example_4/reservoir.ini"]
RUN_PY_PATH = "../run.py"
SIZES = [
    (32, 32, 1),
    (64, 32, 1),
    (128, 64, 1),
    (256, 128, 1),
    (512, 256, 1),
    (1024, 512, 1),
    (2048, 1024, 1),
    (4096, 2048, 1),
    (8192, 4096, 1),
]

MPI_MAX = 16
SIZES.sort()
SIZES = SIZES[::-1]

MPI_PROCESSES_STRONG = np.arange(1, MPI_MAX + 1, 1).tolist()
MPI_PROCESSES_PERFORMANCE = [4**i for i in range(np.log2(MPI_MAX).astype(int) // 2 + 1)]


def create_reservoir_ini(
    base_path: str, output_path: str, nx: int, ny: int, nz: int, correctness: bool = False
):
    config = configparser.ConfigParser()
    config.read(base_path)

    original_size = (
        config["RESERVOIR_INPUT"].getint("NX")
        * config["RESERVOIR_INPUT"].getint("NY")
        * config["RESERVOIR_INPUT"].getint("NZ")
    )
    new_size = nx * ny * nz

    factor = new_size / original_size

    config["RESERVOIR_INPUT"]["NX"] = str(nx)
    config["RESERVOIR_INPUT"]["NY"] = str(ny)
    config["RESERVOIR_INPUT"]["NZ"] = str(nz)

    if correctness:
        # We must decrease the time step if we increase the number of elements
        original_time_step = config["TIME_SETTINGS"].getfloat("TIME_STEP")
        new_time_step = original_time_step / factor**2
        config["TIME_SETTINGS"]["TIME_STEP"] = str(new_time_step)

    with open(output_path, "w") as f:
        config.write(f)

    base_dir = os.path.dirname(base_path)
    output_dir = os.path.dirname(output_path)

    truth_src = os.path.join(base_dir, "truth")
    if os.path.exists(truth_src):
        truth_dst = os.path.join(output_dir, "truth")
        if not os.path.exists(truth_dst):
            shutil.copytree(truth_src, truth_dst)

    reservoir_ini_dst = os.path.join(output_dir, "reservoir.ini")
    shutil.copy2(output_path, reservoir_ini_dst)


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
                    except:
                        pass

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.05)

        memory_data["max"] = max_mem
        memory_data["samples"] = mem_samples
        memory_data["avg"] = np.mean(mem_samples) if mem_samples else 0

    except psutil.NoSuchProcess:
        memory_data["max"] = 0
        memory_data["samples"] = []
        memory_data["avg"] = 0


def run_solver(
    reservoir_path: str, mpi_processes: int, name: str, opt: bool = True
) -> Tuple[Dict, Dict]:
    reservoir_path = os.path.abspath(reservoir_path)
    output_dir = Path(reservoir_path).parent / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_py_dir = os.path.dirname(RUN_PY_PATH)

    cmd = [
        "mpirun",
        "--bind-to",
        "core",
        "-n",
        str(mpi_processes),
        "python3",
        os.path.basename(RUN_PY_PATH),
        "-name",
        name,
        "-reservoir",
        os.path.abspath(reservoir_path),
        "-opt" if opt else "",
        "-ksp_type",
        "gmres",
        "-pc_type",
        "mg",
    ]

    memory_data = {}

    process = subprocess.Popen(cmd, cwd=run_py_dir, stdout=subprocess.PIPE)

    monitor_thread = threading.Thread(target=monitor_memory_thread, args=(process.pid, memory_data))
    monitor_thread.start()

    stdout, stderr = process.communicate()
    monitor_thread.join()

    if process.returncode != 0:
        print(f"Erro ao executar solver: {stderr.decode()}")
        print(f"Stdout: {stdout.decode()}")
        return None, memory_data

    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        print(f"Nenhum arquivo JSON encontrado em {output_dir}")
        print(f"Arquivos no diretório: {list(output_dir.glob('*'))}")
        return None, memory_data

    with open(json_files[0], "r") as f:
        results = json.load(f)
    return results, memory_data


def test_correctness(temp_dir: Path, og_ini_path: Path):
    print("\n=== TESTE DE CORRETUDE ===")

    results_data = []

    for nx, ny, nz in SIZES[1:]:
        total_size = nx * ny * nz
        print(f"Executando para tamanho {nx}x{ny}x{nz} = {total_size} elementos...")

        ini_path = temp_dir / f"reservoir_{total_size}.ini"
        create_reservoir_ini(og_ini_path, str(ini_path), nx, ny, nz, correctness=True)

        results, memory = run_solver(
            str(ini_path), MPI_MAX, f"TPFA_Correct_{total_size}", opt=False
        )
        if results:
            results_data.append(
                {
                    "size": total_size,
                    "nx": nx,
                    "l1_error": float(results["l1_error"][0]),
                    "l2_error": float(results["l2_error"][0]),
                    "linf_error": float(results["linf_error"][0]),
                }
            )

    return results_data


def test_strong_scaling(temp_dir: Path, og_ini_path: Path):
    print("\n=== TESTE DE ESCALABILIDADE FORTE ===")

    nx, ny, nz = SIZES[1]
    total_size = nx * ny * nz

    ini_path = temp_dir / f"reservoir_strong_{total_size}.ini"
    create_reservoir_ini(og_ini_path, str(ini_path), nx, ny, nz)

    results_data = []

    for mpi_procs in MPI_PROCESSES_STRONG:
        print(f"Executando com {mpi_procs} processos MPI...")

        results, memory = run_solver(str(ini_path), mpi_procs, f"TPFA_Strong_{mpi_procs}", opt=True)

        if results:
            results_data.append(
                {
                    "mpi": int(mpi_procs),
                    "time": float(results["total_time"]),
                    "memory_max": float(memory.get("max", 0)),
                    "memory_avg": float(memory.get("avg", 0)),
                    "total_size": total_size,
                }
            )

    return results_data


def test_weak_scaling(temp_dir: Path, og_ini_path: Path):
    print("\n=== TESTE DE ESCALABILIDADE FRACA ===")

    weak_configs = [
        ((128, 128, 1), 1),
        ((256, 128, 1), 2),
        ((256, 256, 1), 4),
        ((512, 256, 1), 8),
        ((512, 512, 1), 16),
    ]

    results_data = []

    for (nx, ny, nz), mpi_procs in weak_configs:
        total_size = nx * ny * nz
        elements_per_proc = total_size / mpi_procs
        print(
            f"Executando {nx}x{ny}x{nz} = {total_size} elementos com {mpi_procs} processos "
            f"({elements_per_proc:.0f} elem/proc)..."
        )

        ini_path = temp_dir / f"reservoir_weak_{total_size}_{mpi_procs}.ini"
        create_reservoir_ini(og_ini_path, str(ini_path), nx, ny, nz)

        results, memory = run_solver(
            str(ini_path), mpi_procs, f"TPFA_Weak_{total_size}_{mpi_procs}", opt=True
        )

        if results:
            results_data.append(
                {
                    "size": int(total_size),
                    "mpi": int(mpi_procs),
                    "time": float(results["total_time"]),
                    "memory_max": float(memory.get("max", 0)),
                    "memory_avg": float(memory.get("avg", 0)),
                    "elements_per_proc": float(elements_per_proc),
                }
            )

    return results_data


def test_performance(temp_dir: Path, og_ini_path: Path):
    print("\n=== TESTE DE DESEMPENHO ===")

    all_results = {mpi: [] for mpi in MPI_PROCESSES_PERFORMANCE}

    for nx, ny, nz in SIZES:
        total_size = nx * ny * nz

        ini_path = temp_dir / f"reservoir_perf_{total_size}.ini"
        create_reservoir_ini(og_ini_path, str(ini_path), nx, ny, nz)

        for mpi_procs in MPI_PROCESSES_PERFORMANCE:
            print(f"Executando {total_size} elementos com {mpi_procs} processos...")

            results, memory = run_solver(
                str(ini_path), mpi_procs, f"TPFA_Perf_{total_size}_{mpi_procs}", opt=True
            )

            if results:
                all_results[mpi_procs].append(
                    {
                        "size": int(total_size),
                        "preprocessing": float(results.get("preprocessing_time", 0)),
                        "updating": float(results.get("updating_time", 0)),
                        "solving": float(results.get("solving_time", 0)),
                        "total": float(results.get("total_time", 0)),
                        "memory": float(memory.get("max", 0)),
                    }
                )

    return all_results


def main():
    print("=" * 60)
    print("ANÁLISE DE DESEMPENHO DO TPFASOLVER")
    print("=" * 60)

    global OG_RESERVOIR_PATHS, RUN_PY_PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OG_RESERVOIR_PATHS = [
        os.path.abspath(os.path.join(script_dir, og_ini_path)) for og_ini_path in OG_RESERVOIR_PATHS
    ]
    RUN_PY_PATH = os.path.abspath(os.path.join(script_dir, RUN_PY_PATH))

    for og_ini_path in OG_RESERVOIR_PATHS:
        case_name = os.path.basename(os.path.dirname(og_ini_path))
        results_dir = os.path.join(script_dir, os.path.join("results", case_name))
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not os.path.exists(og_ini_path):
            print(f"Erro: Arquivo de configuração não encontrado: {og_ini_path}")
            sys.exit(1)

        run_py_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), RUN_PY_PATH))
        if not os.path.exists(run_py_abs):
            print(f"Erro: Script run.py não encontrado: {run_py_abs}")
            sys.exit(1)

        with tempfile.TemporaryDirectory(prefix="INIS_", dir="/tmp") as temp_dir:
            temp_path = Path(temp_dir)
            print(f"Diretório temporário criado: {temp_path}")

            try:
                correctness_results = test_correctness(temp_path, og_ini_path)
                with open(os.path.join(results_dir, f"correctness_{timestamp}.yml"), "w") as f:
                    yaml.dump(correctness_results, f)

                strong_scaling_results = test_strong_scaling(temp_path, og_ini_path)
                with open(os.path.join(results_dir, f"strong_scaling_{timestamp}.yml"), "w") as f:
                    yaml.dump(strong_scaling_results, f)

                weak_scaling_results = test_weak_scaling(temp_path, og_ini_path)
                with open(os.path.join(results_dir, f"weak_scaling_{timestamp}.yml"), "w") as f:
                    yaml.dump(weak_scaling_results, f)

                performance_results = test_performance(temp_path, og_ini_path)
                with open(os.path.join(results_dir, f"performance_{timestamp}.yml"), "w") as f:
                    yaml.dump(performance_results, f)

                print("\n" + "=" * 60)
                print("TESTES CONCLUÍDOS COM SUCESSO!")
                print(f"Resultados salvos em: {results_dir}")
                print(f"  - correctness_{timestamp}.yml")
                print(f"  - strong_scaling_{timestamp}.yml")
                print(f"  - weak_scaling_{timestamp}.yml")
                print(f"  - performance_{timestamp}.yml")
                print("=" * 60)

            except Exception as e:
                print(f"\nErro durante a execução: {e}")
                import traceback

                traceback.print_exc()
                sys.exit(1)


if __name__ == "__main__":
    main()
