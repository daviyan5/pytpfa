#!/usr/bin/env python3
"""
Script de análise de desempenho para o TPFASolver
Executa testes de corretude, escalabilidade e desempenho
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from pathlib import Path
from typing import List, Tuple, Dict
import configparser

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 14

# Configurações hardcoded
OG_RESERVOIR_PATH = "../examples/example_Li/reservoir.ini"
RUN_PY_PATH = "../run.py"
SIZES = [
    (64, 64, 1),
    (128, 128, 1),
    (256, 256, 1),
    (512, 512, 1),
    (1024, 1024, 1),
    (2048, 2048, 1),
    (4096, 4096, 1),
]

MPI_PROCESSES_STRONG = np.arange(1, 16, 1).tolist()
MPI_PROCESSES_PERFORMANCE = [1, 8, 16]


def create_reservoir_ini(base_path: str, output_path: str, nx: int, ny: int, nz: int):
    config = configparser.ConfigParser()
    config.read(base_path)

    config["RESERVOIR_INPUT"]["NX"] = str(nx)
    config["RESERVOIR_INPUT"]["NY"] = str(ny)
    config["RESERVOIR_INPUT"]["NZ"] = str(nz)

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


def run_solver(reservoir_path: str, mpi_processes: int, name: str) -> Tuple[Dict, Dict]:
    reservoir_path = os.path.abspath(reservoir_path)
    output_dir = Path(reservoir_path).parent / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_py_dir = os.path.dirname(RUN_PY_PATH)

    cmd = [
        "mpirun",
        "--use-hwthread-cpus",
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
    ]

    memory_data = {}

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=run_py_dir)

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


def test_correctness(temp_dir: Path):
    print("\n=== TESTE DE CORRETUDE ===")

    results_data = []

    for nx, ny, nz in SIZES:
        total_size = nx * ny * nz
        print(f"Executando para tamanho {nx}x{ny}x{nz} = {total_size} elementos...")

        ini_path = temp_dir / f"reservoir_{total_size}.ini"
        create_reservoir_ini(OG_RESERVOIR_PATH, str(ini_path), nx, ny, nz)

        results, memory = run_solver(str(ini_path), 16, f"TPFA_Correct_{total_size}")

        if results:
            results_data.append(
                {
                    "size": total_size,
                    "nx": nx,
                    "l1_error": results["l1_error"][-1],
                    "l2_error": results["l2_error"][-1],
                    "linf_error": results["linf_error"][-1],
                }
            )

    if results_data:
        fig, ax = plt.subplots(figsize=(10, 7))

        sizes = [r["size"] for r in results_data]
        l1_errors = [r["l1_error"] for r in results_data]
        l2_errors = [r["l2_error"] for r in results_data]
        linf_errors = [r["linf_error"] for r in results_data]

        ax.loglog(sizes, l1_errors, "o-", label="Norma L₁", linewidth=2, markersize=8)
        ax.loglog(sizes, l2_errors, "s-", label="Norma L₂", linewidth=2, markersize=8)
        ax.loglog(sizes, linf_errors, "^-", label="Norma L∞", linewidth=2, markersize=8)

        ax.set_xlabel("Número de Elementos", fontsize=12)
        ax.set_ylabel("Erro Relativo", fontsize=12)
        ax.set_title("Análise de Convergência - Teste de Corretude", fontsize=14, fontweight="bold")
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.grid(True, which="major", ls="-", alpha=0.4)

        plt.tight_layout()
        plt.savefig("teste_corretude.png", dpi=300, bbox_inches="tight")
        plt.show()

    return results_data


def test_strong_scaling(temp_dir: Path):
    print("\n=== TESTE DE ESCALABILIDADE FORTE ===")

    nx, ny, nz = SIZES[-1]
    total_size = nx * ny * nz

    ini_path = temp_dir / f"reservoir_strong_{total_size}.ini"
    create_reservoir_ini(OG_RESERVOIR_PATH, str(ini_path), nx, ny, nz)

    results_data = []

    for mpi_procs in MPI_PROCESSES_STRONG:
        print(f"Executando com {mpi_procs} processos MPI...")

        results, memory = run_solver(str(ini_path), mpi_procs, f"TPFA_Strong_{mpi_procs}")

        if results:
            results_data.append(
                {
                    "mpi": mpi_procs,
                    "time": results["total_time"],
                    "memory_max": memory.get("max", 0),
                    "memory_avg": memory.get("avg", 0),
                }
            )

    if results_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        mpi_procs = [r["mpi"] for r in results_data]
        times = [r["time"] for r in results_data]
        memory_max = [r["memory_max"] for r in results_data]

        base_time = times[0] if times else 1
        speedup = [base_time / t for t in times]
        ideal_speedup = mpi_procs

        ax1.plot(mpi_procs, speedup, "o-", label="Speedup Real", linewidth=2, markersize=8)
        ax1.plot(mpi_procs, ideal_speedup, "--", label="Speedup Ideal", linewidth=2, alpha=0.6)
        ax1.set_xlabel("Número de Processos MPI", fontsize=12)
        ax1.set_ylabel("Speedup", fontsize=12)
        ax1.set_title("Escalabilidade Forte - Speedup", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.5, max(mpi_procs) + 0.5)

        ax2.bar(
            range(len(mpi_procs)),
            memory_max,
            tick_label=mpi_procs,
            color=sns.color_palette("viridis", len(mpi_procs)),
        )
        ax2.set_xlabel("Número de Processos MPI", fontsize=12)
        ax2.set_ylabel("Memória Máxima (MB)", fontsize=12)
        ax2.set_title("Escalabilidade Forte - Uso de Memória", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        for i, (m, mem) in enumerate(zip(mpi_procs, memory_max)):
            ax2.text(i, mem, f"{mem:.1f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle(
            f"Análise de Escalabilidade Forte - {total_size} elementos",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig("teste_escalabilidade_forte.png", dpi=300, bbox_inches="tight")
        plt.show()

    return results_data


def test_weak_scaling(temp_dir: Path):
    print("\n=== TESTE DE ESCALABILIDADE FRACA ===")

    weak_configs = [
        ((8, 8, 1), 1),
        ((16, 8, 1), 2),
        ((16, 16, 1), 4),
        ((32, 16, 1), 8),
        ((32, 32, 1), 16),
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
        create_reservoir_ini(OG_RESERVOIR_PATH, str(ini_path), nx, ny, nz)

        results, memory = run_solver(
            str(ini_path), mpi_procs, f"TPFA_Weak_{total_size}_{mpi_procs}"
        )

        if results:
            results_data.append(
                {
                    "size": total_size,
                    "mpi": mpi_procs,
                    "time": results["total_time"],
                    "memory_max": memory.get("max", 0),
                    "memory_avg": memory.get("avg", 0),
                    "elements_per_proc": elements_per_proc,
                }
            )

    if results_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        mpi_procs = [r["mpi"] for r in results_data]
        times = [r["time"] for r in results_data]
        memory_max = [r["memory_max"] for r in results_data]
        sizes = [r["size"] for r in results_data]

        base_time = times[0] if times else 1
        efficiency = [base_time / t * 100 for t in times]

        ax1.plot(mpi_procs, efficiency, "o-", linewidth=2, markersize=8, color="darkblue")
        ax1.axhline(y=100, color="r", linestyle="--", label="Eficiência Ideal (100%)", alpha=0.6)
        ax1.set_xlabel("Número de Processos MPI", fontsize=12)
        ax1.set_ylabel("Eficiência (%)", fontsize=12)
        ax1.set_title("Escalabilidade Fraca - Eficiência", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 120)

        for i, (m, s) in enumerate(zip(mpi_procs, sizes)):
            ax1.annotate(
                f"{s} elem",
                xy=(m, efficiency[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        memory_per_proc = [m / p for m, p in zip(memory_max, mpi_procs)]
        ax2.bar(
            range(len(mpi_procs)),
            memory_per_proc,
            tick_label=mpi_procs,
            color=sns.color_palette("coolwarm", len(mpi_procs)),
        )
        ax2.set_xlabel("Número de Processos MPI", fontsize=12)
        ax2.set_ylabel("Memória por Processo (MB)", fontsize=12)
        ax2.set_title("Escalabilidade Fraca - Memória por Processo", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        for i, (m, mem) in enumerate(zip(mpi_procs, memory_per_proc)):
            ax2.text(i, mem, f"{mem:.1f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle("Análise de Escalabilidade Fraca", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig("teste_escalabilidade_fraca.png", dpi=300, bbox_inches="tight")
        plt.show()

    return results_data


def test_performance(temp_dir: Path):
    print("\n=== TESTE DE DESEMPENHO ===")

    all_results = {mpi: [] for mpi in MPI_PROCESSES_PERFORMANCE}

    for nx, ny, nz in SIZES[:4]:
        total_size = nx * ny * nz

        ini_path = temp_dir / f"reservoir_perf_{total_size}.ini"
        create_reservoir_ini(OG_RESERVOIR_PATH, str(ini_path), nx, ny, nz)

        for mpi_procs in MPI_PROCESSES_PERFORMANCE:
            print(f"Executando {total_size} elementos com {mpi_procs} processos...")

            results, memory = run_solver(
                str(ini_path), mpi_procs, f"TPFA_Perf_{total_size}_{mpi_procs}"
            )

            if results:
                all_results[mpi_procs].append(
                    {
                        "size": total_size,
                        "preprocessing": results.get("preprocessing_time", 0),
                        "updating": results.get("updating_time", 0),
                        "solving": results.get("solving_time", 0),
                        "total": results.get("total_time", 0),
                        "memory": memory.get("max", 0),
                    }
                )

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    colors = sns.color_palette("Set2", 4)

    for row, mpi_procs in enumerate(MPI_PROCESSES_PERFORMANCE):
        data = all_results[mpi_procs]
        if not data:
            continue

        sizes = [d["size"] for d in data]
        preprocessing = [d["preprocessing"] for d in data]
        updating = [d["updating"] for d in data]
        solving = [d["solving"] for d in data]
        total = [d["total"] for d in data]
        memory = [d["memory"] for d in data]

        axes[row, 0].plot(sizes, preprocessing, "o-", color=colors[0], linewidth=2, markersize=7)
        axes[row, 0].set_title(f"Pré-processamento\n(MPI={mpi_procs})", fontsize=11)
        axes[row, 0].set_xlabel("Elementos", fontsize=10)
        axes[row, 0].set_ylabel("Tempo (s)", fontsize=10)
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_xscale("log")

        axes[row, 1].plot(sizes, updating, "s-", color=colors[1], linewidth=2, markersize=7)
        axes[row, 1].set_title(f"Atualização\n(MPI={mpi_procs})", fontsize=11)
        axes[row, 1].set_xlabel("Elementos", fontsize=10)
        axes[row, 1].set_ylabel("Tempo (s)", fontsize=10)
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_xscale("log")

        axes[row, 2].plot(sizes, solving, "^-", color=colors[2], linewidth=2, markersize=7)
        axes[row, 2].set_title(f"Solução\n(MPI={mpi_procs})", fontsize=11)
        axes[row, 2].set_xlabel("Elementos", fontsize=10)
        axes[row, 2].set_ylabel("Tempo (s)", fontsize=10)
        axes[row, 2].grid(True, alpha=0.3)
        axes[row, 2].set_xscale("log")

        axes[row, 3].plot(sizes, memory, "d-", color=colors[3], linewidth=2, markersize=7)
        axes[row, 3].set_title(f"Memória Máxima\n(MPI={mpi_procs})", fontsize=11)
        axes[row, 3].set_xlabel("Elementos", fontsize=10)
        axes[row, 3].set_ylabel("Memória (MB)", fontsize=10)
        axes[row, 3].grid(True, alpha=0.3)
        axes[row, 3].set_xscale("log")

    plt.suptitle(
        "Análise Detalhada de Desempenho - Decomposição de Tempos e Memória",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig("teste_desempenho_detalhado.png", dpi=300, bbox_inches="tight")
    plt.show()

    return all_results


def main():
    """Função principal"""
    print("=" * 60)
    print("ANÁLISE DE DESEMPENHO DO TPFASOLVER")
    print("=" * 60)

    global OG_RESERVOIR_PATH, RUN_PY_PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OG_RESERVOIR_PATH = os.path.abspath(os.path.join(script_dir, OG_RESERVOIR_PATH))
    RUN_PY_PATH = os.path.abspath(os.path.join(script_dir, RUN_PY_PATH))

    if not os.path.exists(OG_RESERVOIR_PATH):
        print(f"Erro: Arquivo de configuração não encontrado: {OG_RESERVOIR_PATH}")
        sys.exit(1)

    run_py_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), RUN_PY_PATH))
    if not os.path.exists(run_py_abs):
        print(f"Erro: Script run.py não encontrado: {run_py_abs}")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="INIS_", dir="/tmp") as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Diretório temporário criado: {temp_path}")

        try:
            correctness_results = test_correctness(temp_path)
            strong_scaling_results = test_strong_scaling(temp_path)
            weak_scaling_results = test_weak_scaling(temp_path)
            performance_results = test_performance(temp_path)

            print("\n" + "=" * 60)
            print("ANÁLISE CONCLUÍDA COM SUCESSO!")
            print("Gráficos salvos:")
            print("  - teste_corretude.png")
            print("  - teste_escalabilidade_forte.png")
            print("  - teste_escalabilidade_fraca.png")
            print("  - teste_desempenho_detalhado.png")
            print("=" * 60)

        except Exception as e:
            print(f"\nErro durante a execução: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
