#!/usr/bin/env python3

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_correctness(data, output_path):
    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    sizes = [r["size"] for r in data]
    l1_errors = [r["l1_error"] for r in data]
    l2_errors = [r["l2_error"] for r in data]
    linf_errors = [r["linf_error"] for r in data]

    ax.loglog(sizes, l1_errors, "o-", label="Norma L₁", linewidth=2, markersize=8)
    ax.loglog(sizes, l2_errors, "s-", label="Norma L₂", linewidth=2, markersize=8)
    ax.loglog(sizes, linf_errors, "^-", label="Norma L∞", linewidth=2, markersize=8)
    minn = min(sizes)
    minerr = min(min(l1_errors), min(l2_errors), min(linf_errors))

    import numpy as np

    ax.plot(
        [minn, 10 * minn],
        [minerr, minerr * np.exp((-2 / 3) * np.log(10))],
        color="r",
        label="O(n²)",
    )
    ax.set_xlabel("Número de Elementos", fontsize=12)
    ax.set_ylabel("Erro Relativo", fontsize=12)
    ax.set_title("Análise de Convergência - Teste de Corretude", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_strong_scaling(data, output_path):
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    mpi_procs = [r["mpi"] for r in data]
    times = [r["time"] for r in data]
    memory_max = [r["memory_max"] for r in data]
    total_size = data[0]["total_size"] if data else 0

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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_weak_scaling(data, output_path):
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    mpi_procs = [r["mpi"] for r in data]
    times = [r["time"] for r in data]
    memory_max = [r["memory_max"] for r in data]
    sizes = [r["size"] for r in data]

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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_performance(data, output_path):
    if not data:
        return

    MPI_PROCESSES_PERFORMANCE = sorted(list(data.keys()))
    num_mpi_configs = len(MPI_PROCESSES_PERFORMANCE)

    # Calculate global ranges for consistent scaling
    all_times = []
    all_memory = []

    for mpi_procs in MPI_PROCESSES_PERFORMANCE:
        proc_data = data[mpi_procs]
        if not proc_data:
            continue

        # Collect all time values
        for d in proc_data:
            all_times.extend([d["preprocessing"], d["updating"], d["solving"]])
            all_memory.append(d["memory"])

    # Calculate global ranges with small buffer for better visualization
    if all_times:
        time_min = min(all_times) * 0.8
        time_max = max(all_times) * 1.2
    else:
        time_min, time_max = 0.1, 100

    if all_memory:
        memory_min = min(all_memory) * 0.8
        memory_max = max(all_memory) * 1.2
    else:
        memory_min, memory_max = 1, 1000

    fig, axes = plt.subplots(
        num_mpi_configs, 2, figsize=(14, 5 * num_mpi_configs), constrained_layout=True
    )

    if num_mpi_configs == 1:
        axes = axes.reshape(1, -1)

    time_styles = {
        "preprocessing": {"color": "mediumseagreen", "marker": "o", "label": "Pré-processamento"},
        "updating": {"color": "coral", "marker": "s", "label": "Atualização"},
        "solving": {"color": "cornflowerblue", "marker": "^", "label": "Solução"},
    }

    for row, mpi_procs in enumerate(MPI_PROCESSES_PERFORMANCE):
        proc_data = data[mpi_procs]
        if not proc_data:
            continue

        sizes = [d["size"] for d in proc_data]

        # Time plot
        ax_time = axes[row, 0]
        for time_key, style in time_styles.items():
            times = [d[time_key] for d in proc_data]
            ax_time.plot(
                sizes,
                times,
                marker=style["marker"],
                linestyle="-",
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=7,
            )

        ax_time.set_title(f"Decomposição de Tempos (MPI={mpi_procs})", fontsize=12)
        ax_time.set_xlabel("Elementos", fontsize=11)
        ax_time.set_ylabel("Tempo (s)", fontsize=11)
        ax_time.grid(True, which="both", ls="--", alpha=0.4)
        ax_time.set_xscale("log")
        ax_time.set_yscale("log")
        ax_time.set_ylim(time_min, time_max)  # Set consistent time scale
        ax_time.legend(title="Etapa do Processo")

        # Memory plot
        ax_mem = axes[row, 1]
        memory = [d["memory"] for d in proc_data]
        ax_mem.plot(
            sizes,
            memory,
            marker="d",
            linestyle="-",
            color="palevioletred",
            linewidth=2,
            markersize=7,
        )

        ax_mem.set_title(f"Memória Máxima (MPI={mpi_procs})", fontsize=12)
        ax_mem.set_xlabel("Elementos", fontsize=11)
        ax_mem.set_ylabel("Memória (MB)", fontsize=11)
        ax_mem.grid(True, which="both", ls="--", alpha=0.4)
        ax_mem.set_xscale("log")
        ax_mem.set_yscale("log")
        ax_mem.set_ylim(memory_min, memory_max)  # Set consistent memory scale

    plt.suptitle(
        "Análise Detalhada de Desempenho - Tempos e Memória por Configuração MPI",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    if not os.path.exists(results_dir):
        print(f"Erro: Diretório de resultados não encontrado: {results_dir}")
        sys.exit(1)

    yml_files = list(Path(results_dir).glob("*.yml"))
    if not yml_files:
        print(f"Erro: Nenhum arquivo YAML encontrado em {results_dir}")
        sys.exit(1)

    timestamps = set()
    for f in yml_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            timestamp = "_".join(parts[-2:])
            timestamps.add(timestamp)

    if not timestamps:
        print("Erro: Não foi possível encontrar timestamps nos arquivos")
        sys.exit(1)

    latest_timestamp = sorted(timestamps)[-1]
    print(f"Usando timestamp: {latest_timestamp}")

    correctness_file = os.path.join(results_dir, f"correctness_{latest_timestamp}.yml")
    if os.path.exists(correctness_file):
        with open(correctness_file, "r") as f:
            data = yaml.safe_load(f)
        plot_correctness(data, os.path.join(results_dir, f"teste_corretude_{latest_timestamp}.png"))
        print(f"Plotado: teste_corretude_{latest_timestamp}.png")

    strong_file = os.path.join(results_dir, f"strong_scaling_{latest_timestamp}.yml")
    if os.path.exists(strong_file):
        with open(strong_file, "r") as f:
            data = yaml.safe_load(f)
        plot_strong_scaling(
            data, os.path.join(results_dir, f"teste_escalabilidade_forte_{latest_timestamp}.png")
        )
        print(f"Plotado: teste_escalabilidade_forte_{latest_timestamp}.png")

    weak_file = os.path.join(results_dir, f"weak_scaling_{latest_timestamp}.yml")
    if os.path.exists(weak_file):
        with open(weak_file, "r") as f:
            data = yaml.safe_load(f)
        plot_weak_scaling(
            data, os.path.join(results_dir, f"teste_escalabilidade_fraca_{latest_timestamp}.png")
        )
        print(f"Plotado: teste_escalabilidade_fraca_{latest_timestamp}.png")

    performance_file = os.path.join(results_dir, f"performance_{latest_timestamp}.yml")
    if os.path.exists(performance_file):
        with open(performance_file, "r") as f:
            data = yaml.safe_load(f)
        plot_performance(
            data, os.path.join(results_dir, f"teste_desempenho_detalhado_{latest_timestamp}.png")
        )
        print(f"Plotado: teste_desempenho_detalhado_{latest_timestamp}.png")

    print(f"\nGráficos salvos em: {results_dir}")


if __name__ == "__main__":
    main()
