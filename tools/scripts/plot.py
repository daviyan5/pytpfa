#!/usr/bin/env python3

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

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

    fig, ax = plt.subplots(figsize=(12, 8))

    sizes = [r["size"] for r in data]
    l1_errors = [r["l1_error"] for r in data]
    l2_errors = [r["l2_error"] for r in data]
    linf_errors = [r["linf_error"] for r in data]

    ax.loglog(sizes, l1_errors, "o-", label="Norma L₁", linewidth=2, markersize=8)
    ax.loglog(sizes, l2_errors, "s-", label="Norma L₂", linewidth=2, markersize=8)
    ax.loglog(sizes, linf_errors, "^-", label="Norma L∞", linewidth=2, markersize=8)

    minn = min(sizes)
    minerr = min(min(l1_errors), min(l2_errors), min(linf_errors))

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    mpi_procs = [r["mpi"] for r in data]
    times = [r["time"] for r in data]
    memory_max = [r["memory_max"] for r in data]
    total_size = data[0]["total_size"] if data else 0

    base_time = times[0] if times else 1
    speedup = [base_time / t for t in times]
    efficiency = [s / p * 100 for s, p in zip(speedup, mpi_procs)]

    ax1.plot(
        mpi_procs, speedup, "o-", label="Speedup Real", linewidth=2, markersize=8, color="darkblue"
    )
    ax1.plot(mpi_procs, mpi_procs, "--", label="Speedup Ideal", linewidth=2, alpha=0.6, color="red")
    ax1.set_xlabel("Número de Processos MPI", fontsize=12)
    ax1.set_ylabel("Speedup", fontsize=12)
    ax1.set_title("Escalabilidade Forte - Speedup", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, max(mpi_procs) + 0.5)
    ax1.set_ylim(0, max(max(speedup), max(mpi_procs)) * 1.1)

    color_map = plt.cm.viridis(np.linspace(0, 1, len(mpi_procs)))
    bars = ax2.bar(mpi_procs, memory_max, color=color_map, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Número de Processos MPI", fontsize=12)
    ax2.set_ylabel("Memória Máxima (MB)", fontsize=12)
    ax2.set_title("Escalabilidade Forte - Uso de Memória", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, (bar, eff) in enumerate(zip(bars, efficiency)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{eff:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.suptitle(
        f"Análise de Escalabilidade Forte - {total_size:,} elementos",
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

    fig, ax1 = plt.subplots(figsize=(8, 8))

    mpi_procs = [r["mpi"] for r in data]
    times = [r["time"] for r in data]
    sizes = [r["size"] for r in data]
    elements_per_proc = [r["elements_per_proc"] for r in data]

    base_time = times[0] if times else 1
    efficiency = [(base_time / t) * 100 for t in times]

    ax1.plot(mpi_procs, efficiency, "o-", linewidth=2, markersize=8, color="darkblue")
    ax1.axhline(y=100, color="r", linestyle="--", label="Eficiência Ideal (100%)", alpha=0.6)
    ax1.set_xlabel("Número de Processos MPI", fontsize=12)
    ax1.set_ylabel("Eficiência (%)", fontsize=12)
    ax1.set_title("Eficiência de Escalabilidade Fraca", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(120, max(efficiency) * 1.1))
    ax1.set_xlim(0.5, max(mpi_procs) + 0.5)

    for i, (m, e) in enumerate(zip(mpi_procs, efficiency)):
        ax1.annotate(
            f"{e:.1f}%",
            xy=(m, e),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_performance(data, output_path):
    if not data:
        return

    MPI_PROCESSES = sorted([k for k in data.keys() if data[k]])
    if not MPI_PROCESSES:
        return

    num_mpi = len(MPI_PROCESSES)

    fig, axes = plt.subplots(2, num_mpi, figsize=(6 * num_mpi, 10))

    if num_mpi == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif axes.ndim == 1:
        axes = axes.reshape(2, 1)

    time_styles = {
        "preprocessing": {"color": "mediumseagreen", "marker": "o", "label": "Pré-processamento"},
        "updating": {"color": "coral", "marker": "s", "label": "Atualização"},
        "solving": {"color": "cornflowerblue", "marker": "^", "label": "Solução"},
    }

    for col, mpi_procs in enumerate(MPI_PROCESSES):
        proc_data = data[mpi_procs]
        if not proc_data:
            continue

        sizes = [d["size"] for d in proc_data]

        ax_time = axes[0, col]
        for time_key, style in time_styles.items():
            times = [d[time_key] for d in proc_data]
            ax_time.loglog(
                sizes,
                times,
                marker=style["marker"],
                linestyle="-",
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=7,
            )

        ax_time.set_title(f"MPI={mpi_procs} processos", fontsize=11, fontweight="bold")
        ax_time.set_xlabel("Número de Elementos", fontsize=10)
        if col == 0:
            ax_time.set_ylabel("Tempo (s)", fontsize=10)
        ax_time.grid(True, which="both", ls="--", alpha=0.3)
        ax_time.legend(loc="best", fontsize=8)

        ax_mem = axes[1, col]
        memory = [d["memory"] for d in proc_data]
        ax_mem.loglog(
            sizes, memory, marker="d", linestyle="-", color="darkviolet", linewidth=2, markersize=7
        )
        ax_mem.set_title(f"Memória - MPI={mpi_procs}", fontsize=11, fontweight="bold")
        ax_mem.set_xlabel("Número de Elementos", fontsize=10)
        if col == 0:
            ax_mem.set_ylabel("Memória Máxima (MB)", fontsize=10)
        ax_mem.grid(True, which="both", ls="--", alpha=0.3)

    plt.suptitle(
        "Análise de Desempenho - Decomposição de Tempo e Memória", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    if not os.path.exists(results_dir):
        print(f"Erro: Diretório de resultados não encontrado: {results_dir}")
        return

    cases = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    if not cases:
        print(f"Erro: Nenhum caso encontrado em {results_dir}")
        return

    for case in cases:
        print(f"\nProcessando caso: {case}")
        process_case(os.path.join(results_dir, case))


def process_case(results_dir):
    yml_files = list(Path(results_dir).glob("*.yml"))
    if not yml_files:
        print(f"Nenhum arquivo YAML encontrado em {results_dir}")
        return

    timestamps = set()
    for f in yml_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            timestamp = "_".join(parts[-2:])
            timestamps.add(timestamp)

    if not timestamps:
        print("Não foi possível encontrar timestamps nos arquivos")
        return

    latest_timestamp = sorted(timestamps)[-1]
    print(f"Usando timestamp: {latest_timestamp}")

    correctness_file = os.path.join(results_dir, f"correctness_{latest_timestamp}.yml")
    if os.path.exists(correctness_file):
        with open(correctness_file, "r") as f:
            data = yaml.safe_load(f)
        if data:
            plot_correctness(
                data, os.path.join(results_dir, f"teste_corretude_{latest_timestamp}.png")
            )
            print(f"Plotado: teste_corretude_{latest_timestamp}.png")

    strong_file = os.path.join(results_dir, f"strong_scaling_{latest_timestamp}.yml")
    if os.path.exists(strong_file):
        with open(strong_file, "r") as f:
            data = yaml.safe_load(f)
        if data:
            plot_strong_scaling(
                data,
                os.path.join(results_dir, f"teste_escalabilidade_forte_{latest_timestamp}.png"),
            )
            print(f"Plotado: teste_escalabilidade_forte_{latest_timestamp}.png")

    weak_file = os.path.join(results_dir, f"weak_scaling_{latest_timestamp}.yml")
    if os.path.exists(weak_file):
        with open(weak_file, "r") as f:
            data = yaml.safe_load(f)
        if data:
            plot_weak_scaling(
                data,
                os.path.join(results_dir, f"teste_escalabilidade_fraca_{latest_timestamp}.png"),
            )
            print(f"Plotado: teste_escalabilidade_fraca_{latest_timestamp}.png")

    performance_file = os.path.join(results_dir, f"performance_{latest_timestamp}.yml")
    if os.path.exists(performance_file):
        with open(performance_file, "r") as f:
            data = yaml.safe_load(f)
        if data:
            plot_performance(
                data, os.path.join(results_dir, f"teste_desempenho_{latest_timestamp}.png")
            )
            print(f"Plotado: teste_desempenho_{latest_timestamp}.png")

    print(f"Gráficos salvos em: {results_dir}")


if __name__ == "__main__":
    main()
