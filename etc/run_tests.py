import os
import subprocess
import tempfile
import shutil
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import math

MAX_ELEMENTS = 5000000
BASE_PERF_N = 5000000 / (math.sqrt(2) ** 10)

PROCESS_LIST_SCALING = [1, 2, 3, 4, 5, 6]
BASE_WEAK_N = MAX_ELEMENTS // PROCESS_LIST_SCALING[-1]
PERF_N_PROCS = 6
SCRIPT_NAME = "test_gpu.py"
RESULTS_DIR = "results"


def run_test(tmp_dir, device, nprocs, n_elements):
    comm = PETSc.COMM_WORLD

    yaml_filename = f"results_{device}_{n_elements}_{nprocs}.yaml"
    yaml_path = os.path.join(tmp_dir, yaml_filename)

    base_command = [
        "mpirun",
        "-n",
        str(nprocs),
        sys.executable,
        SCRIPT_NAME,
        "-device",
        device,
        "-output",
        tmp_dir,
        "-number",
        str(n_elements),
    ]

    export_cmd = (
        "export OMP_PROC_BIND=close && "
        "export OMP_PLACES=cores && "
        "export PETSC_ARCH=myconfigureopt && "
    )
    full_command = f"{export_cmd} {' '.join(base_command)}"

    try:
        subprocess.run(full_command, check=True, capture_output=True, text=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e.stderr}")
        return None

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Output file not found {yaml_path}")
        return None


def run_performance_tests(tmp_dir):
    results = []
    n = BASE_PERF_N
    nprocs = PERF_N_PROCS

    while n <= MAX_ELEMENTS:
        current_n = int(n)
        for device in ["cpu", "gpu"]:
            print(f"Running Performance Test: {device}, N={current_n}, Procs={nprocs}")
            data = run_test(tmp_dir, device, nprocs, current_n)
            if data:
                data.update(
                    {"test_type": "performance", "device": device, "nprocs": nprocs, "N": current_n}
                )
                results.append(data)
        n *= 1.41421
    return results


def run_strong_scaling_tests(tmp_dir):
    results = []
    n_elements = MAX_ELEMENTS

    for nprocs in PROCESS_LIST_SCALING:
        for device in ["cpu", "gpu"]:
            print(f"Running Strong Scaling: {device}, N={n_elements}, Procs={nprocs}")
            data = run_test(tmp_dir, device, nprocs, n_elements)
            if data:
                data.update(
                    {
                        "test_type": "strong_scaling",
                        "device": device,
                        "nprocs": nprocs,
                        "N": n_elements,
                    }
                )
                results.append(data)
    return results


def run_weak_scaling_tests(tmp_dir):
    results = []

    for nprocs in PROCESS_LIST_SCALING:
        n_elements = BASE_WEAK_N * nprocs
        for device in ["cpu", "gpu"]:
            print(f"Running Weak Scaling: {device}, N={n_elements}, Procs={nprocs}")
            data = run_test(tmp_dir, device, nprocs, n_elements)
            if data:
                data.update(
                    {
                        "test_type": "weak_scaling",
                        "device": device,
                        "nprocs": nprocs,
                        "N": n_elements,
                    }
                )
                results.append(data)
    return results


def plot_results(all_data):
    if not all_data:
        print("No data collected, skipping plotting.")
        return

    df = pd.DataFrame(all_data)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    df_perf = df[df["test_type"] == "performance"]
    if not df_perf.empty:
        g = sns.relplot(
            data=df_perf,
            x="N",
            y="execution_time",
            hue="device",
            style="device",
            kind="line",
            markers=True,
        )
        g.set_titles(f"Performance Test ({PERF_N_PROCS} MPI Processes)")
        g.set_axis_labels("Problem Size (N)", "Execution Time (s)")
        g.set(xscale="log")
        plt.savefig(os.path.join(RESULTS_DIR, "performance.png"))
        plt.close()

    df_strong = df[df["test_type"] == "strong_scaling"].copy()
    if not df_strong.empty:
        t1_map = df_strong[df_strong["nprocs"] == 1].set_index("device")["execution_time"].to_dict()
        df_strong.loc[:, "t1"] = df_strong["device"].map(t1_map)
        df_strong.loc[:, "speedup"] = df_strong["t1"] / df_strong["execution_time"]

        g = sns.relplot(
            data=df_strong,
            x="nprocs",
            y="speedup",
            hue="device",
            style="device",
            kind="line",
            markers=True,
        )
        g.set_titles(f"Strong Scaling (N = {MAX_ELEMENTS})")
        g.set_axis_labels("Number of Processes", "Speedup")

        procs = sorted(df_strong["nprocs"].unique())
        ideal_speedup = [p for p in procs]
        g.ax.plot(procs, ideal_speedup, "k--", label="Ideal Speedup")
        g.ax.legend()
        g.ax.set_ylim(bottom=0)

        plt.savefig(os.path.join(RESULTS_DIR, "strong_scaling.png"))
        plt.close()

    df_weak = df[df["test_type"] == "weak_scaling"].copy()
    if not df_weak.empty:
        t1_map = df_weak[df_weak["nprocs"] == 1].set_index("device")["execution_time"].to_dict()
        df_weak.loc[:, "t1"] = df_weak["device"].map(t1_map)
        df_weak.loc[:, "efficiency"] = (df_weak["t1"] / df_weak["execution_time"]) * 100.0

        g = sns.relplot(
            data=df_weak,
            x="nprocs",
            y="efficiency",
            hue="device",
            style="device",
            kind="line",
            markers=True,
        )
        g.set_titles(f"Weak Scaling (N/Proc = {BASE_WEAK_N})")
        g.set_axis_labels("Number of Processes", "Weak Scaling Efficiency (%)")

        g.ax.axhline(100.0, ls="--", color="k", label="Ideal Efficiency (100%)")
        g.ax.legend()
        g.ax.set_ylim(bottom=0)

        plt.savefig(os.path.join(RESULTS_DIR, "weak_scaling.png"))
        plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    if not os.path.exists(SCRIPT_NAME):
        print(f"Error: {SCRIPT_NAME} not found.")
        return

    try:
        from petsc4py import PETSc
    except ImportError:
        print("Error: petsc4py not found.")
        return

    global PETSc
    from petsc4py import PETSc

    tmp_dir = tempfile.mkdtemp(prefix="petsc_test_", dir="/tmp")
    all_results = []

    try:
        all_results.extend(run_performance_tests(tmp_dir))
        all_results.extend(run_strong_scaling_tests(tmp_dir))
        all_results.extend(run_weak_scaling_tests(tmp_dir))
    finally:
        print(f"Cleaning up temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir)

    plot_results(all_results)
    print(f"Tests complete. Plots saved in '{RESULTS_DIR}' directory.")


if __name__ == "__main__":
    main()
