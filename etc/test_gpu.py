import sys
import time
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import gc

try:
    import pynvml

    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

from petsc4py import PETSc

PETSc.Sys.pushErrorHandler("traceback")


def get_memory_usage():
    ram_used = psutil.Process().memory_info().rss / (1024**2)
    vram_used = 0
    if HAS_NVML:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = mem_info.used / (1024**2)
    return ram_used, vram_used


def check_cuda_support():
    try:
        test_vec = PETSc.Vec().create()
        test_vec.setType("cuda")
        test_vec.destroy()
        return True
    except:
        return False


def setup_problem_coo(comm, N, mat_type, b_vec_type, x_vec_type):
    rank = comm.Get_rank()
    t_start = time.perf_counter()

    h = 1.0 / (N - 1)
    kx, ky = 1.0, 0.1
    beta_x, beta_y = 10.0, 5.0
    alpha = 1.0

    n_total = N * N
    n_local = n_total // comm.size
    rank = comm.rank

    i_start = rank * n_local
    i_end = (rank + 1) * n_local if rank < comm.size - 1 else n_total
    n_local_actual = i_end - i_start

    local_indices = np.arange(i_start, i_end, dtype=np.int32)

    i_coords = local_indices // N
    j_coords = local_indices % N

    is_boundary = (i_coords == 0) | (i_coords == N - 1) | (j_coords == 0) | (j_coords == N - 1)
    is_interior = ~is_boundary

    interior_indices = local_indices[is_interior]
    boundary_indices = local_indices[is_boundary]

    n_interior = len(interior_indices)
    n_boundary = len(boundary_indices)

    interior_rows = np.repeat(interior_indices, 5)
    interior_cols = np.zeros(n_interior * 5, dtype=np.int32)
    interior_vals = np.zeros(n_interior * 5, dtype=np.float64)

    interior_cols[0::5] = interior_indices
    interior_cols[1::5] = interior_indices - 1
    interior_cols[2::5] = interior_indices + 1
    interior_cols[3::5] = interior_indices - N
    interior_cols[4::5] = interior_indices + N

    center_coef = 2 * kx / h**2 + 2 * ky / h**2 + alpha + beta_x / h + beta_y / h
    west_coef = -kx / h**2 - beta_x / h
    east_coef = -kx / h**2
    south_coef = -ky / h**2 - beta_y / h
    north_coef = -ky / h**2

    interior_vals[0::5] = center_coef
    interior_vals[1::5] = west_coef
    interior_vals[2::5] = east_coef
    interior_vals[3::5] = south_coef
    interior_vals[4::5] = north_coef

    boundary_rows = boundary_indices
    boundary_cols = boundary_indices
    boundary_vals = np.ones(n_boundary, dtype=np.float64)

    rows = np.concatenate([interior_rows, boundary_rows])
    cols = np.concatenate([interior_cols, boundary_cols])
    vals = np.concatenate([interior_vals, boundary_vals])

    x_coords = j_coords * h
    y_coords = i_coords * h
    b_array = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords) + np.exp(
        -((x_coords - 0.5) ** 2 + (y_coords - 0.5) ** 2) / 0.1
    )
    b_array[is_boundary] = 0.0

    A = PETSc.Mat().create(comm=comm)
    A.setSizes([(n_local_actual, n_total), (n_local_actual, n_total)])
    A.setType(mat_type)
    A.setFromOptions()

    A.setPreallocationCOO(rows, cols)
    A.setValuesCOO(vals, PETSc.InsertMode.INSERT_VALUES)
    A.assemblyBegin()
    A.assemblyEnd()

    b = A.createVecRight()
    if b_vec_type == "cuda":
        b.setType(PETSc.Vec.Type.CUDA)
    b.setArray(b_array)
    b.assemblyBegin()
    b.assemblyEnd()

    x = A.createVecRight()
    if x_vec_type == "cuda":
        x.setType(PETSc.Vec.Type.CUDA)
    x.set(0.0)

    comm.barrier()
    t_end = time.perf_counter()
    setup_time = t_end - t_start
    if rank == 0:
        print(f"Setup time: {setup_time:.6f}s")

    return A, b, x, setup_time


def run_benchmark(problem_size):
    comm = PETSc.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    KSP_TYPES = ["gmres", "bcgs", "bicg", "fgmres", "tfqmr"]

    if nprocs == 1:
        PC_TYPES_BASE = ["jacobi", "bjacobi", "asm", "ilu"]
    else:
        PC_TYPES_BASE = ["jacobi", "bjacobi", "asm"]

    scenarios = {
        "1. All GPU": {
            "mat_type": "aijcusparse",
            "b_type": "cuda",
            "x_type": "cuda",
            "use_hypre": True,
        },
        "2. A,b on GPU": {
            "mat_type": "aijcusparse",
            "b_type": "cuda",
            "x_type": "standard",
            "use_hypre": True,
        },
        "3. A on GPU": {
            "mat_type": "aijcusparse",
            "b_type": "standard",
            "x_type": "standard",
            "use_hypre": True,
        },
        "4. All CPU": {
            "mat_type": "aij",
            "b_type": "standard",
            "x_type": "standard",
            "use_hypre": True,
        },
    }

    cuda_available = check_cuda_support()
    if rank == 0:
        print(f"CUDA Support: {'Available' if cuda_available else 'Not Available'}")
        print(f"MPI Processes: {nprocs}")
        if not cuda_available:
            print("WARNING: GPU scenarios will be skipped")

    all_results = []

    if rank == 0:
        print(f"\n--- Starting PETSc Benchmark (Grid Size: {problem_size}x{problem_size}) ---")

    for scenario_name, config in scenarios.items():
        if not cuda_available and config["mat_type"] != "aij":
            if rank == 0:
                print(f"\n[{scenario_name}] SKIPPED - CUDA not available")
            continue

        PC_TYPES = PC_TYPES_BASE + (["hypre"] if config["use_hypre"] else [])

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[{scenario_name}] Setting up problem...")
            print(
                f"Testing {len(KSP_TYPES)} KSP × {len(PC_TYPES)} PC = {len(KSP_TYPES)*len(PC_TYPES)} combinations"
            )
            print(f"{'='*60}")

        comm.barrier()
        gc.collect()

        ram_before_setup, vram_before_setup = get_memory_usage()

        try:
            A, b, x, transfer_time = setup_problem_coo(
                comm, problem_size, config["mat_type"], config["b_type"], config["x_type"]
            )
        except Exception as e:
            if rank == 0:
                print(f"[{scenario_name}] FAILED to setup: {str(e)}")
            continue

        ram_after_setup, vram_after_setup = get_memory_usage()
        setup_ram = ram_after_setup - ram_before_setup
        setup_vram = vram_after_setup - vram_before_setup

        for ksp_type in KSP_TYPES:
            for pc_type in PC_TYPES:
                comm.barrier()
                gc.collect()

                ram_before, vram_before = get_memory_usage()

                try:
                    ksp = PETSc.KSP().create(comm)
                    ksp.setOperators(A)
                    ksp.setType(ksp_type)

                    pc = ksp.getPC()
                    pc.setType(pc_type)

                    if pc_type == "hypre" and (
                        config["b_type"] == "cuda" or config["x_type"] == "cuda"
                    ):
                        pc.setFromOptions()
                        pc.setHYPREType("boomeramg")

                    # ksp.setTolerances(rtol=1e-5, max_it=50000)
                    ksp.setFromOptions()

                    x.set(0.0)
                    comm.barrier()

                    solve_start = time.perf_counter()
                    ksp.solve(b, x)
                    comm.barrier()
                    solve_end = time.perf_counter()

                    solve_time = solve_end - solve_start
                    iterations = ksp.getIterationNumber()
                    residual = ksp.getResidualNorm()
                    converged = ksp.is_converged

                    comm.barrier()

                    ram_after, vram_after = get_memory_usage()

                    if rank == 0:
                        if converged:
                            result = {
                                "Scenario": scenario_name,
                                "KSP": ksp_type,
                                "PC": pc_type,
                                "Solver": f"{ksp_type}/{pc_type}",
                                "Setup Time (s)": transfer_time,
                                "Solve Time (s)": solve_time,
                                "Total Time (s)": transfer_time + solve_time,
                                "Iterations": iterations,
                                "Residual": residual,
                                "RAM Usage (MB)": ram_after - ram_before,
                                "VRAM Usage (MB)": vram_after - vram_before,
                                "Setup RAM (MB)": setup_ram,
                                "Setup VRAM (MB)": setup_vram,
                            }
                            all_results.append(result)
                            print(
                                f"  {ksp_type:8s}/{pc_type:8s}: {solve_time:8.4f}s ({iterations:4d} iter)"
                            )
                        else:
                            print(f"  {ksp_type:8s}/{pc_type:8s}: DIVERGED")

                    ksp.destroy()

                except Exception as e:
                    if rank == 0:
                        error_msg = str(e)
                        if "error code" in error_msg:
                            print(f"  {ksp_type:8s}/{pc_type:8s}: FAILED")
                        else:
                            print(f"  {ksp_type:8s}/{pc_type:8s}: FAILED - {error_msg[:50]}")

                gc.collect()
                comm.barrier()

        A.destroy()
        b.destroy()
        x.destroy()

        if rank == 0:
            print()

    if rank == 0:
        return pd.DataFrame(all_results)
    else:
        return None


def create_summary_plot(df, output_dir, mpi_procs, problem_size):
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    scenarios = ["1. All GPU", "2. A,b on GPU", "3. A on GPU", "4. All CPU"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    best_results = []
    for scenario in scenarios:
        subset = df[df["Scenario"] == scenario]
        if len(subset) > 0:
            best = subset.nsmallest(1, "Solve Time (s)").iloc[0]
            best_results.append(best)

    # Plot 1: Best solver comparison
    ax1 = fig.add_subplot(gs[0, :])
    if best_results:
        scenario_names = [b["Scenario"].split(". ")[1] for b in best_results]
        solve_times = [b["Solve Time (s)"] for b in best_results]
        solvers = [b["Solver"] for b in best_results]

        bars = ax1.bar(
            scenario_names,
            solve_times,
            color=colors[: len(scenario_names)],
            edgecolor="black",
            linewidth=1.5,
        )

        for bar, solver in zip(bars, solvers):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.02,
                f"{solver}\n{height:.4f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax1.set_ylabel("Solve Time (seconds)", fontsize=12, fontweight="bold", labelpad=10)
        ax1.set_title("BEST SOLVER FOR EACH SCENARIO", fontsize=14, fontweight="bold", pad=20)
        ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Speedup analysis
    ax3 = fig.add_subplot(gs[1, :])
    speedup_data = []
    cpu_subset = df[df["Scenario"] == "4. All CPU"]
    if len(cpu_subset) > 0:
        cpu_best = cpu_subset["Solve Time (s)"].min()

        for scenario in scenarios[:3]:
            subset = df[df["Scenario"] == scenario]
            if len(subset) > 0:
                gpu_best = subset["Solve Time (s)"].min()
                speedup = cpu_best / gpu_best
                speedup_data.append({"Scenario": scenario.split(". ")[1], "Speedup": speedup})

        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            bars = ax3.bar(
                speedup_df["Scenario"],
                speedup_df["Speedup"],
                color=colors[: len(speedup_data)],
                edgecolor="black",
                linewidth=1.5,
            )
            ax3.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="CPU Baseline")

            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    f"{height:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            ax3.set_ylabel("Speedup Factor", fontsize=12, fontweight="bold", labelpad=10)
            ax3.set_title(
                "GPU Speedup vs CPU (Best Solvers)", fontsize=12, fontweight="bold", pad=20
            )
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis="y")

    # Plot 3: Setup vs Solve time
    ax4 = fig.add_subplot(gs[2, :])
    comparison_data = []
    for scenario in scenarios:
        subset = df[df["Scenario"] == scenario]
        if len(subset) > 0:
            best = subset.nsmallest(1, "Solve Time (s)").iloc[0]
            comparison_data.append(
                {
                    "Scenario": scenario.split(". ")[1],
                    "Setup": best["Setup Time (s)"],
                    "Solve": best["Solve Time (s)"],
                }
            )

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        x = np.arange(len(comp_df))
        width = 0.35

        ax4.bar(
            x - width / 2,
            comp_df["Setup"],
            width,
            label="Setup/Transfer",
            color="#FF6B6B",
            edgecolor="black",
            linewidth=1,
        )
        ax4.bar(
            x + width / 2,
            comp_df["Solve"],
            width,
            label="Solve",
            color="#4ECDC4",
            edgecolor="black",
            linewidth=1,
        )

        ax4.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold", labelpad=10)
        ax4.set_title("Setup vs Solve Time (Best Solvers)", fontsize=12, fontweight="bold", pad=20)
        ax4.set_xticks(x)
        ax4.set_xticklabels(comp_df["Scenario"])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

    # Plot 4: VRAM usage
    ax5 = fig.add_subplot(gs[3, :])
    vram_data = []
    for scenario in scenarios:
        subset = df[df["Scenario"] == scenario]
        if len(subset) > 0:
            best = subset.nsmallest(1, "Solve Time (s)").iloc[0]
            vram_data.append(
                {
                    "Scenario": scenario.split(". ")[1],
                    "Setup VRAM": best.get("Setup VRAM (MB)", 0),
                    "Solve VRAM": best.get("VRAM Usage (MB)", 0),
                }
            )

    if vram_data:
        vram_df = pd.DataFrame(vram_data)
        x = np.arange(len(vram_df))
        width = 0.35

        ax5.bar(
            x - width / 2,
            vram_df["Setup VRAM"],
            width,
            label="Setup VRAM",
            color="#9B59B6",
            edgecolor="black",
            linewidth=1,
        )
        ax5.bar(
            x + width / 2,
            vram_df["Solve VRAM"],
            width,
            label="Solve VRAM",
            color="#E74C3C",
            edgecolor="black",
            linewidth=1,
        )

        ax5.set_ylabel("VRAM Usage (MB)", fontsize=12, fontweight="bold", labelpad=10)
        ax5.set_title("VRAM Memory Usage (Best Solvers)", fontsize=12, fontweight="bold", pad=20)
        ax5.set_xticks(x)
        ax5.set_xticklabels(vram_df["Scenario"])
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"PETSc GPU Benchmark Summary ({mpi_procs} MPI process(es))\n{problem_size}×{problem_size} Grid - Convection-Diffusion",
        fontsize=16,
        fontweight="bold",
    )

    filename = os.path.join(output_dir, f"benchmark_SUMMARY_{mpi_procs}_procs.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"SUMMARY saved: {filename}")
    plt.close()


def plot_results(df, output_dir, mpi_procs, problem_size):
    if df is None or df.empty:
        print("No results to plot.")
        return

    print(f"\n--- Generating Plots for {mpi_procs} process(es) ---")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    create_summary_plot(df, output_dir, mpi_procs, problem_size)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    scenarios = df["Scenario"].unique()

    for idx, scenario in enumerate(scenarios):
        if idx >= 4:
            break
        ax = axes[idx // 2, idx % 2]
        subset = df[df["Scenario"] == scenario].nsmallest(10, "Solve Time (s)")

        if len(subset) > 0:
            y_pos = np.arange(len(subset))
            ax.barh(y_pos, subset["Solve Time (s)"], color="#4ECDC4", edgecolor="black")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(subset["Solver"], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Solve Time (seconds)", fontsize=11, fontweight="bold")
            ax.set_title(f"{scenario}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        f"Top 10 Solvers - {mpi_procs} MPI procs",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    filename = os.path.join(output_dir, f"benchmark_top10_{mpi_procs}_procs.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


def print_summary(df):
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    scenarios = ["1. All GPU", "2. A,b on GPU", "3. A on GPU", "4. All CPU"]

    print("BEST SOLVER FOR EACH SCENARIO:")
    print(f"{'─'*80}")

    for scenario in scenarios:
        subset = df[df["Scenario"] == scenario]
        if len(subset) > 0:
            best = subset.nsmallest(1, "Solve Time (s)").iloc[0]
            print(f"\n{scenario}")
            print(f"  Solver:       {best['Solver']}")
            print(f"  Setup Time:   {best['Setup Time (s)']:.4f} s")
            print(f"  Solve Time:   {best['Solve Time (s)']:.4f} s")
            print(f"  Total Time:   {best['Total Time (s)']:.4f} s")
            print(f"  Iterations:   {best['Iterations']}")

    cpu_subset = df[df["Scenario"] == "4. All CPU"]
    if len(cpu_subset) > 0:
        cpu_best_time = cpu_subset["Solve Time (s)"].min()

        print(f"\n{'='*80}")
        print("SPEEDUP ANALYSIS (vs Best CPU Solver)")
        print(f"{'='*80}")

        for scenario in scenarios[:3]:
            subset = df[df["Scenario"] == scenario]
            if len(subset) > 0:
                gpu_best_time = subset["Solve Time (s)"].min()
                speedup = cpu_best_time / gpu_best_time
                print(f"{scenario:20s}: {speedup:6.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PETSc GPU Benchmark")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save result plots"
    )
    parser.add_argument("--mpi-procs", type=int, required=True, help="Number of MPI processes")
    parser.add_argument("--problem-size", type=int, default=32, help="Grid size")
    args, unknown = parser.parse_known_args()

    rank = PETSc.COMM_WORLD.Get_rank()

    results_df = run_benchmark(args.problem_size)

    if rank == 0:
        if results_df is not None and not results_df.empty:
            print("\n--- Full Results ---")
            pd.set_option("display.max_rows", None)
            print(results_df.to_string())

            csv_file = os.path.join(args.output_dir, f"results_{args.mpi_procs}_procs.csv")
            results_df.to_csv(csv_file, index=False)
            print(f"\nResults saved: {csv_file}")

            print_summary(results_df)
            plot_results(results_df, args.output_dir, args.mpi_procs, args.problem_size)
        else:
            print("\nWARNING: No successful benchmark runs completed!")

    if HAS_NVML:
        pynvml.nvmlShutdown()
