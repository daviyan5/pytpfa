#!/usr/bin/env python3
"""
vram_incremental_test.py - Real incremental GPU allocation test
Simulates TPFA solver allocation including KSP/PC setup
"""

import sys
import os
import subprocess
import time
import argparse
import json
import gc
import numpy as np
from typing import Tuple, List, Dict

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_vram_mb(gpu_id: int = 0) -> Tuple[float, float, float]:
    """Return (used, total, free) in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return float(parts[0]), float(parts[1]), float(parts[2])
    except:
        pass
    return 0.0, 0.0, 0.0


def build_7point_coo_vectorized(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build COO arrays for 7-point stencil using pure NumPy vectorization.
    Returns (rows, cols, vals) - NO FOR LOOPS.
    """
    n = nx * ny * nz
    
    cells = np.arange(n, dtype=np.int32)
    
    i_idx = cells % nx
    j_idx = (cells // nx) % ny
    k_idx = cells // (nx * ny)
    
    # Diagonal (all cells)
    diag_rows = cells
    diag_cols = cells
    diag_vals = 6.0 + np.random.rand(n) * 0.1
    
    # -X neighbors (i > 0)
    mask_xm = i_idx > 0
    xm_rows = cells[mask_xm]
    xm_cols = xm_rows - 1
    xm_vals = -1.0 - np.random.rand(mask_xm.sum()) * 0.01
    
    # +X neighbors (i < nx-1)
    mask_xp = i_idx < nx - 1
    xp_rows = cells[mask_xp]
    xp_cols = xp_rows + 1
    xp_vals = -1.0 - np.random.rand(mask_xp.sum()) * 0.01
    
    # -Y neighbors (j > 0)
    mask_ym = j_idx > 0
    ym_rows = cells[mask_ym]
    ym_cols = ym_rows - nx
    ym_vals = -1.0 - np.random.rand(mask_ym.sum()) * 0.01
    
    # +Y neighbors (j < ny-1)
    mask_yp = j_idx < ny - 1
    yp_rows = cells[mask_yp]
    yp_cols = yp_rows + nx
    yp_vals = -1.0 - np.random.rand(mask_yp.sum()) * 0.01
    
    # -Z neighbors (k > 0)
    stride_z = nx * ny
    mask_zm = k_idx > 0
    zm_rows = cells[mask_zm]
    zm_cols = zm_rows - stride_z
    zm_vals = -1.0 - np.random.rand(mask_zm.sum()) * 0.01
    
    # +Z neighbors (k < nz-1)
    mask_zp = k_idx < nz - 1
    zp_rows = cells[mask_zp]
    zp_cols = zp_rows + stride_z
    zp_vals = -1.0 - np.random.rand(mask_zp.sum()) * 0.01
    
    rows = np.concatenate([diag_rows, xm_rows, xp_rows, ym_rows, yp_rows, zm_rows, zp_rows])
    cols = np.concatenate([diag_cols, xm_cols, xp_cols, ym_cols, yp_cols, zm_cols, zp_cols])
    vals = np.concatenate([diag_vals, xm_vals, xp_vals, ym_vals, yp_vals, zm_vals, zp_vals])
    
    return rows.astype(np.int32), cols.astype(np.int32), vals.astype(np.float64)


def create_and_solve(nx: int, ny: int, nz: int, pc_type: str = "gamg") -> Dict:
    """
    Create matrix, vectors, KSP solver with PC, and do one solve.
    This triggers full GPU allocation including preconditioner structures.
    """
    from petsc4py import PETSc
    
    n = nx * ny * nz
    
    result = {
        "n": n,
        "nx": nx, "ny": ny, "nz": nz,
        "nnz": 0,
        "ksp_iterations": 0,
    }
    
    # Build COO (vectorized)
    t0 = time.time()
    rows, cols, vals = build_7point_coo_vectorized(nx, ny, nz)
    result["time_coo"] = time.time() - t0
    result["nnz"] = len(rows)
    
    # Create matrix
    t0 = time.time()
    A = PETSc.Mat().create(PETSc.COMM_SELF)
    A.setSizes([n, n])
    A.setFromOptions()
    A.setPreallocationCOO(rows, cols)
    A.setValuesCOO(vals, PETSc.InsertMode.INSERT_VALUES)
    result["time_matrix"] = time.time() - t0
    
    # Create vectors
    x = A.createVecRight()
    b = A.createVecLeft()
    x.set(0.0)
    b.setRandom()
    
    # Create KSP with preconditioner
    t0 = time.time()
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setType(PETSc.KSP.Type.FGMRES)
    
    pc = ksp.getPC()
    if pc_type == "gamg":
        pc.setType(PETSc.PC.Type.GAMG)
        pc.setGAMGType("agg")
    elif pc_type == "ilu":
        pc.setType(PETSc.PC.Type.ILU)
    elif pc_type == "jacobi":
        pc.setType(PETSc.PC.Type.JACOBI)
    elif pc_type == "bjacobi":
        pc.setType(PETSc.PC.Type.BJACOBI)
    else:
        pc.setType(PETSc.PC.Type.NONE)
    
    ksp.setOperators(A)
    ksp.setTolerances(rtol=1e-6, max_it=100)
    ksp.setFromOptions()
    result["time_ksp_setup"] = time.time() - t0
    
    # Solve
    t0 = time.time()
    ksp.solve(b, x)
    result["time_solve"] = time.time() - t0
    result["ksp_iterations"] = ksp.getIterationNumber()
    result["ksp_converged"] = ksp.getConvergedReason() 
    result["_refs"] = (A, x, b, ksp)
    
    return result


def measure_allocation(nx: int, ny: int, nz: int, gpu_id: int, pc_type: str) -> Dict:
    """Measure real GPU allocation for given mesh size."""
    n = nx * ny * nz
    
    result = {
        "nx": nx, "ny": ny, "nz": nz,
        "n": n,
        "success": False,
        "vram_before_mb": 0,
        "vram_after_mb": 0,
        "vram_used_mb": 0,
        "error": None,
    }
    
    gc.collect()
    time.sleep(0.5)
    
    used_before, total, _ = get_vram_mb(gpu_id)
    result["vram_before_mb"] = used_before
    result["vram_total_mb"] = total
    
    try:
        solver_result = create_and_solve(nx, ny, nz, pc_type)
        
        # Force GPU sync
        from petsc4py import PETSc
        PETSc.COMM_SELF.barrier()
        time.sleep(0.5)
        
        used_after, _, _ = get_vram_mb(gpu_id)
        result["vram_after_mb"] = used_after
        result["vram_used_mb"] = used_after - used_before
        result["success"] = True
        
        result["nnz"] = solver_result["nnz"]
        result["ksp_iterations"] = solver_result["ksp_iterations"]
        result["time_coo"] = solver_result["time_coo"]
        result["time_matrix"] = solver_result["time_matrix"]
        result["time_ksp_setup"] = solver_result["time_ksp_setup"]
        result["time_solve"] = solver_result["time_solve"]
        result["time_total"] = sum([
            solver_result["time_coo"],
            solver_result["time_matrix"],
            solver_result["time_ksp_setup"],
            solver_result["time_solve"]
        ])
        
        # Cleanup
        A, x, b, ksp = solver_result["_refs"]
        ksp.destroy()
        x.destroy()
        b.destroy()
        A.destroy()
        
        gc.collect()
        time.sleep(0.3)
        
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


def run_incremental_test(
    gpu_id: int = 0,
    start_side: int = 50,
    step_factor: float = 1.2,
    max_vram_percent: float = 90.0,
    max_iterations: int = 25,
    pc_type: str = "gamg"
) -> List[Dict]:
    """Run incremental tests until limit is reached."""
    results = []
    current_side = start_side
    last_success_side = 0
    
    _, total_vram, _ = get_vram_mb(gpu_id)
    max_vram = total_vram * (max_vram_percent / 100)
    
    print(f"\n{'Mesh':>12} {'N':>12} {'VRAM':>10} {'VRAM%':>8} {'B/Elem':>8} {'KSP':>6} {'Time':>8} {'Status':>8}")
    print("-" * 85)
    
    for i in range(max_iterations):
        side = int(current_side)
        nx, ny, nz = side, side, side
        n = nx * ny * nz
        
        result = measure_allocation(nx, ny, nz, gpu_id, pc_type)
        results.append(result)
        
        if result["success"]:
            vram_used = result["vram_used_mb"]
            vram_percent = (result["vram_after_mb"] / total_vram) * 100
            bytes_per_elem = (vram_used * 1024 * 1024) / n if n > 0 else 0
            ksp_its = result.get("ksp_iterations", 0)
            time_total = result.get("time_total", 0)
            
            status = f"{Colors.GREEN}OK{Colors.END}"
            if vram_percent > 80:
                status = f"{Colors.YELLOW}HIGH{Colors.END}"
            
            print(f"{side:>8}^3 {n:>12,} {vram_used:>8.0f}MB {vram_percent:>7.1f}% {bytes_per_elem:>6.0f}B {ksp_its:>6} {time_total:>6.1f}s {status}")
            
            last_success_side = side
            
            if result["vram_after_mb"] >= max_vram:
                print(f"\n{Colors.YELLOW}Reached VRAM limit ({max_vram_percent}%){Colors.END}")
                break
            
            current_side = side * step_factor
            
        else:
            print(f"{side:>8}^3 {n:>12,} {'---':>10} {'---':>8} {'---':>8} {'---':>6} {'---':>8} {Colors.RED}FAIL{Colors.END}")
            if result.get('error'):
                print(f"    Error: {result['error'][:100]}")
            
            if last_success_side > 0:
                current_side = (last_success_side + side) / 2
                if current_side <= last_success_side * 1.05:
                    print(f"\n{Colors.GREEN}Found limit: {last_success_side}^3 = {last_success_side**3:,}{Colors.END}")
                    break
            else:
                current_side = side * 0.7
                if current_side < 20:
                    print(f"\n{Colors.RED}Failed even with small mesh{Colors.END}")
                    break
    
    return results


def analyze_results(results: List[Dict], gpu_id: int = 0) -> Dict:
    """Analyze test results."""
    successful = [r for r in results if r["success"] and r["vram_used_mb"] > 10]
    
    if not successful:
        return {"error": "No successful tests"}
    
    ns = np.array([r["n"] for r in successful])
    vrams = np.array([r["vram_used_mb"] for r in successful])
    
    bytes_per_elem = (vrams * 1024 * 1024) / ns
    
    slope, intercept = np.polyfit(ns / 1e6, vrams, 1)
    
    _, total_vram, _ = get_vram_mb(gpu_id)
    
    target_vram = total_vram * 0.9
    estimated_max_n = int((target_vram - intercept) / slope * 1e6) if slope > 0 else 0
    
    return {
        "total_tests": len(results),
        "successful_tests": len(successful),
        "max_n_tested": int(max(ns)),
        "max_side_tested": int(round(max(ns) ** (1/3))),
        "max_vram_used_mb": float(max(vrams)),
        "bytes_per_element": {
            "mean": float(np.mean(bytes_per_elem)),
            "std": float(np.std(bytes_per_elem)),
            "min": float(np.min(bytes_per_elem)),
            "max": float(np.max(bytes_per_elem)),
        },
        "linear_fit": {
            "slope_mb_per_million": float(slope),
            "intercept_mb": float(intercept),
        },
        "estimated_max_n_90_percent": estimated_max_n,
        "estimated_max_side_90_percent": int(round(estimated_max_n ** (1/3))),
        "vram_total_mb": total_vram,
    }


def print_report(analysis: Dict, pc_type: str):
    """Print final report."""
    print(f"\n{'='*70}")
    print(f"{'FINAL REPORT':^70}")
    print(f"{'='*70}")
    
    print(f"\n{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  Preconditioner: {pc_type.upper()}")
    
    print(f"\n{Colors.BOLD}Tests:{Colors.END}")
    print(f"  Successful: {analysis['successful_tests']}/{analysis['total_tests']}")
    print(f"  Max tested: {analysis['max_side_tested']}^3 = {analysis['max_n_tested']:,}")
    print(f"  Max VRAM: {analysis['max_vram_used_mb']:.0f} MB")
    
    bpe = analysis['bytes_per_element']
    print(f"\n{Colors.BOLD}Bytes per Element:{Colors.END}")
    print(f"  Mean: {bpe['mean']:.0f} +/- {bpe['std']:.0f}")
    print(f"  Range: [{bpe['min']:.0f}, {bpe['max']:.0f}]")
    
    lf = analysis['linear_fit']
    print(f"\n{Colors.BOLD}Linear Model:{Colors.END}")
    print(f"  VRAM (MB) = {lf['slope_mb_per_million']:.1f} * (N / 1M) + {lf['intercept_mb']:.0f}")
    
    print(f"\n{Colors.BOLD}Estimated Max (90% VRAM):{Colors.END}")
    print(f"  {Colors.GREEN}N = {analysis['estimated_max_n_90_percent']:,}{Colors.END}")
    print(f"  {Colors.GREEN}Mesh = {analysis['estimated_max_side_90_percent']}^3{Colors.END}")
    
    slope = lf['slope_mb_per_million']
    intercept = lf['intercept_mb']
    
    print(f"\n{Colors.BOLD}Estimated max by GPU size:{Colors.END}")
    print(f"  {'VRAM':>8} {'N max':>15} {'Mesh':>12}")
    print("  " + "-" * 40)
    
    for vram_gb in [8, 12, 16, 24, 32, 40, 48, 80]:
        vram_mb = vram_gb * 1024
        target = vram_mb * 0.9
        max_n = int((target - intercept) / slope * 1e6) if slope > 0 else 0
        if max_n > 0:
            side = int(round(max_n ** (1/3)))
            print(f"  {vram_gb:>6} GB {max_n:>15,} {side:>8}^3")
        else:
            print(f"  {vram_gb:>6} GB {'N/A':>15} {'N/A':>12}")


def main():
    parser = argparse.ArgumentParser(description="VRAM allocation test with KSP solve")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    # Updated default to start from 1 million (100^3)
    parser.add_argument("--start-side", type=int, default=100, help="Starting mesh side")
    # Updated default to 1.1 (10% steps) to avoid crashing too early
    parser.add_argument("--step-factor", type=float, default=1.1, help="Side increment factor")
    parser.add_argument("--max-vram-percent", type=float, default=90, help="Max VRAM (%%)")
    parser.add_argument("--pc-type", type=str, default="gamg", 
                       choices=["gamg", "ilu", "jacobi", "bjacobi", "none"],
                       help="Preconditioner type")
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print(f"Initializing PETSc with GPU...")
    
    import petsc4py
    petsc4py.init([
        'vram_test',
        '-vec_type', 'cuda',
        '-mat_type', 'aijcusparse',
    ])
    
    from petsc4py import PETSc
    
    used, total, free = get_vram_mb(args.gpu)
    print(f"\n{Colors.BOLD}GPU {args.gpu}:{Colors.END}")
    print(f"  VRAM Total: {total:,.0f} MB ({total/1024:.1f} GB)")
    print(f"  VRAM Used: {used:,.0f} MB ({used/total*100:.1f}%)")
    print(f"  VRAM Free: {free:,.0f} MB")
    
    print(f"\n{Colors.CYAN}Test Configuration:{Colors.END}")
    print(f"  Start side: {args.start_side}")
    print(f"  Step factor: {args.step_factor}")
    print(f"  Max VRAM: {args.max_vram_percent}%")
    print(f"  Preconditioner: {args.pc_type.upper()}")
    
    results = run_incremental_test(
        gpu_id=args.gpu,
        start_side=args.start_side,
        step_factor=args.step_factor,
        max_vram_percent=args.max_vram_percent,
        pc_type=args.pc_type
    )
    
    analysis = analyze_results(results, args.gpu)
    
    if "error" not in analysis:
        print_report(analysis, args.pc_type)
    else:
        print(f"\n{Colors.RED}Analysis failed: {analysis['error']}{Colors.END}")
    
    output = {
        "gpu_id": args.gpu,
        "parameters": {
            "start_side": args.start_side,
            "step_factor": args.step_factor,
            "max_vram_percent": args.max_vram_percent,
            "pc_type": args.pc_type,
        },
        "analysis": analysis,
        "raw_results": results,
    }
    
    output_file = f"vram_test_gpu{args.gpu}_{args.pc_type}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{Colors.GREEN}Results saved to: {output_file}{Colors.END}")


if __name__ == "__main__":
    main()