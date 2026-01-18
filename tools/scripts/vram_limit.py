#!/usr/bin/env python3
"""
vram_limit_finder.py - Find GPU VRAM allocation limit for TPFA solver
"""

import sys
import subprocess
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict


class Colors:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


@dataclass
class VRAMInfo:
    gpu_id: int
    used_mb: float
    total_mb: float
    free_mb: float
    
    @property
    def used_percent(self) -> float:
        return (self.used_mb / self.total_mb) * 100 if self.total_mb > 0 else 0


def get_vram_info(gpu_id: int = 0) -> Optional[VRAMInfo]:
    """Get VRAM info from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        
        parts = result.stdout.strip().split(',')
        if len(parts) >= 4:
            return VRAMInfo(
                gpu_id=int(parts[0].strip()),
                used_mb=float(parts[1].strip()),
                total_mb=float(parts[2].strip()),
                free_mb=float(parts[3].strip())
            )
    except Exception as e:
        print(f"{Colors.RED}Error getting VRAM: {e}{Colors.END}")
    return None


def get_all_gpus_info() -> List[VRAMInfo]:
    """Get info from all GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 4:
                gpus.append(VRAMInfo(
                    gpu_id=int(parts[0].strip()),
                    used_mb=float(parts[1].strip()),
                    total_mb=float(parts[2].strip()),
                    free_mb=float(parts[3].strip())
                ))
        return gpus
    except Exception:
        return []


def estimate_memory_usage(n_elements: int, nnz_per_row: int = 7) -> Dict[str, float]:
    """
    Estimate memory usage for N elements in TPFA pattern.
    
    Structures:
    - CSR Matrix: values (float64), col_indices (int32), row_ptr (int32)
    - Vectors x, b: float64
    """
    nnz_total = int(n_elements * nnz_per_row * 0.93)
    
    values_bytes = nnz_total * 8
    col_indices_bytes = nnz_total * 4
    row_ptr_bytes = (n_elements + 1) * 4
    
    vec_x_bytes = n_elements * 8
    vec_b_bytes = n_elements * 8
    
    petsc_overhead = 0.20
    
    matrix_mb = (values_bytes + col_indices_bytes + row_ptr_bytes) / (1024**2)
    vectors_mb = (vec_x_bytes + vec_b_bytes) / (1024**2)
    
    subtotal = matrix_mb + vectors_mb
    overhead_mb = subtotal * petsc_overhead
    
    return {
        "matrix_mb": matrix_mb,
        "vectors_mb": vectors_mb,
        "overhead_mb": overhead_mb,
        "total_mb": subtotal + overhead_mb,
        "nnz_total": nnz_total,
    }


def n_from_mesh(nx: int, ny: int, nz: int) -> int:
    return nx * ny * nz


def mesh_from_n(n: int) -> Tuple[int, int, int]:
    side = int(round(n ** (1/3)))
    return (side, side, side)


def binary_search_max_n(
    gpu_id: int,
    max_vram_percent: float,
    min_n: int = 100_000,
    max_n: int = 500_000_000,
) -> Tuple[int, List[Dict]]:
    """Binary search to find maximum N that fits in VRAM."""
    vram_info = get_vram_info(gpu_id)
    if not vram_info:
        print(f"{Colors.RED}Error: Could not get GPU {gpu_id} info{Colors.END}")
        return 0, []
    
    target_vram_mb = vram_info.total_mb * (max_vram_percent / 100)
    
    print(f"\n{Colors.CYAN}GPU {gpu_id}: {vram_info.total_mb:.0f} MB total, target: {target_vram_mb:.0f} MB ({max_vram_percent}%){Colors.END}\n")
    
    history = []
    low, high = min_n, max_n
    best_n = min_n
    
    while low <= high:
        mid = (low + high) // 2
        est = estimate_memory_usage(mid)
        
        entry = {
            "n": mid,
            "estimated_mb": est["total_mb"],
            "matrix_mb": est["matrix_mb"],
            "vectors_mb": est["vectors_mb"],
            "fits": est["total_mb"] <= target_vram_mb
        }
        history.append(entry)
        
        if est["total_mb"] <= target_vram_mb:
            best_n = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(f"  Initial estimate: N_max = {best_n:,}")
    
    return best_n, history


def analyze_vram_vs_n(history: List[Dict]) -> Dict:
    """Analyze relationship between VRAM and N."""
    if not history:
        return {}
    
    valid = [h for h in history if h.get("estimated_mb") or h.get("vram_used_mb")]
    
    if len(valid) < 2:
        return {}
    
    ns = np.array([h["n"] for h in valid])
    vrams = np.array([h.get("estimated_mb", h.get("vram_used_mb", 0)) for h in valid])
    
    bytes_per_element = (vrams * 1024**2) / ns
    
    return {
        "avg_bytes_per_element": float(np.mean(bytes_per_element)),
        "std_bytes_per_element": float(np.std(bytes_per_element)),
        "min_bytes_per_element": float(np.min(bytes_per_element)),
        "max_bytes_per_element": float(np.max(bytes_per_element)),
        "linear_coefficient_mb_per_million": float(np.mean(vrams / (ns / 1e6))),
    }


def print_analysis_report(
    gpu_id: int,
    max_n: int,
    history: List[Dict],
    analysis: Dict,
    vram_info: VRAMInfo
):
    """Print detailed analysis report."""
    
    print(f"\n{'='*70}")
    print(f"{'VRAM ANALYSIS REPORT':^70}")
    print(f"{'='*70}")
    
    print(f"\n{Colors.BOLD}GPU {gpu_id}:{Colors.END}")
    print(f"  VRAM Total: {vram_info.total_mb:,.0f} MB ({vram_info.total_mb/1024:.1f} GB)")
    print(f"  VRAM Free: {vram_info.free_mb:,.0f} MB")
    
    print(f"\n{Colors.BOLD}Result:{Colors.END}")
    print(f"  {Colors.GREEN}Max N: {max_n:,} elements{Colors.END}")
    
    mesh = mesh_from_n(max_n)
    print(f"  Equivalent mesh: ~{mesh[0]}x{mesh[1]}x{mesh[2]}")
    
    est = estimate_memory_usage(max_n)
    print(f"\n{Colors.BOLD}Memory Breakdown (N={max_n:,}):{Colors.END}")
    print(f"  Matrix (CSR):     {est['matrix_mb']:>10,.1f} MB")
    print(f"  Vectors (x, b):   {est['vectors_mb']:>10,.1f} MB")
    print(f"  PETSc Overhead:   {est['overhead_mb']:>10,.1f} MB")
    print(f"  {Colors.BOLD}Total Estimated:  {est['total_mb']:>10,.1f} MB{Colors.END}")
    
    if analysis:
        print(f"\n{Colors.BOLD}VRAM vs N Relationship:{Colors.END}")
        print(f"  Bytes per element: {analysis['avg_bytes_per_element']:.1f} +/- {analysis['std_bytes_per_element']:.1f}")
        print(f"  MB per million elements: {analysis['linear_coefficient_mb_per_million']:.1f}")
    
    print(f"\n{Colors.BOLD}Reference Table:{Colors.END}")
    print(f"  {'Mesh':<20} {'N':>15} {'VRAM Est.':>12} {'Fits?':>8}")
    print("  " + "-" * 60)
    
    reference_meshes = [
        (50, 50, 50),
        (100, 100, 100),
        (150, 150, 150),
        (200, 200, 200),
        (250, 250, 250),
        (300, 300, 300),
        (200, 200, 100),
        (200, 100, 50),
    ]
    
    for mesh in reference_meshes:
        n = n_from_mesh(*mesh)
        est = estimate_memory_usage(n)
        fits = est["total_mb"] <= vram_info.total_mb * 0.9
        fits_str = f"{Colors.GREEN}Y{Colors.END}" if fits else f"{Colors.RED}N{Colors.END}"
        print(f"  {mesh[0]}x{mesh[1]}x{mesh[2]:<10} {n:>15,} {est['total_mb']:>10.1f} MB {fits_str:>8}")
    
    if analysis:
        coef = analysis['linear_coefficient_mb_per_million']
        print(f"\n{Colors.BOLD}Approximate Formula:{Colors.END}")
        print(f"  VRAM (MB) = {coef:.2f} * (N / 1,000,000)")
        print(f"  N_max = VRAM_available (MB) / {coef:.2f} * 1,000,000")
        
        print(f"\n{Colors.BOLD}Max N for different GPUs:{Colors.END}")
        vram_sizes = [8, 12, 16, 24, 32, 40, 48, 80]
        for vram_gb in vram_sizes:
            vram_mb = vram_gb * 1024
            max_n_est = int((vram_mb * 0.9) / coef * 1e6)
            mesh_est = mesh_from_n(max_n_est)
            print(f"  {vram_gb:>2} GB: N_max = {max_n_est:>12,} (~{mesh_est[0]}^3)")


def main():
    parser = argparse.ArgumentParser(description="Find GPU VRAM limit for TPFA solver")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument("--max-vram-percent", type=float, default=90,
                       help="Max VRAM percentage to use (default: 90)")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"{'VRAM LIMIT FINDER - TPFA Solver':^70}")
    print(f"{'='*70}")
    
    gpus = get_all_gpus_info()
    if not gpus:
        print(f"{Colors.RED}No NVIDIA GPU found!{Colors.END}")
        sys.exit(1)
    
    print(f"\n{Colors.BOLD}Available GPUs:{Colors.END}")
    for gpu in gpus:
        print(f"  GPU {gpu.gpu_id}: {gpu.total_mb:,.0f} MB ({gpu.total_mb/1024:.1f} GB), "
              f"used: {gpu.used_mb:,.0f} MB ({gpu.used_percent:.1f}%)")
    
    if args.gpu >= len(gpus):
        print(f"{Colors.RED}GPU {args.gpu} does not exist!{Colors.END}")
        sys.exit(1)
    
    vram_info = gpus[args.gpu]
    
    max_n, history = binary_search_max_n(
        gpu_id=args.gpu,
        max_vram_percent=args.max_vram_percent,
    )
    
    analysis = analyze_vram_vs_n(history)
    print_analysis_report(args.gpu, max_n, history, analysis, vram_info)
    
    import json
    output = {
        "gpu_id": args.gpu,
        "vram_total_mb": vram_info.total_mb,
        "max_vram_percent": args.max_vram_percent,
        "max_n": max_n,
        "analysis": analysis,
        "history": history[-20:] if len(history) > 20 else history,
    }
    
    output_file = f"vram_analysis_gpu{args.gpu}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{Colors.GREEN}Results saved to: {output_file}{Colors.END}")


if __name__ == "__main__":
    main()