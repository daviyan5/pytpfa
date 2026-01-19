#!/usr/bin/env python3
"""
vram_test_incremental.py - Fixed Distributed VRAM Test
Uses mpi4py for control logic and PETSc for solvers.
"""

import sys
import os
import subprocess
import time
import argparse
import json
import gc
import numpy as np
from typing import Tuple, Dict

def get_rank_size_env():
    rank = os.environ.get('OMPI_COMM_WORLD_RANK') or \
           os.environ.get('PMI_RANK') or \
           os.environ.get('MV2_COMM_WORLD_RANK') or \
           os.environ.get('SLURM_PROCID')
    size = os.environ.get('OMPI_COMM_WORLD_SIZE') or \
           os.environ.get('PMI_SIZE') or \
           os.environ.get('MV2_COMM_WORLD_SIZE') or \
           os.environ.get('SLURM_NTASKS')
    if rank is not None:
        return int(rank), int(size)
    return 0, 1

RANK, SIZE = get_rank_size_env()

try:
    ngpu_out = subprocess.check_output("nvidia-smi -L | wc -l", shell=True)
    NUM_GPUS = int(ngpu_out.strip())
except:
    NUM_GPUS = 1

PHYSICAL_GPU_ID = RANK % NUM_GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = str(PHYSICAL_GPU_ID)

import petsc4py
petsc4py.init([
    'vram_test', 
    '-vec_type', 'cuda', 
    '-mat_type', 'aijcusparse', 
    '-options_left', '0'
])
from petsc4py import PETSc
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
petsc_comm = PETSc.COMM_WORLD

assert RANK == mpi_comm.Get_rank()

sys.stdout.reconfigure(line_buffering=True)
def log(msg):
    if RANK == 0:
        print(msg, flush=True)

def get_vram_mb(gpu_id: int) -> Tuple[float, float, float]:
    try:
        env = os.environ.copy()
        if 'CUDA_VISIBLE_DEVICES' in env:
            del env['CUDA_VISIBLE_DEVICES']
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5, env=env
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return float(parts[0]), float(parts[1]), float(parts[2])
    except:
        pass
    return 0.0, 0.0, 0.0

def create_distributed_matrix(nx: int, ny: int, nz: int, comm):
    n_global = nx * ny * nz
    
    A = PETSc.Mat().create(comm)
    A.setSizes([n_global, n_global])
    A.setFromOptions()
    
    rstart, rend = A.getOwnershipRange()
    local_size = rend - rstart
    local_rows = np.arange(rstart, rend, dtype=np.int32)
    
    i = local_rows % nx
    j = (local_rows // nx) % ny
    k = local_rows // (nx * ny)
    
    all_rows = [local_rows]
    all_cols = [local_rows]
    all_vals = [np.full(local_size, 6.0 + np.random.rand() * 0.1)]
    
    def add_neighbor(mask, offset):
        rows = local_rows[mask]
        cols = rows + offset
        vals = np.full(len(rows), -1.0 - np.random.rand() * 0.01)
        all_rows.append(rows)
        all_cols.append(cols)
        all_vals.append(vals)

    add_neighbor(i > 0, -1)
    add_neighbor(i < nx-1, 1)
    add_neighbor(j > 0, -nx)
    add_neighbor(j < ny-1, nx)
    add_neighbor(k > 0, -(nx*ny))
    add_neighbor(k < nz-1, (nx*ny))
    
    rows = np.concatenate(all_rows).astype(np.int32)
    cols = np.concatenate(all_cols).astype(np.int32)
    vals = np.concatenate(all_vals).astype(np.float64)
    
    A.setPreallocationCOO(rows, cols)
    A.setValuesCOO(vals, PETSc.InsertMode.INSERT_VALUES)
    
    return A, local_size, len(rows)

def test_distributed_mesh(side: int, physical_gpu_id: int, pc_type: str) -> Dict:
    n_total = side ** 3
    
    result = {
        "side": side, 
        "n_total": n_total, 
        "gpu": physical_gpu_id, 
        "success": False, 
        "error": None
    }
    
    A = x = b = ksp = None
    
    try:
        gc.collect()
        mpi_comm.Barrier()
        
        vram_before, vram_total, _ = get_vram_mb(physical_gpu_id)
        result["vram_before_mb"] = vram_before
        result["vram_total_mb"] = vram_total
        
        mpi_comm.Barrier()
        t0 = time.time()
        A, local_size, local_nnz = create_distributed_matrix(side, side, side, petsc_comm)
        
        A.assemblyBegin()
        A.assemblyEnd()
        t_create = time.time() - t0
        
        result["local_rows"] = local_size
        result["local_nnz"] = local_nnz
        result["time_create"] = t_create
        
        x = A.createVecRight()
        b = A.createVecLeft()
        x.set(0.0)
        b.setRandom()
        
        mpi_comm.Barrier()
        
        vram_after, _, _ = get_vram_mb(physical_gpu_id)
        result["vram_after_load_mb"] = vram_after
        result["vram_used_mb"] = max(0, vram_after - vram_before)
        
        ksp = PETSc.KSP().create(petsc_comm)
        ksp.setType(PETSc.KSP.Type.CG)
        pc = ksp.getPC()
        pc_map = {
            "gamg": PETSc.PC.Type.GAMG, 
            "ilu": PETSc.PC.Type.ILU,
            "jacobi": PETSc.PC.Type.JACOBI, 
            "bjacobi": PETSc.PC.Type.BJACOBI,
            "none": PETSc.PC.Type.NONE
        }
        pc.setType(pc_map.get(pc_type, PETSc.PC.Type.BJACOBI))
        ksp.setOperators(A)
        ksp.setTolerances(rtol=1e-6, max_it=100)
        ksp.setFromOptions()
        
        mpi_comm.Barrier()
        t0 = time.time()
        ksp.solve(b, x)
        
        try:
            PETSc.Device(0).synchronize()
        except:
            pass
            
        t_solve = time.time() - t0
        
        result["time_solve"] = t_solve
        result["ksp_iterations"] = ksp.getIterationNumber()
        result["time_total"] = t_create + t_solve
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    finally:
        if ksp: ksp.destroy()
        if x: x.destroy()
        if b: b.destroy()
        if A: A.destroy()
        gc.collect()
        mpi_comm.Barrier()
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-side", type=int, default=200)
    parser.add_argument("--step-factor", type=float, default=1.1)
    parser.add_argument("--max-vram-percent", type=float, default=90)
    parser.add_argument("--pc-type", type=str, default="bjacobi", 
                        choices=["gamg", "ilu", "jacobi", "bjacobi", "none"])
    args = parser.parse_args()

    mpi_comm.Barrier()
    
    if RANK == 0:
        _, total_vram, _ = get_vram_mb(0)
        log(f"\n{'='*110}")
        log(f" DISTRIBUTED VRAM TEST - Problem SPLIT across {SIZE} GPUs ({total_vram/1024:.0f}GB each)")
        log(f" Rank to GPU Map: {[f'R{i}->G{i%NUM_GPUS}' for i in range(SIZE)]}")
        log(f"{'='*110}")
        log(f" PC: {args.pc_type.upper()}, Target: {args.max_vram_percent}%")
        log(f"{'Mesh':>6} {'Total N':>14} {'N/GPU':>12} | {'VRAM (MB) per GPU':^30} | {'Max%':>5} {'Time':>6} {'Its':>3}")
        log("-" * 110)
    
    mpi_comm.Barrier()
    
    current_side = args.start_side
    last_success = 0
    
    for _ in range(50):
        side = int(current_side)
        n_total = side ** 3
        
        result = test_distributed_mesh(side, PHYSICAL_GPU_ID, args.pc_type)
        
        all_results = mpi_comm.gather(result, root=0)
        
        decision = ("continue", side * args.step_factor)
        
        if RANK == 0:
            all_success = all(r["success"] for r in all_results)
            
            if all_success:
                vrams = [r["vram_after_load_mb"] for r in all_results]
                max_vram_used = max(vrams)
                max_pct = max([(v / r["vram_total_mb"] * 100) for v, r in zip(vrams, all_results)])
                avg_time = sum(r["time_total"] for r in all_results) / SIZE
                iters = all_results[0]["ksp_iterations"]
                
                vram_str = " | ".join([f"{v:.0f}" for v in vrams])
                
                log(f"{side:>6}^3 {n_total:>14,} {n_total//SIZE:>12,} | {vram_str:^30} | {max_pct:>5.1f}% {avg_time:>6.2f}s {iters:>3}")
                
                last_success = side
                
                if max_pct >= args.max_vram_percent:
                    decision = ("stop", side)
                else:
                    decision = ("continue", side * args.step_factor)
            else:
                errors = [r["error"] for r in all_results if r["error"]]
                log(f"{side:>6}^3 {n_total:>14,} {n_total//SIZE:>12,} | FAILED: {errors[0][:50] if errors else 'Unknown'}")
                
                if last_success > 0:
                    next_side = (last_success + side) // 2
                    if next_side <= last_success + 2:
                        decision = ("stop", last_success)
                    else:
                        decision = ("continue", next_side)
                else:
                    decision = ("stop", 0)

        decision = mpi_comm.bcast(decision, root=0)
        
        if decision[0] == "stop":
            break
            
        current_side = decision[1]
        mpi_comm.Barrier()

    if RANK == 0 and last_success > 0:
        log(f"\nMax Distributed Mesh: {last_success}^3 = {last_success**3:,} cells")
        log(f"Cells per GPU: ~{(last_success**3)//SIZE:,}")

if __name__ == "__main__":
    main()