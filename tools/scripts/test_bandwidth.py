#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bandwidth_patch.py - Instruções para adicionar métricas de bandwidth ao seu solver

COMO USAR:
1. Adicione as linhas indicadas abaixo ao seu solver.py
2. Execute sua simulação normalmente
3. O JSON de saída terá as métricas de bandwidth

OU

Execute este script passando o JSON de saída para calcular as métricas:
    python bandwidth_patch.py results.json --hardware-bw 672 --num-gpus 4
"""

import json
import argparse
import sys

# =============================================================================
# PATCH PARA solver.py - ADICIONAR ESTAS LINHAS:
# =============================================================================

PATCH_INSTRUCTIONS = """
================================================================================
MODIFICAÇÕES NO solver.py
================================================================================

1. NO MÉTODO __init__ (após linha ~46), adicione:
--------------------------------------------------------------------------------
        self.out_info["ksp_iterations_per_solve"] = []
        self.out_info["bandwidth_metrics"] = {}
--------------------------------------------------------------------------------

2. NO MÉTODO solve_system (após linha 929), SUBSTITUA:
--------------------------------------------------------------------------------
    def solve_system(self):
        \"\"\"
        Solve the linear system Ax = b
        \"\"\"
        logger.info("Solving the system...", extra={"context": f"Solver SOLVE [{self.iteration}]"})
        self.ksp.solve(self.b, self.x)
        
        # NOVO: Coletar iterações do KSP
        ksp_iters = self.ksp.getIterationNumber()
        self.out_info["ksp_iterations_per_solve"].append(ksp_iters)
        
        self.dmstag_manager.update_from_global("pressure", SL.ELEMENT)
--------------------------------------------------------------------------------

3. NO MÉTODO export_info (ANTES de salvar o JSON), adicione:
--------------------------------------------------------------------------------
            # Calcular métricas de bandwidth
            self._calculate_bandwidth_metrics()
--------------------------------------------------------------------------------

4. ADICIONE ESTE NOVO MÉTODO na classe TPFASolver:
--------------------------------------------------------------------------------
    def _calculate_bandwidth_metrics(self):
        \"\"\"
        Calcula métricas de bandwidth baseado na estrutura real da matriz.
        \"\"\"
        if self.A is None:
            return
            
        # Estrutura da matriz
        mat_info = self.A.getInfo()
        n_rows = self.A.getSize()[0]
        nnz = int(mat_info.get('nz_allocated', 0) or mat_info.get('nz_used', 0))
        
        if nnz == 0:
            return
            
        # Iterações do KSP
        total_ksp_iters = sum(self.out_info.get("ksp_iterations_per_solve", [0]))
        n_solves = len(self.out_info.get("ksp_iterations_per_solve", [1]))
        avg_iters_per_solve = total_ksp_iters / n_solves if n_solves > 0 else 0
        
        # Tipo do solver
        ksp_type = self.ksp.getType() if self.ksp else "unknown"
        pc_type = self.ksp.getPC().getType() if self.ksp else "unknown"
        
        # Bytes por SpMV (formato CSR)
        # values: nnz * 8, col_idx: nnz * 4, row_ptr: (n+1) * 4
        # x read: n * 8 (com cache ~50%), y write: n * 8
        bytes_spmv = nnz * 8 + nnz * 4 + (n_rows + 1) * 4 + n_rows * 8 * 1.5 + n_rows * 8
        
        # FLOPs por SpMV: 2 * nnz (mul + add)
        flops_spmv = 2 * nnz
        
        # Operações adicionais por iteração (depende do solver)
        vector_bytes = n_rows * 8
        if ksp_type.lower() in ['cg']:
            # CG: 1 SpMV + 2 dots + 3 AXPYs
            bytes_per_iter = bytes_spmv + 2 * (2 * vector_bytes) + 3 * (3 * vector_bytes)
            flops_per_iter = flops_spmv + 2 * (2 * n_rows) + 3 * (2 * n_rows)
        elif ksp_type.lower() in ['gmres', 'fgmres']:
            # GMRES: estimativa com restart=30
            k_avg = min(avg_iters_per_solve, 30) / 2
            bytes_per_iter = bytes_spmv + (k_avg + 2) * (2 * vector_bytes) + (k_avg + 1) * (3 * vector_bytes)
            flops_per_iter = flops_spmv + (k_avg + 2) * (2 * n_rows) + (k_avg + 1) * (2 * n_rows)
        elif ksp_type.lower() in ['bcgs', 'bicgstab']:
            # BiCGStab: 2 SpMVs + 4 dots + 6 AXPYs
            bytes_per_iter = 2 * bytes_spmv + 4 * (2 * vector_bytes) + 6 * (3 * vector_bytes)
            flops_per_iter = 2 * flops_spmv + 4 * (2 * n_rows) + 6 * (2 * n_rows)
        else:
            bytes_per_iter = bytes_spmv
            flops_per_iter = flops_spmv
        
        # Métricas totais
        total_bytes = bytes_per_iter * total_ksp_iters
        total_flops = flops_per_iter * total_ksp_iters
        
        # Intensidade aritmética
        ai = flops_per_iter / bytes_per_iter if bytes_per_iter > 0 else 0
        
        # Bandwidth efetiva (se tiver tempo de solve)
        solve_time = self.out_info.get("solving_time", 0)
        effective_bw = (total_bytes / 1e9) / solve_time if solve_time > 0 else 0
        effective_gflops = (total_flops / 1e9) / solve_time if solve_time > 0 else 0
        
        self.out_info["bandwidth_metrics"] = {
            "matrix_rows": n_rows,
            "matrix_nnz": nnz,
            "avg_nnz_per_row": nnz / n_rows if n_rows > 0 else 0,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "total_ksp_iterations": total_ksp_iters,
            "avg_iterations_per_solve": avg_iters_per_solve,
            "bytes_per_iteration": bytes_per_iter,
            "flops_per_iteration": flops_per_iter,
            "bytes_per_cell": bytes_per_iter / n_rows if n_rows > 0 else 0,
            "flops_per_cell": flops_per_iter / n_rows if n_rows > 0 else 0,
            "total_bytes_transferred": total_bytes,
            "total_flops": total_flops,
            "arithmetic_intensity": ai,
            "effective_bandwidth_gb_s": effective_bw,
            "effective_gflops": effective_gflops,
        }
--------------------------------------------------------------------------------

================================================================================
"""

def calculate_bandwidth_from_json(json_path: str, hardware_bw_gb_s: float = 672.0, num_gpus: int = 4):
    """
    Calcula métricas de bandwidth a partir do JSON de saída do solver.
    
    Se o JSON já tiver bandwidth_metrics (após aplicar o patch), usa esses valores.
    Caso contrário, faz estimativa baseada no número de elementos e tempo.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_hardware_bw = hardware_bw_gb_s * num_gpus
    
    print("\n" + "="*80)
    print("ANÁLISE DE BANDWIDTH - DADOS DO SOLVER REAL")
    print("="*80)
    
    # Info básica
    n_elements = data.get("n_elements", 0)
    n_iterations = data.get("n_iterations", 1)
    solve_time = data.get("solving_time", 0)
    total_time = data.get("total_time", 0)
    
    print(f"\n--- INFORMAÇÕES DA SIMULAÇÃO ---")
    print(f"Elementos (células): {n_elements:,}")
    print(f"Passos de tempo: {n_iterations}")
    print(f"Tempo de solve: {solve_time:.2f} s")
    print(f"Tempo total: {total_time:.2f} s")
    
    # Se tiver métricas de bandwidth do patch
    bw = data.get("bandwidth_metrics", {})
    
    if bw:
        print(f"\n--- ESTRUTURA DA MATRIZ ---")
        print(f"Linhas: {bw.get('matrix_rows', 'N/A'):,}")
        print(f"Non-zeros: {bw.get('matrix_nnz', 'N/A'):,}")
        print(f"NNZ/linha: {bw.get('avg_nnz_per_row', 'N/A'):.2f}")
        print(f"Solver: {bw.get('ksp_type', 'N/A').upper()} + {bw.get('pc_type', 'N/A').upper()}")
        
        print(f"\n--- MÉTRICAS POR ITERAÇÃO ---")
        print(f"Bytes por iteração: {bw.get('bytes_per_iteration', 0)/1e6:.2f} MB")
        print(f"FLOPs por iteração: {bw.get('flops_per_iteration', 0)/1e6:.2f} MFLOPs")
        print(f"Bytes por célula: {bw.get('bytes_per_cell', 0):.2f}")
        print(f"FLOPs por célula: {bw.get('flops_per_cell', 0):.2f}")
        
        print(f"\n--- TOTAIS ---")
        print(f"Iterações KSP totais: {bw.get('total_ksp_iterations', 'N/A')}")
        print(f"Bytes totais: {bw.get('total_bytes_transferred', 0)/1e9:.2f} GB")
        print(f"FLOPs totais: {bw.get('total_flops', 0)/1e9:.2f} GFLOPs")
        
        ai = bw.get('arithmetic_intensity', 0)
        eff_bw = bw.get('effective_bandwidth_gb_s', 0)
        eff_gflops = bw.get('effective_gflops', 0)
        
        bw_efficiency = (eff_bw / total_hardware_bw) * 100 if total_hardware_bw > 0 else 0
        
        print(f"\n--- BANDWIDTH ---")
        print(f"Bandwidth efetiva: {eff_bw:.2f} GB/s")
        print(f"Bandwidth de pico ({num_gpus} GPUs): {total_hardware_bw:.0f} GB/s")
        print(f"Eficiência: {bw_efficiency:.2f}%")
        print(f"Desempenho: {eff_gflops:.2f} GFLOP/s")
        
        print(f"\n--- CLASSIFICAÇÃO ROOFLINE ---")
        ridge_point = 24.0  # Aproximado para RTX 6000
        classification = "MEMORY-BOUND" if ai < ridge_point else "COMPUTE-BOUND"
        print(f"Intensidade aritmética: {ai:.4f} FLOP/Byte")
        print(f"Ridge point: ~{ridge_point:.1f} FLOP/Byte")
        print(f"Classificação: {classification}")
        
        print(f"\n--- VALORES PARA YAML ---")
        print(f"""
bandwidth:
  analysis:
    bytes_per_cell: {bw.get('bytes_per_cell', 0):.1f}
    flops_per_cell: {bw.get('flops_per_cell', 0):.1f}
    stencil_size: 7
    arithmetic_intensity: {ai:.4f}
    
# Resultados medidos:
# - Bandwidth efetiva: {eff_bw:.2f} GB/s
# - Eficiência: {bw_efficiency:.2f}%
# - Classificação: {classification}
""")
    else:
        print("\n⚠️  JSON não contém bandwidth_metrics.")
        print("    Aplique o patch ao solver.py e execute novamente.")
        print("\n    Ou use estimativa baseada em valores típicos:")
        
        # Estimativa grosseira
        # Assumindo 7-pt stencil, ~7 nnz/row
        est_nnz = n_elements * 7
        est_bytes_spmv = est_nnz * 8 + est_nnz * 4 + (n_elements + 1) * 4 + n_elements * 8 * 2.5
        est_flops_spmv = 2 * est_nnz
        
        # Assumindo CG
        vector_bytes = n_elements * 8
        est_bytes_iter = est_bytes_spmv + 2 * (2 * vector_bytes) + 3 * (3 * vector_bytes)
        est_flops_iter = est_flops_spmv + 10 * n_elements
        
        est_ai = est_flops_iter / est_bytes_iter if est_bytes_iter > 0 else 0
        
        print(f"\n    Estimativa (7-pt stencil, CG):")
        print(f"    - Bytes por célula: ~{est_bytes_iter/n_elements:.1f}")
        print(f"    - FLOPs por célula: ~{est_flops_iter/n_elements:.1f}")
        print(f"    - Intensidade aritmética: ~{est_ai:.4f} FLOP/Byte")
        print(f"    - Classificação: MEMORY-BOUND (AI << 24)")
    
    print("\n" + "="*80)
    
    return bw


def print_patch():
    """Imprime as instruções de patch."""
    print(PATCH_INSTRUCTIONS)


def main():
    parser = argparse.ArgumentParser(
        description='Análise de Bandwidth para solver TPFA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Ver instruções de patch
  python bandwidth_patch.py --show-patch
  
  # Analisar JSON de resultado
  python bandwidth_patch.py output.json --hardware-bw 672 --num-gpus 4
"""
    )
    parser.add_argument('json_file', nargs='?', help='Arquivo JSON de saída do solver')
    parser.add_argument('--show-patch', action='store_true', help='Mostra instruções de patch')
    parser.add_argument('--hardware-bw', type=float, default=672.0, 
                        help='Bandwidth de pico por GPU (GB/s)')
    parser.add_argument('--num-gpus', type=int, default=4, help='Número de GPUs')
    
    args = parser.parse_args()
    
    if args.show_patch:
        print_patch()
        return
    
    if args.json_file:
        calculate_bandwidth_from_json(args.json_file, args.hardware_bw, args.num_gpus)
    else:
        print("Uso: python bandwidth_patch.py <arquivo.json> [opções]")
        print("     python bandwidth_patch.py --show-patch")
        print("\nUse --help para mais informações.")


if __name__ == '__main__':
    main()