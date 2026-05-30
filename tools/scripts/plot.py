#!/usr/bin/env python3
"""
Script para geração de gráficos do TCC - Simulador de Reservatórios
Estilo acadêmico: sem títulos (títulos são definidos no LaTeX via caption)
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from typing import Dict, List, Any

# Configuração global de estilo acadêmico
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Paleta de cores consistente
COLORS = {
    'cpu': '#2ecc71',
    'gpu': '#e74c3c',
    'prep': '#3498db',
    'solve': '#e74c3c',
    'update': '#f39c12',
    'ideal': '#95a5a6',
    'strong': '#3498db',
    'weak': '#e74c3c',
    'efficiency': '#1abc9c',
    # Cores para precondicionadores
    'gamg': '#2ecc71',
    'jacobi': '#e74c3c',
    'bjacobi': '#3498db',
}

# Cores para comparação de solvers
SOLVER_COLORS = sns.color_palette("husl", 10)


def find_latest_results(base_path: str, test_type: str) -> str:
    """Encontra o diretório mais recente de resultados para um tipo de teste."""
    pattern = os.path.join(base_path, test_type, "*")
    dirs = glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"Nenhum resultado encontrado para {test_type} em {base_path}")
    return max(dirs)


def load_yaml(filepath: str) -> Dict:
    """Carrega um arquivo YAML."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_all_results(base_path: str) -> Dict[str, Any]:
    """Carrega todos os resultados disponíveis."""
    results = {}
    test_types = ['performance', 'strong', 'weak', 'correctness', 'solvers']
    
    for test_type in test_types:
        try:
            test_dir = find_latest_results(base_path, test_type)
            files = glob(os.path.join(test_dir, "*.yaml")) + glob(os.path.join(test_dir, "*.yml"))
            if files:
                results[test_type] = load_yaml(files[0])
                print(f"✓ Carregado: {test_type} de {os.path.basename(test_dir)}")
        except FileNotFoundError as e:
            print(f"⚠ Aviso: {e}")
    
    return results


# =============================================================================
# FUNÇÕES DE EXTRAÇÃO DE DADOS
# =============================================================================

def extract_performance_data(perf_results: Dict) -> Dict[str, List]:
    """Extrai dados de performance CPU vs GPU."""
    data = {
        'mpi': [], 'total_cells': [],
        'cpu_total': [], 'gpu_total': [],
        'cpu_prep': [], 'cpu_solve': [], 'cpu_update': [],
        'gpu_prep': [], 'gpu_solve': [], 'gpu_update': [],
        'speedup_total': [], 'speedup_solve': [],
        'cpu_ram': [], 'gpu_ram': [], 'gpu_vram': []
    }
    
    root = perf_results.get('results', perf_results)
    perf_data = root.get('performance', root)
    
    for r in perf_data.get('results', []):
        data['mpi'].append(r.get('mpi', 1))
        data['total_cells'].append(r.get('total_cells', 0))
        
        cpu = r.get('cpu', {}).get('summary', {})
        gpu = r.get('gpu', {}).get('summary', {})
        
        data['cpu_total'].append(cpu.get('avg_total', 0))
        data['cpu_prep'].append(cpu.get('avg_preprocessing', 0))
        data['cpu_solve'].append(cpu.get('avg_solving', 0))
        data['cpu_update'].append(cpu.get('avg_updating', 0))
        data['cpu_ram'].append(cpu.get('max_ram_mb', 0) / 1024)
        
        data['gpu_total'].append(gpu.get('avg_total', 0))
        data['gpu_prep'].append(gpu.get('avg_preprocessing', 0))
        data['gpu_solve'].append(gpu.get('avg_solving', 0))
        data['gpu_update'].append(gpu.get('avg_updating', 0))
        data['gpu_ram'].append(gpu.get('max_ram_mb', 0) / 1024)
        data['gpu_vram'].append(gpu.get('max_vram_mb', 0) / 1024)
        
        data['speedup_total'].append(r.get('speedup', 1.0))
        data['speedup_solve'].append(r.get('speedup_solving', 1.0))
    
    return data


def extract_scaling_data(scaling_results: Dict, scaling_type: str = 'strong') -> Dict[str, List]:
    """Extrai dados de escalabilidade forte ou fraca."""
    data = {
        'gpus': [], 'mpi': [], 'total_cells': [],
        'time': [], 'prep': [], 'solve': [], 'update': [],
        'speedup': [], 'efficiency': [],
        'ram': [], 'vram': []
    }
    
    key = 'strong_scaling' if scaling_type == 'strong' else 'weak_scaling'
    root = scaling_results.get('results', scaling_results)
    scaling_data = root.get(key, root)
    
    for r in scaling_data.get('results', []):
        data['gpus'].append(r.get('gpus', r.get('mpi', 1)))
        data['mpi'].append(r.get('mpi', r.get('gpus', 1)))
        data['total_cells'].append(r.get('total_cells', 0))
        
        s = r.get('summary', {})
        data['time'].append(s.get('avg_total', 0))
        data['prep'].append(s.get('avg_preprocessing', 0))
        data['solve'].append(s.get('avg_solving', 0))
        data['update'].append(s.get('avg_updating', 0))
        data['ram'].append(s.get('max_ram_mb', 0) / 1024)
        data['vram'].append(s.get('max_vram_mb', 0) / 1024)
        data['speedup'].append(s.get('speedup', 1.0))
        data['efficiency'].append(s.get('efficiency', 100.0))
    
    return data


def extract_correctness_data(corr_results: Dict) -> Dict[str, Dict]:
    """Extrai dados de acurácia/convergência."""
    data = {}
    root = corr_results.get('results', corr_results)
    
    for case in root.get('correctness', []):
        name = case['name']
        data[name] = {
            'description': case.get('description', name),
            'expected_order': case.get('expected_order', 2.0),
            'cells': [],
            'h': [],
            'l2_error': [],
            'orders': case.get('orders', []),
            'average_order': case.get('average_order', 0)
        }
        
        for r in case.get('results', []):
            data[name]['cells'].append(r['total_cells'])
            data[name]['h'].append(r['h'])
            data[name]['l2_error'].append(r['l2_error'])
    
    return data


def extract_solver_data(solver_results: Dict) -> Dict[str, Any]:
    """Extrai dados de comparação de solvers."""
    data = {
        'solvers': [],
        'config': {},
        'analysis': {}
    }
    
    root = solver_results.get('results', solver_results)
    solver_comp = root.get('solver_comparison', root)
    
    data['config'] = solver_comp.get('config', {})
    data['analysis'] = solver_comp.get('analysis', {})
    
    for solver in solver_comp.get('solvers', []):
        solver_data = {
            'name': solver.get('name', 'Unknown'),
            'ksp_type': solver.get('ksp_type', ''),
            'pc_type': solver.get('pc_type', ''),
            'description': solver.get('description', ''),
            'summary': solver.get('summary', {}),
            'runs': solver.get('runs', [])
        }
        
        summary = solver_data['summary']
        solver_data['avg_solving_time'] = summary.get('avg_solving_time', 0)
        solver_data['avg_total_time'] = summary.get('avg_total_time', 0)
        solver_data['avg_iterations'] = summary.get('avg_iterations', 0)
        solver_data['speedup'] = summary.get('speedup_vs_first', 1.0)
        solver_data['max_vram_mb'] = summary.get('max_vram_mb', 0)
        solver_data['max_ram_mb'] = summary.get('max_ram_mb', 0)
        solver_data['score'] = summary.get('score', 0)
        
        data['solvers'].append(solver_data)
    
    return data


# =============================================================================
# FUNÇÕES DE PLOTAGEM - PERFORMANCE
# =============================================================================

def plot_decomposicao_cpu(data: Dict, output_path: str):
    """Gráfico de decomposição de tempo CPU."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data['mpi']))
    width = 0.6
    
    prep = np.array(data['cpu_prep'])
    solve = np.array(data['cpu_solve'])
    update = np.array(data['cpu_update'])
    
    ax.bar(x, prep, width, label='Pré-processamento', color=COLORS['prep'])
    ax.bar(x, solve, width, bottom=prep, label='Solver', color=COLORS['solve'])
    ax.bar(x, update, width, bottom=prep + solve, label='Atualização', color=COLORS['update'])
    
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Tempo (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(data['mpi'])
    ax.legend(loc='upper right')
    
    totals = prep + solve + update
    for i, t in enumerate(totals):
        ax.annotate(f'{t:.0f}s', xy=(i, t), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_decomposicao_gpu(data: Dict, output_path: str):
    """Gráfico de decomposição de tempo GPU."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data['mpi']))
    width = 0.6
    
    prep = np.array(data['gpu_prep'])
    solve = np.array(data['gpu_solve'])
    update = np.array(data['gpu_update'])
    
    ax.bar(x, prep, width, label='Pré-processamento', color=COLORS['prep'])
    ax.bar(x, solve, width, bottom=prep, label='Solver', color=COLORS['solve'])
    ax.bar(x, update, width, bottom=prep + solve, label='Atualização', color=COLORS['update'])
    
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Tempo (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(data['mpi'])
    ax.legend(loc='upper right')
    
    totals = prep + solve + update
    for i, t in enumerate(totals):
        ax.annotate(f'{t:.0f}s', xy=(i, t), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_comparacao_solver(data: Dict, output_path: str):
    """Gráfico de comparação CPU vs GPU para o solver."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data['mpi']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['cpu_solve'], width, label='CPU', color=COLORS['cpu'])
    bars2 = ax.bar(x + width/2, data['gpu_solve'], width, label='GPU', color=COLORS['gpu'])
    
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Tempo do Solver (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(data['mpi'])
    ax.legend()
    ax.set_yscale('log')
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_speedup_solver(data: Dict, output_path: str):
    """Gráfico de speedup do solver."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data['mpi'], data['speedup_solve'], 'o-', color=COLORS['gpu'],
            linewidth=2, markersize=10, label='Speedup do Solver')
    
    # Linha vertical indicando limite de GPUs
    num_gpus = 4
    ax.axvline(x=num_gpus, color=COLORS['ideal'], linestyle='--',
               linewidth=1.5, label=f'Limite de GPUs ({num_gpus})')
    
    for x, y in zip(data['mpi'], data['speedup_solve']):
        ax.annotate(f'{y:.1f}×', xy=(x, y), xytext=(5, 5),
                   textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Speedup (CPU/GPU)')
    ax.legend(loc='upper right')
    ax.set_xticks(data['mpi'])
    ax.set_ylim(0, max(data['speedup_solve']) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_speedup_total(data: Dict, output_path: str):
    """Gráfico de speedup total vs solver."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data['mpi'], data['speedup_total'], 'o-', color=COLORS['strong'],
            linewidth=2, markersize=10, label='Speedup Total')
    ax.plot(data['mpi'], data['speedup_solve'], 's-', color=COLORS['weak'],
            linewidth=2, markersize=10, label='Speedup Solver')
    
    num_gpus = 4
    ax.axvline(x=num_gpus, color=COLORS['ideal'], linestyle='--',
               linewidth=1.5, label=f'Limite de GPUs ({num_gpus})')
    
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Speedup (CPU/GPU)')
    ax.legend(loc='upper right')
    ax.set_xticks(data['mpi'])
    ax.set_ylim(0, max(data['speedup_solve']) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


# =============================================================================
# FUNÇÕES DE PLOTAGEM - ESCALABILIDADE
# =============================================================================

def plot_strong_scaling_speedup(data: Dict, output_path: str):
    """Gráfico de speedup para escalabilidade forte."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gpus = np.array(data['gpus'])
    speedup = np.array(data['speedup'])
    
    ax.plot(gpus, speedup, 'o-', color=COLORS['strong'],
            linewidth=2, markersize=12, label='Speedup Obtido')
    ax.plot(gpus, gpus, '--', color=COLORS['ideal'],
            linewidth=1.5, label='Ideal (Linear)')
    
    for x, y in zip(gpus, speedup):
        ax.annotate(f'{y:.2f}×', xy=(x, y), xytext=(5, 5),
                   textcoords='offset points', fontsize=11)
    
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Speedup')
    ax.legend(loc='upper left')
    ax.set_xticks(gpus)
    ax.set_ylim(0, max(gpus) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_strong_scaling_efficiency(data: Dict, output_path: str):
    """Gráfico de eficiência para escalabilidade forte."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data['gpus'], data['efficiency'], 'o-', color=COLORS['strong'],
            linewidth=2, markersize=12, label='Eficiência Obtida')
    ax.axhline(y=100, color=COLORS['ideal'], linestyle='--',
               linewidth=1.5, label='Ideal (100%)')
    
    for x, y in zip(data['gpus'], data['efficiency']):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=11)
    
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Eficiência (%)')
    ax.legend(loc='lower left')
    ax.set_xticks(data['gpus'])
    ax.set_ylim(80, 105)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_weak_scaling_efficiency(data: Dict, output_path: str):
    """Gráfico de eficiência para escalabilidade fraca."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data['gpus'], data['efficiency'], 'o-', color=COLORS['weak'],
            linewidth=2, markersize=12, label='Eficiência Obtida')
    ax.axhline(y=100, color=COLORS['ideal'], linestyle='--',
               linewidth=1.5, label='Ideal (100%)')
    
    for x, y in zip(data['gpus'], data['efficiency']):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=11)
    
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Eficiência (%)')
    ax.legend(loc='lower left')
    ax.set_xticks(data['gpus'])
    ax.set_ylim(70, 105)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_memoria_scaling(strong_data: Dict, weak_data: Dict, output_path: str):
    """Gráfico de memória para ambas escalabilidades."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gpus = strong_data['gpus']
    
    ax.plot(gpus, strong_data['ram'], 'o-', color=COLORS['strong'],
            linewidth=2, markersize=10, label='RAM - Strong Scaling')
    ax.plot(gpus, weak_data['ram'], 's-', color=COLORS['weak'],
            linewidth=2, markersize=10, label='RAM - Weak Scaling')
    ax.plot(gpus, strong_data['vram'], 'o--', color=COLORS['strong'],
            linewidth=2, markersize=10, alpha=0.7, label='VRAM - Strong Scaling')
    ax.plot(gpus, weak_data['vram'], 's--', color=COLORS['weak'],
            linewidth=2, markersize=10, alpha=0.7, label='VRAM - Weak Scaling')
    
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Memória (GB)')
    ax.legend(loc='upper left')
    ax.set_xticks(gpus)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


# =============================================================================
# FUNÇÕES DE PLOTAGEM - ACURÁCIA
# =============================================================================

def plot_convergence(corr_data: Dict, output_path: str):
    """Gráfico de convergência combinado."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = sns.color_palette("husl", len(corr_data))
    
    for i, (name, d) in enumerate(corr_data.items()):
        if name == 'example_2':
            continue
        
        label = d.get('description', name)
        ax.loglog(d['cells'], d['l2_error'], f'{markers[i % len(markers)]}-',
                 color=colors[i], linewidth=2, markersize=10, label=label)
    
    # Linha de referência
    x_ref = np.array([1e3, 1e7])
    ax.loglog(x_ref, 1e-4 * (x_ref/1e3)**(-2/3), 'k--',
              linewidth=1.5, alpha=0.5, label=r'$O(N^{-2/3})$')
    
    ax.set_xlabel('Número de Células')
    ax.set_ylabel('Erro $L_2$')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_convergence_individual(case_name: str, case_data: Dict, output_path: str):
    """Gráfico de convergência individual por caso."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cells = case_data['cells']
    errors = case_data['l2_error']
    
    ax.loglog(cells, errors, 'o-', color=COLORS['strong'],
              linewidth=2, markersize=12, label='Erro $L_2$ Obtido')
    
    if len(cells) >= 2:
        log_cells = np.log10(cells)
        log_errors = np.log10(errors)
        slope, intercept = np.polyfit(log_cells, log_errors, 1)
        
        x_fit = np.array([min(cells), max(cells)])
        y_fit = 10**(slope * np.log10(x_fit) + intercept)
        ax.loglog(x_fit, y_fit, 'k--', linewidth=1.5, alpha=0.5,
                 label=f'Tendência ($O(N^{{{slope:.2f}}})$)')
    
    ax.set_xlabel('Número de Células')
    ax.set_ylabel('Erro $L_2$')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


# =============================================================================
# FUNÇÕES DE PLOTAGEM - COMPARAÇÃO DE SOLVERS
# =============================================================================

def plot_solver_comparison_time(solver_data: Dict, output_path: str):
    """Gráfico de comparação de tempo entre solvers."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    solvers = solver_data['solvers']
    names = [s['name'] for s in solvers]
    solving_times = [s['avg_solving_time'] for s in solvers]
    
    colors = [SOLVER_COLORS[i % len(SOLVER_COLORS)] for i in range(len(solvers))]
    
    bars = ax.barh(names, solving_times, color=colors, edgecolor='black', linewidth=0.5)
    
    # Destacar o melhor solver
    best_idx = np.argmin(solving_times)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    for i, (bar, time) in enumerate(zip(bars, solving_times)):
        width = bar.get_width()
        label = f'{time:.2f}s'
        if i == best_idx:
            label += ' ★'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
               label, va='center', fontsize=10)
    
    ax.set_xlabel('Tempo de Resolução (s)')
    ax.set_ylabel('Solver + Precondicionador')
    ax.set_xlim(0, max(solving_times) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_solver_comparison_speedup(solver_data: Dict, output_path: str):
    """Gráfico de speedup relativo ao solver baseline."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    solvers = solver_data['solvers']
    names = [s['name'] for s in solvers]
    speedups = [s['speedup'] for s in solvers]
    
    colors = [SOLVER_COLORS[i % len(SOLVER_COLORS)] for i in range(len(solvers))]
    
    bars = ax.barh(names, speedups, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline')
    
    for bar, spd in zip(bars, speedups):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{spd:.2f}×', va='center', fontsize=10)
    
    ax.set_xlabel('Speedup (relativo ao primeiro solver)')
    ax.set_ylabel('Solver + Precondicionador')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_solver_comparison_memory(solver_data: Dict, output_path: str):
    """Gráfico de uso de memória por solver."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    solvers = solver_data['solvers']
    names = [s['name'] for s in solvers]
    vram = [s['max_vram_mb'] / 1024 for s in solvers]
    ram = [s['max_ram_mb'] / 1024 for s in solvers]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.barh(x - width/2, vram, width, label='VRAM', color=COLORS['gpu'])
    ax.barh(x + width/2, ram, width, label='RAM', color=COLORS['cpu'])
    
    ax.set_xlabel('Memória (GB)')
    ax.set_ylabel('Solver + Precondicionador')
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_solver_by_preconditioner(solver_data: Dict, output_path: str):
    """Gráfico de comparação agrupada por precondicionador."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    analysis = solver_data.get('analysis', {})
    by_pc = analysis.get('by_preconditioner', {})
    
    if not by_pc:
        solvers = solver_data['solvers']
        by_pc = {}
        for s in solvers:
            pc = s['pc_type'].upper()
            if pc not in by_pc:
                by_pc[pc] = {'times': [], 'names': []}
            by_pc[pc]['times'].append(s['avg_solving_time'])
            by_pc[pc]['names'].append(s['ksp_type'].upper())
    
    precond_names = list(by_pc.keys())
    n_groups = len(precond_names)
    
    pc_colors = {'GAMG': COLORS['gamg'], 'JACOBI': COLORS['jacobi'], 'BJACOBI': COLORS['bjacobi']}
    
    x = np.arange(n_groups)
    width = 0.6
    
    times = []
    labels = []
    for pc in precond_names:
        pc_data = by_pc[pc]
        if isinstance(pc_data, dict):
            if 'best_time' in pc_data:
                times.append(pc_data['best_time'])
                labels.append(pc_data.get('best_solver', pc))
            elif 'times' in pc_data:
                times.append(min(pc_data['times']))
                idx = pc_data['times'].index(min(pc_data['times']))
                labels.append(pc_data['names'][idx])
            else:
                times.append(0)
                labels.append(pc)
    
    colors = [pc_colors.get(pc, COLORS['ideal']) for pc in precond_names]
    bars = ax.bar(x, times, width, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, t, label in zip(bars, times, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{t:.2f}s\n({label})', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Precondicionador')
    ax.set_ylabel('Melhor Tempo de Resolução (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(precond_names)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_solver_iterations_vs_time(solver_data: Dict, output_path: str):
    """Gráfico de iterações vs tempo por solver."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    solvers = solver_data['solvers']
    
    for i, s in enumerate(solvers):
        time = s['avg_solving_time']
        iters = s['avg_iterations']
        color = SOLVER_COLORS[i % len(SOLVER_COLORS)]
        
        ax.scatter(iters, time, s=200, c=[color], label=s['name'],
                  edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(s['name'], xy=(iters, time), xytext=(5, 5),
                   textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Número de Iterações')
    ax.set_ylabel('Tempo de Resolução (s)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


def plot_solver_score(solver_data: Dict, output_path: str):
    """Gráfico de score combinado dos solvers."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    solvers = solver_data['solvers']
    names = [s['name'] for s in solvers]
    scores = [s.get('score', 0) for s in solvers]
    
    colors = [SOLVER_COLORS[i % len(SOLVER_COLORS)] for i in range(len(solvers))]
    
    bars = ax.barh(names, scores, color=colors, edgecolor='black', linewidth=0.5)
    
    best_idx = np.argmin(scores)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        label = f'{score:.3f}'
        if i == best_idx:
            label += ' ★'
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               label, va='center', fontsize=10)
    
    ax.set_xlabel('Score (menor é melhor)')
    ax.set_ylabel('Solver + Precondicionador')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Salvo: {output_path}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal para geração de todos os gráficos."""
    base_path = "./results_jarvis"
    output_dir = "./figures"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GERADOR DE GRÁFICOS DO TCC")
    print("=" * 60)
    print(f"\nDiretório de resultados: {base_path}")
    print(f"Diretório de saída: {output_dir}\n")
    
    results = load_all_results(base_path)
    
    if not results:
        print("\n⚠ Nenhum resultado encontrado. Verifique o diretório de resultados.")
        return
    
    # Gráficos de Performance
    if 'performance' in results:
        print("\n--- Gerando gráficos de Performance ---")
        perf_data = extract_performance_data(results['performance'])
        
        plot_decomposicao_cpu(perf_data, os.path.join(output_dir, "fig_decomposicao_cpu.png"))
        plot_decomposicao_gpu(perf_data, os.path.join(output_dir, "fig_decomposicao_gpu.png"))
        plot_comparacao_solver(perf_data, os.path.join(output_dir, "fig_comparacao_solver.png"))
        plot_speedup_solver(perf_data, os.path.join(output_dir, "fig_speedup_solver.png"))
        plot_speedup_total(perf_data, os.path.join(output_dir, "fig_speedup_total.png"))
    
    # Gráficos de Escalabilidade Forte
    if 'strong' in results:
        print("\n--- Gerando gráficos de Escalabilidade Forte ---")
        strong_data = extract_scaling_data(results['strong'], 'strong')
        
        if strong_data['gpus']:
            plot_strong_scaling_speedup(strong_data, os.path.join(output_dir, "fig_strong_scaling_speedup.png"))
            plot_strong_scaling_efficiency(strong_data, os.path.join(output_dir, "fig_strong_scaling_efficiency.png"))
    
    # Gráficos de Escalabilidade Fraca
    if 'weak' in results:
        print("\n--- Gerando gráficos de Escalabilidade Fraca ---")
        weak_data = extract_scaling_data(results['weak'], 'weak')
        
        if weak_data['gpus']:
            plot_weak_scaling_efficiency(weak_data, os.path.join(output_dir, "fig_weak_scaling_efficiency.png"))
    
    # Gráfico combinado de memória
    if 'strong' in results and 'weak' in results:
        if strong_data['gpus'] and weak_data['gpus']:
            plot_memoria_scaling(strong_data, weak_data, os.path.join(output_dir, "fig_memoria_scaling.png"))
    
    # Gráficos de Acurácia
    if 'correctness' in results:
        print("\n--- Gerando gráficos de Acurácia ---")
        corr_data = extract_correctness_data(results['correctness'])
        
        plot_convergence(corr_data, os.path.join(output_dir, "fig_convergence.png"))
        
        for name, d in corr_data.items():
            if name != 'example_2':
                plot_convergence_individual(name, d, os.path.join(output_dir, f"fig_convergence_{name}.png"))
    
    # Gráficos de Comparação de Solvers
    if 'solvers' in results:
        print("\n--- Gerando gráficos de Comparação de Solvers ---")
        solver_data = extract_solver_data(results['solvers'])
        
        plot_solver_comparison_time(solver_data, os.path.join(output_dir, "fig_solver_time.png"))
        plot_solver_comparison_speedup(solver_data, os.path.join(output_dir, "fig_solver_speedup.png"))
        plot_solver_comparison_memory(solver_data, os.path.join(output_dir, "fig_solver_memory.png"))
        plot_solver_by_preconditioner(solver_data, os.path.join(output_dir, "fig_solver_by_precond.png"))
        plot_solver_iterations_vs_time(solver_data, os.path.join(output_dir, "fig_solver_iters_vs_time.png"))
        plot_solver_score(solver_data, os.path.join(output_dir, "fig_solver_score.png"))
    
    # Resumo final
    print("\n" + "=" * 60)
    print("GERAÇÃO DE GRÁFICOS CONCLUÍDA!")
    print("=" * 60)
    
    generated = sorted(glob(os.path.join(output_dir, "*.png")))
    print(f"\nArquivos gerados ({len(generated)}):")
    for f in generated:
        print(f"  - {os.path.basename(f)}")


if __name__ == "__main__":
    main()