#!/usr/bin/env python3
"""
Script para geração de gráficos do TCC - Simulador de Reservatórios
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

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

COLORS = {
    'cpu': '#2ecc71', 'gpu': '#e74c3c', 'prep': '#3498db',
    'solve': '#e74c3c', 'update': '#f39c12', 'ideal': '#95a5a6',
    'strong': '#3498db', 'weak': '#e74c3c',
}

def find_latest_results(base_path, test_type):
    pattern = os.path.join(base_path, test_type, "*")
    dirs = glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"Nenhum resultado encontrado para {test_type} em {base_path}")
    return max(dirs)

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def load_all_results(base_path):
    results = {}
    for test_type in ['performance', 'strong', 'weak', 'correctness']:
        try:
            test_dir = find_latest_results(base_path, test_type)
            files = glob(os.path.join(test_dir, "*.yaml")) + glob(os.path.join(test_dir, "*.yml"))
            if files:
                results[test_type] = load_yaml(files[0])
        except FileNotFoundError as e:
            print(f"Aviso: {e}")
    return results

def extract_performance_data(perf_results):
    data = {k: [] for k in ['mpi', 'cpu_total', 'gpu_total', 'cpu_prep', 'cpu_solve', 'cpu_update',
                            'gpu_prep', 'gpu_solve', 'gpu_update', 'speedup_total', 'speedup_solve',
                            'cpu_ram', 'gpu_ram', 'gpu_vram']}
    root = perf_results.get('results', perf_results)
    perf_data = root.get('performance', root)
    for r in perf_data.get('results', []):
        data['mpi'].append(r.get('mpi', 1))
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

def extract_scaling_data(scaling_results, scaling_type='strong'):
    data = {k: [] for k in ['gpus', 'time', 'prep', 'solve', 'update', 'speedup', 'efficiency', 'ram', 'vram']}
    key = 'strong_scaling' if scaling_type == 'strong' else 'weak_scaling'
    root = scaling_results.get('results', scaling_results)
    for r in root.get(key, root).get('results', []):
        data['gpus'].append(r.get('mpi', r.get('gpus', 1)))
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

def extract_correctness_data(corr_results):
    data = {}
    for case in corr_results['results']['correctness']:
        name = case['name']
        data[name] = {'cells': [], 'h': [], 'l2_error': []}
        for r in case['results']:
            data[name]['cells'].append(r['total_cells'])
            data[name]['h'].append(r['h'])
            data[name]['l2_error'].append(r['l2_error'])
    return data

def plot_decomposicao_cpu(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data['mpi']))
    ax.bar(x, data['cpu_prep'], 0.6, label='Pré-processamento', color=COLORS['prep'])
    ax.bar(x, data['cpu_solve'], 0.6, bottom=data['cpu_prep'], label='Solver', color=COLORS['solve'])
    ax.bar(x, data['cpu_update'], 0.6, bottom=np.array(data['cpu_prep'])+np.array(data['cpu_solve']), label='Atualização', color=COLORS['update'])
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Tempo (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(data['mpi'])
    ax.legend(loc='upper right')
    totals = np.array(data['cpu_prep']) + np.array(data['cpu_solve']) + np.array(data['cpu_update'])
    for i, t in enumerate(totals):
        ax.annotate(f'{t:.0f}s', xy=(i, t), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_decomposicao_gpu(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data['mpi']))
    ax.bar(x, data['gpu_prep'], 0.6, label='Pré-processamento', color=COLORS['prep'])
    ax.bar(x, data['gpu_solve'], 0.6, bottom=data['gpu_prep'], label='Solver', color=COLORS['solve'])
    ax.bar(x, data['gpu_update'], 0.6, bottom=np.array(data['gpu_prep'])+np.array(data['gpu_solve']), label='Atualização', color=COLORS['update'])
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Tempo (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(data['mpi'])
    ax.legend(loc='upper right')
    totals = np.array(data['gpu_prep']) + np.array(data['gpu_solve']) + np.array(data['gpu_update'])
    for i, t in enumerate(totals):
        ax.annotate(f'{t:.0f}s', xy=(i, t), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_comparacao_solver(data, output_path):
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
        ax.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_speedup_solver(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['mpi'], data['speedup_solve'], 'o-', color=COLORS['gpu'], linewidth=2, markersize=10, label='Speedup do Solver')
    ax.axvline(x=4, color=COLORS['ideal'], linestyle='--', linewidth=1.5, label='Limite de GPUs (4)')
    for x, y in zip(data['mpi'], data['speedup_solve']):
        ax.annotate(f'{y:.1f}×', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Speedup (CPU/GPU)')
    ax.legend(loc='upper right')
    ax.set_xticks(data['mpi'])
    ax.set_ylim(0, max(data['speedup_solve']) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_speedup_total(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['mpi'], data['speedup_total'], 'o-', color=COLORS['strong'], linewidth=2, markersize=10, label='Speedup Total')
    ax.plot(data['mpi'], data['speedup_solve'], 's-', color=COLORS['weak'], linewidth=2, markersize=10, label='Speedup Solver')
    ax.axvline(x=4, color=COLORS['ideal'], linestyle='--', linewidth=1.5, label='Limite de GPUs (4)')
    ax.set_xlabel('Número de Processos MPI')
    ax.set_ylabel('Speedup (CPU/GPU)')
    ax.legend(loc='upper right')
    ax.set_xticks(data['mpi'])
    ax.set_ylim(0, max(data['speedup_solve']) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_strong_scaling_efficiency(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['gpus'], data['efficiency'], 'o-', color=COLORS['strong'], linewidth=2, markersize=12, label='Eficiência Obtida')
    ax.axhline(y=100, color=COLORS['ideal'], linestyle='--', linewidth=1.5, label='Ideal (100%)')
    for x, y in zip(data['gpus'], data['efficiency']):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=11)
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Eficiência (%)')
    ax.legend(loc='lower left')
    ax.set_xticks(data['gpus'])
    ax.set_ylim(80, 105)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_strong_scaling_speedup(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    gpus = np.array(data['gpus'])
    ax.plot(data['gpus'], data['speedup'], 'o-', color=COLORS['strong'], linewidth=2, markersize=12, label='Speedup Obtido')
    ax.plot(data['gpus'], gpus, '--', color=COLORS['ideal'], linewidth=1.5, label='Ideal (Linear)')
    for x, y in zip(data['gpus'], data['speedup']):
        ax.annotate(f'{y:.2f}×', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=11)
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Speedup')
    ax.legend(loc='upper left')
    ax.set_xticks(data['gpus'])
    ax.set_ylim(0, max(data['gpus']) + 0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_weak_scaling_efficiency(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['gpus'], data['efficiency'], 'o-', color=COLORS['weak'], linewidth=2, markersize=12, label='Eficiência Obtida')
    ax.axhline(y=100, color=COLORS['ideal'], linestyle='--', linewidth=1.5, label='Ideal (100%)')
    for x, y in zip(data['gpus'], data['efficiency']):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=11)
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Eficiência (%)')
    ax.legend(loc='lower left')
    ax.set_xticks(data['gpus'])
    ax.set_ylim(70, 105)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_memoria_scaling(strong_data, weak_data, output_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    gpus = strong_data['gpus']
    ax.plot(gpus, strong_data['ram'], 'o-', color=COLORS['strong'], linewidth=2, markersize=10, label='RAM - Strong Scaling')
    ax.plot(gpus, weak_data['ram'], 's-', color=COLORS['weak'], linewidth=2, markersize=10, label='RAM - Weak Scaling')
    ax.plot(gpus, strong_data['vram'], 'o--', color=COLORS['strong'], linewidth=2, markersize=10, alpha=0.7, label='VRAM - Strong Scaling')
    ax.plot(gpus, weak_data['vram'], 's--', color=COLORS['weak'], linewidth=2, markersize=10, alpha=0.7, label='VRAM - Weak Scaling')
    ax.set_xlabel('Número de GPUs')
    ax.set_ylabel('Memória (GB)')
    ax.legend(loc='upper left')
    ax.set_xticks(gpus)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_convergence(corr_data, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    markers = ['o', 's', '^']
    colors = sns.color_palette("husl", 3)
    label_map = {'example_1': 'Caso 1: Linear', 'example_3': 'Caso 2: Não-linear', 'example_Li': 'Caso 3: Li Case 4'}
    for i, (name, d) in enumerate(corr_data.items()):
        if name == 'example_2':
            continue
        ax.loglog(d['cells'], d['l2_error'], f'{markers[i%3]}-', color=colors[i%3], linewidth=2, markersize=10, label=label_map.get(name, name))
    x_ref = np.array([1e5, 1e7])
    ax.loglog(x_ref, 1e-6 * (x_ref/1e5)**(-2/3), 'k--', linewidth=1.5, alpha=0.5, label=r'$O(N^{-2/3})$')
    ax.set_xlabel('Número de Células')
    ax.set_ylabel('Erro $L_2$')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def plot_convergence_individual(case_name, case_data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(case_data['cells'], case_data['l2_error'], 'o-', color=COLORS['strong'], linewidth=2, markersize=12, label='Erro $L_2$ Obtido')
    if len(case_data['cells']) >= 2:
        x_ref = np.array([min(case_data['cells']), max(case_data['cells'])])
        slope = -1 if 'Li' in case_name else -2/3
        y_ref = case_data['l2_error'][0] * (x_ref / case_data['cells'][0]) ** slope
        ax.loglog(x_ref, y_ref, 'k--', linewidth=1.5, alpha=0.5, label=f'$O(N^{{{slope:.2f}}})$')
    ax.set_xlabel('Número de Células')
    ax.set_ylabel('Erro $L_2$')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Salvo: {output_path}")

def create_example_data():
    return {
        'performance': {'results': {'performance': {'results': [
            {'mpi': 1, 'speedup': 9.27, 'speedup_solving': 32.76,
             'cpu': {'summary': {'avg_total': 6111.1, 'avg_preprocessing': 85.8, 'avg_solving': 5586.3, 'avg_updating': 438.9, 'max_ram_mb': 31523}},
             'gpu': {'summary': {'avg_total': 659.1, 'avg_preprocessing': 85.8, 'avg_solving': 170.5, 'avg_updating': 402.8, 'max_ram_mb': 28788, 'max_vram_mb': 14632}}},
            {'mpi': 2, 'speedup': 9.21, 'speedup_solving': 26.66,
             'cpu': {'summary': {'avg_total': 3139.4, 'avg_preprocessing': 44.5, 'avg_solving': 2882.5, 'avg_updating': 212.4, 'max_ram_mb': 36398}},
             'gpu': {'summary': {'avg_total': 340.7, 'avg_preprocessing': 45.1, 'avg_solving': 108.1, 'avg_updating': 187.4, 'max_ram_mb': 40615, 'max_vram_mb': 17746}}},
            {'mpi': 4, 'speedup': 9.22, 'speedup_solving': 25.67,
             'cpu': {'summary': {'avg_total': 1694.1, 'avg_preprocessing': 25.5, 'avg_solving': 1560.6, 'avg_updating': 107.9, 'max_ram_mb': 37935}},
             'gpu': {'summary': {'avg_total': 183.7, 'avg_preprocessing': 27.8, 'avg_solving': 60.8, 'avg_updating': 95.0, 'max_ram_mb': 42883, 'max_vram_mb': 18614}}},
            {'mpi': 8, 'speedup': 7.16, 'speedup_solving': 15.08,
             'cpu': {'summary': {'avg_total': 943.8, 'avg_preprocessing': 15.6, 'avg_solving': 871.0, 'avg_updating': 57.1, 'max_ram_mb': 40482}},
             'gpu': {'summary': {'avg_total': 131.7, 'avg_preprocessing': 15.8, 'avg_solving': 57.8, 'avg_updating': 58.2, 'max_ram_mb': 45750, 'max_vram_mb': 20544}}},
            {'mpi': 16, 'speedup': 6.07, 'speedup_solving': 10.22,
             'cpu': {'summary': {'avg_total': 720.4, 'avg_preprocessing': 9.9, 'avg_solving': 670.7, 'avg_updating': 39.7, 'max_ram_mb': 45532}},
             'gpu': {'summary': {'avg_total': 118.8, 'avg_preprocessing': 11.3, 'avg_solving': 65.6, 'avg_updating': 41.8, 'max_ram_mb': 51649, 'max_vram_mb': 23000}}},
        ]}}},
        'strong': {'results': {'strong_scaling': {'results': [
            {'mpi': 1, 'summary': {'avg_total': 662.88, 'avg_preprocessing': 87.0, 'avg_solving': 170.5, 'avg_updating': 405.3, 'max_ram_mb': 28774, 'max_vram_mb': 14632, 'speedup': 1.0, 'efficiency': 100.0}},
            {'mpi': 2, 'summary': {'avg_total': 339.99, 'avg_preprocessing': 44.7, 'avg_solving': 108.4, 'avg_updating': 186.9, 'max_ram_mb': 41677, 'max_vram_mb': 17712, 'speedup': 1.95, 'efficiency': 97.5}},
            {'mpi': 3, 'summary': {'avg_total': 235.49, 'avg_preprocessing': 31.8, 'avg_solving': 76.7, 'avg_updating': 127.0, 'max_ram_mb': 42086, 'max_vram_mb': 18125, 'speedup': 2.81, 'efficiency': 93.8}},
            {'mpi': 4, 'summary': {'avg_total': 182.88, 'avg_preprocessing': 25.5, 'avg_solving': 59.5, 'avg_updating': 97.9, 'max_ram_mb': 43315, 'max_vram_mb': 18614, 'speedup': 3.62, 'efficiency': 90.6}},
        ]}}},
        'weak': {'results': {'weak_scaling': {'results': [
            {'mpi': 1, 'summary': {'avg_total': 471.1, 'avg_preprocessing': 62.1, 'avg_solving': 120.4, 'avg_updating': 288.6, 'max_ram_mb': 20787, 'max_vram_mb': 10547, 'speedup': 1.0, 'efficiency': 100.0}},
            {'mpi': 2, 'summary': {'avg_total': 533.1, 'avg_preprocessing': 67.2, 'avg_solving': 159.4, 'avg_updating': 306.5, 'max_ram_mb': 59315, 'max_vram_mb': 25190, 'speedup': 0.884, 'efficiency': 88.4}},
            {'mpi': 3, 'summary': {'avg_total': 552.7, 'avg_preprocessing': 70.8, 'avg_solving': 168.8, 'avg_updating': 313.1, 'max_ram_mb': 88391, 'max_vram_mb': 37683, 'speedup': 0.852, 'efficiency': 85.2}},
            {'mpi': 4, 'summary': {'avg_total': 564.6, 'avg_preprocessing': 75.3, 'avg_solving': 168.2, 'avg_updating': 321.1, 'max_ram_mb': 118058, 'max_vram_mb': 50480, 'speedup': 0.834, 'efficiency': 83.4}},
        ]}}},
        'correctness': {'results': {'correctness': [
            {'name': 'example_1', 'results': [{'total_cells': 125000, 'h': 4.0, 'l2_error': 2.99e-10}, {'total_cells': 512000, 'h': 2.5, 'l2_error': 1.81e-10}, {'total_cells': 1728000, 'h': 1.67, 'l2_error': 1.21e-10}, {'total_cells': 5832000, 'h': 1.11, 'l2_error': 8.04e-11}]},
            {'name': 'example_3', 'results': [{'total_cells': 125000, 'h': 4.0, 'l2_error': 2.02e-6}, {'total_cells': 512000, 'h': 2.5, 'l2_error': 1.08e-6}, {'total_cells': 1728000, 'h': 1.67, 'l2_error': 6.70e-7}, {'total_cells': 5832000, 'h': 1.11, 'l2_error': 4.25e-7}]},
            {'name': 'example_Li', 'results': [{'total_cells': 625, 'h': 80.0, 'l2_error': 2.73e-3}, {'total_cells': 2500, 'h': 40.0, 'l2_error': 1.82e-3}, {'total_cells': 62500, 'h': 8.0, 'l2_error': 8.00e-4}, {'total_cells': 250000, 'h': 4.0, 'l2_error': 5.71e-4}, {'total_cells': 6250000, 'h': 0.8, 'l2_error': 1.51e-4}, {'total_cells': 9000000, 'h': 0.67, 'l2_error': 1.38e-4}]},
        ]}}
    }

def main():
    base_path, output_dir = "./results_jarvis", "./figures"
    os.makedirs(output_dir, exist_ok=True)
    print("="*60 + "\nGERADOR DE GRÁFICOS DO TCC\n" + "="*60)
    print(f"\nDiretório de resultados: {base_path}\nDiretório de saída: {output_dir}\n")
    
    results = load_all_results(base_path)
    if not results:
        print("\nUsando dados de exemplo...")
        results = create_example_data()
    else:
        example = create_example_data()
        for key in ['performance', 'strong', 'weak', 'correctness']:
            if key not in results:
                results[key] = example[key]
    
    if 'performance' in results:
        print("\n--- Gerando gráficos de Performance ---")
        perf_data = extract_performance_data(results['performance'])
        plot_decomposicao_cpu(perf_data, os.path.join(output_dir, "fig_decomposicao_cpu.png"))
        plot_decomposicao_gpu(perf_data, os.path.join(output_dir, "fig_decomposicao_gpu.png"))
        plot_comparacao_solver(perf_data, os.path.join(output_dir, "fig_comparacao_solver.png"))
        plot_speedup_solver(perf_data, os.path.join(output_dir, "fig_speedup_solver.png"))
        plot_speedup_total(perf_data, os.path.join(output_dir, "fig_speedup_total.png"))
    
    if 'strong' in results:
        print("\n--- Gerando gráficos de Escalabilidade Forte ---")
        strong_data = extract_scaling_data(results['strong'], 'strong')
        if strong_data['gpus']:
            plot_strong_scaling_efficiency(strong_data, os.path.join(output_dir, "fig_strong_scaling_efficiency.png"))
            plot_strong_scaling_speedup(strong_data, os.path.join(output_dir, "fig_strong_scaling_speedup.png"))
    
    if 'weak' in results:
        print("\n--- Gerando gráficos de Escalabilidade Fraca ---")
        weak_data = extract_scaling_data(results['weak'], 'weak')
        if weak_data['gpus']:
            plot_weak_scaling_efficiency(weak_data, os.path.join(output_dir, "fig_weak_scaling_efficiency.png"))
    
    if 'strong' in results and 'weak' in results and strong_data['gpus'] and weak_data['gpus']:
        plot_memoria_scaling(strong_data, weak_data, os.path.join(output_dir, "fig_memoria_scaling.png"))
    
    if 'correctness' in results:
        print("\n--- Gerando gráficos de Acurácia ---")
        corr_data = extract_correctness_data(results['correctness'])
        plot_convergence(corr_data, os.path.join(output_dir, "fig_convergence.png"))
        for name, d in corr_data.items():
            if name != 'example_2':
                plot_convergence_individual(name, d, os.path.join(output_dir, f"fig_convergence_{name}.png"))
    
    print("\n" + "="*60 + "\nGERAÇÃO DE GRÁFICOS CONCLUÍDA!\n" + "="*60)
    generated = sorted(glob(os.path.join(output_dir, "*.png")))
    print(f"\nArquivos gerados ({len(generated)}):")
    for f in generated:
        print(f"  - {os.path.basename(f)}")

if __name__ == "__main__":
    main()