"""
reference.py - Solução Analítica de Li para Poço em Reservatório 2D
VERSÃO CORRIGIDA com tratamento da singularidade do poço

A solução de Li (séries de Fourier) assume um poço PONTUAL (singularidade matemática).
A discretização numérica usa um VOLUME FINITO para o poço.
Essa incompatibilidade causa erros locais que não convergem com refinamento de malha.

CORREÇÃO: Excluir a célula do poço do cálculo do erro, permitindo observar
a convergência O(h²) nas demais regiões do domínio.

Referência: Li, D. (1995) - Petroleum Engineering Handbook
"""

import os
import configparser
import numpy as np
from pathlib import Path


def _load_reservoir_parameters(reservoir_ini_path=None):
    """
    Carrega parâmetros do arquivo reservoir.ini
    
    Args:
        reservoir_ini_path: Caminho para o arquivo .ini (opcional)
                           Se None, procura no diretório pai
    """
    if reservoir_ini_path is None:
        current_dir = Path(__file__).parent
        reservoir_ini_path = current_dir.parent / "reservoir.ini"

    if not Path(reservoir_ini_path).exists():
        raise FileNotFoundError(f"Could not find reservoir.ini at {reservoir_ini_path}")

    config = configparser.ConfigParser()
    config.read(reservoir_ini_path)

    input_section = config["RESERVOIR_INPUT"]
    well_section = config["WELL_1"]

    # Dimensões do reservatório
    a = input_section.getfloat("LX")
    b = input_section.getfloat("LY")
    h = input_section.getfloat("LZ")

    # Posição do poço
    well_x = well_section.getfloat("BLOCK_COORD_X")
    well_y = well_section.getfloat("BLOCK_COORD_Y")

    # Propriedades do fluido e rocha
    Pi = (
        input_section.getfloat("PRESSURE")
        if "PRESSURE" in input_section
        else config["INITIAL_CONDITION"].getfloat("PRESSURE")
    )
    Bo = input_section.getfloat("B")
    mu = input_section.getfloat("MU")
    por = input_section.getfloat("PORO")
    k = input_section.getfloat("KX")
    cf = input_section.getfloat("CFLUID")
    cr = input_section.getfloat("CPORO")
    c = cr + cf

    # Vazão do poço
    well_rate_field_units = abs(well_section.getfloat("VALUE"))
    Qj = well_rate_field_units * 5.614
    Q = -Bo * Qj / 5.614 / h

    # Parâmetros adimensionais
    alpha = 157.952 * (por * c * mu) / k
    beta = 886.905 * (Bo * mu) / k

    params = {
        "a": a,
        "b": b,
        "h": h,
        "well_x": well_x,
        "well_y": well_y,
        "Pi": Pi,
        "Bo": Bo,
        "mu": mu,
        "por": por,
        "k": k,
        "cf": cf,
        "cr": cr,
        "c": c,
        "Q": Q,
        "Qj": Qj,
        "alpha": alpha,
        "beta": beta,
    }

    return params


def _li_solution(x, y, t, params):
    """
    Calcula a solução de Li (séries de Fourier) para pontos (x, y) no tempo t.
    """
    a = params["a"]
    b = params["b"]
    l = params["well_x"]
    q = params["well_y"]
    Pi = params["Pi"]
    Q = params["Q"]
    alpha = params["alpha"]
    beta = params["beta"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if t == 0.0:
        result_shape = np.broadcast(x, y).shape
        return np.full(result_shape, Pi, dtype=float)

    x_bcast, y_bcast = np.broadcast_arrays(x, y)

    series_terms = 100
    m = np.arange(1, series_terms + 1)
    n = np.arange(1, series_terms + 1)
    pi_sq = np.pi**2

    m_col = m.reshape(-1, 1)
    n_col = n.reshape(-1, 1)

    m_op_shape = (-1,) + (1,) * x_bcast.ndim
    n_op_shape = (-1,) + (1,) * y_bcast.ndim
    m_op = m.reshape(m_op_shape)
    n_op = n.reshape(n_op_shape)

    # Termo d (série em x)
    m2_a2 = m_col**2 / a**2
    C_d_m = (
        (1 / (pi_sq * m2_a2))
        * (1 - np.exp(-pi_sq / alpha * m2_a2 * t))
        * np.cos(m_col * np.pi * l / a)
    )
    cos_mx = np.cos(m_op * np.pi * x_bcast / a)
    d = np.einsum("m,m...->...", C_d_m.flatten(), cos_mx)

    # Termo f (série em y)
    n2_b2 = n_col**2 / b**2
    C_f_n = (
        (1 / (pi_sq * n2_b2))
        * (1 - np.exp(-pi_sq / alpha * n2_b2 * t))
        * np.cos(n_col * np.pi * q / b)
    )
    cos_ny = np.cos(n_op * np.pi * y_bcast / b)
    f = np.einsum("n,n...->...", C_f_n.flatten(), cos_ny)

    # Termo g (série dupla em x e y)
    n_row = n.reshape(1, -1)
    lambda_mn_sq = (m_col**2 / a**2) + (n_row**2 / b**2)
    C_mn = (
        (1 / (pi_sq * lambda_mn_sq))
        * (1 - np.exp(-pi_sq / alpha * lambda_mn_sq * t))
        * np.cos(m_col * np.pi * l / a)
        * np.cos(n_row * np.pi * q / b)
    )

    V_mx = np.cos(m_op * np.pi * x_bcast / a)
    W_ny = np.cos(n_op * np.pi * y_bcast / b)
    g = np.einsum("mn,m...,n...->...", C_mn, V_mx, W_ny)

    # Solução completa
    P_result = Pi - beta * Q / (a * b) * (t / alpha + 2 * d + 2 * f + 4 * g)

    return P_result


def _find_well_cell_mask(x, y, params):
    """
    Encontra a célula mais próxima do poço.
    
    Esta função identifica a ÚNICA célula cujo centro está mais próximo
    da posição do poço. Funciona independente do refinamento da malha.
    
    Args:
        x, y: Arrays com coordenadas dos centros das células
        params: Parâmetros do reservatório (contém well_x, well_y)
    
    Returns:
        Boolean array com True apenas para a célula mais próxima do poço
    """
    well_x = params["well_x"]
    well_y = params["well_y"]
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Calcular distância de cada célula ao poço
    dist_sq = (x - well_x)**2 + (y - well_y)**2
    
    # Encontrar a célula mais próxima
    min_idx = np.argmin(dist_sq)
    
    # Criar máscara com apenas essa célula
    mask = np.zeros_like(dist_sq, dtype=bool)
    mask.flat[min_idx] = True
    
    return mask


def analytical(x, y, z, t, exclude_well=True):
    """
    Solução analítica de Li para o problema do poço.
    
    Args:
        x, y, z: Coordenadas dos pontos (z é ignorado para problema 2D)
        t: Tempo
        exclude_well: Se True, retorna NaN para a célula do poço
    
    Returns:
        Pressão nos pontos especificados.
        Para a célula do poço (se exclude_well=True), retorna NaN.
    """
    params = _load_reservoir_parameters()
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = float(t)
    
    # Calcular solução de Li para todos os pontos
    P_result = _li_solution(x, y, t, params)
    
    # Converter para array se necessário para poder modificar
    P_result = np.atleast_1d(np.asarray(P_result, dtype=float))
    
    # Marcar a célula do poço com NaN se exclude_well=True
    if exclude_well and t > 0:
        well_mask = _find_well_cell_mask(x, y, params)
        P_result[well_mask] = np.nan
    
    # Retornar escalar se input era escalar
    if P_result.size == 1:
        return P_result.item()
    
    return P_result


def analytical_full(x, y, z, t):
    """
    Solução analítica de Li SEM exclusão da célula do poço.
    """
    return analytical(x, y, z, t, exclude_well=False)


def get_parameters():
    """Retorna cópia dos parâmetros carregados."""
    return _load_reservoir_parameters().copy()


# =============================================================================
# Testes
# =============================================================================
if __name__ == "__main__":
    print("Testing Li analytical solution with well cell exclusion...")
    print("=" * 70)

    # Simular diferentes refinamentos de malha
    test_cases = [
        ("25x25", 25, 25),
        ("50x50", 50, 50),
        ("100x100", 100, 100),
    ]
    
    # Parâmetros do domínio (hardcoded para teste)
    Lx, Ly = 2000.0, 2000.0
    well_x, well_y = 1000.0, 1000.0
    
    for name, nx, ny in test_cases:
        print(f"\n{name} mesh:")
        
        dx, dy = Lx / nx, Ly / ny
        
        # Gerar centros das células
        x_centers = np.array([(i + 0.5) * dx for i in range(nx)])
        y_centers = np.array([(j + 0.5) * dy for j in range(ny)])
        
        X, Y = np.meshgrid(x_centers, y_centers)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Encontrar célula mais próxima do poço
        dist_sq = (x_flat - well_x)**2 + (y_flat - well_y)**2
        min_idx = np.argmin(dist_sq)
        min_dist = np.sqrt(dist_sq[min_idx])
        
        well_cell_x = x_flat[min_idx]
        well_cell_y = y_flat[min_idx]
        
        print(f"  Cell size: dx={dx}, dy={dy}")
        print(f"  Well at: ({well_x}, {well_y})")
        print(f"  Closest cell center: ({well_cell_x}, {well_cell_y})")
        print(f"  Distance to well: {min_dist:.2f}")
        
        # Verificar quantas células têm NaN
        params = {
            "a": Lx, "b": Ly, "h": 1.0,
            "well_x": well_x, "well_y": well_y,
            "Pi": 2000.0, "Q": -1.0,
            "alpha": 1.0, "beta": 1.0,
        }
        
        mask = _find_well_cell_mask(x_flat, y_flat, params)
        n_excluded = np.sum(mask)
        print(f"  Cells excluded: {n_excluded}")