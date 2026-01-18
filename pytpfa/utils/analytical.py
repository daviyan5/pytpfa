"""
analytical.py - VERSÃO CORRIGIDA

Este módulo gera funções analíticas para o método de soluções manufaturadas.
A formulação corresponde exatamente à equação implementada no solver.

A equação do solver (em unidades de campo) é:
    V/αc · ∂/∂t[φ/B] - βc · ∇·[k/(μB) · ∇p] = q_sc

Onde:
    αc = 5.615 (conversão bbl → ft³)
    βc = 1.127 (fator de transmissibilidade em unidades de campo)
    B = B_ref / (1 + c_f · (p - p_ref))  (formation volume factor)
    φ = φ_ref · (1 + c_φ · (p - p_ref))  (porosidade)
"""

import numpy as np
import sympy as sp

ALPHA_C = 5.615
BETA_C = 1.127

def _ensure_sympy_expr(expr):
    """
    Garante que expr seja uma expressão SymPy.
    Se for string, converte para expressão SymPy.
    """
    if isinstance(expr, str):
        x, y, z, t = sp.symbols("x y z t")
        expr_clean = expr.lower().replace("^", "**")
        return sp.sympify(expr_clean, locals={"x": x, "y": y, "z": z, "t": t})
    return expr


def analytical(expr, params=None):
    """
    Converte uma expressão SymPy p(x,y,z,t) em uma função numérica.
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), expr, "numpy")
    return func


def initial_condition(expr, params=None):
    """
    Condição inicial: p(x, y, z, t=0)
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    initial = expr.subs(t_sym, 0)
    
    if initial.is_constant():
        const_val = float(initial)
        return lambda x, y, z: np.full_like(np.asarray(x, dtype=float), const_val)
    
    func = sp.lambdify((x_sym, y_sym, z_sym), initial, "numpy")
    return func


def dirichlet(expr, params=None):
    """
    Condição de contorno de Dirichlet: p = p_boundary
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), expr, "numpy")
    return func


def neumann(expr, params):
    """
    Condição de contorno de Neumann: derivada normal da pressão (∂p/∂n)
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym, nx_sym, ny_sym, nz_sym = sp.symbols("x y z t nx ny nz")
    
    dpdx = sp.diff(expr, x_sym)
    dpdy = sp.diff(expr, y_sym)
    dpdz = sp.diff(expr, z_sym)
    
    normal_derivative = dpdx * nx_sym + dpdy * ny_sym + dpdz * nz_sym
    
    func = sp.lambdify(
        (x_sym, y_sym, z_sym, t_sym, nx_sym, ny_sym, nz_sym), 
        normal_derivative, 
        "numpy"
    )
    return func


def source_term(expr, params):
    """
    Calcula o termo fonte q para que a solução analítica seja satisfeita.
    
    A equação do solver é:
        (1/αc) · ∂/∂t[φ/B] - βc · ∇·[k/(μB) · ∇p] = q_sc / V
    
    PARÂMETROS ESPERADOS:
        phi_ref: porosidade de referência
        pore_compressibility: c_φ
        fluid_compressibility: c_f
        formation_volume_factor: B_ref
        permeability: k (escalar ou tupla)
        viscosity: μ
        initial_pressure: p_ref (pode ser escalar, callable, ou None)
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    
    # Extrair parâmetros
    phi_ref = params["phi_ref"]
    c_phi = params["pore_compressibility"]
    c_f = params["fluid_compressibility"]
    B_ref = params["formation_volume_factor"]
    mu = params["viscosity"]
    
    # Permeabilidade (pode ser escalar ou tensor)
    k = params["permeability"]
    if isinstance(k, (list, tuple)):
        kx, ky, kz = k
    else:
        kx = ky = kz = k
    
    # Pressão de referência
    p_ref_val = params.get("initial_pressure", None)
    if p_ref_val is None:
        # Se não especificado, usa a solução em t=0
        p_ref = expr.subs(t_sym, 0)
    elif callable(p_ref_val):
        # Se é uma função, usa a solução em t=0 (não podemos avaliar função simbólica)
        p_ref = expr.subs(t_sym, 0)
    elif isinstance(p_ref_val, (int, float)):
        p_ref = sp.Float(p_ref_val)
    else:
        p_ref = p_ref_val
    
    # Formation Volume Factor: B = B_ref / (1 + c_f · (p - p_ref))
    B = B_ref / (1 + c_f * (expr - p_ref))
    
    # Porosidade: φ = φ_ref · (1 + c_φ · (p - p_ref))
    phi = phi_ref * (1 + c_phi * (expr - p_ref))
    
    # TERMO DE ACUMULAÇÃO: (1/αc) · ∂(φ/B)/∂t
    phi_over_B = phi / B
    accumulation = sp.diff(phi_over_B, t_sym) / ALPHA_C
    
    # TERMO DE FLUXO: βc · ∇·[k/(μB) · ∇p]
    dpdx = sp.diff(expr, x_sym)
    dpdy = sp.diff(expr, y_sym)
    dpdz = sp.diff(expr, z_sym)
    
    flux_x = BETA_C * (kx / (mu * B)) * dpdx
    flux_y = BETA_C * (ky / (mu * B)) * dpdy
    flux_z = BETA_C * (kz / (mu * B)) * dpdz
    
    div_flux = sp.diff(flux_x, x_sym) + sp.diff(flux_y, y_sym) + sp.diff(flux_z, z_sym)
    
    source = accumulation - div_flux
    source = sp.simplify(source)
    
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), source, "numpy")
    return func