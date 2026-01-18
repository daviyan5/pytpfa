"""
analytical.py - VERSÃO CORRIGIDA

Este módulo gera funções analíticas para o método de soluções manufaturadas.
A formulação corresponde EXATAMENTE à discretização implementada no solver.

============================================================================
EQUAÇÃO DO SOLVER (forma contínua que é discretizada):
============================================================================

    (1/αc) · ∂(φ/B)/∂t = βc · ∇·[k/(μB) · ∇p] + q

Onde:
    αc = 5.615 (conversão bbl → ft³)
    βc = 1.127 (fator de transmissibilidade em unidades de campo)

============================================================================
LINEARIZAÇÃO DO TERMO DE ACUMULAÇÃO:
============================================================================

O solver usa a seguinte linearização para ∂(φ/B)/∂t:

    ∂(φ/B)/∂t ≈ [φ_ref · c_φ / B + φ · c_f / B_ref] · ∂p/∂t

Esta linearização vem de:
    
    ∂(φ/B)/∂t = (∂φ/∂t)/B - φ·(∂B/∂t)/B²
    
    Com:
        ∂φ/∂t = φ_ref · c_φ · ∂p/∂t
        ∂B/∂t ≈ -B²/B_ref · c_f · ∂p/∂t

Resultando no coeficiente de acumulação γ:
    
    γ = (V/αc) · [φ_ref · c_φ / B + φ · c_f / B_ref]

E o termo de acumulação:
    
    (V/αc) · ∂(φ/B)/∂t ≈ γ · ∂p/∂t

============================================================================
TERMO FONTE PARA MMS:
============================================================================

Para que p(x,y,z,t) seja solução exata, o termo fonte q deve satisfazer:

    q = (1/αc) · [φ_ref · c_φ / B + φ · c_f / B_ref] · ∂p/∂t 
        - βc · ∇·[k/(μB) · ∇p]

Onde:
    B = B_ref / (1 + c_f · (p - p_ref))
    φ = φ_ref · (1 + c_φ · (p - p_ref))

============================================================================
"""

import numpy as np
import sympy as sp

# Constantes de conversão de unidades de campo
ALPHA_C = 5.615  # Conversão bbl → ft³
BETA_C = 1.127   # Fator de transmissibilidade


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
    
    Parâmetros:
        expr: Expressão SymPy ou string representando p(x,y,z,t)
        params: Dicionário de parâmetros (não utilizado, mantido por compatibilidade)
    
    Retorna:
        Função f(x, y, z, t) -> pressão
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), expr, "numpy")
    return func


def initial_condition(expr, params=None):
    """
    Condição inicial: p(x, y, z, t=0)
    
    Parâmetros:
        expr: Expressão SymPy ou string representando p(x,y,z,t)
        params: Dicionário de parâmetros (não utilizado)
    
    Retorna:
        Função f(x, y, z) -> pressão inicial
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
    
    Parâmetros:
        expr: Expressão SymPy ou string representando p(x,y,z,t)
        params: Dicionário de parâmetros (não utilizado)
    
    Retorna:
        Função f(x, y, z, t) -> pressão na fronteira
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), expr, "numpy")
    return func


def neumann(expr, params):
    """
    Condição de contorno de Neumann: ∂p/∂n (derivada normal da pressão)
    
    Parâmetros:
        expr: Expressão SymPy ou string representando p(x,y,z,t)
        params: Dicionário de parâmetros (não utilizado diretamente)
    
    Retorna:
        Função f(x, y, z, t, nx, ny, nz) -> derivada normal
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
    Calcula o termo fonte q para que a solução analítica p(x,y,z,t) seja satisfeita.
    
    IMPORTANTE: Este cálculo usa a MESMA LINEARIZAÇÃO do solver para o termo
    de acumulação, garantindo consistência entre a solução manufaturada e a
    discretização numérica.
    
    A equação resolvida pelo solver é:
    
        (1/αc) · ∂(φ/B)/∂t = βc · ∇·[k/(μB) · ∇p] + q
    
    Com a linearização:
    
        ∂(φ/B)/∂t ≈ [φ_ref · c_φ / B + φ · c_f / B_ref] · ∂p/∂t
    
    Portanto, o termo fonte é:
    
        q = (1/αc) · [φ_ref · c_φ / B + φ · c_f / B_ref] · ∂p/∂t 
            - βc · ∇·[k/(μB) · ∇p]
    
    ============================================================================
    PARÂMETROS ESPERADOS:
    ============================================================================
        phi_ref: porosidade de referência (φ_ref)
        pore_compressibility: compressibilidade da rocha (c_φ) [1/psi]
        fluid_compressibility: compressibilidade do fluido (c_f) [1/psi]
        formation_volume_factor: fator volume de formação de referência (B_ref) [bbl/STB]
        permeability: permeabilidade (k) - escalar ou tupla (kx, ky, kz) [mD]
        viscosity: viscosidade do fluido (μ) [cp]
        initial_pressure: pressão de referência (p_ref) - escalar, callable, ou None [psi]
    
    ============================================================================
    CASOS ESPECIAIS:
    ============================================================================
    
    1. FLUIDO E ROCHA INCOMPRESSÍVEIS (c_f = c_φ = 0):
       - B = B_ref (constante)
       - φ = φ_ref (constante)
       - Termo de acumulação = 0
       - Problema se torna estacionário (solução não depende de condição inicial)
       - Útil para verificação de transmissibilidade
    
    2. SOLUÇÃO ESTACIONÁRIA (∂p/∂t = 0):
       - Termo de acumulação = 0 independente das compressibilidades
       - q = -βc · ∇·[k/(μB) · ∇p]
       - Útil para verificação do operador de difusão
    
    3. SOLUÇÃO LINEAR EM ESPAÇO (p = a·x + b·y + c·z + f(t)):
       - ∇²p = 0
       - Se B = constante: ∇·[k/(μB) · ∇p] = 0
       - Solução exata para TPFA em malhas uniformes
    
    4. SOLUÇÃO QUADRÁTICA EM ESPAÇO (p = a·x² + b·y² + c·z² + ...):
       - ∇²p = constante
       - Solução exata para TPFA em malhas uniformes com k constante
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    
    # =========================================================================
    # Extrair parâmetros
    # =========================================================================
    phi_ref = params["phi_ref"]
    c_phi = params["pore_compressibility"]
    c_f = params["fluid_compressibility"]
    B_ref = params["formation_volume_factor"]
    mu = params["viscosity"]
    
    # Permeabilidade (pode ser escalar ou tensor diagonal)
    k = params["permeability"]
    if isinstance(k, (list, tuple)):
        kx, ky, kz = k
    else:
        kx = ky = kz = k
    
    # =========================================================================
    # Pressão de referência (p_ref)
    # =========================================================================
    # O solver usa INITIAL_PRESSURE como p_ref, que é a solução avaliada em t=0
    # nos centros das células. Para MMS, usamos a expressão simbólica em t=0.
    p_ref_val = params.get("initial_pressure", None)
    
    if p_ref_val is None:
        # Se não especificado, usa a solução em t=0
        p_ref = expr.subs(t_sym, 0)
    elif callable(p_ref_val):
        # Se é uma função, avalia simbolicamente
        # Isso assume que a função pode ser expressa como p_ref(x,y,z)
        p_ref = expr.subs(t_sym, 0)
    elif isinstance(p_ref_val, (int, float)):
        # Valor escalar constante
        p_ref = sp.Float(p_ref_val)
    else:
        # Assume que já é uma expressão SymPy
        p_ref = p_ref_val
    
    # =========================================================================
    # Propriedades do fluido e da rocha (dependentes da pressão)
    # =========================================================================
    # Formation Volume Factor: B = B_ref / (1 + c_f · (p - p_ref))
    B = B_ref / (1 + c_f * (expr - p_ref))
    
    # Porosidade: φ = φ_ref · (1 + c_φ · (p - p_ref))
    phi = phi_ref * (1 + c_phi * (expr - p_ref))
    
    # =========================================================================
    # TERMO DE ACUMULAÇÃO (usando a linearização do solver)
    # =========================================================================
    # O solver usa:
    #   γ = (V/αc) · [φ_ref · c_φ / B + φ · c_f / B_ref]
    #   termo_acumulacao = γ · ∂p/∂t / V = (1/αc) · [φ_ref · c_φ / B + φ · c_f / B_ref] · ∂p/∂t
    #
    # NOTA: Esta é a linearização correta que corresponde exatamente ao solver.
    # NÃO usamos sp.diff(phi/B, t_sym) diretamente, pois isso daria a derivada
    # exata, que é diferente da linearização usada pelo solver.
    
    dpdt = sp.diff(expr, t_sym)
    
    # Coeficiente de acumulação linearizado (por unidade de volume)
    accumulation_coeff = (phi_ref * c_phi / B) + (phi * c_f / B_ref)
    
    # Termo de acumulação completo
    accumulation = accumulation_coeff * dpdt / ALPHA_C
    
    # =========================================================================
    # TERMO DE FLUXO (difusão)
    # =========================================================================
    # ∇·[k/(μB) · ∇p] = ∂/∂x[kx/(μB) · ∂p/∂x] + ∂/∂y[ky/(μB) · ∂p/∂y] + ∂/∂z[kz/(μB) · ∂p/∂z]
    
    dpdx = sp.diff(expr, x_sym)
    dpdy = sp.diff(expr, y_sym)
    dpdz = sp.diff(expr, z_sym)
    
    # Fluxos em cada direção (incluindo βc)
    flux_x = BETA_C * (kx / (mu * B)) * dpdx
    flux_y = BETA_C * (ky / (mu * B)) * dpdy
    flux_z = BETA_C * (kz / (mu * B)) * dpdz
    
    # Divergência do fluxo
    div_flux = sp.diff(flux_x, x_sym) + sp.diff(flux_y, y_sym) + sp.diff(flux_z, z_sym)
    
    # =========================================================================
    # TERMO FONTE
    # =========================================================================
    # Da equação: (1/αc) · ∂(φ/B)/∂t = βc · ∇·[k/(μB) · ∇p] + q
    # Isolando q: q = acumulação - difusão
    
    source = accumulation - div_flux
    
    # Simplificar a expressão (pode demorar para expressões complexas)
    source = sp.simplify(source)
    
    # Converter para função numérica
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), source, "numpy")
    
    return func

def get_accumulation_coefficient(expr, params):
    """
    Retorna apenas o coeficiente de acumulação linearizado.
    Útil para debug e verificação.
    
    Retorna:
        Função f(x, y, z, t) -> coeficiente γ/(V·αc)
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    
    phi_ref = params["phi_ref"]
    c_phi = params["pore_compressibility"]
    c_f = params["fluid_compressibility"]
    B_ref = params["formation_volume_factor"]
    
    p_ref_val = params.get("initial_pressure", None)
    if p_ref_val is None:
        p_ref = expr.subs(t_sym, 0)
    elif isinstance(p_ref_val, (int, float)):
        p_ref = sp.Float(p_ref_val)
    else:
        p_ref = expr.subs(t_sym, 0)
    
    B = B_ref / (1 + c_f * (expr - p_ref))
    phi = phi_ref * (1 + c_phi * (expr - p_ref))
    
    coeff = (phi_ref * c_phi / B) + (phi * c_f / B_ref)
    
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), coeff, "numpy")
    return func


def get_diffusion_term(expr, params):
    """
    Retorna apenas o termo de difusão: βc · ∇·[k/(μB) · ∇p]
    Útil para debug e verificação.
    
    Retorna:
        Função f(x, y, z, t) -> termo de difusão
    """
    expr = _ensure_sympy_expr(expr)
    x_sym, y_sym, z_sym, t_sym = sp.symbols("x y z t")
    
    phi_ref = params["phi_ref"]
    c_phi = params["pore_compressibility"]
    c_f = params["fluid_compressibility"]
    B_ref = params["formation_volume_factor"]
    mu = params["viscosity"]
    
    k = params["permeability"]
    if isinstance(k, (list, tuple)):
        kx, ky, kz = k
    else:
        kx = ky = kz = k
    
    p_ref_val = params.get("initial_pressure", None)
    if p_ref_val is None:
        p_ref = expr.subs(t_sym, 0)
    elif isinstance(p_ref_val, (int, float)):
        p_ref = sp.Float(p_ref_val)
    else:
        p_ref = expr.subs(t_sym, 0)
    
    B = B_ref / (1 + c_f * (expr - p_ref))
    
    dpdx = sp.diff(expr, x_sym)
    dpdy = sp.diff(expr, y_sym)
    dpdz = sp.diff(expr, z_sym)
    
    flux_x = BETA_C * (kx / (mu * B)) * dpdx
    flux_y = BETA_C * (ky / (mu * B)) * dpdy
    flux_z = BETA_C * (kz / (mu * B)) * dpdz
    
    div_flux = sp.diff(flux_x, x_sym) + sp.diff(flux_y, y_sym) + sp.diff(flux_z, z_sym)
    
    func = sp.lambdify((x_sym, y_sym, z_sym, t_sym), sp.simplify(div_flux), "numpy")
    return func


def verify_source_term(expr, params, x, y, z, t, tolerance=1e-10):
    """
    Verifica se o termo fonte calculado é consistente.
    
    Para soluções simples (linear, quadrática), o termo fonte deve ser:
    - Zero para soluções lineares com B constante
    - Constante para soluções quadráticas com B constante
    
    Parâmetros:
        expr: Expressão da solução
        params: Dicionário de parâmetros
        x, y, z, t: Arrays de coordenadas e tempo para teste
        tolerance: Tolerância para verificação
    
    Retorna:
        dict com informações de verificação
    """
    source_func = source_term(expr, params)
    q_values = source_func(x, y, z, t)
    
    return {
        "min": np.min(q_values),
        "max": np.max(q_values),
        "mean": np.mean(q_values),
        "std": np.std(q_values),
        "is_constant": np.std(q_values) < tolerance,
        "is_zero": np.abs(np.mean(q_values)) < tolerance and np.std(q_values) < tolerance,
    }

import numpy as np
import sympy as sp

def test_linear_incompressible():
    """
    Teste 1: Solução linear com fluido/rocha incompressíveis
    
    p(x,y,z,t) = x + 2y + 3z + 100
    
    Esperado:
    - c_f = c_phi = 0 → B = B_ref, φ = φ_ref (constantes)
    - Coeficiente de acumulação = 0
    - ∇p = (1, 2, 3), constante
    - ∇²p = 0
    - ∇·[k/(μB) · ∇p] = 0 para k, B constantes
    - Termo fonte q = 0
    """
    print("=" * 70)
    print("TESTE 1: Solução linear, fluido/rocha incompressíveis")
    print("=" * 70)
    
    x, y, z, t = sp.symbols("x y z t")
    p_expr = x + 2*y + 3*z + 100
    
    params = {
        "phi_ref": 0.2,
        "pore_compressibility": 0.0,      # INCOMPRESSÍVEL
        "fluid_compressibility": 0.0,     # INCOMPRESSÍVEL
        "formation_volume_factor": 1.0,
        "permeability": 100.0,
        "viscosity": 1.0,
        "initial_pressure": None,
    }
    
    # Gerar pontos de teste
    xx = np.linspace(0, 200, 5)
    yy = np.linspace(0, 100, 5)
    zz = np.linspace(0, 20, 5)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
    T = np.ones_like(X) * 10.0
    
    # Calcular termo fonte
    q_func = source_term(p_expr, params)
    q_values = q_func(X, Y, Z, T)
    
    print(f"Solução: p = x + 2y + 3z + 100")
    print(f"Parâmetros: c_f = c_φ = 0 (incompressível)")
    print(f"Termo fonte q:")
    print(f"  min = {np.min(q_values):.2e}")
    print(f"  max = {np.max(q_values):.2e}")
    print(f"  mean = {np.mean(q_values):.2e}")
    
    assert np.allclose(q_values, 0, atol=1e-12), "FALHA: Termo fonte deveria ser zero!"
    print("✓ PASSOU: Termo fonte é zero como esperado\n")


def test_quadratic_incompressible():
    """
    Teste 2: Solução quadrática com fluido/rocha incompressíveis
    
    p(x,y,z,t) = x² + y² + z²
    
    Esperado:
    - c_f = c_phi = 0 → B = B_ref, φ = φ_ref (constantes)
    - ∇p = (2x, 2y, 2z)
    - ∇²p = 6
    - ∇·[k/(μB) · ∇p] = k/(μB) · 6 = constante
    - Termo fonte q = -βc · k/(μB) · 6 = constante
    """
    print("=" * 70)
    print("TESTE 2: Solução quadrática, fluido/rocha incompressíveis")
    print("=" * 70)
    
    x, y, z, t = sp.symbols("x y z t")
    p_expr = x**2 + y**2 + z**2
    
    params = {
        "phi_ref": 0.2,
        "pore_compressibility": 0.0,
        "fluid_compressibility": 0.0,
        "formation_volume_factor": 1.0,
        "permeability": 100.0,
        "viscosity": 1.0,
        "initial_pressure": None,
    }
    
    # Calcular termo fonte esperado
    # q = -βc · k/(μB) · ∇²p = -1.127 · 100/(1·1) · 6 = -676.2
    k = params["permeability"]
    mu = params["viscosity"]
    B = params["formation_volume_factor"]
    laplacian_p = 6.0  # ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z² = 2 + 2 + 2 = 6
    
    q_expected = -BETA_C * (k / (mu * B)) * laplacian_p
    
    # Gerar pontos de teste
    xx = np.linspace(0, 200, 5)
    yy = np.linspace(0, 100, 5)
    zz = np.linspace(0, 20, 5)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
    T = np.ones_like(X) * 10.0
    
    # Calcular termo fonte
    q_func = source_term(p_expr, params)
    q_values = q_func(X, Y, Z, T)
    
    print(f"Solução: p = x² + y² + z²")
    print(f"Parâmetros: c_f = c_φ = 0 (incompressível)")
    print(f"Termo fonte q esperado: {q_expected:.4f}")
    print(f"Termo fonte q calculado:")
    print(f"  min = {np.min(q_values):.4f}")
    print(f"  max = {np.max(q_values):.4f}")
    print(f"  mean = {np.mean(q_values):.4f}")
    print(f"  std = {np.std(q_values):.2e}")
    
    assert np.allclose(q_values, q_expected, rtol=1e-10), "FALHA: Termo fonte não é constante!"
    print("✓ PASSOU: Termo fonte é constante como esperado\n")


def test_linear_compressible():
    """
    Teste 3: Solução linear com fluido/rocha COMPRESSÍVEIS
    
    p(x,y,z,t) = x + 100
    
    IMPORTANTE: Com compressibilidade, mesmo uma solução linear em espaço
    resulta em termo fonte não-zero, pois:
    - B = B_ref / (1 + c_f · (p - p_ref)) depende de x
    - ∇·[k/(μB) · ∇p] = k/(μ) · ∂/∂x[(1/B) · ∂p/∂x] ≠ 0
    """
    print("=" * 70)
    print("TESTE 3: Solução linear, fluido/rocha COMPRESSÍVEIS")
    print("=" * 70)
    
    x, y, z, t = sp.symbols("x y z t")
    p_expr = x + 100
    
    params = {
        "phi_ref": 0.2,
        "pore_compressibility": 1e-5,     # COMPRESSÍVEL
        "fluid_compressibility": 1e-5,    # COMPRESSÍVEL
        "formation_volume_factor": 1.0,
        "permeability": 100.0,
        "viscosity": 1.0,
        "initial_pressure": None,
    }
    
    # Gerar pontos de teste
    xx = np.linspace(0, 200, 10)
    yy = np.linspace(0, 100, 5)
    zz = np.linspace(0, 20, 5)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
    T = np.ones_like(X) * 10.0
    
    # Calcular termo fonte
    q_func = source_term(p_expr, params)
    q_values = q_func(X, Y, Z, T)
    
    print(f"Solução: p = x + 100")
    print(f"Parâmetros: c_f = c_φ = 1e-5 (compressível)")
    print(f"Termo fonte q:")
    print(f"  min = {np.min(q_values):.6f}")
    print(f"  max = {np.max(q_values):.6f}")
    print(f"  mean = {np.mean(q_values):.6f}")
    print(f"  std = {np.std(q_values):.6f}")
    
    # Com compressibilidade, termo fonte NÃO é zero
    print(f"NOTA: Com compressibilidade, termo fonte não é zero nem constante.")
    print("      Isso é esperado e o solver deve reproduzir a solução exata")
    print("      SE o termo fonte for calculado consistentemente.\n")


def test_transient_incompressible():
    """
    Teste 4: Solução transiente com fluido/rocha incompressíveis
    
    p(x,y,z,t) = x + t
    
    Esperado:
    - c_f = c_phi = 0 → coeficiente de acumulação = 0
    - ∂p/∂t = 1
    - Termo de acumulação = 0 · 1 = 0
    - ∇²p = 0
    - Termo fonte q = 0
    
    NOTA: Com c_f = c_phi = 0, não há termo de acumulação!
    O problema se torna quase-estático.
    """
    print("=" * 70)
    print("TESTE 4: Solução transiente, fluido/rocha incompressíveis")
    print("=" * 70)
    
    x, y, z, t = sp.symbols("x y z t")
    p_expr = x + t
    
    params = {
        "phi_ref": 0.2,
        "pore_compressibility": 0.0,
        "fluid_compressibility": 0.0,
        "formation_volume_factor": 1.0,
        "permeability": 100.0,
        "viscosity": 1.0,
        "initial_pressure": None,
    }
    
    # Gerar pontos de teste
    xx = np.linspace(0, 200, 5)
    yy = np.linspace(0, 100, 5)
    zz = np.linspace(0, 20, 5)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
    T = np.ones_like(X) * 10.0
    
    # Calcular termo fonte
    q_func = source_term(p_expr, params)
    q_values = q_func(X, Y, Z, T)
    
    print(f"Solução: p = x + t")
    print(f"Parâmetros: c_f = c_φ = 0 (incompressível)")
    print(f"∂p/∂t = 1, mas coeficiente de acumulação = 0")
    print(f"Termo fonte q:")
    print(f"  min = {np.min(q_values):.2e}")
    print(f"  max = {np.max(q_values):.2e}")
    
    assert np.allclose(q_values, 0, atol=1e-12), "FALHA: Termo fonte deveria ser zero!"
    print("✓ PASSOU: Termo fonte é zero (sem acumulação para incompressível)\n")


def test_transient_compressible():
    """
    Teste 5: Solução transiente com fluido/rocha compressíveis
    
    p(x,y,z,t) = 100 + t  (uniforme em espaço, variável no tempo)
    
    Esperado:
    - ∇p = 0 → termo de difusão = 0
    - ∂p/∂t = 1
    - p - p_ref = t (onde p_ref = p(t=0) = 100)
    - B = B_ref / (1 + c_f · t)
    - φ = φ_ref · (1 + c_φ · t)
    - Coef. acumulação = φ_ref · c_φ / B + φ · c_f / B_ref
    - Termo fonte q = (1/αc) · coef · 1 = coef / αc
    """
    print("=" * 70)
    print("TESTE 5: Solução transiente uniforme, fluido/rocha compressíveis")
    print("=" * 70)
    
    x, y, z, t = sp.symbols("x y z t")
    p_expr = 100 + t
    
    phi_ref = 0.2
    c_phi = 1e-5
    c_f = 1e-5
    B_ref = 1.0
    
    params = {
        "phi_ref": phi_ref,
        "pore_compressibility": c_phi,
        "fluid_compressibility": c_f,
        "formation_volume_factor": B_ref,
        "permeability": 100.0,
        "viscosity": 1.0,
        "initial_pressure": None,
    }
    
    # Calcular termo fonte esperado em t = 10
    t_val = 10.0
    p_val = 100 + t_val  # = 110
    p_ref_val = 100      # = p(t=0)
    
    B_val = B_ref / (1 + c_f * (p_val - p_ref_val))  # = 1/(1 + 1e-5 * 10)
    phi_val = phi_ref * (1 + c_phi * (p_val - p_ref_val))  # = 0.2 * (1 + 1e-5 * 10)
    
    accum_coeff = (phi_ref * c_phi / B_val) + (phi_val * c_f / B_ref)
    dpdt = 1.0
    
    q_expected = accum_coeff * dpdt / ALPHA_C
    
    # Gerar pontos de teste (todos no mesmo x,y,z já que p é uniforme)
    X = np.array([100.0])
    Y = np.array([50.0])
    Z = np.array([10.0])
    T = np.array([t_val])
    
    # Calcular termo fonte
    q_func = source_term(p_expr, params)
    q_values = q_func(X, Y, Z, T)
    
    print(f"Solução: p = 100 + t")
    print(f"Parâmetros: c_f = c_φ = 1e-5 (compressível)")
    print(f"Em t = {t_val}:")
    print(f"  p = {p_val}, p_ref = {p_ref_val}")
    print(f"  B = {B_val:.6f}")
    print(f"  φ = {phi_val:.6f}")
    print(f"  Coef. acumulação = {accum_coeff:.2e}")
    print(f"  Termo fonte q esperado = {q_expected:.6e}")
    print(f"  Termo fonte q calculado = {q_values[0]:.6e}")
    
    assert np.allclose(q_values, q_expected, rtol=1e-10), "FALHA: Termo fonte incorreto!"
    print("✓ PASSOU: Termo fonte correto para solução transiente compressível\n")


def test_accumulation_coefficient():
    """
    Teste 6: Verificar coeficiente de acumulação diretamente
    """
    print("=" * 70)
    print("TESTE 6: Verificação do coeficiente de acumulação")
    print("=" * 70)
    
    x, y, z, t = sp.symbols("x y z t")
    p_expr = 100 + x + t
    
    phi_ref = 0.2
    c_phi = 1e-5
    c_f = 1e-5
    B_ref = 1.0
    
    params = {
        "phi_ref": phi_ref,
        "pore_compressibility": c_phi,
        "fluid_compressibility": c_f,
        "formation_volume_factor": B_ref,
        "permeability": 100.0,
        "viscosity": 1.0,
        "initial_pressure": None,
    }
    
    # Testar em um ponto específico
    x_val, y_val, z_val, t_val = 50.0, 50.0, 10.0, 5.0
    
    p_val = 100 + x_val + t_val  # = 155
    p_ref_val = 100 + x_val      # = 150 (em t=0)
    
    B_val = B_ref / (1 + c_f * (p_val - p_ref_val))
    phi_val = phi_ref * (1 + c_phi * (p_val - p_ref_val))
    
    expected_coeff = (phi_ref * c_phi / B_val) + (phi_val * c_f / B_ref)
    
    coeff_func = get_accumulation_coefficient(p_expr, params)
    calculated_coeff = coeff_func(x_val, y_val, z_val, t_val)
    
    print(f"Solução: p = 100 + x + t")
    print(f"Ponto: (x={x_val}, y={y_val}, z={z_val}, t={t_val})")
    print(f"p = {p_val}, p_ref = {p_ref_val}")
    print(f"Coeficiente esperado: {expected_coeff:.2e}")
    print(f"Coeficiente calculado: {calculated_coeff:.2e}")
    
    assert np.allclose(calculated_coeff, expected_coeff, rtol=1e-10), "FALHA!"
    print("✓ PASSOU\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTES DO ANALYTICAL.PY")
    print("=" * 70 + "\n")
    
    test_linear_incompressible()
    test_quadratic_incompressible()
    test_linear_compressible()
    test_transient_incompressible()
    test_transient_compressible()
    test_accumulation_coefficient()