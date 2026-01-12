#!/usr/bin/env python3
"""
Solução Analítica 3D - Extensão do Case Study 4 de Li (2012)

Equação resolvida:
    ∂P/∂t = η∇²P + (q·S/V)·δ(r - r_poço)

Solução:
    P = Pi + (q·S/V)·[t + (1/η)·Σ_séries]
"""

import configparser
import numpy as np
from pathlib import Path

_cache = None


def _load_params():
    global _cache
    if _cache is not None:
        return _cache

    ini_path = Path(__file__).parent.parent / "reservoir.ini"
    config = configparser.ConfigParser()
    config.read(ini_path)

    inp = config["RESERVOIR_INPUT"]
    well = config["WELL_1"]

    # Geometria
    a = inp.getfloat("LX")
    b = inp.getfloat("LY")
    c = inp.getfloat("LZ")

    # Posição do poço
    l = well.getfloat("BLOCK_COORD_X")
    m = well.getfloat("BLOCK_COORD_Y")
    p = well.getfloat("BLOCK_COORD_Z")

    # Propriedades
    Pi = config["INITIAL_CONDITION"].getfloat("PRESSURE")
    phi = inp.getfloat("PORO")
    k = inp.getfloat("KX")
    mu = inp.getfloat("MU")
    B = inp.getfloat("B")
    c_t = inp.getfloat("CPORO") + inp.getfloat("CFLUID")

    # Vazão
    q = well.getfloat("VALUE")

    # Coeficientes (consistentes com TPFASolver)
    ALPHA_C = 5.615
    BETA_C = 1.127
    eta = (BETA_C * ALPHA_C * k) / (phi * mu * c_t)
    S = (ALPHA_C * B) / (phi * c_t)

    _cache = {"a": a, "b": b, "c": c, "l": l, "m": m, "p": p, "Pi": Pi, "q": q, "eta": eta, "S": S}
    return _cache


def analytical(x, y, z, t):
    """
    P(x,y,z,t) = Pi + (q·S/V)·[t + (1/η)·(2d + 2f + 2h + 4gxy + 4gxz + 4gyz + 8gxyz)]
    """
    par = _load_params()
    a, b, c = par["a"], par["b"], par["c"]
    l, m, p = par["l"], par["m"], par["p"]
    Pi, q, eta, S = par["Pi"], par["q"], par["eta"], par["S"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = float(t)

    if t == 0.0:
        shape = np.broadcast(x, y, z).shape
        return np.full(shape, Pi) if shape else Pi

    V = a * b * c
    N = 30  # termos da série
    pi2 = np.pi**2

    # Índices
    i = np.arange(1, N + 1)
    j = np.arange(1, N + 1)
    k_idx = np.arange(1, N + 1)

    # Função auxiliar: (1 - exp(-π²ηλ²t)) / (π²λ²)
    def E(lam2):
        return (1.0 - np.exp(-pi2 * eta * lam2 * t)) / (pi2 * lam2)

    # Broadcast arrays
    x, y, z = np.broadcast_arrays(x, y, z)
    flat_x, flat_y, flat_z = x.ravel(), y.ravel(), z.ravel()
    n_pts = flat_x.size

    # Pré-calcular cossenos no poço
    cos_il = np.cos(i * np.pi * l / a)
    cos_jm = np.cos(j * np.pi * m / b)
    cos_kp = np.cos(k_idx * np.pi * p / c)

    # λ² para séries 1D
    lam2_i = (i / a) ** 2
    lam2_j = (j / b) ** 2
    lam2_k = (k_idx / c) ** 2

    # Cossenos para todos os pontos: shape (N, n_pts)
    cos_ix = np.cos(np.outer(i * np.pi / a, flat_x))
    cos_jy = np.cos(np.outer(j * np.pi / b, flat_y))
    cos_kz = np.cos(np.outer(k_idx * np.pi / c, flat_z))

    # === Séries 1D ===
    d = np.sum((E(lam2_i) * cos_il)[:, None] * cos_ix, axis=0)
    f = np.sum((E(lam2_j) * cos_jm)[:, None] * cos_jy, axis=0)
    h = np.sum((E(lam2_k) * cos_kp)[:, None] * cos_kz, axis=0)

    # === Séries 2D ===
    # g_xy
    I, J = np.meshgrid(i, j, indexing="ij")
    lam2_ij = (I / a) ** 2 + (J / b) ** 2
    C_ij = E(lam2_ij) * np.cos(I * np.pi * l / a) * np.cos(J * np.pi * m / b)
    g_xy = np.einsum("ij,ip,jp->p", C_ij, cos_ix, cos_jy)

    # g_xz
    I, K = np.meshgrid(i, k_idx, indexing="ij")
    lam2_ik = (I / a) ** 2 + (K / c) ** 2
    C_ik = E(lam2_ik) * np.cos(I * np.pi * l / a) * np.cos(K * np.pi * p / c)
    g_xz = np.einsum("ik,ip,kp->p", C_ik, cos_ix, cos_kz)

    # g_yz
    J, K = np.meshgrid(j, k_idx, indexing="ij")
    lam2_jk = (J / b) ** 2 + (K / c) ** 2
    C_jk = E(lam2_jk) * np.cos(J * np.pi * m / b) * np.cos(K * np.pi * p / c)
    g_yz = np.einsum("jk,jp,kp->p", C_jk, cos_jy, cos_kz)

    # === Série 3D ===
    N3 = 15  # menos termos para 3D
    i3, j3, k3 = np.arange(1, N3 + 1), np.arange(1, N3 + 1), np.arange(1, N3 + 1)
    I3, J3, K3 = np.meshgrid(i3, j3, k3, indexing="ij")
    lam2_ijk = (I3 / a) ** 2 + (J3 / b) ** 2 + (K3 / c) ** 2
    C_ijk = (
        E(lam2_ijk)
        * np.cos(I3 * np.pi * l / a)
        * np.cos(J3 * np.pi * m / b)
        * np.cos(K3 * np.pi * p / c)
    )

    cos_ix3 = np.cos(np.outer(i3 * np.pi / a, flat_x))
    cos_jy3 = np.cos(np.outer(j3 * np.pi / b, flat_y))
    cos_kz3 = np.cos(np.outer(k3 * np.pi / c, flat_z))
    g_xyz = np.einsum("ijk,ip,jp,kp->p", C_ijk, cos_ix3, cos_jy3, cos_kz3)

    # === Solução ===
    series = 2 * d + 2 * f + 2 * h + 4 * g_xy + 4 * g_xz + 4 * g_yz + 8 * g_xyz
    P = Pi + (q * S / V) * (t + series / eta)

    result = P.reshape(x.shape)
    return float(result) if result.ndim == 0 else result


if __name__ == "__main__":
    par = _load_params()
    print(f"Domínio: {par['a']} x {par['b']} x {par['c']} ft")
    print(f"Poço em: ({par['l']}, {par['m']}, {par['p']}) ft")
    print(f"η = {par['eta']:.2e} ft²/dia")
    print(f"τ = L²/η = {(par['a']/2)**2 / par['eta']:.2f} dias")
    print()

    t = 365.0
    P_centro = analytical(par["l"], par["m"], par["p"], t)
    P_canto = analytical(0, 0, 0, t)
    print(f"t = {t} dias:")
    print(f"  P(centro) = {P_centro:.2f} psi")
    print(f"  P(canto)  = {P_canto:.2f} psi")
    print(f"  ΔP = {par['Pi'] - P_centro:.2f} psi")
