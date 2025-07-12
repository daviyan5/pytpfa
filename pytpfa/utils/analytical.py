import numpy as np
import sympy as sp

x, y, z, t = sp.symbols("x y z t")
nx, ny, nz = sp.symbols("nx ny nz")  # Normal vector components


def analytical(p_expr, params=None):
    """
    Converts a SymPy expression for p(x,y,z,t) into a numerical function.
    """
    return sp.lambdify((x, y, z, t), p_expr, "numpy")


def dirichlet(p_expr, params=None):
    """
    The Dirichlet condition is just the analytical solution itself, evaluated at the boundary.
    """
    return analytical(p_expr)


def neumann(p_expr, params):
    """
    Calculates the normal flux (e.g., K * grad(p) · n) and returns a numerical function.
    Note: This is for a simplified flux term. Your Eq. 8.1 is more complex.
    """
    kx, ky, kz = params["permeability"]
    mu_ref = params["mu_ref"]

    flux_vec_x = -(kx / mu_ref) * sp.diff(p_expr, x)
    flux_vec_y = -(ky / mu_ref) * sp.diff(p_expr, y)
    flux_vec_z = -(kz / mu_ref) * sp.diff(p_expr, z)

    normal_flux = flux_vec_x * nx + flux_vec_y * ny + flux_vec_z * nz

    return sp.lambdify((x, y, z, t, nx, ny, nz), normal_flux, "numpy")


def source_term(p_expr, params):
    """
    Calculates the source term q by rearranging the PDE: q = Accumulation + Divergence_of_Flux
    This assumes a PDE of the form: ∂(φρ)/∂t - ∇·(ρk/μ ∇p) = q
    """
    p_ref = params["p_ref"]
    c_phi = params["porosity_compressibility"]
    c_rho = params["fluid_compressibility"]
    rho_ref = params["rho_ref"]
    mu_ref = params["mu_ref"]
    phi_ref = params["phi_ref"]
    kx, ky, kz = params["permeability"]

    rho = rho_ref * (1 + c_rho * (p_expr - p_ref))
    phi = phi_ref * (1 + c_phi * (p_expr - p_ref))
    mu = mu_ref

    accumulation = sp.diff(phi * rho, t)

    flux_vec_x = -(rho * kx / mu) * sp.diff(p_expr, x)
    flux_vec_y = -(rho * ky / mu) * sp.diff(p_expr, y)
    flux_vec_z = -(rho * kz / mu) * sp.diff(p_expr, z)

    div_flux = -(sp.diff(flux_vec_x, x) + sp.diff(flux_vec_y, y) + sp.diff(flux_vec_z, z))

    source_expr = accumulation - div_flux

    return sp.lambdify((x, y, z, t), source_expr, "numpy")


if __name__ == "__main__":
    p_manufactured = sp.sin(x) * sp.sin(y) * sp.cos(t)

    parameters = {
        "permeability": (1.0e-12, 1.0e-12, 1.0e-12),
        "rho_ref": 1000,  # kg/m^3
        "p_ref": 1.0e5,  # Pa
        "phi_ref": 0.2,
        "mu_ref": 1.0e-3,  # Pa.s
        "fluid_compressibility": 4.0e-10,  # 1/Pa
        "porosity_compressibility": 3.0e-10,  # 1/Pa
    }

    p_sol = get_manufactured_solution(p_manufactured)
    neumann_bc = get_neumann_condition(p_manufactured, parameters)
    source = get_source_term(p_manufactured, parameters)

    pressure_val = p_sol(0.5, 0.5, 0, 1.0)
    print(f"Pressure at (0.5, 0.5, 0) at t=1s is: {pressure_val:.4f}")

    source_val = source(0.5, 0.5, 0, 1.0)
    print(f"Source term at (0.5, 0.5, 0) at t=1s is: {source_val:.4e}")

    neumann_val = neumann_bc(1.0, 0.5, 0, 1.0, 1, 0, 0)
    print(f"Neumann value at (1, 0.5, 0) for normal (1,0,0) at t=1s is: {neumann_val:.4e}")
