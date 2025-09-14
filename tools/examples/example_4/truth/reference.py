import os
import configparser
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io


_params_cache = None


def _load_reservoir_parameters():
    """Load parameters from reservoir.ini file"""
    global _params_cache

    if _params_cache is not None:
        return _params_cache

    current_dir = Path(__file__).parent
    reservoir_ini_path = current_dir.parent / "reservoir.ini"

    if not reservoir_ini_path.exists():
        raise FileNotFoundError(f"Could not find reservoir.ini at {reservoir_ini_path}")

    config = configparser.ConfigParser()
    config.read(reservoir_ini_path)

    input_section = config["RESERVOIR_INPUT"]
    well1_section = config["WELL_1"]  # Injector
    well2_section = config["WELL_2"]  # Producer

    # Reservoir dimensions
    a = input_section.getfloat("LX")  # 1000 ft
    b = input_section.getfloat("LY")  # 1000 ft
    h = input_section.getfloat("LZ")  # 1 ft

    # Initial pressure
    Pi = (
        input_section.getfloat("PRESSURE")
        if "PRESSURE" in input_section
        else config["INITIAL_CONDITION"].getfloat("PRESSURE")
    )

    # Fluid and rock properties
    Bo = input_section.getfloat("B")
    mu = input_section.getfloat("MU")
    por = input_section.getfloat("PORO")
    k = input_section.getfloat("KX")
    cf = input_section.getfloat("CFLUID")
    cr = input_section.getfloat("CPORO")
    c = cr + cf  # Total compressibility

    # Well locations and rates
    x_inj = well1_section.getfloat("BLOCK_COORD_X")  # 50 ft
    y_inj = well1_section.getfloat("BLOCK_COORD_Y")  # 50 ft
    x_prod = well2_section.getfloat("BLOCK_COORD_X")  # 950 ft
    y_prod = well2_section.getfloat("BLOCK_COORD_Y")  # 950 ft

    # Well rates (converting from field units)
    # Assuming injection rate is positive and production rate is negative in the INI
    Q_inj_field = abs(well1_section.getfloat("VALUE"))  # ft³/day
    Q_prod_field = abs(well2_section.getfloat("VALUE"))  # ft³/day

    # Convert to specific rates (STB/D-ft)
    Q_inj = -Bo * Q_inj_field / 5.614 / h  # Negative for injection
    Q_prod = Bo * Q_prod_field / 5.614 / h  # Positive for production

    # Dimensionless groups
    alpha = 157.952 * (por * c * mu) / k  # Time group
    beta = 886.905 * (Bo * mu) / k  # Pressure group

    _params_cache = {
        "a": a,
        "b": b,
        "h": h,
        "x_inj": x_inj,
        "y_inj": y_inj,
        "x_prod": x_prod,
        "y_prod": y_prod,
        "Pi": Pi,
        "Bo": Bo,
        "mu": mu,
        "por": por,
        "k": k,
        "cf": cf,
        "cr": cr,
        "c": c,
        "Q_inj": Q_inj,
        "Q_prod": Q_prod,
        "Q_inj_field": Q_inj_field,
        "Q_prod_field": Q_prod_field,
        "alpha": alpha,
        "beta": beta,
    }

    return _params_cache


def _single_well_contribution(x, y, x_well, y_well, Q, a, b, t, alpha, beta, series_terms=100):
    """
    Calculate pressure contribution from a single well with no-flow boundaries.
    Uses the method of images and Fourier series solution.
    """
    if t == 0.0:
        return 0.0

    # Ensure numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Series indices
    m = np.arange(1, series_terms + 1)
    n = np.arange(1, series_terms + 1)
    pi_sq = np.pi**2

    # Reshape for broadcasting
    m_col = m.reshape(-1, 1)
    n_col = n.reshape(-1, 1)

    # X-direction contribution
    m_op_shape = (-1,) + (1,) * x.ndim
    m_op = m.reshape(m_op_shape)

    m2_a2 = m_col**2 / a**2
    C_d_m = (
        (1 / (pi_sq * m2_a2))
        * (1 - np.exp(-pi_sq / alpha * m2_a2 * t))
        * np.cos(m_col * np.pi * x_well / a)
    )
    cos_mx = np.cos(m_op * np.pi * x / a)
    d = np.einsum("m,m...->...", C_d_m.flatten(), cos_mx)

    # Y-direction contribution
    n_op_shape = (-1,) + (1,) * y.ndim
    n_op = n.reshape(n_op_shape)

    n2_b2 = n_col**2 / b**2
    C_f_n = (
        (1 / (pi_sq * n2_b2))
        * (1 - np.exp(-pi_sq / alpha * n2_b2 * t))
        * np.cos(n_col * np.pi * y_well / b)
    )
    cos_ny = np.cos(n_op * np.pi * y / b)
    f = np.einsum("n,n...->...", C_f_n.flatten(), cos_ny)

    # Cross terms (m,n)
    n_row = n.reshape(1, -1)
    lambda_mn_sq = (m_col**2 / a**2) + (n_row**2 / b**2)
    C_mn = (
        (1 / (pi_sq * lambda_mn_sq))
        * (1 - np.exp(-pi_sq / alpha * lambda_mn_sq * t))
        * np.cos(m_col * np.pi * x_well / a)
        * np.cos(n_row * np.pi * y_well / b)
    )

    V_mx = np.cos(m_op * np.pi * x / a)
    W_ny = np.cos(n_op * np.pi * y / b)
    g = np.einsum("mn,m...,n...->...", C_mn, V_mx, W_ny)

    # Total contribution from this well
    delta_P = -beta * Q / (a * b) * (t / alpha + 2 * d + 2 * f + 4 * g)

    return delta_P


def analytical(x, y, z, t):
    """
    Analytical solution for 1/4 five-spot pattern with compressible flow.
    Two wells: injector at (x_inj, y_inj) and producer at (x_prod, y_prod).
    """
    params = _load_reservoir_parameters()

    a = params["a"]
    b = params["b"]
    x_inj = params["x_inj"]
    y_inj = params["y_inj"]
    x_prod = params["x_prod"]
    y_prod = params["y_prod"]
    Pi = params["Pi"]
    Q_inj = params["Q_inj"]
    Q_prod = params["Q_prod"]
    alpha = params["alpha"]
    beta = params["beta"]

    # Convert to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = float(t)

    # Initial condition
    if t == 0.0:
        result_shape = np.broadcast(x, y).shape
        return np.full(result_shape, Pi, dtype=float)

    # Broadcast x and y to same shape
    x_bcast, y_bcast = np.broadcast_arrays(x, y)

    # Contribution from injection well
    P_inj = _single_well_contribution(x_bcast, y_bcast, x_inj, y_inj, Q_inj, a, b, t, alpha, beta)

    # Contribution from production well
    P_prod = _single_well_contribution(
        x_bcast, y_bcast, x_prod, y_prod, Q_prod, a, b, t, alpha, beta
    )

    # Total pressure (superposition)
    P_result = Pi + P_inj + P_prod

    # Return scalar if input was scalar
    if P_result.ndim == 0 or P_result.size == 1:
        return P_result.item()
    else:
        return P_result


def get_parameters():
    """Return a copy of the loaded parameters"""
    return _load_reservoir_parameters().copy()


if __name__ == "__main__":
    print("Testing 1/4 Five-Spot Pattern analytical solution...")

    try:
        params = get_parameters()
        print("\nLoaded parameters:")
        print(f"  Reservoir: {params['a']:.0f} x {params['b']:.0f} x {params['h']:.0f} ft")
        print(f"  Injector location: ({params['x_inj']:.0f}, {params['y_inj']:.0f}) ft")
        print(f"  Producer location: ({params['x_prod']:.0f}, {params['y_prod']:.0f}) ft")
        print(f"  Injection rate: {params['Q_inj_field']:.2f} ft³/day")
        print(f"  Production rate: {params['Q_prod_field']:.2f} ft³/day")
        print(f"  Initial pressure: {params['Pi']:.0f} psi")
        print(f"  Permeability: {params['k']:.1f} md")
        print(f"  Porosity: {params['por']:.2f}")

        # Test at specific points
        test_points = [
            (params["x_inj"], params["y_inj"], 0, 365),  # At injector
            (params["x_prod"], params["y_prod"], 0, 365),  # At producer
            (500, 500, 0, 365),  # Center
            (250, 250, 0, 365),  # Quarter point
            (750, 750, 0, 365),  # Three-quarter point
        ]

        print(f"\nPressure at t=365 days:")
        for x, y, z, t in test_points:
            p = analytical(x, y, z, t)
            print(f"  P({x:4.0f}, {y:4.0f}) = {p:7.2f} psi")

        # Pressure profile along diagonal
        print("\nPressure profile along diagonal (y=x) at t=365 days:")
        x_diag = np.linspace(0, 1000, 11)
        y_diag = x_diag
        p_diag = analytical(x_diag, y_diag, 0, 365)
        for i in range(0, len(x_diag), 2):
            print(f"  ({x_diag[i]:4.0f}, {y_diag[i]:4.0f}): P = {p_diag[i]:7.2f} psi")

        # Create animation
        print("\nGenerating pressure animation...")
        grid_resolution = 50
        x = np.linspace(0, 1000, grid_resolution)
        y = np.linspace(0, 1000, grid_resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        times = np.linspace(0, 365, 25)

        # Calculate pressure range for consistent color scale
        P_final = analytical(X, Y, Z, 365)
        vmin = P_final.min() - 50
        vmax = P_final.max() + 50

        fig, ax = plt.subplots(figsize=(10, 8))
        frames = []

        for i, t in enumerate(times):
            ax.clear()

            P = analytical(X, Y, Z, t)

            p_min, p_max, p_avg = P.min(), P.max(), P.mean()
            print(
                f"  Frame {i+1}/{len(times)} (t={t:5.1f} days) | "
                f"Min: {p_min:7.2f}, Max: {p_max:7.2f}, Avg: {p_avg:7.2f} psi"
            )

            # Create contour plot
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(X, Y, P, levels=levels, cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.contour(X, Y, P, levels=levels[::2], colors="black", alpha=0.3, linewidths=0.5)

            # Mark well locations
            ax.plot(params["x_inj"], params["y_inj"], "bo", markersize=10, label="Injector")
            ax.plot(params["x_prod"], params["y_prod"], "ro", markersize=10, label="Producer")

            ax.set_title(f"1/4 Five-Spot Pattern - Pressure at t = {t:.1f} days")
            ax.set_xlabel("X (ft)")
            ax.set_ylabel("Y (ft)")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if i == 0:
                cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Pressure (psi)")

            # Save frame for GIF
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
            buf.seek(0)
            frames.append(Image.open(buf))

            plt.pause(0.01)

        # Save animation
        gif_path = "five_spot_pressure_animation.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )

        print(f"\nAnimation saved as {gif_path}")
        plt.show()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
