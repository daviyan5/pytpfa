import os
import configparser
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io


_params_cache = None


def _load_reservoir_parameters():
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
    well_section = config["WELL_1"]

    a = input_section.getfloat("LX")
    b = input_section.getfloat("LY")
    h = input_section.getfloat("LZ")

    l = a / 2.0
    q = b / 2.0

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

    well_rate_field_units = abs(well_section.getfloat("VALUE"))

    Qj = well_rate_field_units * 5.614
    Q = -Bo * Qj / 5.614 / h

    alpha = 157.952 * (por * c * mu) / k
    beta = 886.905 * (Bo * mu) / k

    _params_cache = {
        "a": a,
        "b": b,
        "h": h,
        "l": l,
        "q": q,
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

    return _params_cache


def analytical(x, y, z, t):
    params = _load_reservoir_parameters()
    a = params["a"]
    b = params["b"]
    l = params["l"]
    q = params["q"]
    Pi = params["Pi"]
    Q = params["Q"]
    alpha = params["alpha"]
    beta = params["beta"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = float(t)

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

    m2_a2 = m_col**2 / a**2
    C_d_m = (
        (1 / (pi_sq * m2_a2))
        * (1 - np.exp(-pi_sq / alpha * m2_a2 * t))
        * np.cos(m_col * np.pi * l / a)
    )
    cos_mx = np.cos(m_op * np.pi * x_bcast / a)
    d = np.einsum("m,m...->...", C_d_m.flatten(), cos_mx)

    n2_b2 = n_col**2 / b**2
    C_f_n = (
        (1 / (pi_sq * n2_b2))
        * (1 - np.exp(-pi_sq / alpha * n2_b2 * t))
        * np.cos(n_col * np.pi * q / b)
    )
    cos_ny = np.cos(n_op * np.pi * y_bcast / b)
    f = np.einsum("n,n...->...", C_f_n.flatten(), cos_ny)

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

    P_result = Pi - beta * Q / (a * b) * (t / alpha + 2 * d + 2 * f + 4 * g)

    if P_result.ndim == 0 or P_result.size == 1:
        return P_result.item()
    else:
        return P_result


def get_parameters():
    return _load_reservoir_parameters().copy()


if __name__ == "__main__":
    print("Testing Case Study 4 analytical solution...")

    try:
        params = get_parameters()
        print("Loaded parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        test_points = [
            (1000, 840, 0, 365),
            (0, 840, 0, 365),
            (2000, 840, 0, 365),
            (1000, 1000, 0, 365),
        ]

        print(f"\nTest results at t=365 days:")
        for x, y, z, t in test_points:
            p = analytical(x, y, z, t)
            print(f"  P({x}, {y}, {z}, {t}) = {p:.2f} psi")

        x_array = np.linspace(0, 2000, 5)
        y_fixed = 840.0
        z_fixed = 0.0
        t_fixed = 365.0

        p_array = analytical(x_array, y_fixed, z_fixed, t_fixed)
        print(f"\nPressure profile at y={y_fixed} ft, t={t_fixed} days:")
        for i, (xi, pi) in enumerate(zip(x_array, p_array)):
            print(f"  x={xi:6.1f} ft: P={pi:.2f} psi")

        grid_resolution = 100
        x = np.linspace(0, 2000, grid_resolution)
        y = np.linspace(0, 2000, grid_resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        times = np.linspace(0, 365, 50)
        vmin, vmax = 2000, 2200

        fig, ax = plt.subplots(figsize=(10, 8))
        frames = []

        for i, t in enumerate(times):
            ax.clear()

            P = analytical(X, Y, Z, t)

            p_min, p_max, p_avg = P.min(), P.max(), P.mean()
            print(
                f"Frame {i+1}/{len(times)} (t={t:5.1f} days) | "
                f"Min: {p_min:7.2f}, Max: {p_max:7.2f}, Avg: {p_avg:7.2f} psi"
            )

            P_clipped = np.clip(P, vmin, vmax)

            im = ax.contourf(X, Y, P_clipped, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"Pressure Evolution - t = {t:.1f} days")
            ax.set_xlabel("X (ft)")
            ax.set_ylabel("Y (ft)")
            ax.set_aspect("equal")

            if i == 0:
                cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Pressure (psi)")

            plt.draw()
            plt.pause(0.01)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            frames.append(Image.open(buf))

        gif_path = "pressure_animation_fine.gif"
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
