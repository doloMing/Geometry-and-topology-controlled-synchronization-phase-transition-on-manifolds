from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from paths import DATA_DIR


PRODUCT_CASES = [
    {"case": r"$S^3\times S^3$", "p": 3, "q": 3, "chi": 0, "conditioned": False, "role": "continuous control"},
    {"case": r"$S^2\times S^2$", "p": 2, "q": 2, "chi": 4, "conditioned": True, "role": "discontinuous test"},
    {"case": r"$S^3\times S^5$", "p": 3, "q": 5, "chi": 0, "conditioned": False, "role": "continuous control"},
    {"case": r"$S^2\times S^4$", "p": 2, "q": 4, "chi": 4, "conditioned": True, "role": "discontinuous test"},
]
PRODUCT_DELTAS = [0.35, 0.50]
PRODUCT_SEEDS = range(15)
PRODUCT_N = 256
PRODUCT_RELAX_STEPS = 850
PRODUCT_K_GRID_SIZE = 61
PRODUCT_DT = 0.012
PRODUCT_BURN_FRACTION = 0.60
PRODUCT_SAMPLE_EVERY = 20
PRODUCT_K_WINDOW = (0.08, 1.85)
PRODUCT_RNG_BASE = 122029
PRODUCT_RAW_PATH = DATA_DIR / "fig05_product_formal_raw.csv"
PRODUCT_SUMMARY_PATH = DATA_DIR / "fig05_product_formal_summary.csv"

TABLEVI_CASES = [
    {
        "case": r"$\mathbb{T}^{2}$",
        "case_plain": "T^2",
        "kind": "torus",
        "dimension": 2,
        "chi": 0,
        "parameter": 2,
        "conditioned": False,
        "role": "continuous control",
    },
    {
        "case": r"$\mathbb{CP}^{2}$",
        "case_plain": "CP^2",
        "kind": "projective",
        "dimension": 4,
        "chi": 3,
        "parameter": 2,
        "conditioned": True,
        "role": "discontinuous test",
    },
    {
        "case": r"$\mathbb{T}^{4}$",
        "case_plain": "T^4",
        "kind": "torus",
        "dimension": 4,
        "chi": 0,
        "parameter": 4,
        "conditioned": False,
        "role": "continuous control",
    },
    {
        "case": r"$\mathbb{CP}^{3}$",
        "case_plain": "CP^3",
        "kind": "projective",
        "dimension": 6,
        "chi": 4,
        "parameter": 3,
        "conditioned": True,
        "role": "discontinuous test",
    },
]
TABLEVI_DELTAS = [0.35, 0.50]
TABLEVI_SEEDS = range(15)
TABLEVI_N = 256
TABLEVI_RELAX_STEPS = 1800
TABLEVI_K_GRID_SIZE = 121
TABLEVI_DT = 0.012
TABLEVI_BURN_FRACTION = 0.60
TABLEVI_SAMPLE_EVERY = 20
TABLEVI_K_WINDOW = (0.06, 4.80)
TABLEVI_RNG_BASE = 60292026
TABLEVI_CONDITIONING_NOISE = 0.10
TABLEVI_RAW_PATH = DATA_DIR / "fig05_tablevi_formal_raw.csv"
TABLEVI_SUMMARY_PATH = DATA_DIR / "fig05_tablevi_formal_summary.csv"


def planned_extension_cases() -> pd.DataFrame:
    """Return the planned non-spherical and product manifold cases."""
    rows = [
        {
            "block": "product",
            "case": r"$S^3\times S^3$",
            "dimension": 6,
            "euler_characteristic": 0,
            "ensemble": "ordinary random skew drift",
            "expected_role": "continuous control",
        },
        {
            "block": "product",
            "case": r"$S^2\times S^2$",
            "dimension": 4,
            "euler_characteristic": 4,
            "ensemble": "axial sign-conditioned skew drift",
            "expected_role": "discontinuous test",
        },
        {
            "block": "product",
            "case": r"$S^3\times S^5$",
            "dimension": 8,
            "euler_characteristic": 0,
            "ensemble": "ordinary random skew drift",
            "expected_role": "continuous control",
        },
        {
            "block": "product",
            "case": r"$S^2\times S^4$",
            "dimension": 6,
            "euler_characteristic": 4,
            "ensemble": "axial sign-conditioned skew drift",
            "expected_role": "discontinuous test",
        },
        {
            "block": "Table VI",
            "case": r"$\mathbb{T}^2$",
            "dimension": 2,
            "euler_characteristic": 0,
            "ensemble": "ordinary frequency drift",
            "expected_role": "continuous control",
        },
        {
            "block": "Table VI",
            "case": r"$\mathbb{CP}^2$",
            "dimension": 4,
            "euler_characteristic": 3,
            "ensemble": "axial sign-conditioned Hermitian drift",
            "expected_role": "discontinuous test",
        },
        {
            "block": "Table VI",
            "case": r"$\mathbb{T}^4$",
            "dimension": 4,
            "euler_characteristic": 0,
            "ensemble": "ordinary frequency drift",
            "expected_role": "continuous control",
        },
        {
            "block": "Table VI",
            "case": r"$\mathbb{CP}^3$",
            "dimension": 6,
            "euler_characteristic": 4,
            "ensemble": "axial sign-conditioned Hermitian drift",
            "expected_role": "discontinuous test",
        },
    ]
    return pd.DataFrame(rows)


def random_sphere_points(rng, n_points: int, sphere_dim: int) -> np.ndarray:
    ambient_dim = sphere_dim + 1
    x = rng.normal(size=(n_points, ambient_dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x[:, 0] += 0.01
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lorentzian_quantiles(n_points: int, delta: float, rng, clip: float = 0.995) -> np.ndarray:
    q = (np.arange(n_points) + 0.5) / n_points
    q = np.clip(q, 1.0 - clip, clip)
    rates = delta * np.tan(np.pi * (q - 0.5))
    rng.shuffle(rates)
    return rates


def random_skew_directions(rng, n_points: int, ambient_dim: int) -> np.ndarray:
    A = rng.normal(size=(n_points, ambient_dim, ambient_dim))
    A = 0.5 * (A - np.swapaxes(A, 1, 2))
    norm = np.sqrt(np.sum(A * A, axis=(1, 2), keepdims=True))
    return np.sqrt(2.0) * A / np.maximum(norm, 1.0e-12)


def axial_skew_direction(sphere_dim: int) -> np.ndarray:
    ambient_dim = sphere_dim + 1
    A = np.zeros((ambient_dim, ambient_dim))
    for pair_index in range(sphere_dim // 2):
        i = 2 * pair_index
        j = 2 * pair_index + 1
        A[i, j] = -1.0
        A[j, i] = 1.0
    norm = np.sqrt(np.sum(A * A))
    if norm == 0.0:
        return A
    return np.sqrt(2.0) * A / norm


def product_skew_drifts(
    rng,
    n_points: int,
    first_dim: int,
    second_dim: int,
    delta: float,
    conditioned: bool,
    noise: float = 0.35,
):
    first_ambient = first_dim + 1
    second_ambient = second_dim + 1
    if conditioned:
        A0 = axial_skew_direction(first_dim)
        B0 = axial_skew_direction(second_dim)
        A = np.repeat(A0[None, :, :], n_points, axis=0)
        B = np.repeat(B0[None, :, :], n_points, axis=0)
        A = (1.0 - noise) * A + noise * random_skew_directions(rng, n_points, first_ambient)
        B = (1.0 - noise) * B + noise * random_skew_directions(rng, n_points, second_ambient)
        A_norm = np.sqrt(np.sum(A * A, axis=(1, 2), keepdims=True))
        B_norm = np.sqrt(np.sum(B * B, axis=(1, 2), keepdims=True))
        A = np.sqrt(2.0) * A / np.maximum(A_norm, 1.0e-12)
        B = np.sqrt(2.0) * B / np.maximum(B_norm, 1.0e-12)
    else:
        A = random_skew_directions(rng, n_points, first_ambient)
        B = random_skew_directions(rng, n_points, second_ambient)
    rates = lorentzian_quantiles(n_points, delta, rng)
    signs = rng.choice([-1.0, 1.0], size=n_points)
    signed_rates = rates * signs if conditioned else rates
    return A * signed_rates[:, None, None], B * signed_rates[:, None, None]


@njit(fastmath=True)
def integrate_product_sphere_sweep(
    x,
    y,
    omega_x,
    omega_y,
    k_values,
    dt,
    relax_steps,
    burn_step,
    sample_every,
):
    n_points, first_ambient = x.shape
    second_ambient = y.shape[1]
    n_k = len(k_values)
    result = np.zeros((2, n_k))
    for scan in range(2):
        for scan_index in range(n_k):
            k_index = scan_index if scan == 0 else n_k - 1 - scan_index
            K = k_values[k_index]
            sample_sum = 0.0
            sample_count = 0
            for step in range(relax_steps):
                rx = np.zeros(first_ambient)
                ry = np.zeros(second_ambient)
                for i in range(n_points):
                    for j in range(first_ambient):
                        rx[j] += x[i, j]
                    for j in range(second_ambient):
                        ry[j] += y[i, j]
                for j in range(first_ambient):
                    rx[j] /= n_points
                for j in range(second_ambient):
                    ry[j] /= n_points

                for i in range(n_points):
                    proj_x = 0.0
                    proj_y = 0.0
                    for j in range(first_ambient):
                        proj_x += x[i, j] * rx[j]
                    for j in range(second_ambient):
                        proj_y += y[i, j] * ry[j]

                    norm_x = 0.0
                    for j in range(first_ambient):
                        drift_j = 0.0
                        for ell in range(first_ambient):
                            drift_j += omega_x[i, j, ell] * x[i, ell]
                        value = x[i, j] + dt * (drift_j + K * (rx[j] - proj_x * x[i, j]))
                        x[i, j] = value
                        norm_x += value * value
                    inv_x = 1.0 / np.sqrt(norm_x)
                    for j in range(first_ambient):
                        x[i, j] *= inv_x

                    norm_y = 0.0
                    for j in range(second_ambient):
                        drift_j = 0.0
                        for ell in range(second_ambient):
                            drift_j += omega_y[i, j, ell] * y[i, ell]
                        value = y[i, j] + dt * (drift_j + K * (ry[j] - proj_y * y[i, j]))
                        y[i, j] = value
                        norm_y += value * value
                    inv_y = 1.0 / np.sqrt(norm_y)
                    for j in range(second_ambient):
                        y[i, j] *= inv_y

                if step >= burn_step and step % sample_every == 0:
                    rx_norm_sq = 0.0
                    ry_norm_sq = 0.0
                    for j in range(first_ambient):
                        component = 0.0
                        for i in range(n_points):
                            component += x[i, j]
                        component /= n_points
                        rx_norm_sq += component * component
                    for j in range(second_ambient):
                        component = 0.0
                        for i in range(n_points):
                            component += y[i, j]
                        component /= n_points
                        ry_norm_sq += component * component
                    sample_sum += np.sqrt(0.5 * (rx_norm_sq + ry_norm_sq))
                    sample_count += 1
            result[scan, k_index] = sample_sum / max(sample_count, 1)
    return result


def run_product_sphere_case(
    first_dim: int,
    second_dim: int,
    delta: float,
    seed: int,
    conditioned: bool,
    n_points: int = 384,
    relax_steps: int = 1200,
    k_grid_size: int = 81,
):
    rng = np.random.default_rng(seed)
    x = random_sphere_points(rng, n_points, first_dim)
    y = random_sphere_points(rng, n_points, second_dim)
    omega_x, omega_y = product_skew_drifts(rng, n_points, first_dim, second_dim, delta, conditioned)
    k_ref = delta
    k_values = np.geomspace(0.08 * k_ref, 1.85 * k_ref, k_grid_size)
    scans = integrate_product_sphere_sweep(
        x,
        y,
        omega_x,
        omega_y,
        k_values,
        0.012,
        relax_steps,
        int(0.60 * relax_steps),
        20,
    )
    return summarize_scan(k_values, k_ref, scans)


def warm_up_product_integrator():
    rng = np.random.default_rng(1)
    x = random_sphere_points(rng, 8, 1)
    y = random_sphere_points(rng, 8, 1)
    omega_x = np.zeros((8, 2, 2))
    omega_y = np.zeros((8, 2, 2))
    integrate_product_sphere_sweep(
        x,
        y,
        omega_x,
        omega_y,
        np.geomspace(0.1, 0.2, 3),
        PRODUCT_DT,
        5,
        2,
        1,
    )


def run_product_formal_experiment():
    warm_up_product_integrator()
    raw_rows = []
    summary_rows = []
    for case_index, case in enumerate(PRODUCT_CASES):
        for delta in PRODUCT_DELTAS:
            for seed_index in PRODUCT_SEEDS:
                seed = PRODUCT_RNG_BASE + 10000 * case_index + 1000 * int(round(100 * delta)) + seed_index
                rng = np.random.default_rng(seed)
                x = random_sphere_points(rng, PRODUCT_N, case["p"])
                y = random_sphere_points(rng, PRODUCT_N, case["q"])
                omega_x, omega_y = product_skew_drifts(
                    rng,
                    PRODUCT_N,
                    case["p"],
                    case["q"],
                    delta,
                    conditioned=case["conditioned"],
                )
                k_ref = delta
                k_values = np.geomspace(PRODUCT_K_WINDOW[0] * k_ref, PRODUCT_K_WINDOW[1] * k_ref, PRODUCT_K_GRID_SIZE)
                scans = integrate_product_sphere_sweep(
                    x,
                    y,
                    omega_x,
                    omega_y,
                    k_values,
                    PRODUCT_DT,
                    PRODUCT_RELAX_STEPS,
                    int(PRODUCT_BURN_FRACTION * PRODUCT_RELAX_STEPS),
                    PRODUCT_SAMPLE_EVERY,
                )
                scan_summary = summarize_scan(k_values, k_ref, scans)
                summary_rows.append(
                    {
                        "case": case["case"],
                        "case_index": case_index,
                        "p": case["p"],
                        "q": case["q"],
                        "dimension": case["p"] + case["q"],
                        "chi": case["chi"],
                        "conditioned": case["conditioned"],
                        "role": case["role"],
                        "delta": delta,
                        "seed_index": seed_index,
                        "seed": seed,
                        **scan_summary,
                    }
                )
                for scan_idx, direction in enumerate(["forward", "backward"]):
                    for k_idx, K in enumerate(k_values):
                        raw_rows.append(
                            {
                                "case": case["case"],
                                "case_index": case_index,
                                "p": case["p"],
                                "q": case["q"],
                                "dimension": case["p"] + case["q"],
                                "chi": case["chi"],
                                "conditioned": case["conditioned"],
                                "role": case["role"],
                                "delta": delta,
                                "seed_index": seed_index,
                                "seed": seed,
                                "direction": direction,
                                "K_index": k_idx,
                                "K": float(K),
                                "K_over_Kref": float(K / k_ref),
                                "R": float(scans[scan_idx, k_idx]),
                                "N": PRODUCT_N,
                                "relax_steps": PRODUCT_RELAX_STEPS,
                                "K_grid_size": PRODUCT_K_GRID_SIZE,
                            }
                        )
    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    raw.to_csv(PRODUCT_RAW_PATH, index=False)
    summary.to_csv(PRODUCT_SUMMARY_PATH, index=False)
    return raw, summary


def load_or_run_product_formal(recompute: bool = False):
    if recompute or not (PRODUCT_RAW_PATH.exists() and PRODUCT_SUMMARY_PATH.exists()):
        return run_product_formal_experiment()
    return pd.read_csv(PRODUCT_RAW_PATH), pd.read_csv(PRODUCT_SUMMARY_PATH)


def torus_initial_angles(rng, n_points: int, torus_dim: int) -> np.ndarray:
    return rng.uniform(0.0, 2.0 * np.pi, size=(n_points, torus_dim))


def torus_order_parameter(theta: np.ndarray) -> np.ndarray:
    z = np.exp(1j * theta)
    return np.mean(z, axis=0)


def integrate_torus_sweep(
    theta,
    omega,
    k_values,
    dt,
    relax_steps,
    burn_step,
    sample_every,
):
    n_k = len(k_values)
    result = np.zeros((2, n_k))
    for scan in range(2):
        for scan_index in range(n_k):
            k_index = scan_index if scan == 0 else n_k - 1 - scan_index
            K = k_values[k_index]
            samples = []
            for step in range(relax_steps):
                z_mean = torus_order_parameter(theta)
                force = np.imag(z_mean[None, :] * np.exp(-1j * theta))
                theta = theta + dt * (omega + K * force)
                if step >= burn_step and step % sample_every == 0:
                    samples.append(float(np.sqrt(np.mean(np.abs(torus_order_parameter(theta)) ** 2))))
            result[scan, k_index] = float(np.mean(samples))
    return result


def random_projective_states(rng, n_points: int, complex_dim: int) -> np.ndarray:
    z = rng.normal(size=(n_points, complex_dim)) + 1j * rng.normal(size=(n_points, complex_dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z[:, 0] += 0.01
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def random_hermitian_directions(rng, n_points: int, complex_dim: int) -> np.ndarray:
    A = rng.normal(size=(n_points, complex_dim, complex_dim)) + 1j * rng.normal(size=(n_points, complex_dim, complex_dim))
    H = 0.5 * (A + np.conjugate(np.swapaxes(A, 1, 2)))
    traces = np.trace(H, axis1=1, axis2=2) / complex_dim
    H = H - traces[:, None, None] * np.eye(complex_dim)[None, :, :]
    norm = np.sqrt(np.real(np.sum(np.conjugate(H) * H, axis=(1, 2), keepdims=True)))
    return H / np.maximum(norm, 1.0e-12)


def axial_hermitian_direction(complex_dim: int) -> np.ndarray:
    values = np.linspace(-1.0, 1.0, complex_dim)
    values = values - np.mean(values)
    H = np.diag(values)
    norm = np.sqrt(np.real(np.sum(np.conjugate(H) * H)))
    return H / max(norm, 1.0e-12)


def projective_drifts(rng, n_points: int, complex_dim: int, delta: float, conditioned: bool, noise: float = 0.35):
    if conditioned:
        base = axial_hermitian_direction(complex_dim)
        H = np.repeat(base[None, :, :], n_points, axis=0)
        H = (1.0 - noise) * H + noise * random_hermitian_directions(rng, n_points, complex_dim)
    else:
        H = random_hermitian_directions(rng, n_points, complex_dim)
    rates = lorentzian_quantiles(n_points, delta, rng)
    signs = rng.choice([-1.0, 1.0], size=n_points)
    signed_rates = rates * signs if conditioned else rates
    return H * signed_rates[:, None, None]


def projective_order_parameter(z: np.ndarray) -> float:
    n_points, complex_dim = z.shape
    Pbar = np.einsum("ni,nj->ij", z, np.conjugate(z)) / n_points
    trace_p2 = np.real(np.trace(Pbar @ Pbar))
    return float(np.sqrt(max((trace_p2 - 1.0 / complex_dim) / (1.0 - 1.0 / complex_dim), 0.0)))


def integrate_projective_sweep(
    z,
    H,
    k_values,
    dt,
    relax_steps,
    burn_step,
    sample_every,
):
    n_points = z.shape[0]
    n_k = len(k_values)
    result = np.zeros((2, n_k))
    for scan in range(2):
        for scan_index in range(n_k):
            k_index = scan_index if scan == 0 else n_k - 1 - scan_index
            K = k_values[k_index]
            samples = []
            for step in range(relax_steps):
                Pbar = np.einsum("ni,nj->ij", z, np.conjugate(z)) / n_points
                drift = -1j * np.einsum("nij,nj->ni", H, z)
                pz = z @ Pbar.T
                scalar = np.real(np.sum(np.conjugate(z) * pz, axis=1))
                coupling = K * (pz - scalar[:, None] * z)
                z = z + dt * (drift + coupling)
                z /= np.linalg.norm(z, axis=1, keepdims=True)
                if step >= burn_step and step % sample_every == 0:
                    samples.append(projective_order_parameter(z))
            result[scan, k_index] = float(np.mean(samples))
    return result


def summarize_scan(k_values: np.ndarray, k_ref: float, scans: np.ndarray) -> dict[str, float]:
    forward = scans[0]
    backward = scans[1]
    jumps = np.abs(np.diff(forward))
    scan_difference = np.abs(backward - forward)
    jump_index = int(np.argmax(jumps))
    scan_index = int(np.argmax(scan_difference))
    return {
        "jump_size": float(jumps[jump_index]),
        "max_scan_difference": float(scan_difference[scan_index]),
        "K_at_jump": float((k_values / k_ref)[jump_index]),
        "K_at_max_scan_difference": float((k_values / k_ref)[scan_index]),
    }


def tablevi_torus_frequencies(rng: np.random.Generator, n_points: int, torus_dim: int, delta: float) -> np.ndarray:
    """Use independent Lorentzian quantiles for each angular coordinate."""
    omega = np.zeros((n_points, torus_dim))
    for coordinate in range(torus_dim):
        omega[:, coordinate] = lorentzian_quantiles(n_points, delta, rng)
    return omega


def add_tablevi_continuation_metrics(raw: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Add low- and high-coupling endpoint metrics to the Table VI summary."""
    required = {"forward_R_lowK", "backward_R_lowK", "forward_R_highK", "hysteresis_area"}
    if required.issubset(summary.columns):
        return summary
    rows = []
    for keys, group in raw.groupby(["case", "case_plain", "chi", "role", "delta", "seed_index"]):
        case, case_plain, chi, role, delta, seed_index = keys
        forward = group[group["direction"] == "forward"].sort_values("K_over_Kref")
        backward = group[group["direction"] == "backward"].sort_values("K_over_Kref")
        difference = np.abs(backward["R"].to_numpy() - forward["R"].to_numpy())
        x_values = forward["K_over_Kref"].to_numpy()
        rows.append(
            {
                "case": case,
                "case_plain": case_plain,
                "chi": chi,
                "role": role,
                "delta": delta,
                "seed_index": seed_index,
                "forward_R_lowK": float(forward["R"].iloc[0]),
                "backward_R_lowK": float(backward["R"].iloc[0]),
                "forward_R_highK": float(forward["R"].iloc[-1]),
                "hysteresis_area": float(np.trapezoid(difference, x_values)),
            }
        )
    metrics = pd.DataFrame(rows)
    drop_columns = [
        column
        for column in metrics.columns
        if column in summary.columns and column not in ["case", "case_plain", "chi", "role", "delta", "seed_index"]
    ]
    if drop_columns:
        summary = summary.drop(columns=drop_columns)
    return summary.merge(metrics, on=["case", "case_plain", "chi", "role", "delta", "seed_index"], how="left")


def run_tablevi_case(case: dict, case_index: int, delta: float, seed_index: int) -> tuple[pd.DataFrame, dict]:
    """Run one finite-N bidirectional sweep for a Table VI representative manifold."""
    seed = (
        TABLEVI_RNG_BASE
        + 100000 * case_index
        + 1000 * int(round(100 * delta))
        + 10000 * seed_index
        + 97 * case["parameter"]
    )
    rng = np.random.default_rng(seed)
    k_ref = delta
    k_values = np.geomspace(TABLEVI_K_WINDOW[0] * k_ref, TABLEVI_K_WINDOW[1] * k_ref, TABLEVI_K_GRID_SIZE)
    burn_step = int(TABLEVI_BURN_FRACTION * TABLEVI_RELAX_STEPS)

    if case["kind"] == "torus":
        theta = torus_initial_angles(rng, TABLEVI_N, case["parameter"])
        omega = tablevi_torus_frequencies(rng, TABLEVI_N, case["parameter"], delta)
        scans = integrate_torus_sweep(
            theta,
            omega,
            k_values,
            TABLEVI_DT,
            TABLEVI_RELAX_STEPS,
            burn_step,
            TABLEVI_SAMPLE_EVERY,
        )
    elif case["kind"] == "projective":
        complex_dim = case["parameter"] + 1
        z = random_projective_states(rng, TABLEVI_N, complex_dim)
        H = projective_drifts(
            rng,
            TABLEVI_N,
            complex_dim,
            delta,
            conditioned=case["conditioned"],
            noise=TABLEVI_CONDITIONING_NOISE,
        )
        scans = integrate_projective_sweep(
            z,
            H,
            k_values,
            TABLEVI_DT,
            TABLEVI_RELAX_STEPS,
            burn_step,
            TABLEVI_SAMPLE_EVERY,
        )
    else:
        raise ValueError(f"Unknown Table VI case kind: {case['kind']}")

    scan_summary = summarize_scan(k_values, k_ref, scans)
    summary = {
        "case": case["case"],
        "case_plain": case["case_plain"],
        "case_index": case_index,
        "kind": case["kind"],
        "dimension": case["dimension"],
        "chi": case["chi"],
        "parameter": case["parameter"],
        "conditioned": case["conditioned"],
        "role": case["role"],
        "delta": delta,
        "seed_index": seed_index,
        "seed": seed,
        "N": TABLEVI_N,
        "relax_steps": TABLEVI_RELAX_STEPS,
        "K_grid_size": TABLEVI_K_GRID_SIZE,
        "conditioning_noise": TABLEVI_CONDITIONING_NOISE if case["kind"] == "projective" else np.nan,
        "forward_R_lowK": float(scans[0, 0]),
        "backward_R_lowK": float(scans[1, 0]),
        "forward_R_highK": float(scans[0, -1]),
        **scan_summary,
    }

    rows = []
    common_keys = [
        "case",
        "case_plain",
        "case_index",
        "kind",
        "dimension",
        "chi",
        "parameter",
        "conditioned",
        "role",
        "delta",
        "seed_index",
        "seed",
        "N",
        "relax_steps",
        "K_grid_size",
        "conditioning_noise",
    ]
    for scan_index, direction in enumerate(["forward", "backward"]):
        for k_index, K in enumerate(k_values):
            rows.append(
                {
                    **{key: summary[key] for key in common_keys},
                    "direction": direction,
                    "K_index": k_index,
                    "K": float(K),
                    "K_over_Kref": float(K / k_ref),
                    "R": float(scans[scan_index, k_index]),
                }
            )
    return pd.DataFrame(rows), summary


def run_tablevi_formal_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the formal Table VI non-spherical finite-N experiment."""
    raw_frames = []
    summaries = []
    total = len(TABLEVI_CASES) * len(TABLEVI_DELTAS) * len(list(TABLEVI_SEEDS))
    completed = 0
    for case_index, case in enumerate(TABLEVI_CASES):
        for delta in TABLEVI_DELTAS:
            for seed_index in TABLEVI_SEEDS:
                raw, summary = run_tablevi_case(case, case_index, delta, seed_index)
                raw_frames.append(raw)
                summaries.append(summary)
                completed += 1
                print(f"Completed Table VI sweep {completed}/{total}: {case['case_plain']}, delta={delta}, seed={seed_index}")
    raw = pd.concat(raw_frames, ignore_index=True)
    summary = add_tablevi_continuation_metrics(raw, pd.DataFrame(summaries))
    raw.to_csv(TABLEVI_RAW_PATH, index=False)
    summary.to_csv(TABLEVI_SUMMARY_PATH, index=False)
    return raw, summary


def load_or_run_tablevi_formal(recompute: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load or run the formal Table VI non-spherical finite-N experiment."""
    if recompute or not (TABLEVI_RAW_PATH.exists() and TABLEVI_SUMMARY_PATH.exists()):
        return run_tablevi_formal_experiment()
    raw = pd.read_csv(TABLEVI_RAW_PATH)
    summary = add_tablevi_continuation_metrics(raw, pd.read_csv(TABLEVI_SUMMARY_PATH))
    return raw, summary
