from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def kappa_sphere(D: int) -> float:
    """Return the averaged tangent-projection factor for S^{D-1} in R^D."""
    return (D - 1) / D


def hopf_threshold_theory(D: int, delta: float) -> float:
    """Return K_c for the Lorentzian Hopf-rotation ensemble."""
    return delta / kappa_sphere(D)


def frequency_density_at_zero(delta: float, distribution: str) -> float:
    """Return g_delta(0) for a normalized even unimodal density."""
    if distribution == "lorentzian":
        return 1.0 / (np.pi * delta)
    if distribution == "gaussian":
        return 1.0 / (np.sqrt(2.0 * np.pi) * delta)
    if distribution == "uniform":
        return 1.0 / (2.0 * delta)
    raise ValueError(f"Unknown distribution: {distribution}")


def frequency_density(omega: np.ndarray | float, delta: float, distribution: str):
    """Evaluate an even frequency density used in the spectral tests."""
    if distribution == "lorentzian":
        return delta / (np.pi * (np.asarray(omega) ** 2 + delta**2))
    if distribution == "gaussian":
        return np.exp(-0.5 * (np.asarray(omega) / delta) ** 2) / (np.sqrt(2.0 * np.pi) * delta)
    if distribution == "uniform":
        omega_arr = np.asarray(omega)
        return np.where(np.abs(omega_arr) <= delta, 0.5 / delta, 0.0)
    raise ValueError(f"Unknown distribution: {distribution}")


def critical_coupling_theory(D: int, delta: float, distribution: str) -> float:
    """Return K_c=1/[kappa(M) pi g_delta(0)] for the Hopf-rotation ensemble."""
    return 1.0 / (kappa_sphere(D) * np.pi * frequency_density_at_zero(delta, distribution))


def spectral_susceptibility_hopf(
    D: int,
    delta: float,
    distribution: str,
    epsilon_factor: float,
) -> float:
    """Evaluate the finite-epsilon scalar Hopf-rotation susceptibility.

    The finite-epsilon quantity is
        s_epsilon = kappa(S^{D-1}) int epsilon g(w)/(epsilon^2+w^2) dw.
    It converges to kappa(S^{D-1}) pi g(0) as epsilon -> 0+.
    """
    from scipy.special import erfcx

    epsilon = epsilon_factor * delta
    ratio = epsilon / delta

    if distribution == "lorentzian":
        integral = 1.0 / (delta + epsilon)
    elif distribution == "gaussian":
        integral = np.sqrt(np.pi / 2.0) * erfcx(ratio / np.sqrt(2.0)) / delta
    elif distribution == "uniform":
        integral = np.arctan(1.0 / ratio) / delta
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return kappa_sphere(D) * integral


def scan_spectral_thresholds(
    D_values: Iterable[int],
    delta_values: Iterable[float],
    distributions: Iterable[str],
    epsilon_factors: Iterable[float],
) -> pd.DataFrame:
    """Compute finite-epsilon linear thresholds from the susceptibility."""
    records = []
    for distribution in distributions:
        for D in D_values:
            for delta in delta_values:
                Kc_theory = critical_coupling_theory(D, float(delta), distribution)
                for epsilon_factor in epsilon_factors:
                    susceptibility = spectral_susceptibility_hopf(
                        D=D,
                        delta=float(delta),
                        distribution=distribution,
                        epsilon_factor=float(epsilon_factor),
                    )
                    Kc_numeric = 1.0 / susceptibility
                    records.append(
                        {
                            "distribution": distribution,
                            "D": D,
                            "delta": float(delta),
                            "epsilon_factor": float(epsilon_factor),
                            "susceptibility": susceptibility,
                            "Kc_numeric": Kc_numeric,
                            "Kc_theory": Kc_theory,
                            "collapse": Kc_numeric / Kc_theory,
                        }
                    )
    return pd.DataFrame.from_records(records)


def complex_structure(D: int) -> np.ndarray:
    """Return the canonical block complex structure on R^D."""
    if D % 2 != 0:
        raise ValueError("The Hopf ensemble requires an even ambient dimension D.")
    J = np.zeros((D, D), dtype=float)
    for j in range(0, D, 2):
        J[j, j + 1] = -1.0
        J[j + 1, j] = 1.0
    return J


def random_sphere_points(rng: np.random.Generator, n: int, D: int) -> np.ndarray:
    """Draw n approximately uniform points on S^{D-1}."""
    x = rng.normal(size=(n, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def balanced_sphere_points(rng: np.random.Generator, n: int, D: int) -> np.ndarray:
    """Draw paired antipodal points to suppress finite-N incoherent noise."""
    half = n // 2
    x_half = random_sphere_points(rng, half, D)
    x = np.vstack([x_half, -x_half])
    if 2 * half < n:
        x = np.vstack([x, random_sphere_points(rng, 1, D)])
    rng.shuffle(x, axis=0)
    return x


def deterministic_lorentzian_sample(
    n: int,
    delta: float,
    rng: np.random.Generator,
    clip_quantile: float = 0.995,
) -> np.ndarray:
    """Draw a shuffled deterministic Lorentzian quantile sample.

    The clipping avoids extremely rare frequencies that dominate finite-N
    integration cost while preserving the central density g_delta(0).
    """
    q = (np.arange(n) + 0.5) / n
    q = np.clip(q, 1.0 - clip_quantile, clip_quantile)
    omega = delta * np.tan(np.pi * (q - 0.5))
    rng.shuffle(omega)
    return omega


def sphere_order_parameter(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the embedded order parameter and its norm."""
    r = x.mean(axis=0)
    return r, float(np.linalg.norm(r))


def integrate_hopf_sphere(
    D: int,
    K: float,
    delta: float,
    n: int,
    seed: int,
    dt: float,
    steps: int,
    sample_every: int = 10,
    initial_bias: float = 0.0,
    balanced_initial: bool = False,
) -> pd.DataFrame:
    """Integrate the Hopf-rotation model on S^{D-1}.

    The finite-N equation is
        dot{x_i} = omega_i J x_i + K (I - x_i x_i^T) r.
    Retraction is implemented by normalizing after each Euler step.
    """
    rng = np.random.default_rng(seed)
    J = complex_structure(D)
    if balanced_initial:
        x = balanced_sphere_points(rng, n, D)
    else:
        x = random_sphere_points(rng, n, D)
    if initial_bias > 0:
        x[:, 0] += initial_bias
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    omega = deterministic_lorentzian_sample(n, delta, rng)
    records = []
    for step in range(steps + 1):
        if step % sample_every == 0:
            _, R = sphere_order_parameter(x)
            records.append({"time": step * dt, "R": R})
        r = x.mean(axis=0)
        drift = omega[:, None] * (x @ J.T)
        proj_coeff = x @ r
        coupling = K * (r[None, :] - proj_coeff[:, None] * x)
        x = x + dt * (drift + coupling)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return pd.DataFrame.from_records(records)


def estimate_growth_rate(trace: pd.DataFrame, fit_fraction: float = 0.45) -> float:
    """Estimate the early exponential growth rate from log R(t)."""
    data = trace.copy()
    data = data[data["R"] > 1e-12]
    if len(data) < 5:
        return np.nan
    t_max = data["time"].min() + fit_fraction * (data["time"].max() - data["time"].min())
    fit = data[data["time"] <= t_max]
    if len(fit) < 5:
        fit = data.iloc[: max(5, len(data) // 3)]
    coeff = np.polyfit(fit["time"].to_numpy(), np.log(fit["R"].to_numpy()), 1)
    return float(coeff[0])


def estimate_threshold_from_growth_with_status(growth_table: pd.DataFrame) -> tuple[float, str]:
    """Estimate K_c by linear interpolation through lambda(K)=0.

    The status field records whether the scan brackets zero. A bracketed
    estimate is suitable for the main threshold figure. Boundary statuses
    identify parameter sets whose K window should be enlarged or whose growth
    fits should be rerun with longer integration time.
    """
    table = growth_table.sort_values("K")
    values = table[["K", "lambda"]].dropna().to_numpy()
    if len(values) < 2:
        return np.nan, "insufficient"
    for (k0, l0), (k1, l1) in zip(values[:-1], values[1:]):
        if l0 == 0:
            return float(k0), "interpolated"
        if l0 * l1 < 0:
            return float(k0 - l0 * (k1 - k0) / (l1 - l0)), "interpolated"
    idx = int(np.argmin(np.abs(values[:, 1])))
    if np.all(values[:, 1] < 0):
        return float(values[idx, 0]), "all_negative"
    if np.all(values[:, 1] > 0):
        return float(values[idx, 0]), "all_positive"
    return float(values[idx, 0]), "nearest"


def estimate_threshold_from_growth(growth_table: pd.DataFrame) -> float:
    """Estimate K_c by linear interpolation through lambda(K)=0."""
    value, _ = estimate_threshold_from_growth_with_status(growth_table)
    return value


def growth_threshold_summary(growth: pd.DataFrame) -> pd.DataFrame:
    """Summarize growth-rate scans into seed-level threshold estimates."""
    records = []
    group_cols = ["D", "delta", "N", "seed"]
    for keys, group in growth.groupby(group_cols, sort=False):
        D, delta, n, seed = keys
        Kc_est, status = estimate_threshold_from_growth_with_status(group)
        records.append(
            {
                "D": D,
                "delta": delta,
                "N": n,
                "seed": seed,
                "Kc_est": Kc_est,
                "Kc_theory": hopf_threshold_theory(D, delta),
                "collapse": Kc_est * kappa_sphere(D) / delta,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def _hopf_growth_task(
    D: int,
    delta: float,
    n: int,
    K: float,
    seed: int,
    dt: float,
    steps: int,
    sample_every: int,
    initial_bias: float,
    fit_fraction: float,
) -> dict:
    """Run one Hopf-sphere growth-rate task."""
    trace = integrate_hopf_sphere(
        D=D,
        K=float(K),
        delta=float(delta),
        n=int(n),
        seed=int(seed),
        dt=dt,
        steps=steps,
        sample_every=sample_every,
        initial_bias=initial_bias,
        balanced_initial=True,
    )
    lam = estimate_growth_rate(trace, fit_fraction=fit_fraction)
    return {
        "D": D,
        "delta": delta,
        "N": n,
        "K": float(K),
        "Kc_theory": hopf_threshold_theory(D, delta),
        "seed": seed,
        "lambda": lam,
    }


def scan_growth_rates(
    D_values: Iterable[int],
    delta_values: Iterable[float],
    n_values: Iterable[int],
    k_grid_size: int,
    seeds: Iterable[int],
    dt: float,
    steps: int,
    sample_every: int,
    k_window: tuple[float, float] = (0.70, 1.30),
    initial_bias: float = 0.04,
    fit_fraction: float = 0.50,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run a growth-rate scan around the theoretical threshold."""
    tasks = []
    for D in D_values:
        for delta in delta_values:
            Kc = hopf_threshold_theory(D, delta)
            K_values = np.linspace(k_window[0] * Kc, k_window[1] * Kc, k_grid_size)
            for n in n_values:
                for K in K_values:
                    for seed in seeds:
                        tasks.append(
                            (
                                D,
                                float(delta),
                                int(n),
                                float(K),
                                int(seed),
                                dt,
                                steps,
                                sample_every,
                                initial_bias,
                                fit_fraction,
                            )
                        )
    if n_jobs == 1:
        records = [_hopf_growth_task(*task) for task in tasks]
    else:
        from joblib import Parallel, delayed

        records = Parallel(n_jobs=n_jobs, batch_size="auto", verbose=10)(
            delayed(_hopf_growth_task)(*task) for task in tasks
        )
    return pd.DataFrame.from_records(records)


def steady_R_hopf_sphere(
    D: int,
    K: float,
    delta: float,
    n: int,
    seed: int,
    dt: float,
    steps: int,
    burn_fraction: float = 0.5,
) -> float:
    """Return the late-time mean order parameter for the Hopf-sphere model."""
    trace = integrate_hopf_sphere(
        D=D,
        K=K,
        delta=delta,
        n=n,
        seed=seed,
        dt=dt,
        steps=steps,
        sample_every=max(1, steps // 200),
        initial_bias=1e-3,
    )
    t_cut = trace["time"].min() + burn_fraction * (trace["time"].max() - trace["time"].min())
    return float(trace.loc[trace["time"] >= t_cut, "R"].mean())


def random_skew_symmetric_matrices(
    rng: np.random.Generator,
    n: int,
    D: int,
    delta: float,
) -> np.ndarray:
    """Draw random skew-symmetric matrices in so(D) with controlled scale."""
    mats = np.empty((n, D, D), dtype=float)
    for i in range(n):
        a = rng.normal(size=(D, D))
        omega = 0.5 * (a - a.T)
        omega *= delta / max(np.linalg.norm(omega), 1e-12)
        mats[i] = omega
    return mats


def integrate_general_sphere(
    D: int,
    K: float,
    delta: float,
    n: int,
    seed: int,
    dt: float,
    steps: int,
    sample_every: int = 10,
    initial_bias: float = 0.0,
) -> pd.DataFrame:
    """Integrate the generalized sphere model with random so(D) drift fields."""
    rng = np.random.default_rng(seed)
    x = random_sphere_points(rng, n, D)
    if initial_bias > 0:
        x[:, 0] += initial_bias
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    omega = random_skew_symmetric_matrices(rng, n, D, delta)
    records = []
    for step in range(steps + 1):
        if step % sample_every == 0:
            _, R = sphere_order_parameter(x)
            records.append({"time": step * dt, "R": R})
        r = x.mean(axis=0)
        drift = np.einsum("nij,nj->ni", omega, x)
        proj_coeff = x @ r
        coupling = K * (r[None, :] - proj_coeff[:, None] * x)
        x = x + dt * (drift + coupling)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return pd.DataFrame.from_records(records)


def steady_R_general_sphere(
    D: int,
    K: float,
    delta: float,
    n: int,
    seed: int,
    dt: float,
    steps: int,
    burn_fraction: float = 0.5,
) -> float:
    """Return the late-time mean order parameter for the general sphere model."""
    trace = integrate_general_sphere(
        D=D,
        K=K,
        delta=delta,
        n=n,
        seed=seed,
        dt=dt,
        steps=steps,
        sample_every=max(1, steps // 200),
        initial_bias=1e-3,
    )
    t_cut = trace["time"].min() + burn_fraction * (trace["time"].max() - trace["time"].min())
    return float(trace.loc[trace["time"] >= t_cut, "R"].mean())


def relax_general_sphere_state(
    x: np.ndarray,
    omega: np.ndarray,
    K: float,
    dt: float,
    steps: int,
    burn_fraction: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Relax one finite-N state at fixed K and return the late-time mean R."""
    burn_step = int(np.floor(burn_fraction * steps))
    sample_every = max(1, steps // 100)
    R_values = []
    for step in range(steps):
        r = x.mean(axis=0)
        drift = np.einsum("nij,nj->ni", omega, x)
        proj_coeff = x @ r
        coupling = K * (r[None, :] - proj_coeff[:, None] * x)
        x = x + dt * (drift + coupling)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if step >= burn_step and step % sample_every == 0:
            _, R = sphere_order_parameter(x)
            R_values.append(R)
    if not R_values:
        _, R = sphere_order_parameter(x)
        R_values.append(R)
    return x, float(np.mean(R_values))


def general_sphere_continuation_sweep(
    D: int,
    K_values: np.ndarray,
    delta: float,
    n: int,
    seed: int,
    dt: float,
    relax_steps: int,
    initial_bias: float = 1e-3,
    burn_fraction: float = 0.5,
) -> pd.DataFrame:
    """Run a forward and backward K-sweep with fixed drifts and continued states."""
    rng = np.random.default_rng(seed)
    x = random_sphere_points(rng, n, D)
    if initial_bias > 0:
        x[:, 0] += initial_bias
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    omega = random_skew_symmetric_matrices(rng, n, D, delta)

    records = []
    for direction, K_sequence in (("forward", K_values), ("backward", K_values[::-1])):
        for K in K_sequence:
            x, R = relax_general_sphere_state(
                x=x,
                omega=omega,
                K=float(K),
                dt=dt,
                steps=relax_steps,
                burn_fraction=burn_fraction,
            )
            records.append({"direction": direction, "K": float(K), "R": R})
    return pd.DataFrame.from_records(records)


def threshold_proxy_from_steady_scan(scan: pd.DataFrame, R_cut: float = 0.08) -> float:
    """Estimate K_c from a steady-state R(K) scan."""
    table = scan.sort_values("K")
    values = table[["K", "R"]].dropna().to_numpy()
    if len(values) == 0:
        return np.nan
    above = values[:, 1] >= R_cut
    if not np.any(above):
        return float(values[np.argmax(values[:, 1]), 0])
    idx = int(np.argmax(above))
    if idx == 0:
        return float(values[0, 0])
    k0, r0 = values[idx - 1]
    k1, r1 = values[idx]
    if r1 == r0:
        return float(k1)
    return float(k0 + (R_cut - r0) * (k1 - k0) / (r1 - r0))


def scan_steady_thresholds(
    D_values: Iterable[int],
    delta_values: Iterable[float],
    seeds: Iterable[int],
    n: int,
    dt: float,
    steps: int,
    k_grid_size: int,
    R_cut: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run steady-state threshold scans and return raw and summary tables."""
    raw_records = []
    summary_records = []
    for D in D_values:
        for delta in delta_values:
            Kc = hopf_threshold_theory(D, delta)
            K_values = np.linspace(0.65 * Kc, 1.45 * Kc, k_grid_size)
            for seed in seeds:
                seed_records = []
                for K in K_values:
                    R = steady_R_hopf_sphere(D, float(K), float(delta), n, int(seed), dt, steps)
                    rec = {
                        "D": D,
                        "delta": delta,
                        "seed": seed,
                        "K": float(K),
                        "R": R,
                        "Kc_theory": Kc,
                    }
                    raw_records.append(rec)
                    seed_records.append(rec)
                Kc_est = threshold_proxy_from_steady_scan(pd.DataFrame(seed_records), R_cut=R_cut)
                summary_records.append(
                    {
                        "D": D,
                        "delta": delta,
                        "seed": seed,
                        "Kc_est": Kc_est,
                        "Kc_theory": Kc,
                        "collapse": Kc_est * kappa_sphere(D) / delta,
                    }
                )
    return pd.DataFrame(raw_records), pd.DataFrame(summary_records)


@dataclass
class SweepResult:
    raw: pd.DataFrame
    summary: pd.DataFrame


def hysteresis_sphere_family(
    D_values: Iterable[int],
    delta_values: Iterable[float],
    seeds: Iterable[int],
    n: int,
    dt: float,
    relax_steps: int,
    k_grid_size: int,
    k_window: tuple[float, float] = (0.5, 1.8),
    burn_fraction: float = 0.5,
) -> SweepResult:
    """Run forward/backward continuation on hyperspheres with so(D) drifts."""
    records = []
    summary = []
    for D in D_values:
        manifold_dimension = D - 1
        sphere = f"S^{manifold_dimension}"
        chi = 2 if manifold_dimension % 2 == 0 else 0
        for delta in delta_values:
            Kc_ref = delta / kappa_sphere(D)
            K_values = np.linspace(k_window[0] * Kc_ref, k_window[1] * Kc_ref, k_grid_size)
            for seed in seeds:
                sweep = general_sphere_continuation_sweep(
                    D=D,
                    K_values=K_values,
                    delta=float(delta),
                    n=n,
                    seed=int(seed),
                    dt=dt,
                    relax_steps=relax_steps,
                    burn_fraction=burn_fraction,
                )
                for _, row in sweep.iterrows():
                    records.append(
                        {
                            "sphere": sphere,
                            "ambient_dimension": D,
                            "manifold_dimension": manifold_dimension,
                            "chi": chi,
                            "delta": float(delta),
                            "seed": int(seed),
                            "direction": row["direction"],
                            "K": float(row["K"]),
                            "R": float(row["R"]),
                            "Kc_ref": Kc_ref,
                        }
                    )
                forward_table = sweep[sweep["direction"] == "forward"].sort_values("K")
                backward_table = sweep[sweep["direction"] == "backward"].sort_values("K")
                f = forward_table["R"].to_numpy()
                b = backward_table["R"].to_numpy()
                area = float(np.trapezoid(np.abs(b - f), K_values))
                jump = float(np.max(np.abs(np.diff(f))))
                summary.append(
                    {
                        "sphere": sphere,
                        "ambient_dimension": D,
                        "manifold_dimension": manifold_dimension,
                        "chi": chi,
                        "delta": float(delta),
                        "seed": int(seed),
                        "hysteresis_area": area,
                        "jump_size": jump,
                        "Kc_ref": Kc_ref,
                    }
                )
    return SweepResult(raw=pd.DataFrame(records), summary=pd.DataFrame(summary))


def cp2_random_state(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw random normalized complex vectors in C^3."""
    z = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def cp2_random_skew_hermitian(rng: np.random.Generator, n: int, delta: float) -> np.ndarray:
    """Draw traceless skew-Hermitian matrices with controlled scale."""
    mats = np.empty((n, 3, 3), dtype=complex)
    for i in range(n):
        a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        h = 0.5 * (a + a.conj().T)
        h -= np.trace(h) * np.eye(3) / 3.0
        h *= delta / max(np.linalg.norm(h), 1e-12)
        mats[i] = 1j * h
    return mats


def cp2_order(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the CP^2 centered-projector order parameter."""
    P = z[:, :, None] * z[:, None, :].conj()
    r = P.mean(axis=0) - np.eye(3) / 3.0
    R = float(np.sqrt(np.real(np.trace(r @ r))))
    return r, R


def integrate_cp2(
    K: float,
    delta: float,
    n: int,
    seed: int,
    dt: float,
    steps: int,
    sample_every: int = 10,
    initial_bias: float = 0.0,
) -> pd.DataFrame:
    """Integrate the CP^2 rank-one projector model."""
    rng = np.random.default_rng(seed)
    z = cp2_random_state(rng, n)
    if initial_bias > 0:
        z[:, 0] += initial_bias
        z /= np.linalg.norm(z, axis=1, keepdims=True)
    omega = cp2_random_skew_hermitian(rng, n, delta)
    records = []
    for step in range(steps + 1):
        if step % sample_every == 0:
            _, R = cp2_order(z)
            records.append({"time": step * dt, "R": R})
        r, _ = cp2_order(z)
        rz = z @ r.T
        inner = np.sum(z.conj() * rz, axis=1)
        tangent = rz - inner[:, None] * z
        drift = np.einsum("nij,nj->ni", omega, z)
        z = z + dt * (drift + K * tangent)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
    return pd.DataFrame.from_records(records)


def steady_R_cp2(K: float, delta: float, n: int, seed: int, dt: float, steps: int) -> float:
    """Return the late-time mean CP^2 order parameter."""
    trace = integrate_cp2(K, delta, n, seed, dt, steps, sample_every=max(1, steps // 200), initial_bias=1e-3)
    t_cut = trace["time"].min() + 0.5 * (trace["time"].max() - trace["time"].min())
    return float(trace.loc[trace["time"] >= t_cut, "R"].mean())


def cp2_sweep(
    delta_values: Iterable[float],
    seeds: Iterable[int],
    n: int,
    dt: float,
    steps: int,
    k_grid_size: int,
) -> SweepResult:
    """Run forward/backward K sweeps for CP^2."""
    records = []
    summary = []
    for delta in delta_values:
        K_values = np.linspace(0.4 * delta, 4.0 * delta, k_grid_size)
        for seed in seeds:
            forward = []
            backward = []
            for K in K_values:
                R = steady_R_cp2(float(K), float(delta), n, int(seed), dt, steps)
                forward.append(R)
                records.append({"manifold": "CP^2", "delta": delta, "seed": seed, "direction": "forward", "K": float(K), "R": R})
            for K in K_values[::-1]:
                R = steady_R_cp2(float(K), float(delta), n, int(seed) + 100000, dt, steps)
                backward.append(R)
                records.append({"manifold": "CP^2", "delta": delta, "seed": seed, "direction": "backward", "K": float(K), "R": R})
            f = np.asarray(forward)
            b = np.asarray(backward[::-1])
            area = float(np.trapezoid(np.abs(b - f), K_values))
            jump = float(np.max(np.abs(np.diff(f))))
            summary.append({"manifold": "CP^2", "delta": delta, "seed": seed, "hysteresis_area": area, "jump_size": jump})
    return SweepResult(raw=pd.DataFrame(records), summary=pd.DataFrame(summary))


def defect_index_samples(samples_per_manifold: int, seed: int = 1) -> pd.DataFrame:
    """Return exact Poincare-Hopf reference values, not simulated data.

    This helper supports auxiliary checks of defect counts. The manuscript
    figures use the dedicated data-generation routines in the figure scripts.
    """
    manifolds = [
        ("S^2", 2, 2),
        ("S^4", 2, 2),
        ("S^2 x S^2", 4, 4),
        ("CP^2", 3, 3),
        ("T^2", 0, 0),
        ("U(2)", 0, 0),
    ]
    records = []
    for name, chi, min_defects in manifolds:
        for sample in range(samples_per_manifold):
            records.append(
                {
                    "manifold": name,
                    "sample": sample,
                    "chi": chi,
                    "index_sum": chi,
                    "defect_count": min_defects,
                    "data_source": "exact Poincare-Hopf reference",
                }
            )
    return pd.DataFrame.from_records(records)
