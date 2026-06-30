from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit
from paths import DATA_DIR, FIGURE_DIR, ensure_directories
from plotting import set_paper_style, save_figure
from representative_manifold_dynamics import PRODUCT_CASES, TABLEVI_CASES, load_or_run_product_formal, load_or_run_tablevi_formal


RNG_BASE = 91023
MANIFOLD_DIMS = list(range(1, 11))
DELTAS = [0.35, 0.50]
SEEDS = range(15)

N = 384
DT = 0.012
RELAX_STEPS = 1200
K_GRID_SIZE = 81
K_WINDOW = (0.08, 1.85)
BURN_FRACTION = 0.60
SAMPLE_EVERY = 20

CONDITIONING_NOISE = 0.35
HIGH_DIMS = [3, 4, 5, 6]
HIGH_SEEDS = range(2)
HIGH_N = 1024
HIGH_RELAX_STEPS = 3000
HIGH_K_GRID_SIZE = 181

RAW_PATH = DATA_DIR / "fig05_construction_conditioned_raw.csv"
SUMMARY_PATH = DATA_DIR / "fig05_construction_conditioned_summary.csv"
FIT_PATH = DATA_DIR / "fig05_construction_conditioned_fit.csv"
MERGED_PATH = DATA_DIR / "fig05_construction_conditioned_merged.csv"
CONSTRUCTION_PATH = DATA_DIR / "fig05_construction_conditioned_ensemble.csv"
HIGH_PATH = DATA_DIR / "fig05_construction_conditioned_high_resolution_D3_D6.csv"


@njit(fastmath=True)
def integrate_bidirectional_sweep(x, omega, k_values, dt, relax_steps, burn_step, sample_every):
    n_points, ambient_dim = x.shape
    n_k = len(k_values)
    result = np.zeros((2, n_k))
    for scan in range(2):
        for scan_index in range(n_k):
            k_index = scan_index if scan == 0 else n_k - 1 - scan_index
            K = k_values[k_index]
            sample_sum = 0.0
            sample_count = 0
            for step in range(relax_steps):
                r = np.zeros(ambient_dim)
                for i in range(n_points):
                    for j in range(ambient_dim):
                        r[j] += x[i, j]
                for j in range(ambient_dim):
                    r[j] /= n_points

                for i in range(n_points):
                    proj = 0.0
                    for j in range(ambient_dim):
                        proj += x[i, j] * r[j]

                    norm_sq = 0.0
                    for j in range(ambient_dim):
                        drift_j = 0.0
                        for ell in range(ambient_dim):
                            drift_j += omega[i, j, ell] * x[i, ell]
                        value = x[i, j] + dt * (drift_j + K * (r[j] - proj * x[i, j]))
                        x[i, j] = value
                        norm_sq += value * value

                    inv_norm = 1.0 / np.sqrt(norm_sq)
                    for j in range(ambient_dim):
                        x[i, j] *= inv_norm

                if step >= burn_step and step % sample_every == 0:
                    r_norm_sq = 0.0
                    for j in range(ambient_dim):
                        component = 0.0
                        for i in range(n_points):
                            component += x[i, j]
                        component /= n_points
                        r_norm_sq += component * component
                    sample_sum += np.sqrt(r_norm_sq)
                    sample_count += 1

            result[scan, k_index] = sample_sum / max(sample_count, 1)
    return result


def euler_sphere(manifold_dim: int) -> int:
    return 2 if manifold_dim % 2 == 0 else 0


def kappa_for_sphere(manifold_dim: int) -> float:
    return manifold_dim / (manifold_dim + 1)


def random_points(rng, n_points: int, ambient_dim: int) -> np.ndarray:
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


def axial_direction(manifold_dim: int) -> np.ndarray:
    ambient_dim = manifold_dim + 1
    A = np.zeros((ambient_dim, ambient_dim))
    for p in range(manifold_dim // 2):
        i = 2 * p
        j = 2 * p + 1
        A[i, j] = -1.0
        A[j, i] = 1.0
    norm = np.sqrt(np.sum(A * A))
    if norm == 0.0:
        return A
    return np.sqrt(2.0) * A / norm


def sign_conditioned_skew_directions(rng, n_points: int, manifold_dim: int) -> np.ndarray:
    ambient_dim = manifold_dim + 1
    base = axial_direction(manifold_dim)
    A = np.repeat(base[None, :, :], n_points, axis=0)
    B = random_skew_directions(rng, n_points, ambient_dim)
    A = (1.0 - CONDITIONING_NOISE) * A + CONDITIONING_NOISE * B
    norm = np.sqrt(np.sum(A * A, axis=(1, 2), keepdims=True))
    A = np.sqrt(2.0) * A / np.maximum(norm, 1.0e-12)
    signs = rng.choice([-1.0, 1.0], size=n_points)
    return A * signs[:, None, None]


def build_drifts(manifold_dim: int, delta: float, n_points: int, seed: int, conditioned: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if conditioned:
        directions = sign_conditioned_skew_directions(rng, n_points, manifold_dim)
    else:
        directions = random_skew_directions(rng, n_points, manifold_dim + 1)
    rates = lorentzian_quantiles(n_points, delta, rng)
    return directions * rates[:, None, None]


def run_sweep_with_drifts(manifold_dim, delta, seed, omega, n_points, relax_steps, k_grid_size):
    rng = np.random.default_rng(seed + 7919)
    x = random_points(rng, n_points, manifold_dim + 1)
    K_ref = delta / kappa_for_sphere(manifold_dim)
    K_values = np.geomspace(K_WINDOW[0] * K_ref, K_WINDOW[1] * K_ref, k_grid_size)
    R_scans = integrate_bidirectional_sweep(
        x,
        omega,
        K_values,
        DT,
        relax_steps,
        int(BURN_FRACTION * relax_steps),
        SAMPLE_EVERY,
    )
    return K_values, K_ref, R_scans


def summarize_scan(K_values, K_ref, R_scans):
    R_forward = R_scans[0]
    R_backward = R_scans[1]
    jumps = np.abs(np.diff(R_forward))
    scan_diff = np.abs(R_backward - R_forward)
    jump_index = int(np.argmax(jumps))
    diff_index = int(np.argmax(scan_diff))
    return {
        "jump_size": float(jumps[jump_index]),
        "max_scan_difference": float(scan_diff[diff_index]),
        "K_at_jump": float((K_values / K_ref)[jump_index]),
        "K_at_max_scan_difference": float((K_values / K_ref)[diff_index]),
    }


def construction_status(manifold_dim: int) -> str:
    if euler_sphere(manifold_dim) == 0:
        return "ordinary random skew drift"
    return "axial sign-conditioned skew drift"


def fit_stable_branch(K_over_Kref, R):
    mask = (R > 0.04) & (R < 0.72)
    if np.sum(mask) < 8:
        return np.nan, np.nan, np.nan, int(np.sum(mask))
    x2 = R[mask] ** 2
    X = np.column_stack([np.ones(np.sum(mask)), x2, x2**2])
    y = K_over_Kref[mask]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[1]), float(coef[2]), float(1.0 - ss_res / max(ss_tot, 1.0e-12)), int(np.sum(mask))


def warm_up_numba():
    rng = np.random.default_rng(0)
    x = random_points(rng, 8, 4)
    omega = np.zeros((8, 4, 4))
    integrate_bidirectional_sweep(x, omega, np.geomspace(0.1, 0.2, 3), DT, 5, 2, 1)


def run_main_experiment():
    warm_up_numba()
    raw_rows = []
    summary_rows = []
    fit_rows = []
    construction_rows = []
    start = time.time()
    for manifold_dim in MANIFOLD_DIMS:
        for delta in DELTAS:
            for seed_index in SEEDS:
                base_seed = RNG_BASE + 10000 * manifold_dim + 1000 * int(round(100 * delta)) + seed_index
                conditioned = euler_sphere(manifold_dim) != 0
                seed = base_seed
                omega = build_drifts(manifold_dim, delta, N, seed, conditioned=conditioned)
                K_values, K_ref, R_scans = run_sweep_with_drifts(
                    manifold_dim, delta, seed, omega, N, RELAX_STEPS, K_GRID_SIZE
                )
                scan_summary = summarize_scan(K_values, K_ref, R_scans)
                b2, b4, fit_r2, n_fit = fit_stable_branch(K_values / K_ref, R_scans[0])
                construction_rows.append(
                    {
                        "manifold_dimension": manifold_dim,
                        "chi": euler_sphere(manifold_dim),
                        "delta": delta,
                        "seed_index": seed_index,
                        "base_seed": base_seed,
                        "seed": seed,
                        "construction_status": construction_status(manifold_dim),
                        "conditioned": conditioned,
                    }
                )
                summary_rows.append(
                    {
                        "manifold": rf"$S^{manifold_dim}$",
                        "manifold_dimension": manifold_dim,
                        "chi": euler_sphere(manifold_dim),
                        "delta": delta,
                        "seed_index": seed_index,
                        "seed": seed,
                        "conditioned": conditioned,
                        **scan_summary,
                    }
                )
                fit_rows.append(
                    {
                        "manifold": rf"$S^{manifold_dim}$",
                        "manifold_dimension": manifold_dim,
                        "chi": euler_sphere(manifold_dim),
                        "delta": delta,
                        "seed_index": seed_index,
                        "seed": seed,
                        "cubic_slope_fit": b2,
                        "quintic_slope_fit": b4,
                        "fit_r2": fit_r2,
                        "n_fit": n_fit,
                    }
                )
                for scan_idx, direction in enumerate(["forward", "backward"]):
                    for k_idx, K in enumerate(K_values):
                        raw_rows.append(
                            {
                                "manifold": rf"$S^{manifold_dim}$",
                                "manifold_dimension": manifold_dim,
                                "chi": euler_sphere(manifold_dim),
                                "delta": delta,
                                "seed_index": seed_index,
                                "seed": seed,
                                "conditioned": conditioned,
                                "direction": direction,
                                "K_index": k_idx,
                                "K": float(K),
                                "K_over_Kref": float(K / K_ref),
                                "R": float(R_scans[scan_idx, k_idx]),
                                "K_ref": float(K_ref),
                                "N": N,
                                "relax_steps": RELAX_STEPS,
                                "K_grid_size": K_GRID_SIZE,
                            }
                        )
                if len(summary_rows) % 20 == 0:
                    elapsed = (time.time() - start) / 60.0
                    print(f"Completed {len(summary_rows)}/300 final sweeps in {elapsed:.1f} min")

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    fit_summary = pd.DataFrame(fit_rows)
    construction = pd.DataFrame(construction_rows)
    merged = summary.merge(
        fit_summary,
        on=["manifold", "manifold_dimension", "chi", "delta", "seed_index", "seed"],
        how="left",
    )
    raw.to_csv(RAW_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    fit_summary.to_csv(FIT_PATH, index=False)
    construction.to_csv(CONSTRUCTION_PATH, index=False)
    merged.to_csv(MERGED_PATH, index=False)
    return raw, summary, fit_summary, construction, merged


def run_high_resolution_curves():
    warm_up_numba()
    rows = []
    for manifold_dim in HIGH_DIMS:
        for seed_index in HIGH_SEEDS:
            delta = 0.35
            base_seed = RNG_BASE + 700000 + 10000 * manifold_dim + seed_index
            conditioned = euler_sphere(manifold_dim) != 0
            seed = base_seed
            omega = build_drifts(manifold_dim, delta, HIGH_N, seed, conditioned=conditioned)
            K_values, K_ref, R_scans = run_sweep_with_drifts(
                manifold_dim, delta, seed, omega, HIGH_N, HIGH_RELAX_STEPS, HIGH_K_GRID_SIZE
            )
            for scan_idx, direction in enumerate(["forward", "backward"]):
                for k_idx, K in enumerate(K_values):
                    rows.append(
                        {
                            "manifold": rf"$S^{manifold_dim}$",
                            "manifold_dimension": manifold_dim,
                            "chi": euler_sphere(manifold_dim),
                            "delta": delta,
                            "seed_index": seed_index,
                            "seed": seed,
                            "conditioned": conditioned,
                            "direction": direction,
                            "K_index": k_idx,
                            "K": float(K),
                            "K_over_Kref": float(K / K_ref),
                            "R": float(R_scans[scan_idx, k_idx]),
                            "K_ref": float(K_ref),
                            "N": HIGH_N,
                            "relax_steps": HIGH_RELAX_STEPS,
                            "K_grid_size": HIGH_K_GRID_SIZE,
                            "construction_status": construction_status(manifold_dim),
                        }
                    )
    high = pd.DataFrame(rows)
    high.to_csv(HIGH_PATH, index=False)
    return high


def load_or_run(recompute: bool = False):
    ensure_directories()
    if recompute or not (RAW_PATH.exists() and SUMMARY_PATH.exists() and FIT_PATH.exists() and CONSTRUCTION_PATH.exists()):
        raw, summary, fit_summary, construction, merged = run_main_experiment()
    else:
        raw = pd.read_csv(RAW_PATH)
        summary = pd.read_csv(SUMMARY_PATH)
        fit_summary = pd.read_csv(FIT_PATH)
        construction = pd.read_csv(CONSTRUCTION_PATH)
        merged = pd.read_csv(MERGED_PATH)
    if recompute or not HIGH_PATH.exists():
        high = run_high_resolution_curves()
    else:
        high = pd.read_csv(HIGH_PATH)
    product_raw, product_summary = load_or_run_product_formal(recompute=recompute)
    tablevi_raw, tablevi_summary = load_or_run_tablevi_formal(recompute=recompute)
    return raw, summary, fit_summary, construction, merged, high, product_raw, product_summary, tablevi_raw, tablevi_summary


def make_figure(
    summary: pd.DataFrame,
    high: pd.DataFrame,
    product_raw: pd.DataFrame | None = None,
    product_summary: pd.DataFrame | None = None,
    tablevi_raw: pd.DataFrame | None = None,
    tablevi_summary: pd.DataFrame | None = None,
):
    set_paper_style()
    title_size = 9.5
    label_size = 9.5
    tick_size = 8.5
    legend_size = 7.2
    legend_title_size = 8.2
    panel_label_size = 12
    palette = dict(zip(MANIFOLD_DIMS, sns.color_palette("viridis", n_colors=len(MANIFOLD_DIMS))))
    curve_palette = dict(zip(HIGH_DIMS, sns.color_palette("viridis", n_colors=len(HIGH_DIMS))))
    curve_markers = {3: "o", 4: "s", 5: "o", 6: "s"}
    legend_style = dict(
        frameon=True,
        fontsize=legend_size,
        title_fontsize=legend_title_size,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.35,
        handletextpad=0.45,
    )

    fig, axes = plt.subplots(3, 3, figsize=(11.45, 9.45))
    axes = axes.ravel()

    curve = (
        high[high["direction"] == "forward"]
        .groupby(["manifold_dimension", "K_index"], as_index=False)
        .agg(K_over_Kref=("K_over_Kref", "mean"), R_mean=("R", "mean"), R_std=("R", "std"))
    )
    for manifold_dim in HIGH_DIMS:
        part = curve[curve["manifold_dimension"] == manifold_dim].sort_values("K_over_Kref")
        color = curve_palette[manifold_dim]
        axes[0].scatter(
            part["K_over_Kref"],
            part["R_mean"],
            s=11,
            color=color,
            marker=curve_markers[manifold_dim],
            alpha=0.84,
            edgecolors="white",
            linewidths=0.25,
            label=rf"$D={manifold_dim}$",
        )
    for manifold_dim, y_offset in [(4, -0.08), (6, 0.06)]:
        part = curve[curve["manifold_dimension"] == manifold_dim].sort_values("K_over_Kref")
        R_values = part["R_mean"].to_numpy()
        K_values = part["K_over_Kref"].to_numpy()
        jump_index = int(np.argmax(np.abs(np.diff(R_values))))
        jump_K = K_values[jump_index + 1]
        jump_R = R_values[jump_index + 1]
        axes[0].annotate(
            r"Jump",
            xy=(jump_K, jump_R),
            xytext=(jump_K * 1.55, np.clip(jump_R + y_offset, 0.12, 0.94)),
            arrowprops=dict(arrowstyle="->", lw=0.8, color=curve_palette[manifold_dim]),
            color=curve_palette[manifold_dim],
            fontsize=legend_size,
            ha="left",
            va="center",
        )
    axes[0].set_xscale("log")
    axes[0].set_title(r"Phase transition on hyperspheres")
    axes[0].set_xlabel(r"$K/K_{\rm ref}$")
    axes[0].set_ylabel(r"$R=\|r\|$")
    axes[0].legend(loc="lower right", ncol=2, **legend_style)

    marker_by_chi = {0: "o", 2: "s"}
    label_by_chi = {0: r"$\chi(S^D)=0$", 2: r"$\chi(S^D)=2$"}
    jitter_rng = np.random.default_rng(20260629)

    for chi_value in [0, 2]:
        part = summary[summary["chi"] == chi_value]
        x_values = part["manifold_dimension"].to_numpy(dtype=float) - 1.0
        jitter = jitter_rng.uniform(-0.17, 0.17, size=len(part))
        colors = [palette[int(dim)] for dim in part["manifold_dimension"]]
        axes[1].scatter(
            x_values + jitter,
            part["jump_size"],
            s=15,
            marker=marker_by_chi[chi_value],
            c=colors,
            alpha=0.62,
            edgecolors="none",
            label=label_by_chi[chi_value],
        )
    sns.pointplot(
        data=summary,
        x="manifold_dimension",
        y="jump_size",
        order=MANIFOLD_DIMS,
        color="black",
        errorbar=("ci", 95),
        markers="_",
        linestyles="none",
        ax=axes[1],
    )
    axes[1].set_title(r"Jump size")
    axes[1].set_xlabel(r"Manifold dimension $D$")
    axes[1].set_ylabel(r"$\max |\Delta R|$")
    axes[1].legend(loc="best", **legend_style)

    for chi_value in [0, 2]:
        part = summary[summary["chi"] == chi_value]
        x_values = part["manifold_dimension"].to_numpy(dtype=float) - 1.0
        jitter = jitter_rng.uniform(-0.17, 0.17, size=len(part))
        colors = [palette[int(dim)] for dim in part["manifold_dimension"]]
        axes[2].scatter(
            x_values + jitter,
            part["max_scan_difference"],
            s=15,
            marker=marker_by_chi[chi_value],
            c=colors,
            alpha=0.62,
            edgecolors="none",
            label=label_by_chi[chi_value],
        )
    sns.pointplot(
        data=summary,
        x="manifold_dimension",
        y="max_scan_difference",
        order=MANIFOLD_DIMS,
        color="black",
        errorbar=("ci", 95),
        markers="_",
        linestyles="none",
        ax=axes[2],
    )
    axes[2].set_title(r"Maximum scan difference")
    axes[2].set_xlabel(r"Manifold dimension $D$")
    axes[2].set_ylabel(r"$\max_K |R_{\rm b}(K)-R_{\rm f}(K)|$")
    axes[2].legend(loc="best", **legend_style)

    if product_raw is None or product_summary is None:
        product_raw, product_summary = load_or_run_product_formal(recompute=False)
    product_order = [case["case"] for case in PRODUCT_CASES]
    product_palette = dict(zip(product_order, sns.color_palette("viridis", n_colors=len(product_order))))
    product_marker_by_chi = {0: "o", 4: "s"}
    product_label_by_chi = {0: r"$\chi=0$", 4: r"$\chi=4$"}

    product_curve = (
        product_raw[(product_raw["direction"] == "forward") & (product_raw["delta"] == 0.35)]
        .groupby(["case", "K_index"], as_index=False)
        .agg(K_over_Kref=("K_over_Kref", "mean"), R_mean=("R", "mean"), R_std=("R", "std"))
    )
    for case in PRODUCT_CASES:
        case_label = case["case"]
        part = product_curve[product_curve["case"] == case_label].sort_values("K_over_Kref")
        marker = product_marker_by_chi[case["chi"]]
        color = product_palette[case_label]
        axes[3].scatter(
            part["K_over_Kref"],
            part["R_mean"],
            marker=marker,
            s=12,
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.25,
            label=case_label,
        )
    for case_label, y_offset in [(r"$S^2\times S^2$", -0.10), (r"$S^2\times S^4$", 0.06)]:
        part = product_curve[product_curve["case"] == case_label].sort_values("K_over_Kref")
        R_values = part["R_mean"].to_numpy()
        K_values = part["K_over_Kref"].to_numpy()
        jump_index = int(np.argmax(np.abs(np.diff(R_values))))
        jump_K = K_values[jump_index + 1]
        jump_R = R_values[jump_index + 1]
        axes[3].annotate(
            r"Jump",
            xy=(jump_K, jump_R),
            xytext=(jump_K * 1.55, np.clip(jump_R + y_offset, 0.14, 0.94)),
            arrowprops=dict(arrowstyle="->", lw=0.8, color=product_palette[case_label]),
            color=product_palette[case_label],
            fontsize=legend_size,
            ha="left",
            va="center",
        )
    axes[3].set_xscale("log")
    axes[3].set_title(r"Phase transition on product manifolds")
    axes[3].set_xlabel(r"$K/K_{\rm ref}$")
    axes[3].set_ylabel(r"$R$")
    axes[3].legend(loc="lower right", ncol=1, **legend_style)

    product_rng = np.random.default_rng(20260630)
    for chi_value in [0, 4]:
        part = product_summary[product_summary["chi"] == chi_value]
        x_values = part["case_index"].to_numpy(dtype=float)
        jitter = product_rng.uniform(-0.16, 0.16, size=len(part))
        colors = [product_palette[case] for case in part["case"]]
        axes[4].scatter(
            x_values + jitter,
            part["jump_size"],
            marker=product_marker_by_chi[chi_value],
            c=colors,
            s=16,
            alpha=0.66,
            edgecolors="none",
            label=product_label_by_chi[chi_value],
        )
        axes[5].scatter(
            x_values + jitter,
            part["max_scan_difference"],
            marker=product_marker_by_chi[chi_value],
            c=colors,
            s=16,
            alpha=0.66,
            edgecolors="none",
            label=product_label_by_chi[chi_value],
        )
    for ax, y_column in [(axes[4], "jump_size"), (axes[5], "max_scan_difference")]:
        sns.pointplot(
            data=product_summary,
            x="case_index",
            y=y_column,
            order=list(range(len(product_order))),
            color="black",
            errorbar=("ci", 95),
            markers="_",
            linestyles="none",
            ax=ax,
        )
        ax.set_xticks(list(range(len(product_order))))
        ax.set_xticklabels(product_order, rotation=16, ha="right")
        ax.set_xlabel(r"Product manifold")
    axes[4].set_title(r"Jump size")
    axes[4].set_ylabel(r"$\max |\Delta R|$")
    axes[4].legend(loc="best", **legend_style)
    axes[5].set_title(r"Maximum scan difference")
    axes[5].set_ylabel(r"$\max_K |R_{\rm b}(K)-R_{\rm f}(K)|$")
    axes[5].legend(loc="best", **legend_style)

    if tablevi_raw is None or tablevi_summary is None:
        tablevi_raw, tablevi_summary = load_or_run_tablevi_formal(recompute=False)
    tablevi_order = [case["case"] for case in TABLEVI_CASES]
    tablevi_palette = dict(zip(tablevi_order, sns.color_palette("viridis", n_colors=len(tablevi_order))))
    tablevi_marker_by_chi = {0: "o", 3: "s", 4: "s"}
    tablevi_label_by_chi = {0: r"$\chi=0$", 3: r"$\chi=3$", 4: r"$\chi=4$"}

    tablevi_curve = (
        tablevi_raw[(tablevi_raw["direction"] == "forward") & (tablevi_raw["delta"] == 0.35)]
        .groupby(["case", "K_index"], as_index=False)
        .agg(K_over_Kref=("K_over_Kref", "mean"), R_mean=("R", "mean"), R_std=("R", "std"))
    )
    for case in TABLEVI_CASES:
        case_label = case["case"]
        part = tablevi_curve[tablevi_curve["case"] == case_label].sort_values("K_over_Kref")
        marker = tablevi_marker_by_chi[case["chi"]]
        color = tablevi_palette[case_label]
        axes[6].scatter(
            part["K_over_Kref"],
            part["R_mean"],
            marker=marker,
            s=12,
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.25,
            label=case_label,
        )
    for case_label, y_offset in [(r"$\mathbb{CP}^{2}$", -0.10), (r"$\mathbb{CP}^{3}$", 0.06)]:
        part = tablevi_curve[tablevi_curve["case"] == case_label].sort_values("K_over_Kref")
        R_values = part["R_mean"].to_numpy()
        K_values = part["K_over_Kref"].to_numpy()
        jump_index = int(np.argmax(np.abs(np.diff(R_values))))
        jump_K = K_values[jump_index + 1]
        jump_R = R_values[jump_index + 1]
        axes[6].annotate(
            r"Jump",
            xy=(jump_K, jump_R),
            xytext=(jump_K * 1.55, np.clip(jump_R + y_offset, 0.14, 0.94)),
            arrowprops=dict(arrowstyle="->", lw=0.8, color=tablevi_palette[case_label]),
            color=tablevi_palette[case_label],
            fontsize=legend_size,
            ha="left",
            va="center",
        )
    axes[6].set_xscale("log")
    axes[6].set_ylim(0.0, 1.04)
    axes[6].set_title(r"Phase transition on non-spherical manifolds")
    axes[6].set_xlabel(r"$K/K_{\rm ref}$")
    axes[6].set_ylabel(r"$R$")
    axes[6].legend(loc="lower right", ncol=1, **legend_style)

    tablevi_rng = np.random.default_rng(20260701)
    for chi_value in [0, 3, 4]:
        part = tablevi_summary[tablevi_summary["chi"] == chi_value]
        x_values = part["case_index"].to_numpy(dtype=float)
        jitter = tablevi_rng.uniform(-0.16, 0.16, size=len(part))
        colors = [tablevi_palette[case] for case in part["case"]]
        axes[7].scatter(
            x_values + jitter,
            part["jump_size"],
            marker=tablevi_marker_by_chi[chi_value],
            c=colors,
            s=16,
            alpha=0.66,
            edgecolors="none",
            label=tablevi_label_by_chi[chi_value],
        )
        axes[8].scatter(
            x_values + jitter,
            part["max_scan_difference"],
            marker=tablevi_marker_by_chi[chi_value],
            c=colors,
            s=16,
            alpha=0.66,
            edgecolors="none",
            label=tablevi_label_by_chi[chi_value],
        )
    for ax, y_column in [(axes[7], "jump_size"), (axes[8], "max_scan_difference")]:
        sns.pointplot(
            data=tablevi_summary,
            x="case_index",
            y=y_column,
            order=list(range(len(tablevi_order))),
            color="black",
            errorbar=("ci", 95),
            markers="_",
            linestyles="none",
            ax=ax,
        )
        ax.set_xticks(list(range(len(tablevi_order))))
        ax.set_xticklabels(tablevi_order, rotation=16, ha="right")
        ax.set_xlabel(r"Non-spherical manifold")
    axes[7].set_title(r"Jump size")
    axes[7].set_ylabel(r"$\max |\Delta R|$")
    axes[7].legend(loc="best", **legend_style)
    axes[8].set_title(r"Maximum scan difference")
    axes[8].set_ylabel(r"$\max_K |R_{\rm b}(K)-R_{\rm f}(K)|$")
    axes[8].legend(loc="best", **legend_style)

    for ax in axes:
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontsize(tick_size)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965], w_pad=1.35, h_pad=1.75)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"], axes):
        box = ax.get_position()
        fig.text(box.x0 - 0.018, box.y1 + 0.014, label, ha="left", va="bottom", fontsize=panel_label_size, fontweight="normal")

    save_figure(fig, FIGURE_DIR / "fig05_direct_dynamics")
    fig.savefig(FIGURE_DIR / "fig05_direct_dynamics_600dpi.png", dpi=600, bbox_inches="tight")
    return fig
