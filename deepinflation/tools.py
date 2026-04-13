"""Tools for Inflation Agent

Two main tools for analyzing inflation potentials:
- analyze_potential: Compute observables (ns, r, A_s) for all trajectories (~1s)
- plot_potential: Generate 3-panel diagnostic plot (~2s)
  - Panel 1: V(φ) with trajectory markers (φ_end, N=50, N=60)
  - Panel 2: Slow-roll parameters (ε, η) vs φ
  - Panel 3: Predicted (ns, r) vs observational posteriors
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .inflation import compute_observables_all_trajectories, generate_plot_data

# Load observational posterior data
_PROJECT_ROOT = Path(__file__).parent.parent
_BK18_DATA_PATH = _PROJECT_ROOT / "data/bk18_planck_posterior.npz"
_BK18_DATA = np.load(_BK18_DATA_PATH) if _BK18_DATA_PATH.exists() else None
_PACT_DATA_PATH = _PROJECT_ROOT / "data/planck_act_posterior.npz"
_PACT_DATA = np.load(_PACT_DATA_PATH) if _PACT_DATA_PATH.exists() else None

logger = logging.getLogger(__name__)
TAB10 = colormaps["tab10"]


def analyze_potential(expression: str) -> str:
    """Compute ns, r, A_s for all inflation trajectories. Returns JSON with observables.

    Expression must use only 'phi' and concrete numbers (NO symbolic parameters like M, V0).

    Args:
        expression: V(φ) with concrete values only.
            Valid: 'phi^2' or 'phi**2', '(1-exp(-sqrt(2/3)*phi))^2'.
            Invalid: 'M*phi^2', 'V0*exp(-phi)'.
    """
    logger.debug("[Analyze] V(φ) = %s", expression)
    try:
        trajectories = compute_observables_all_trajectories(expression)
        if not trajectories:
            return json.dumps({"success": False, "error": "No valid trajectories"}, indent=2)
        return json.dumps(
            {
                "success": True,
                "expression": expression,
                "num_trajectories": len(trajectories),
                "trajectories": trajectories,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def plot_potential(expression: str, output_path: str = "./potential_plot.png") -> str:
    """Generate 3-panel plot: [1] V(φ) with trajectory markers, [2] slow-roll params (ε,η), [3] ns-r plane with observational contours.

    Returns JSON with plot_path.

    Args:
        expression: V(φ) with concrete values.
            Valid: 'phi^2', '(1-exp(-0.816*phi))^2'.
            Invalid: 'V0*phi^2'.
        output_path: Save path (default: './potential_plot.png').
    """
    logger.debug("[Plot] V(φ) = %s", expression)
    try:
        phi, V, eps, eta = generate_plot_data(expression)
        trajectories_60 = compute_observables_all_trajectories(expression, N=60.0)
        trajectories_50 = compute_observables_all_trajectories(expression, N=50.0)
        trajectory_pairs = list(zip(trajectories_60, trajectories_50, strict=True))
        trajectories_N = (
            [compute_observables_all_trajectories(expression, N=float(N_val)) for N_val in np.linspace(50, 60, 11)]
            if trajectory_pairs
            else []
        )

        # Limit the potential panel to the visible inflation window.
        phi_min, phi_max = phi[0], phi[-1]
        if trajectory_pairs:
            all_phi = [p for t60, t50 in trajectory_pairs for p in [t60["phi_end"], t50["phi_N"], t60["phi_N"]]]
            phi_min = max(phi[0], min(all_phi) - 3)
            phi_max = min(phi[-1], max(all_phi) + 3)

        positive_V_indices = np.flatnonzero(V > 0)
        if positive_V_indices.size:
            phi_min = max(phi_min, phi[positive_V_indices[0]])
            phi_max = min(phi_max, phi[positive_V_indices[-1]])

        mask = (phi >= phi_min) & (phi <= phi_max)
        phi_plot, V_plot, eps_plot, eta_plot = phi[mask], V[mask], eps[mask], eta[mask]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        fig.suptitle(f"V(φ) = {expression}", fontsize=11, y=0.98)

        # Panel 1: potential with trajectory markers.
        axes[0].plot(phi_plot, V_plot, linewidth=2, color="#2E86AB", alpha=0.8)

        if trajectory_pairs:
            colors = TAB10(np.arange(len(trajectory_pairs)))
            for i, (t60, t50) in enumerate(trajectory_pairs):
                V_end = V[np.argmin(np.abs(phi - t60["phi_end"]))]
                V_50 = V[np.argmin(np.abs(phi - t50["phi_N"]))]
                V_60 = V[np.argmin(np.abs(phi - t60["phi_N"]))]

                axes[0].scatter(t60["phi_end"], V_end, s=60, c=[colors[i]], marker="x", linewidths=2.5, zorder=10)
                axes[0].scatter(
                    [t50["phi_N"], t60["phi_N"]],
                    [V_50, V_60],
                    s=[40, 60],
                    c=[colors[i], colors[i]],
                    marker="o",
                    edgecolors="black",
                    linewidths=1,
                    zorder=9,
                )

            legend = []
            if len(trajectory_pairs) > 1:
                legend.extend(
                    [
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=colors[i],
                            markeredgecolor="black",
                            markersize=7,
                            linewidth=0,
                            label=f"Trajectory #{i + 1}",
                        )
                        for i in range(len(trajectory_pairs))
                    ]
                )
            legend.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="x",
                        color=colors[0],
                        markersize=8,
                        markeredgewidth=2.5,
                        linewidth=0,
                        label="φ_end",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[0],
                        markeredgecolor="black",
                        markersize=6,
                        linewidth=0,
                        label="N=50",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[0],
                        markeredgecolor="black",
                        markersize=8,
                        linewidth=0,
                        label="N=60",
                    ),
                ]
            )
            axes[0].legend(handles=legend, fontsize=9, loc="best", framealpha=0.95, edgecolor="gray")

        axes[0].set(xlabel="φ", ylabel="V(φ)", title="Potential")
        axes[0].grid(True, alpha=0.3, linestyle=":")

        # Panel 2: slow-roll parameters.
        valid_eps = np.isfinite(eps_plot) & (eps_plot > 0) & (eps_plot < 1e2)
        valid_eta = np.isfinite(eta_plot) & (np.abs(eta_plot) > 0) & (np.abs(eta_plot) < 1e2)
        if np.any(valid_eps):
            axes[1].plot(phi_plot[valid_eps], eps_plot[valid_eps], label="ε", linewidth=2, color="#A23B72", alpha=0.8)
        if np.any(valid_eta):
            axes[1].plot(
                phi_plot[valid_eta], np.abs(eta_plot[valid_eta]), label="|η|", linewidth=2, color="#F18F01", alpha=0.8
            )
        axes[1].axhline(1, color="black", linestyle="--", alpha=0.5, linewidth=1.5)
        axes[1].set(
            xlabel="φ", ylabel="Slow-roll parameters", title="Slow-roll Parameters", yscale="log", ylim=[1e-4, 1e2]
        )
        handles, _ = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, which="both", linestyle=":")

        # Panel 3: observational contours then model predictions.
        posterior_legend = []
        posterior_specs = [
            (1, _BK18_DATA, "P_bk18", "levels_bk18", TAB10(0), "BK18 + Planck 2018"),
            (4, _PACT_DATA, "P_pact", "levels_pact", TAB10(1), "BK18 + Planck + ACT DR6"),
        ]
        for zorder, data, P_key, levels_key, color, label in posterior_specs:
            if data is None:
                continue

            ns, r, P = data["ns"], data["r"], data[P_key]
            levels = data[levels_key]
            axes[2].contourf(ns, r, P, levels=[levels[0], levels[1]], colors=[color], alpha=0.4, zorder=zorder)
            axes[2].contourf(ns, r, P, levels=[levels[1], P.max()], colors=[color], alpha=0.8, zorder=zorder + 1)
            axes[2].contour(ns, r, P, levels=levels, colors=[color], linewidths=1.2, alpha=0.9, zorder=zorder + 2)
            posterior_legend.append(
                Patch(
                    facecolor=color,
                    alpha=0.8,
                    edgecolor=color,
                    linewidth=1.2,
                    label=label,
                )
            )

        axes[2].grid(True, alpha=0.3, linestyle=":", zorder=0)
        axes[2].set(xlim=[0.95, 1.0])

        if trajectory_pairs:
            for i, (t60, t50) in enumerate(trajectory_pairs):
                color = TAB10((i + 2) % 10)

                ns_line = [traj[i]["ns"] for traj in trajectories_N if traj and len(traj) > i]
                r_line = [traj[i]["r"] for traj in trajectories_N if traj and len(traj) > i]

                if len(ns_line) > 1:
                    axes[2].plot(ns_line, r_line, "-", color=color, alpha=0.7, linewidth=2.5, zorder=7)

                axes[2].scatter(
                    t50["ns"],
                    t50["r"],
                    s=40,
                    c=[color],
                    marker="o",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=8,
                    alpha=0.95,
                )
                axes[2].scatter(
                    t60["ns"],
                    t60["r"],
                    s=60,
                    c=[color],
                    marker="o",
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=9,
                    alpha=0.95,
                )

            legend = list(posterior_legend)
            if len(trajectory_pairs) > 1:
                legend.extend(
                    [
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linewidth=2.5,
                            color=TAB10((i + 2) % 10),
                            markerfacecolor=TAB10((i + 2) % 10),
                            markeredgecolor="black",
                            markersize=5,
                            label=f"Trajectory #{i + 1}",
                        )
                        for i in range(len(trajectory_pairs))
                    ]
                )
            legend.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=TAB10(2),
                        markeredgecolor="black",
                        markersize=5,
                        linewidth=0,
                        label="N=50",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=TAB10(2),
                        markeredgecolor="black",
                        markersize=7,
                        linewidth=0,
                        label="N=60",
                    ),
                ]
            )
            axes[2].legend(handles=legend, fontsize=9, framealpha=0.95, edgecolor="gray")
            r_all = [t["r"] for t in trajectories_60 + trajectories_50]
            axes[2].set_ylim([0.0, min(0.26, max(max(r_all) * 1.4, 0.07))])
        else:
            axes[2].text(
                0.5,
                0.5,
                "No valid trajectories",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
                fontsize=11,
                color="gray",
            )
            axes[2].set_ylim([0.0, 0.26])

        axes[2].set(xlabel="$n_s$", ylabel="$r$", title="Observables vs CMB Constraints")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("[Plot] Saved to %s", output_file.absolute())

        return json.dumps({"success": True, "plot_path": str(output_file.absolute())}, indent=2)

    except Exception as e:
        return json.dumps({"success": False, "error": f"Plot error: {e}"}, indent=2)
