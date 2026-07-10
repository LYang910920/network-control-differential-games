"""Plotting and table output for the foundation example runner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cybercontrol.baseline_diagnostics import RANDOM_BASELINE_COUNT, RANDOM_BASELINE_SEED, control_baseline_rows, game_baseline_rows, save_control_baseline_plot, save_game_baseline_plot, write_baseline_table
from cybercontrol.fbs import TimeSeries, degree_control_model, degree_game_model, node_control_model, node_game_model
from cybercontrol.foundation_evaluation import evaluate_degree_control_policy, evaluate_degree_game_strategy, evaluate_node_control_policy, evaluate_node_game_strategy
from cybercontrol.foundation_profiles import CONTINUOUS_IMPULSE_PROFILE, DEGREE_CONTROL_PROFILE, DEGREE_GAME_PROFILE, NODE_CONTROL_PROFILE, NODE_GAME_PROFILE
from cybercontrol.graph_data import DegreeData
from cybercontrol.plotting import apply_clean_axes, plot_time_series, save_publication_figure

LOGGER = logging.getLogger(__name__)
LINE_WIDTH = 2.0
FIGSIZE_SINGLE = (7.4, 4.2)
FIGSIZE_STACKED = (7.4, 5.8)


def save_degree_distribution(D: DegreeData, out_dir: Path) -> None:
    """Save the empirical degree table and its guide figure."""

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"k": D.k.astype(int), "N_k": D.Nk, "P(k)": D.pk}).to_csv(
        out_dir / "degree_distribution.csv",
        index=False,
    )
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    ax.bar(D.k, D.pk, color="tab:blue", alpha=0.85)
    apply_clean_axes(
        ax,
        xlabel="degree k",
        ylabel="P(k)",
        title=f"Empirical degree distribution; average degree={D.kbar:.2f}",
    )
    savefig(out_dir / "degree_distribution")

def savefig(path: Path) -> None:
    """Save the current figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    save_publication_figure(plt.gcf(), path, metadata={"source": "foundation example"})
    plt.close()
    LOGGER.info(f"saved {path}")


def mark_event_times(ax, event_times: np.ndarray, *, color: str = "0.35",
                     label: str | None = None) -> None:
    """Mark impulse times on a state axis without implying continuous control."""
    for idx, tau in enumerate(np.asarray(event_times, dtype=float)):
        ax.axvline(
            tau,
            color=color,
            linestyle=":",
            linewidth=1.3,
            alpha=0.55,
            label=label if idx == 0 else None,
        )


def plot_impulse_events(ax, event_times: np.ndarray, heights: np.ndarray, label: str,
                        *, color: str = "tab:red") -> None:
    """Plot impulse magnitudes only at discrete event times."""
    event_times = np.asarray(event_times, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if len(event_times) == 0:
        return
    ax.vlines(event_times, 0.0, heights, color=color, linewidth=2.2, label=label)
    ax.scatter(event_times, heights, color=color, s=24, zorder=3)


def save_fbs_convergence(results: dict[str, TimeSeries], out_dir: Path) -> None:
    """Save residual curves for all forward-backward sweep examples."""
    rows = []
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    for name, result in results.items():
        history = np.asarray(result.controls.get("fbs_delta", []), dtype=float)
        if len(history) == 0:
            continue
        iterations = np.arange(1, len(history) + 1)
        ax.semilogy(iterations, history, marker="o", linewidth=LINE_WIDTH, label=name.replace("_", " "))
        tolerance = float(result.value.get("tolerance", 1e-4))
        converged = bool(result.value.get("converged", history[-1] < tolerance))
        for iteration, delta in zip(iterations, history):
            rows.append(
                {
                    "example": name,
                    "iteration": int(iteration),
                    "max_control_change": float(delta),
                    "tolerance": tolerance,
                    "below_tolerance": bool(delta < tolerance),
                    "run_converged": converged,
                }
            )

    if not rows:
        return
    pd.DataFrame(rows).to_csv(out_dir / "fbs_convergence.csv", index=False)
    apply_clean_axes(ax, xlabel="FBS iteration", ylabel="max control/strategy change", title="Forward-backward sweep convergence")
    ax.axhline(1e-4, color="0.45", linestyle=":", linewidth=1.4, label="tolerance=1e-4")
    ax.legend(frameon=False, fontsize=8)
    savefig(out_dir / "fbs_convergence")


def save_baseline_comparison(
    results: dict[str, TimeSeries],
    D: DegreeData,
    A: np.ndarray,
    out_dir: Path,
) -> None:
    """Save one baseline-comparison figure per model."""
    mixed_plot = out_dir / "baseline_comparison.png"
    if mixed_plot.exists():
        mixed_plot.unlink()

    all_rows: list[dict[str, object]] = []

    if "degree_control" in results:
        result = results["degree_control"]
        model = degree_control_model(D)
        rows = control_baseline_rows(
            result.controls["control"],
            float(result.value["cost"]),
            lambda u: evaluate_degree_control_policy(D, result.t, u)[1],
            random_upper=model.control_bound,
            seed=RANDOM_BASELINE_SEED + 11,
            model_field="example",
            model_name="degree_control",
        )
        all_rows.extend(rows)
        save_control_baseline_plot(
            rows,
            out_dir / "degree_control_baseline_comparison.png",
            title="Degree-k continuous control vs baselines",
            ylabel="cost (lower is better)",
            random_label=f"{RANDOM_BASELINE_COUNT} random smooth controls",
        )

    if "node_control" in results:
        result = results["node_control"]
        model = node_control_model(A)
        rows = control_baseline_rows(
            result.controls["control"],
            float(result.value["cost"]),
            lambda u: evaluate_node_control_policy(A, result.t, u)[1],
            random_upper=model.control_bound,
            seed=RANDOM_BASELINE_SEED + 21,
            model_field="example",
            model_name="node_control",
        )
        all_rows.extend(rows)
        save_control_baseline_plot(
            rows,
            out_dir / "node_control_baseline_comparison.png",
            title="Node-level continuous control vs baselines",
            ylabel="cost (lower is better)",
            random_label=f"{RANDOM_BASELINE_COUNT} random smooth controls",
        )

    if "degree_game" in results:
        result = results["degree_game"]
        model = degree_game_model(D)
        rows = game_baseline_rows(
            result.controls["attack"],
            result.controls["defense"],
            lambda attack, defense: evaluate_degree_game_strategy(D, result.t, attack, defense)[1:],
            attack_upper=model.attack_bound,
            defense_upper=model.defense_bound,
            seed=RANDOM_BASELINE_SEED + 31,
            model_field="example",
            model_name="degree_game",
        )
        all_rows.extend(rows)
        save_game_baseline_plot(
            rows,
            out_dir / "degree_game_baseline_comparison.png",
            title="Degree-k continuous differential game: unilateral baselines",
        )

    if "node_game" in results:
        result = results["node_game"]
        model = node_game_model(A)
        rows = game_baseline_rows(
            result.controls["attack"],
            result.controls["defense"],
            lambda attack, defense: evaluate_node_game_strategy(A, result.t, attack, defense)[1:],
            attack_upper=model.attack_bound,
            defense_upper=model.defense_bound,
            seed=RANDOM_BASELINE_SEED + 41,
            model_field="example",
            model_name="node_game",
        )
        all_rows.extend(rows)
        save_game_baseline_plot(
            rows,
            out_dir / "node_game_baseline_comparison.png",
            title="Node-level continuous differential game: unilateral baselines",
        )

    if all_rows:
        write_baseline_table(all_rows, out_dir / "baseline_summary.csv")
        LOGGER.info(f"saved model-specific baseline comparison figures to {out_dir}")


def plot_degree_control(result: TimeSeries, D: DegreeData, out_dir: Path) -> None:
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    plot_time_series(ax, result.t, result.x @ D.pk, "state: population-weighted degree-class mean infection", linestyle="-")
    plot_time_series(
        ax,
        result.t,
        result.controls["control"] @ D.pk,
        "continuous control: population-weighted degree-class mean",
        linestyle="-.",
    )
    for j in sorted(set([0, len(D.k) // 2, len(D.k) - 1])):
        plot_time_series(ax, result.t, result.x[:, j], f"state: degree k={int(D.k[j])}", linestyle="--", linewidth=1.5)
    apply_clean_axes(
        ax,
        xlabel="time",
        ylabel="infection share / control intensity",
        title=f"Degree-k continuous optimal control (population-weighted means), cost={result.value['cost']:.2f}",
    )
    ax.legend(frameon=False, fontsize=8)
    savefig(out_dir / "degree_control_trajectory")


def plot_degree_game(result: TimeSeries, D: DegreeData, out_dir: Path) -> None:
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    plot_time_series(ax, result.t, result.x @ D.pk, "state: population-weighted degree-class mean infection", linestyle="-")
    plot_time_series(
        ax,
        result.t,
        result.controls["attack"] @ D.pk,
        "continuous attack strategy: population-weighted degree-class mean",
        linestyle="--",
    )
    plot_time_series(
        ax,
        result.t,
        result.controls["defense"] @ D.pk,
        "continuous defense strategy: population-weighted degree-class mean",
        linestyle="-.",
    )
    high = int(np.argmax(D.k))
    plot_time_series(ax, result.t, result.x[:, high], f"state: high degree k={int(D.k[high])}", linestyle=":", linewidth=1.8)
    apply_clean_axes(
        ax,
        xlabel="time",
        ylabel="infection share / strategy intensity",
        title=f"Degree-k continuous game (population-weighted means), JA={result.value['JA']:.2f}, JD={result.value['JD']:.2f}",
    )
    ax.legend(frameon=False, fontsize=8)
    savefig(out_dir / "degree_game_trajectory")


def plot_node_control(result: TimeSeries, out_dir: Path) -> None:
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    plot_time_series(ax, result.t, result.x.mean(axis=1), "state: node mean infection (all nodes)", linestyle="-")
    plot_time_series(ax, result.t, result.x.max(axis=1), "state: max infection over nodes", linestyle="--")
    plot_time_series(
        ax,
        result.t,
        result.controls["control"].mean(axis=1),
        "continuous control: node mean",
        linestyle="-.",
    )
    apply_clean_axes(
        ax,
        xlabel="time",
        ylabel="infection share / control intensity",
        title=f"Node-level continuous optimal control (node aggregates), cost={result.value['cost']:.2f}",
    )
    ax.legend(frameon=False, fontsize=8)
    savefig(out_dir / "node_control_trajectory")


def plot_node_game(result: TimeSeries, out_dir: Path) -> None:
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    plot_time_series(ax, result.t, result.x.mean(axis=1), "state: node mean infection (all nodes)", linestyle="-")
    plot_time_series(
        ax,
        result.t,
        result.controls["attack"].mean(axis=1),
        "continuous attack strategy: node mean",
        linestyle="--",
    )
    plot_time_series(
        ax,
        result.t,
        result.controls["defense"].mean(axis=1),
        "continuous defense strategy: node mean",
        linestyle="-.",
    )
    apply_clean_axes(
        ax,
        xlabel="time",
        ylabel="infection share / strategy intensity",
        title=f"Node-level continuous game (node aggregates), JA={result.value['JA']:.2f}, JD={result.value['JD']:.2f}",
    )
    ax.legend(frameon=False, fontsize=8)
    savefig(out_dir / "node_game_trajectory")


def plot_continuous_impulse(result: TimeSeries, out_dir: Path) -> None:
    impulse_times = result.controls["impulse_times"]
    impulse_heights = result.controls["impulse_heights"]
    continuous_control = result.controls["continuous_control"]
    controlled_count = int(result.value["controlled_count"])
    impulse_fraction = result.value["impulse_fraction"]
    continuous_min = result.value["continuous_control_min"]
    continuous_max = result.value["continuous_control_max"]

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_STACKED, sharex=True, height_ratios=[2.0, 1.0])
    ax_state, ax_control = axes
    plot_time_series(ax_state, result.t, result.x.mean(axis=1), "state: node mean infection (all nodes)", linestyle="-")
    plot_time_series(ax_state, result.t, result.x.max(axis=1), "state: max infection over nodes", linestyle="--")
    mark_event_times(ax_state, impulse_times, label="impulse time")
    apply_clean_axes(
        ax_state,
        ylabel="infection share",
        title=f"Continuous-impulse control on {controlled_count} high-degree nodes",
    )
    ax_state.legend(frameon=False, ncol=2, fontsize=8)

    plot_time_series(
        ax_control,
        result.t,
        continuous_control,
        "continuous control: time-varying level",
        color="tab:blue",
        linestyle="-.",
    )
    plot_impulse_events(ax_control, impulse_times, impulse_heights, "impulse control: discrete fraction", color="tab:red")
    ax_control.set_ylim(0.0, max(0.65, float(np.max(impulse_heights)) + 0.08))
    apply_clean_axes(
        ax_control,
        xlabel="time",
        ylabel="control intensity / impulse fraction",
        title=f"continuous range={continuous_min:.2f}-{continuous_max:.2f}; impulse fraction={impulse_fraction:.2f}",
    )
    ax_control.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save_publication_figure(
        fig,
        out_dir / "continuous_impulse_trajectory.png",
        metadata={
            "model": "continuous-impulse example",
            "control_type": "continuous plus impulse control",
        },
    )
    plt.close(fig)
    LOGGER.info(f"saved {out_dir / 'continuous_impulse_trajectory.png'}")


def write_parameter_summary(out_dir: Path, args: argparse.Namespace, D: DegreeData, A: np.ndarray) -> None:
    """Write a compact parameter map so readers do not need to hunt literals."""
    rows: list[dict[str, object]] = []

    def add(model: str, parameter: str, value: object, meaning: str) -> None:
        rows.append({"model": model, "parameter": parameter, "value": value, "meaning": meaning})

    def add_array(model: str, parameter: str, values: np.ndarray, meaning: str) -> None:
        arr = np.asarray(values, dtype=float)
        add(model, parameter, np.array2string(arr, precision=4, separator=";"), meaning)
        add(
            model,
            f"{parameter}_summary",
            f"mean={arr.mean():.4g}; min={arr.min():.4g}; max={arr.max():.4g}",
            "Summary statistics for the resolved heterogeneous array.",
        )
        add(
            model,
            f"matched_mean_homogeneous_{parameter}",
            f"{arr.mean():.6g}",
            "Scalar matched-mean homogeneous value to use as a simple baseline profile.",
        )

    add("input graph", "full_nodes", D.node_degree.size, "Number of nodes used to compute the empirical degree distribution.")
    add("input graph", "degree_classes", len(D.k), "Number of observed degree classes in degree-level models.")
    add("input graph", "node_level_state_dimension", A.shape[0], "Reduced node count used by dense node-level PMP/game examples.")
    add("input graph", "normalization", args.normalize, "Adjacency scaling used before node-level ODE solves.")

    add(DEGREE_CONTROL_PROFILE.label, "horizon", DEGREE_CONTROL_PROFILE.horizon, "Total simulated time.")
    add(DEGREE_CONTROL_PROFILE.label, "time_grid_steps", args.steps, "Number of intervals in the degree-level FBS time grid.")
    add(DEGREE_CONTROL_PROFILE.label, "beta", DEGREE_CONTROL_PROFILE.beta, "Infection/contact rate.")
    add(DEGREE_CONTROL_PROFILE.label, "delta", DEGREE_CONTROL_PROFILE.delta, "Natural recovery/removal rate before control.")
    add(DEGREE_CONTROL_PROFILE.label, "u_max", DEGREE_CONTROL_PROFILE.control_max, "Upper bound for continuous healing control u_k(t).")
    add(DEGREE_CONTROL_PROFILE.label, "state/control weights", f"{DEGREE_CONTROL_PROFILE.state_weight}/{DEGREE_CONTROL_PROFILE.control_weight}", "Objective tradeoff between infection and control effort.")
    add(DEGREE_CONTROL_PROFILE.label, "heterogeneity_profile", "degree_correlated_sis_params(strength=0.28)", "Deterministic degree-correlated physical/economic heterogeneity; no random seed is used.")
    degree_control_params = degree_control_model(D)
    for name in ("susceptibility", "infectivity", "recovery", "state_weight", "control_weight", "control_bound"):
        add_array(DEGREE_CONTROL_PROFILE.label, f"resolved_{name}_by_degree_class", getattr(degree_control_params, name), "Class-specific heterogeneous array used by the degree-level FBS solve.")

    add(DEGREE_GAME_PROFILE.label, "horizon", DEGREE_GAME_PROFILE.horizon, "Total simulated time.")
    add(DEGREE_GAME_PROFILE.label, "time_grid_steps", args.steps, "Number of intervals in the degree-level game time grid.")
    add(DEGREE_GAME_PROFILE.label, "beta", DEGREE_GAME_PROFILE.beta, "Infection/contact rate under attack and defense.")
    add(DEGREE_GAME_PROFILE.label, "delta", DEGREE_GAME_PROFILE.delta, "Natural recovery/removal rate before defense.")
    add(DEGREE_GAME_PROFILE.label, "attack/defense bounds", f"{DEGREE_GAME_PROFILE.attack_max}/{DEGREE_GAME_PROFILE.defense_max}", "Upper bounds for continuous attack and defense strategies.")
    add(DEGREE_GAME_PROFILE.label, "payoff weights", f"{DEGREE_GAME_PROFILE.reward_attacker}/{DEGREE_GAME_PROFILE.loss_defender}", "Attacker reward and defender loss weights on infection.")
    add(DEGREE_GAME_PROFILE.label, "heterogeneity_profile", "degree_correlated_sis_params(strength=0.22)", "Deterministic degree-correlated physical/economic heterogeneity; no random seed is used.")
    degree_game_params = degree_game_model(D)
    for name in ("susceptibility", "infectivity", "recovery", "attack_reward", "defense_loss", "attack_cost", "defense_cost", "attack_bound", "defense_bound"):
        add_array(DEGREE_GAME_PROFILE.label, f"resolved_{name}_by_degree_class", getattr(degree_game_params, name), "Class-specific heterogeneous array used by the degree-level game solve and baseline comparison.")

    node_steps = max(24, args.steps // 3)
    add(NODE_CONTROL_PROFILE.label, "horizon", NODE_CONTROL_PROFILE.horizon, "Total simulated time.")
    add(NODE_CONTROL_PROFILE.label, "time_grid_steps", node_steps, "Number of intervals in the node-level FBS time grid.")
    add(NODE_CONTROL_PROFILE.label, "beta", NODE_CONTROL_PROFILE.beta, "Node-level infection/contact rate.")
    add(NODE_CONTROL_PROFILE.label, "delta", NODE_CONTROL_PROFILE.delta, "Node-level natural recovery/removal rate before control.")
    add(NODE_CONTROL_PROFILE.label, "u_max", NODE_CONTROL_PROFILE.control_max, "Upper bound for continuous node control u_i(t).")
    add(NODE_CONTROL_PROFILE.label, "initial infected nodes", 2, "The two highest-degree nodes start with infection 0.12; others start at 0.02.")
    add(NODE_CONTROL_PROFILE.label, "heterogeneity_profile", "degree_correlated_node_sis_params(strength=0.28)", "Deterministic degree-correlated node heterogeneity; no random seed is used.")
    node_control_params = node_control_model(A)
    for name in ("susceptibility", "infectivity", "recovery", "state_weight", "control_weight", "control_bound"):
        add_array(NODE_CONTROL_PROFILE.label, f"resolved_{name}_by_node", getattr(node_control_params, name), "Node-specific heterogeneous array used by the node-level FBS solve.")

    add(NODE_GAME_PROFILE.label, "horizon", NODE_GAME_PROFILE.horizon, "Total simulated time.")
    add(NODE_GAME_PROFILE.label, "time_grid_steps", node_steps, "Number of intervals in the node-level game time grid.")
    add(NODE_GAME_PROFILE.label, "beta", NODE_GAME_PROFILE.beta, "Node-level infection/contact rate under attack and defense.")
    add(NODE_GAME_PROFILE.label, "delta", NODE_GAME_PROFILE.delta, "Node-level natural recovery/removal rate before defense.")
    add(NODE_GAME_PROFILE.label, "attack/defense bounds", f"{NODE_GAME_PROFILE.attack_max}/{NODE_GAME_PROFILE.defense_max}", "Upper bounds for continuous attack and defense strategies.")
    add(NODE_GAME_PROFILE.label, "heterogeneity_profile", "degree_correlated_node_sis_params(strength=0.22)", "Deterministic degree-correlated node heterogeneity; no random seed is used.")
    node_game_params = node_game_model(A)
    for name in ("susceptibility", "infectivity", "recovery", "attack_reward", "defense_loss", "attack_cost", "defense_cost", "attack_bound", "defense_bound"):
        add_array(NODE_GAME_PROFILE.label, f"resolved_{name}_by_node", getattr(node_game_params, name), "Node-specific heterogeneous array used by the node-level game solve and baseline comparison.")

    add("continuous-impulse simulation", "horizon", CONTINUOUS_IMPULSE_PROFILE.horizon, "Total simulated time.")
    add("continuous-impulse simulation", "beta", CONTINUOUS_IMPULSE_PROFILE.beta, "Node-level infection/contact rate between impulse events.")
    add("continuous-impulse simulation", "delta", CONTINUOUS_IMPULSE_PROFILE.delta, "Natural recovery/removal rate between impulse events.")
    add("continuous-impulse simulation", "continuous_control_range", f"{CONTINUOUS_IMPULSE_PROFILE.continuous_lower}-{CONTINUOUS_IMPULSE_PROFILE.continuous_upper}", "Bounds for the time-varying continuous control level.")
    add("continuous-impulse simulation", "impulse_times", ", ".join(str(tau) for tau in CONTINUOUS_IMPULSE_PROFILE.impulse_times), "Discrete event times where the state is jumped.")
    add("continuous-impulse simulation", "impulse_fraction", CONTINUOUS_IMPULSE_PROFILE.impulse_fraction, "Fraction by which controlled high-degree node states are reduced at each impulse.")
    add("baselines", "random_smooth_controls", RANDOM_BASELINE_COUNT, "Random continuous controls/strategies used in each baseline comparison.")

    pd.DataFrame(rows).to_csv(out_dir / "parameter_summary.csv", index=False)
