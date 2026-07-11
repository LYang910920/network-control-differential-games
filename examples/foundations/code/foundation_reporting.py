"""Plotting and table output for the foundation example runner."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cybercontrol.baseline_diagnostics import (
    RANDOM_BASELINE_COUNT,
    RANDOM_BASELINE_SEED,
    control_baseline_rows,
    game_baseline_rows,
    save_control_baseline_plot,
    save_game_baseline_plot,
    write_baseline_table,
)
from cybercontrol.fbs import (
    TimeSeries,
    degree_control_model,
    degree_game_model,
    node_control_model,
    node_game_model,
)
from cybercontrol.foundation_evaluation import (
    evaluate_degree_control_policy,
    evaluate_degree_game_strategy,
    evaluate_node_control_policy,
    evaluate_node_game_strategy,
)
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


def mark_event_times(
    ax, event_times: np.ndarray, *, color: str = "0.35", label: str | None = None
) -> None:
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


def plot_impulse_events(
    ax, event_times: np.ndarray, heights: np.ndarray, label: str, *, color: str = "tab:red"
) -> None:
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
        ax.semilogy(
            iterations, history, marker="o", linewidth=LINE_WIDTH, label=name.replace("_", " ")
        )
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
    apply_clean_axes(
        ax,
        xlabel="FBS iteration",
        ylabel="max control/strategy change",
        title="Forward-backward sweep convergence",
    )
    ax.axhline(1e-4, color="0.45", linestyle=":", linewidth=1.4, label="tolerance=1e-4")
    ax.legend(frameon=False, fontsize=8, loc="best")
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
            lambda u: evaluate_degree_control_policy(D, result.t, u, model=result.model)[1],
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
            lambda u: evaluate_node_control_policy(A, result.t, u, model=result.model)[1],
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
            lambda attack, defense: evaluate_degree_game_strategy(
                D, result.t, attack, defense, model=result.model
            )[1:],
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
            lambda attack, defense: evaluate_node_game_strategy(
                A, result.t, attack, defense, model=result.model
            )[1:],
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
    fig = plt.figure(figsize=(7.4, 5.1), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=(5.0, 0.8))
    ax = fig.add_subplot(grid[0])
    legend_ax = fig.add_subplot(grid[1])
    plot_time_series(
        ax,
        result.t,
        result.x @ D.pk,
        "weighted mean state",
        linestyle="-",
    )
    plot_time_series(
        ax,
        result.t,
        result.controls["control"] @ D.pk,
        "weighted mean control",
        linestyle="-.",
    )
    for j in sorted(set([0, len(D.k) // 2, len(D.k) - 1])):
        plot_time_series(
            ax,
            result.t,
            result.x[:, j],
            f"state: degree $k={int(D.k[j])}$",
            linestyle="--",
            linewidth=1.5,
        )
    apply_clean_axes(
        ax,
        xlabel="time",
        ylabel="infection share / control intensity",
        title=f"Degree-k continuous optimal control (population-weighted means), cost={result.value['cost']:.2f}",
    )
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="center",
        ncol=3,
    )
    save_publication_figure(
        fig,
        out_dir / "degree_control_trajectory",
        metadata={"source": "foundation example"},
    )
    plt.close(fig)
    LOGGER.info(f"saved {out_dir / 'degree_control_trajectory'}")


def plot_degree_game(result: TimeSeries, D: DegreeData, out_dir: Path) -> None:
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    plot_time_series(
        ax,
        result.t,
        result.x @ D.pk,
        "state: population-weighted degree-class mean infection",
        linestyle="-",
    )
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
    plot_time_series(
        ax,
        result.t,
        result.x[:, high],
        f"state: high degree k={int(D.k[high])}",
        linestyle=":",
        linewidth=1.8,
    )
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
    plot_time_series(
        ax, result.t, result.x.mean(axis=1), "state: node mean infection (all nodes)", linestyle="-"
    )
    plot_time_series(
        ax, result.t, result.x.max(axis=1), "state: max infection over nodes", linestyle="--"
    )
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
    plot_time_series(
        ax, result.t, result.x.mean(axis=1), "state: node mean infection (all nodes)", linestyle="-"
    )
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

    fig, axes = plt.subplots(
        2,
        1,
        figsize=FIGSIZE_STACKED,
        sharex=True,
        height_ratios=[2.0, 1.0],
        layout="constrained",
    )
    ax_state, ax_control = axes
    plot_time_series(
        ax_state,
        result.t,
        result.x.mean(axis=1),
        "state: node mean infection (all nodes)",
        linestyle="-",
    )
    plot_time_series(
        ax_state, result.t, result.x.max(axis=1), "state: max infection over nodes", linestyle="--"
    )
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
    plot_impulse_events(
        ax_control,
        impulse_times,
        impulse_heights,
        "impulse control: discrete fraction",
        color="tab:red",
    )
    ax_control.set_ylim(0.0, max(0.65, float(np.max(impulse_heights)) + 0.08))
    apply_clean_axes(
        ax_control,
        xlabel="time",
        ylabel="action magnitude",
        title=f"Flow {continuous_min:.2f}-{continuous_max:.2f}; jump fraction {impulse_fraction:.2f}",
    )
    ax_control.legend(frameon=False, ncol=2, fontsize=8)
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
