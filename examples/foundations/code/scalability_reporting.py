"""Figures and Markdown reports for scalability experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd

from cybercontrol.plotting import panel_label, publication_style, save_publication_figure
from cybercontrol.scalability import COMPARISON_DEFAULTS


def plot_filename(model_level: str, summary: pd.DataFrame) -> str:
    node_min = int(summary["nodes"].min())
    node_max = int(summary["nodes"].max())
    if model_level == "compare":
        return f"degree_node_fbs_comparison_{node_min}_{node_max}.png"
    return f"{model_level}_control_scalability_{node_min}_{node_max}.png"


def compact_node_tick(value: float, _position: int | None = None) -> str:
    """Readable log-axis labels for network sizes."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{int(value)}"


def plot_paired_comparison(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> Path:
    """Plot the paired degree-level versus node-level epidemic FBS comparison."""
    with publication_style():
        fig, axes = plt.subplots(3, 1, figsize=(7.16, 7.0), sharex=True)
    repeated_graphs = bool(raw.groupby("nodes")["repeat"].nunique().max() > 1)
    styles = {
        "degree": {
            "color": "tab:blue",
            "marker": "o",
            "linestyle": "-",
            "label": "degree FBS (degree-class state)",
        },
        "node": {
            "color": "tab:orange",
            "marker": "s",
            "linestyle": "--",
            "label": "node FBS (explicit sparse graph)",
        },
    }
    x_ticks = sorted(summary["nodes"].unique())
    x_formatter = FuncFormatter(compact_node_tick)

    ax = axes[0]
    for model_level, group in summary.groupby("model_level", sort=False):
        style = styles[str(model_level)]
        raw_group = raw[raw["model_level"] == model_level]
        ax.scatter(
            raw_group["nodes"], raw_group["fbs_seconds"], color=style["color"], alpha=0.25, s=28
        )
        ax.plot(
            group["nodes"],
            group["fbs_seconds_median"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            color=style["color"],
            label=style["label"],
        )
        if repeated_graphs:
            ax.fill_between(
                group["nodes"].to_numpy(float),
                group["fbs_seconds_min"].to_numpy(float),
                group["fbs_seconds_max"].to_numpy(float),
                color=style["color"],
                alpha=0.10,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("FBS solve time (s)", fontsize=9)
    panel_label(ax, "(a) runtime", fontsize=9)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")

    ax = axes[1]
    for model_level, group in summary.groupby("model_level", sort=False):
        style = styles[str(model_level)]
        ax.plot(
            group["nodes"],
            group["state_dimension_median"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            color=style["color"],
            label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("FBS state dimension", fontsize=9)
    panel_label(ax, "(b) state dimension", fontsize=9)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")

    ax = axes[2]
    graph_summary = (
        raw.drop_duplicates(["nodes", "repeat", "seed"])
        .groupby("nodes", as_index=False)
        .agg(
            max_degree_median=("max_degree", "median"),
            max_degree_min=("max_degree", "min"),
            max_degree_max=("max_degree", "max"),
        )
    )
    ax.plot(
        graph_summary["nodes"],
        graph_summary["max_degree_median"],
        marker="^",
        linestyle="-.",
        linewidth=2.0,
        color="tab:green",
        label="median maximum degree" if repeated_graphs else "maximum degree (one graph)",
    )
    if repeated_graphs:
        ax.fill_between(
            graph_summary["nodes"].to_numpy(float),
            graph_summary["max_degree_min"].to_numpy(float),
            graph_summary["max_degree_max"].to_numpy(float),
            color="tab:green",
            alpha=0.12,
            label="range over graph seeds",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Maximum degree", fontsize=9)
    panel_label(ax, "(c) graph growth", fontsize=9)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.set_xlabel("Nodes in each synthetic scale-free graph", fontsize=9)

    fig.tight_layout(h_pad=1.2)
    path = out_dir / plot_filename("compare", summary)
    save_publication_figure(
        fig,
        path,
        metadata={
            "figure_type": "scalability comparison",
            "network": "synthetic scale-free",
            "scale": "log-log",
        },
    )
    plt.close(fig)
    return path


def plot_scalability(
    summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path, model_level: str
) -> Path:
    if model_level == "compare":
        return plot_paired_comparison(summary, raw, out_dir)

    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6))
    color = "tab:blue" if model_level == "degree" else "tab:orange"
    node_min = int(summary["nodes"].min())
    node_max = int(summary["nodes"].max())
    tick_candidates = (500, 1000, 1500, 2000, 4000, 6000, 8000, 10000)
    x_ticks = [node_min] + [tick for tick in tick_candidates if node_min < tick <= node_max]

    ax = axes[0]
    for _, row in raw.iterrows():
        ax.scatter(row["nodes"], row["fbs_seconds"], color=color, alpha=0.35, s=28)
    ax.plot(
        summary["nodes"],
        summary["fbs_seconds_median"],
        marker="o",
        linewidth=2.2,
        color=color,
        label="median FBS runtime",
    )
    ax.fill_between(
        summary["nodes"].to_numpy(float),
        summary["fbs_seconds_min"].to_numpy(float),
        summary["fbs_seconds_max"].to_numpy(float),
        color=color,
        alpha=0.15,
        label="min-max over repeats",
    )
    ax.set_xlabel("number of nodes in synthetic SF network")
    ax.set_ylabel("FBS solve time (seconds)")
    panel_label(ax, f"(a) {model_level}-level FBS runtime")
    margin = max(25, int(0.025 * (node_max - node_min + 1)))
    ax.set_xlim(node_min - margin, node_max + margin)
    ax.set_xticks(x_ticks)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(
        summary["nodes"],
        summary["state_dimension_median"],
        marker="s",
        linewidth=2.0,
        color="tab:green",
        label="median state dimension",
    )
    ax2 = ax.twinx()
    ax2.plot(
        summary["nodes"],
        summary["fbs_iterations_median"],
        marker="^",
        linewidth=2.0,
        color="tab:purple",
        label="median FBS iterations",
    )
    ax.set_xlabel("number of nodes in synthetic SF network")
    ax.set_ylabel("state dimension", color="tab:green")
    ax2.set_ylabel("FBS iterations", color="tab:purple")
    panel_label(ax, "(b) state dimension and iterations")
    ax.set_xlim(node_min - margin, node_max + margin)
    ax.set_xticks(x_ticks)
    ax.grid(True, alpha=0.25)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    path = out_dir / plot_filename(model_level, summary)
    save_publication_figure(
        fig,
        path,
        metadata={
            "figure_type": "scalability analysis",
            "network": "synthetic scale-free",
            "model_level": model_level,
            "node_range": f"{node_min}-{node_max}",
        },
    )
    plt.close(fig)
    return path


def write_report(
    summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path, model_level: str, plot_path: Path
) -> None:
    if model_level == "compare":
        largest_node = int(summary["nodes"].max())
        largest = summary[summary["nodes"] == largest_node].sort_values("model_level")
        rows = "\n".join(
            f"| {row.model_level} | {row.solver_type} | {row.state_dimension_median:.0f} | "
            f"{row.fbs_iterations_median:.0f} | {row.fbs_seconds_median:.3f} | {bool(row.all_runs_converged)} |"
            for row in largest.itertuples(index=False)
        )
        graph_summary = (
            raw.drop_duplicates(["nodes", "repeat", "seed"])
            .groupby("nodes", as_index=False)
            .agg(
                edges_median=("edges", "median"),
                mean_degree_median=("mean_degree", "median"),
                max_degree_median=("max_degree", "median"),
                max_degree_min=("max_degree", "min"),
                max_degree_max=("max_degree", "max"),
            )
        )
        graph_rows = "\n".join(
            f"| {int(row.nodes)} | {int(row.edges_median)} | {row.mean_degree_median:.2f} | "
            f"{row.max_degree_median:.0f} | {row.max_degree_min:.0f}-{row.max_degree_max:.0f} |"
            for row in graph_summary.itertuples(index=False)
        )
        params = COMPARISON_DEFAULTS
        text = f"""# Paired Degree-Level vs Node-Level FBS Comparison

This run compares a heterogeneous-mean-field (HMF) degree approximation with an explicit-adjacency node model on the same synthetic Barabasi-Albert graphs. Both rows use the same scalar rates, objective weights, initial degree rule, time grid and graph seed. They are related resolutions, not algebraically identical propagation equations. The CSV reports `initial_rhs_aggregation_rmse` to quantify their derivative gap at the common initial condition.

## What Was Measured

- Network sizes: {", ".join(str(int(n)) for n in sorted(raw["nodes"].unique()))} nodes.
- Repeats per size: {int(raw["repeat"].max())}.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m={int(raw["attachment_m"].iloc[0])}`.
- Epidemic/control model: normalized SIS, `T={params["horizon"]}`, `beta={params["beta"]}`, `delta={params["delta"]}`, state weight `{params["state_weight"]}`, control weight `{params["control_weight"]}`, `u_max={params["control_max"]}`.
- Contact-pressure normalization: each graph's maximum degree, shared by the degree-level and node-level rows.
- Numerical solver for both rows: fixed-grid RK4 inside forward-backward sweep.
- Node-level implementation: sparse adjacency matrix; state/control/costate are still node-indexed.
- Time grid: `{int(raw["time_grid_steps"].iloc[0])}` intervals over the control horizon.
- Maximum FBS iterations: `{int(raw["max_fbs_iterations"].iloc[0])}`.
- FBS tolerance: `{float(raw["tolerance"].iloc[0]):.0e}`.
- Runtime column: `fbs_seconds`, measuring only the FBS solve after graph generation.

## Main Output

| File | Meaning |
| --- | --- |
| `{plot_path.name}` | Paired runtime, state-dimension, and max-degree comparison. |
| `paired_fbs_comparison.csv` | One row per model level, size, and repeat. |
| `paired_fbs_comparison_summary.csv` | Median/min/max runtime and convergence summary by model level and size. |

## Quick Reading

At {largest_node} nodes:

| Model level | Solver label | FBS state dimension | FBS iterations | Median FBS solve time (seconds) | All runs converged |
| --- | --- | ---: | ---: | ---: | --- |
{rows}

## Synthetic SF Degree Growth Check

| Nodes | Edges | Mean degree | Median max degree | Max-degree range |
| ---: | ---: | ---: | ---: | ---: |
{graph_rows}

This plot compares computational scaling of the degree-HMF approximation and the explicit-adjacency node formulation. The degree row keeps one state/control/costate per observed degree class; the node row keeps one per graph node. Sparse matrix products keep the million-node run practical, but the node-indexed FBS state is still much larger. Runtime differences must be interpreted together with the reported approximation gap.
"""
        (out_dir / "scalability_summary.md").write_text(text)
        return

    largest = summary.sort_values("nodes").iloc[-1]
    solver_type = str(raw["solver_type"].iloc[0])
    text = f"""# Scalability Analysis

This run measures `{model_level}`-level forward-backward sweep (FBS) optimal control on synthetic Barabasi-Albert scale-free networks.

## What Was Measured

- Network sizes: {", ".join(str(int(n)) for n in sorted(raw["nodes"].unique()))} nodes.
- Repeats per size: {int(raw["repeat"].max())}.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m={int(raw["attachment_m"].iloc[0])}`.
- Numerical solver: `{solver_type}`.
- Time grid: `{int(raw["time_grid_steps"].iloc[0])}` intervals over the control horizon.
- Maximum FBS iterations: `{int(raw["max_fbs_iterations"].iloc[0])}`.
- FBS tolerance: `{float(raw["tolerance"].iloc[0]):.0e}`.
- Runtime column: `fbs_seconds`, measuring the FBS solve after graph generation and preprocessing.
- Convergence check: `final_delta < tolerance` for each run.

## Main Output

| File | Meaning |
| --- | --- |
| `{plot_path.name}` | Runtime and state-dimension/iteration trends as SF network size grows. |
| `{model_level}_control_scalability.csv` | One row per size and repeat. |
| `{model_level}_control_scalability_summary.csv` | Median/min/max runtime and convergence summary by size. |

## Quick Reading

At {int(largest["nodes"])} nodes, the median FBS solve time was {largest["fbs_seconds_median"]:.3f} seconds over {int(largest["repeats"])} repeat(s). All runs at that size converged: {bool(largest["all_runs_converged"])}.

Read runtime columns with the solver type. The default `--model-level compare` run contrasts degree-HMF and sparse explicit-adjacency FBS with matched rates, objective weights, grid and graph seed. Standalone `degree` and `node` runs are scaling diagnostics and should not be treated as an exact-equation benchmark.
"""
    (out_dir / "scalability_summary.md").write_text(text)
