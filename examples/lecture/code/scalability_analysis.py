# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Scalability analysis for forward-backward sweep optimal control.

The default experiment uses synthetic Barabasi-Albert scale-free networks with
100 to 2000 nodes in steps of 100. It runs the degree-level SIS optimal-control FBS solver,
records runtime and convergence diagnostics, and writes one clear summary plot.

Degree-level FBS is the default because the state dimension is the number of
observed degree classes, not the number of nodes. This makes the 2000-node
experiment lightweight enough for a teaching repository while still showing how
runtime changes as the underlying network grows.

The graph work is intentionally delegated to libraries: igraph generates the
scale-free networks and degree vectors, pandas aggregates timing summaries, and
Matplotlib renders the final diagnostic plot. The control solver receives only
the reduced mathematical objects it actually needs.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CODE_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from network_control_examples import (  # noqa: E402
    DegreeData,
    graph_to_model_matrix,
    solve_degree_control,
    solve_node_control,
)


DEFAULT_SIZES = tuple(range(100, 2001, 100))


def parse_sizes(raw: str) -> list[int]:
    sizes = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be a comma-separated list of positive integers")
    return sizes


def synthetic_scale_free_graph(nodes: int, attachment_m: int, seed: int) -> ig.Graph:
    """Generate the synthetic network with igraph instead of hand-rolled graph code."""
    m = max(1, min(attachment_m, nodes - 1))
    ig.set_random_number_generator(random.Random(seed))
    graph = ig.Graph.Barabasi(n=nodes, m=m, directed=False)
    graph.es["weight"] = [1.0] * graph.ecount()
    return graph


def degree_distribution_from_igraph(graph: ig.Graph) -> DegreeData:
    """Reduce a node-level graph to the degree-k state used by the FBS solver."""
    degree = np.asarray(graph.degree(), dtype=int)
    k, Nk = np.unique(degree, return_counts=True)
    pk = Nk / Nk.sum()
    return DegreeData(k=k.astype(float), Nk=Nk.astype(int), pk=pk.astype(float), kbar=float(k @ pk), node_degree=degree)


def igraph_to_networkx(graph: ig.Graph) -> nx.Graph:
    """Use igraph's converter only for the optional node-level matrix workflow."""
    return nx.Graph(graph.to_networkx())


def run_degree_trial(graph: ig.Graph, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    prep_start = time.perf_counter()
    degree_data = degree_distribution_from_igraph(graph)
    prep_seconds = time.perf_counter() - prep_start

    solve_start = time.perf_counter()
    result = solve_degree_control(degree_data, steps=steps, iterations=iterations, tol=tol)
    solve_seconds = time.perf_counter() - solve_start

    return {
        "model_level": "degree",
        "state_dimension": float(len(degree_data.k)),
        "degree_classes": float(len(degree_data.k)),
        "prep_seconds": prep_seconds,
        "fbs_seconds": solve_seconds,
        "total_seconds": prep_seconds + solve_seconds,
        "fbs_iterations": float(result.value["iterations"]),
        "final_delta": float(result.value["final_delta"]),
        "converged": float(result.value["converged"]),
        "cost": float(result.value["cost"]),
    }


def run_node_trial(graph: ig.Graph, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    prep_start = time.perf_counter()
    A, _ = graph_to_model_matrix(igraph_to_networkx(graph), directed=False, normalize="max-degree")
    prep_seconds = time.perf_counter() - prep_start

    solve_start = time.perf_counter()
    result = solve_node_control(A, steps=steps, iterations=iterations, tol=tol)
    solve_seconds = time.perf_counter() - solve_start

    return {
        "model_level": "node",
        "state_dimension": float(A.shape[0]),
        "degree_classes": float("nan"),
        "prep_seconds": prep_seconds,
        "fbs_seconds": solve_seconds,
        "total_seconds": prep_seconds + solve_seconds,
        "fbs_iterations": float(result.value["iterations"]),
        "final_delta": float(result.value["final_delta"]),
        "converged": float(result.value["converged"]),
        "cost": float(result.value["cost"]),
    }


def run_scalability_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for nodes in args.sizes:
        for repeat in range(1, args.repeats + 1):
            # Each row is one reproducible graph/control solve; the summary plot
            # then separates median behavior from repeat-to-repeat variation.
            seed = args.seed + 1000 * repeat + nodes
            graph = synthetic_scale_free_graph(nodes, args.attachment_m, seed)
            if args.model_level == "degree":
                stats = run_degree_trial(graph, steps=args.steps, iterations=args.iterations, tol=args.tolerance)
            else:
                stats = run_node_trial(graph, steps=args.steps, iterations=args.iterations, tol=args.tolerance)

            row: dict[str, float | int | str] = {
                "network_model": "Barabasi-Albert scale-free",
                "nodes": nodes,
                "edges": graph.ecount(),
                "attachment_m": args.attachment_m,
                "repeat": repeat,
                "seed": seed,
                "time_grid_steps": args.steps,
                "max_fbs_iterations": args.iterations,
                "tolerance": args.tolerance,
            }
            row.update(stats)
            rows.append(row)
            print(
                f"n={nodes:4d} repeat={repeat} level={args.model_level} "
                f"fbs={stats['fbs_seconds']:.3f}s iterations={int(stats['fbs_iterations'])} "
                f"converged={bool(stats['converged'])}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model_level", "nodes"], as_index=False).agg(
        fbs_seconds_median=("fbs_seconds", "median"),
        fbs_seconds_min=("fbs_seconds", "min"),
        fbs_seconds_max=("fbs_seconds", "max"),
        total_seconds_median=("total_seconds", "median"),
        fbs_iterations_median=("fbs_iterations", "median"),
        state_dimension_median=("state_dimension", "median"),
        degree_classes_median=("degree_classes", "median"),
        converged_runs=("converged", "sum"),
        repeats=("repeat", "count"),
    )
    grouped["all_runs_converged"] = grouped["converged_runs"] == grouped["repeats"]
    return grouped


def plot_filename(model_level: str, summary: pd.DataFrame) -> str:
    node_min = int(summary["nodes"].min())
    node_max = int(summary["nodes"].max())
    return f"{model_level}_control_scalability_{node_min}_{node_max}.png"


def plot_scalability(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path, model_level: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    color = "tab:blue" if model_level == "degree" else "tab:orange"
    node_min = int(summary["nodes"].min())
    node_max = int(summary["nodes"].max())
    x_ticks = [node_min] + [tick for tick in (500, 1000, 1500, 2000) if node_min < tick <= node_max]

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
    ax.set_title(f"{model_level.title()}-level FBS runtime")
    ax.set_xlim(node_min - 25, node_max + 25)
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
    ax.set_title("Why runtime changes")
    ax.set_xlim(node_min - 25, node_max + 25)
    ax.set_xticks(x_ticks)
    ax.grid(True, alpha=0.25)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(f"Scalability analysis on synthetic scale-free networks ({node_min}-{node_max} nodes)")
    fig.tight_layout()
    path = out_dir / plot_filename(model_level, summary)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path, model_level: str, plot_path: Path) -> None:
    largest = summary.sort_values("nodes").iloc[-1]
    text = f"""# Scalability Analysis

This run measures `{model_level}`-level forward-backward sweep (FBS) optimal control on synthetic Barabasi-Albert scale-free networks.

## What Was Measured

- Network sizes: {", ".join(str(int(n)) for n in sorted(raw["nodes"].unique()))} nodes.
- Repeats per size: {int(raw["repeat"].max())}.
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

For degree-level models, the FBS state dimension is the number of observed degree classes, so it grows much more slowly than the number of nodes. This is why degree-level analysis is a useful scalability baseline before attempting full node-level FBS.
"""
    (out_dir / "scalability_summary.md").write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SF-network scalability analysis for FBS optimal control.")
    parser.add_argument("--model-level", choices=["degree", "node"], default="degree")
    parser.add_argument("--sizes", type=parse_sizes, default=list(DEFAULT_SIZES), help="Comma-separated node counts.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attachment-m", type=int, default=3)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scalability_degree_sf"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = run_scalability_experiment(args)
    summary = summarize(df)

    raw_path = args.output_dir / f"{args.model_level}_control_scalability.csv"
    summary_path = args.output_dir / f"{args.model_level}_control_scalability_summary.csv"
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    legacy_plot = args.output_dir / f"{args.model_level}_control_scalability.png"
    if legacy_plot.exists():
        legacy_plot.unlink()
    plot_path = plot_scalability(summary, df, args.output_dir, args.model_level)
    write_report(summary, df, args.output_dir, args.model_level, plot_path)
    print(f"saved scalability outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
