# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see COPYRIGHT_AND_LICENSE.md.

"""CLI orchestration for paired degree/node FBS scalability experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

from cybercontrol.fbs import solve_degree_control, solve_node_control
from cybercontrol.graph_data import graph_to_model_matrix
from cybercontrol.scalability import (  # noqa: F401 - compatibility exports
    COMPARISON_DEFAULTS,
    DEFAULT_COMPARE_SIZES,
    DEFAULT_DEGREE_SIZES,
    DEFAULT_NODE_SIZES,
    NODE_SCALABILITY_DEFAULTS,
    comparison_degree_jacobian,
    comparison_degree_initial_state,
    comparison_degree_rhs,
    comparison_node_adjoint_rhs,
    comparison_node_initial_state,
    comparison_node_rhs,
    integrate_comparison_degree_adjoint,
    integrate_comparison_degree_state,
    integrate_comparison_node_adjoint,
    integrate_comparison_node_state,
    integrate_sparse_adjoint,
    integrate_sparse_state,
    run_comparison_degree_fbs,
    run_comparison_node_fbs,
    run_sparse_node_fbs,
    sparse_node_adjoint_rhs,
    sparse_node_initial_state,
    sparse_node_rhs,
    update_comparison_node_control,
)
from cybercontrol.scalability_graphs import (  # noqa: F401 - compatibility exports
    degree_distribution_from_degree,
    degree_distribution_from_igraph,
    igraph_to_networkx,
    igraph_to_sparse_adjacency,
    igraph_to_sparse_model_matrix,
    synthetic_scale_free_graph,
)

try:
    from .scalability_reporting import plot_scalability, write_report
except ImportError:
    from scalability_reporting import plot_scalability, write_report


def parse_sizes(raw: str) -> list[int]:
    sizes = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError(
            "sizes must be a comma-separated list of positive integers"
        )
    return sizes


def run_degree_trial(
    graph: ig.Graph, *, steps: int, iterations: int, tol: float
) -> dict[str, float]:
    prep_start = time.perf_counter()
    degree_data = degree_distribution_from_igraph(graph)
    prep_seconds = time.perf_counter() - prep_start

    solve_start = time.perf_counter()
    result = solve_degree_control(degree_data, steps=steps, iterations=iterations, tol=tol)
    solve_seconds = time.perf_counter() - solve_start

    return {
        "model_level": "degree",
        "solver_type": "adaptive_solve_ivp",
        "state_dimension": float(len(degree_data.k)),
        "degree_classes": float(len(degree_data.k)),
        "matrix_nonzeros": float("nan"),
        "prep_seconds": prep_seconds,
        "fbs_seconds": solve_seconds,
        "total_seconds": prep_seconds + solve_seconds,
        "fbs_iterations": float(result.value["iterations"]),
        "final_delta": float(result.value["final_delta"]),
        "converged": float(result.value["converged"]),
        "cost": float(result.value["cost"]),
        "horizon": float("nan"),
        "beta": float("nan"),
        "delta": float("nan"),
        "state_weight": float("nan"),
        "control_weight": float("nan"),
        "control_max": float("nan"),
        "damping": float("nan"),
        "mean_initial_state": float("nan"),
        "mean_final_state": float("nan"),
        "mean_control": float("nan"),
        "max_control": float("nan"),
    }


def run_node_trial(
    graph: ig.Graph, *, steps: int, iterations: int, tol: float, solver: str
) -> dict[str, float]:
    prep_start = time.perf_counter()
    if solver == "dense":
        A, _ = graph_to_model_matrix(
            igraph_to_networkx(graph), directed=False, normalize="max-degree"
        )
    else:
        A = igraph_to_sparse_model_matrix(graph)
    prep_seconds = time.perf_counter() - prep_start

    solve_start = time.perf_counter()
    if solver == "dense":
        result = solve_node_control(A, steps=steps, iterations=iterations, tol=tol)
        stats = {
            "fbs_iterations": float(result.value["iterations"]),
            "final_delta": float(result.value["final_delta"]),
            "converged": float(result.value["converged"]),
            "cost": float(result.value["cost"]),
            "horizon": float("nan"),
            "beta": float("nan"),
            "delta": float("nan"),
            "state_weight": float("nan"),
            "control_weight": float("nan"),
            "control_max": float("nan"),
            "damping": float("nan"),
            "mean_initial_state": float("nan"),
            "mean_final_state": float("nan"),
            "mean_control": float("nan"),
            "max_control": float("nan"),
        }
        state_dimension = float(A.shape[0])
        nonzeros = float(np.count_nonzero(A))
    else:
        stats = run_sparse_node_fbs(A, steps=steps, iterations=iterations, tol=tol)
        state_dimension = float(A.shape[0])
        nonzeros = float(A.nnz)
    solve_seconds = time.perf_counter() - solve_start

    return {
        "model_level": "node",
        "solver_type": solver,
        "state_dimension": state_dimension,
        "degree_classes": float("nan"),
        "matrix_nonzeros": nonzeros,
        "prep_seconds": prep_seconds,
        "fbs_seconds": solve_seconds,
        "total_seconds": prep_seconds + solve_seconds,
        **stats,
    }


def run_paired_comparison_trials(
    graph: ig.Graph,
    *,
    degree: np.ndarray,
    steps: int,
    iterations: int,
    tol: float,
) -> list[dict[str, float | str]]:
    """Compare a degree-HMF approximation with explicit adjacency on one graph."""
    degree_float = degree.astype(float, copy=False)
    degree_data = degree_distribution_from_degree(degree.astype(int, copy=False))
    A = igraph_to_sparse_adjacency(graph)
    node_initial = comparison_node_initial_state(degree_float)
    node_derivative = comparison_node_rhs(
        node_initial, np.zeros_like(node_initial), A, max(float(degree_float.max(initial=1.0)), 1.0)
    )
    degree_initial = comparison_degree_initial_state(degree_data)
    degree_derivative = comparison_degree_rhs(
        degree_initial, np.zeros_like(degree_initial), degree_data
    )
    aggregated_node_derivative = np.array(
        [node_derivative[degree_float == degree_class].mean() for degree_class in degree_data.k]
    )
    initial_rhs_aggregation_rmse = float(
        np.sqrt(degree_data.pk @ ((aggregated_node_derivative - degree_derivative) ** 2))
    )
    shared = {
        "horizon": float(COMPARISON_DEFAULTS["horizon"]),
        "beta": float(COMPARISON_DEFAULTS["beta"]),
        "delta": float(COMPARISON_DEFAULTS["delta"]),
        "state_weight": float(COMPARISON_DEFAULTS["state_weight"]),
        "control_weight": float(COMPARISON_DEFAULTS["control_weight"]),
        "control_max": float(COMPARISON_DEFAULTS["control_max"]),
        "damping": float(COMPARISON_DEFAULTS["damping"]),
        "mean_degree": float(degree_float.mean()) if len(degree_float) else float("nan"),
        "max_degree": float(degree_float.max(initial=0.0)),
        "initial_rhs_aggregation_rmse": initial_rhs_aggregation_rmse,
    }

    rows: list[dict[str, float | str]] = []
    for model_level, runner in (
        (
            "degree",
            lambda: run_comparison_degree_fbs(
                degree_data, steps=steps, iterations=iterations, tol=tol
            ),
        ),
        (
            "node",
            lambda: run_comparison_node_fbs(
                A, degree_float, steps=steps, iterations=iterations, tol=tol
            ),
        ),
    ):
        solve_start = time.perf_counter()
        stats = runner()
        solve_seconds = time.perf_counter() - solve_start
        rows.append(
            {
                "model_level": model_level,
                "comparison_type": "degree_hmf_vs_explicit_adjacency",
                "prep_seconds": 0.0,
                "fbs_seconds": solve_seconds,
                "total_seconds": solve_seconds,
                **shared,
                **stats,
            }
        )
    return rows


def run_scalability_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for nodes in args.sizes:
        for repeat in range(1, args.repeats + 1):
            # Each row is one reproducible graph/control solve; the summary plot
            # then separates median behavior from repeat-to-repeat variation.
            seed = args.seed + 1000 * repeat + nodes
            graph = synthetic_scale_free_graph(nodes, args.attachment_m, seed)
            degree = np.asarray(graph.degree(), dtype=np.int64)
            mean_degree = float(degree.mean()) if len(degree) else float("nan")
            max_degree = float(degree.max(initial=0))
            if args.model_level == "compare":
                trial_rows = run_paired_comparison_trials(
                    graph,
                    degree=degree,
                    steps=args.steps,
                    iterations=args.iterations,
                    tol=args.tolerance,
                )
            elif args.model_level == "degree":
                stats = run_degree_trial(
                    graph, steps=args.steps, iterations=args.iterations, tol=args.tolerance
                )
                trial_rows = [stats]
            else:
                stats = run_node_trial(
                    graph,
                    steps=args.steps,
                    iterations=args.iterations,
                    tol=args.tolerance,
                    solver=args.node_solver,
                )
                trial_rows = [stats]

            for stats in trial_rows:
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
                    "mean_degree": mean_degree,
                    "max_degree": max_degree,
                }
                row.update(stats)
                rows.append(row)
                print(
                    f"n={nodes:4d} repeat={repeat} level={stats['model_level']} "
                    f"solver={stats.get('solver_type', 'degree')} "
                    f"fbs={float(stats['fbs_seconds']):.3f}s iterations={int(float(stats['fbs_iterations']))} "
                    f"converged={bool(stats['converged'])}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model_level", "solver_type", "nodes"], as_index=False).agg(
        fbs_seconds_median=("fbs_seconds", "median"),
        fbs_seconds_min=("fbs_seconds", "min"),
        fbs_seconds_max=("fbs_seconds", "max"),
        total_seconds_median=("total_seconds", "median"),
        fbs_iterations_median=("fbs_iterations", "median"),
        state_dimension_median=("state_dimension", "median"),
        degree_classes_median=("degree_classes", "median"),
        matrix_nonzeros_median=("matrix_nonzeros", "median"),
        mean_degree_median=("mean_degree", "median"),
        max_degree_median=("max_degree", "median"),
        max_degree_min=("max_degree", "min"),
        max_degree_max=("max_degree", "max"),
        converged_runs=("converged", "sum"),
        repeats=("repeat", "count"),
    )
    grouped["all_runs_converged"] = grouped["converged_runs"] == grouped["repeats"]
    return grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SF-network scalability analysis for FBS optimal control."
    )
    parser.add_argument(
        "--model-level",
        choices=["compare", "degree", "node"],
        default="compare",
        help="Use compare for paired degree/node FBS; standalone modes are diagnostics.",
    )
    parser.add_argument(
        "--sizes", type=parse_sizes, default=None, help="Comma-separated node counts."
    )
    parser.add_argument("--node-solver", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--attachment-m", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sizes is None:
        if args.model_level == "compare":
            args.sizes = list(DEFAULT_COMPARE_SIZES)
        elif args.model_level == "node":
            args.sizes = list(DEFAULT_NODE_SIZES)
        else:
            args.sizes = list(DEFAULT_DEGREE_SIZES)
    if args.output_dir is None:
        default_folder = (
            "scalability_degree_node_sf"
            if args.model_level == "compare"
            else f"scalability_{args.model_level}_sf"
        )
        args.output_dir = Path("results") / default_folder
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = run_scalability_experiment(args)
    summary = summarize(df)

    prefix = (
        "paired_fbs_comparison"
        if args.model_level == "compare"
        else f"{args.model_level}_control_scalability"
    )
    raw_path = args.output_dir / f"{prefix}.csv"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
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
