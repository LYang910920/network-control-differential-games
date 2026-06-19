# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Scalability analysis for forward-backward sweep optimal control.

The default experiment uses synthetic Barabasi-Albert scale-free networks with
100 to 2000 nodes in steps of 100. It runs the degree-level SIS optimal-control
FBS solver, records runtime and convergence diagnostics, and writes one clear
summary plot.

Degree-level FBS is the default because the state dimension is the number of
observed degree classes, not the number of nodes. This makes the 2000-node
experiment lightweight enough for a teaching repository while still showing how
runtime changes as the underlying network grows.

The optional node-level experiment is heavier and uses sparse matrices. Its
default grid is 1000, 2000, ..., 10000 nodes. This keeps the mathematical object
node-indexed, while using igraph and scipy.sparse for the work that should be
handled by libraries rather than hand-written graph code.

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
import scipy.sparse as sp


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


DEFAULT_DEGREE_SIZES = tuple(range(100, 2001, 100))
DEFAULT_NODE_SIZES = tuple(range(1000, 10001, 1000))


NODE_SCALABILITY_DEFAULTS = {
    "horizon": 8.0,
    "beta": 0.55,
    "delta": 0.20,
    "state_weight": 1.0,
    "control_weight": 8.0,
    "control_max": 0.8,
    "damping": 0.25,
}


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


def igraph_to_sparse_model_matrix(graph: ig.Graph) -> sp.csr_matrix:
    """Return sparse A where A[i,j] is the pressure from node j to node i."""
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64)
    n = graph.vcount()
    if edges.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)

    src = edges[:, 0]
    dst = edges[:, 1]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    max_degree = float(np.asarray(A.sum(axis=1)).ravel().max(initial=0.0))
    return A / max_degree if max_degree > 0 else A


def sparse_node_initial_state(A: sp.csr_matrix) -> np.ndarray:
    """Seed the highest-degree nodes without materializing a dense matrix."""
    n = A.shape[0]
    x0 = np.full(n, 0.015, dtype=np.float64)
    degree = np.asarray(A.getnnz(axis=1), dtype=np.int64)
    count = min(3, n)
    if count:
        hubs = np.argsort(-degree)[:count]
        x0[hubs] = 0.10
    return x0


def sparse_node_rhs(
    x: np.ndarray,
    u: np.ndarray,
    A: sp.csr_matrix,
    *,
    beta: float,
    delta: float,
) -> np.ndarray:
    pressure = A @ x
    return beta * (1.0 - x) * pressure - (delta + u) * x


def integrate_sparse_state(
    A: sp.csr_matrix,
    t: np.ndarray,
    x0: np.ndarray,
    u: np.ndarray,
    *,
    beta: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward RK4 integration for sparse node-level SIS dynamics."""
    x = np.empty((len(t), A.shape[0]), dtype=np.float64)
    pressure = np.empty_like(x)
    x[0] = x0
    pressure[0] = A @ x0
    for idx in range(len(t) - 1):
        h = float(t[idx + 1] - t[idx])
        u0 = u[idx]
        u1 = u[idx + 1]
        umid = 0.5 * (u0 + u1)
        y0 = x[idx]
        k1 = sparse_node_rhs(y0, u0, A, beta=beta, delta=delta)
        k2 = sparse_node_rhs(np.clip(y0 + 0.5 * h * k1, 0.0, 1.0), umid, A, beta=beta, delta=delta)
        k3 = sparse_node_rhs(np.clip(y0 + 0.5 * h * k2, 0.0, 1.0), umid, A, beta=beta, delta=delta)
        k4 = sparse_node_rhs(np.clip(y0 + h * k3, 0.0, 1.0), u1, A, beta=beta, delta=delta)
        x[idx + 1] = np.clip(y0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0, 1.0)
        pressure[idx + 1] = A @ x[idx + 1]
    return x, pressure


def sparse_node_adjoint_rhs(
    lam: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    pressure: np.ndarray,
    A: sp.csr_matrix,
    *,
    beta: float,
    delta: float,
    state_weight: float,
) -> np.ndarray:
    diag_part = (-beta * pressure - delta - u) * lam
    offdiag_part = beta * (A.T @ ((1.0 - x) * lam))
    return -state_weight - offdiag_part - diag_part


def integrate_sparse_adjoint(
    A: sp.csr_matrix,
    t: np.ndarray,
    x: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
    *,
    beta: float,
    delta: float,
    state_weight: float,
) -> np.ndarray:
    """Backward RK4 integration for the sparse node-level costate equation."""
    lam = np.zeros_like(x)
    for idx in range(len(t) - 2, -1, -1):
        h = -float(t[idx + 1] - t[idx])
        l1 = lam[idx + 1]
        x1 = x[idx + 1]
        x0 = x[idx]
        p1 = pressure[idx + 1]
        p0 = pressure[idx]
        u1 = u[idx + 1]
        u0 = u[idx]
        xmid = 0.5 * (x0 + x1)
        pmid = 0.5 * (p0 + p1)
        umid = 0.5 * (u0 + u1)

        k1 = sparse_node_adjoint_rhs(l1, x1, u1, p1, A, beta=beta, delta=delta, state_weight=state_weight)
        k2 = sparse_node_adjoint_rhs(l1 + 0.5 * h * k1, xmid, umid, pmid, A, beta=beta, delta=delta, state_weight=state_weight)
        k3 = sparse_node_adjoint_rhs(l1 + 0.5 * h * k2, xmid, umid, pmid, A, beta=beta, delta=delta, state_weight=state_weight)
        k4 = sparse_node_adjoint_rhs(l1 + h * k3, x0, u0, p0, A, beta=beta, delta=delta, state_weight=state_weight)
        lam[idx] = l1 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return lam


def run_sparse_node_fbs(A: sp.csr_matrix, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    """Sparse node-level PMP/FBS sweep for large synthetic graph experiments."""
    params = NODE_SCALABILITY_DEFAULTS
    t = np.linspace(0.0, params["horizon"], steps + 1)
    u = np.zeros((len(t), A.shape[0]), dtype=np.float64)
    x0 = sparse_node_initial_state(A)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_u = u.copy()
        x, pressure = integrate_sparse_state(A, t, x0, u, beta=params["beta"], delta=params["delta"])
        lam = integrate_sparse_adjoint(
            A,
            t,
            x,
            pressure,
            u,
            beta=params["beta"],
            delta=params["delta"],
            state_weight=params["state_weight"],
        )
        candidate = np.clip(lam * x / params["control_weight"], 0.0, params["control_max"])
        u = params["damping"] * candidate + (1.0 - params["damping"]) * old_u
        delta_u = float(np.max(np.abs(u - old_u)))
        delta_history.append(delta_u)
        if delta_u < tol:
            break

    x, _ = integrate_sparse_state(A, t, x0, u, beta=params["beta"], delta=params["delta"])
    running = params["state_weight"] * x.mean(axis=1) + 0.5 * params["control_weight"] * np.mean(u * u, axis=1)
    if hasattr(np, "trapezoid"):
        cost = float(np.trapezoid(running, t))
    else:
        cost = float(np.trapz(running, t))
    final_delta = float(delta_history[-1]) if delta_history else float("nan")
    return {
        "fbs_iterations": float(len(delta_history)),
        "final_delta": final_delta,
        "converged": float(final_delta < tol),
        "cost": cost,
        "mean_initial_state": float(x0.mean()),
        "mean_final_state": float(x[-1].mean()),
        "mean_control": float(u.mean()),
        "max_control": float(u.max()),
        "horizon": float(params["horizon"]),
        "beta": float(params["beta"]),
        "delta": float(params["delta"]),
        "state_weight": float(params["state_weight"]),
        "control_weight": float(params["control_weight"]),
        "control_max": float(params["control_max"]),
        "damping": float(params["damping"]),
    }


def run_degree_trial(graph: ig.Graph, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    prep_start = time.perf_counter()
    degree_data = degree_distribution_from_igraph(graph)
    prep_seconds = time.perf_counter() - prep_start

    solve_start = time.perf_counter()
    result = solve_degree_control(degree_data, steps=steps, iterations=iterations, tol=tol)
    solve_seconds = time.perf_counter() - solve_start

    return {
        "model_level": "degree",
        "solver_type": "degree",
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


def run_node_trial(graph: ig.Graph, *, steps: int, iterations: int, tol: float, solver: str) -> dict[str, float]:
    prep_start = time.perf_counter()
    if solver == "dense":
        A, _ = graph_to_model_matrix(igraph_to_networkx(graph), directed=False, normalize="max-degree")
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
                stats = run_node_trial(
                    graph,
                    steps=args.steps,
                    iterations=args.iterations,
                    tol=args.tolerance,
                    solver=args.node_solver,
                )

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
                f"solver={stats.get('solver_type', 'degree')} "
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
        matrix_nonzeros_median=("matrix_nonzeros", "median"),
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
    ax.set_title(f"{model_level.title()}-level FBS runtime")
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
    ax.set_title("Why runtime changes")
    ax.set_xlim(node_min - margin, node_max + margin)
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
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m={int(raw["attachment_m"].iloc[0])}`.
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

For degree-level models, the FBS state dimension is the number of observed degree classes, so it grows much more slowly than the number of nodes. For node-level models, the state, costate, and control are indexed by node. The optional large-node run therefore uses a sparse adjacency matrix and reports convergence of a node-indexed FBS sweep rather than converting the graph to a dense teaching matrix.
"""
    (out_dir / "scalability_summary.md").write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SF-network scalability analysis for FBS optimal control.")
    parser.add_argument("--model-level", choices=["degree", "node"], default="degree")
    parser.add_argument("--sizes", type=parse_sizes, default=None, help="Comma-separated node counts.")
    parser.add_argument("--node-solver", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attachment-m", type=int, default=3)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sizes is None:
        args.sizes = list(DEFAULT_NODE_SIZES if args.model_level == "node" else DEFAULT_DEGREE_SIZES)
    if args.output_dir is None:
        args.output_dir = Path(f"results/scalability_{args.model_level}_sf")
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
