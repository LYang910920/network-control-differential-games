"""Run one graph-to-control example, then explore games and hybrid models."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

from .fbs import solve_degree_control, solve_degree_game, solve_node_control, solve_node_game
from .foundation_evaluation import evaluate_degree_control_policy
from .graph_data import DegreeData, degree_distribution_from_graph, load_graph
from .heterogeneity import DegreeSISParams
from .models import MalwareParams, controlled_sir_rhs, isolation_jump
from .numerics import project_simplex3, rk4_integrate, trapezoid_integral

ROOT = Path(__file__).resolve().parents[2]


def run_example(args: argparse.Namespace) -> pd.DataFrame:
    """Graph -> degree classes -> PMP control -> matched-model comparisons."""
    import matplotlib.pyplot as plt

    graph, source = load_graph(args)
    if graph.is_directed():
        raise ValueError(
            "This degree closure uses an undirected graph; use a node model for directed data."
        )
    degree = degree_distribution_from_graph(graph)
    model = DegreeSISParams(
        susceptibility=args.beta,
        recovery=args.recovery,
        state_weight=3.0,
        control_weight=args.control_cost,
        control_bound=1.2,
    ).resolve(degree.k, degree.pk)
    result = solve_degree_control(
        degree, steps=args.steps, iterations=args.iterations, tol=args.tolerance, model=model
    )
    control = result.controls["control"]

    # All policies use the same graph, initial state, time grid and objective.
    zero = np.zeros_like(control)
    x_zero, cost_zero = evaluate_degree_control_policy(degree, result.t, zero, model=model)
    constants = []
    for intensity in np.linspace(0.0, 1.2, 9):
        policy = np.full_like(control, intensity)
        state, cost = evaluate_degree_control_policy(degree, result.t, policy, model=model)
        constants.append((cost, state, policy))
    cost_constant, x_constant, u_constant = min(constants, key=lambda item: item[0])
    policies = {
        "No control": (x_zero, zero, cost_zero),
        "Best constant on 9-point grid": (x_constant, u_constant, cost_constant),
        "PMP candidate": (result.x, control, result.value["cost"]),
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"k": degree.k, "N_k": degree.Nk, "P_k": degree.pk}).to_csv(
        out / "degree_distribution.csv", index=False
    )
    rows, trajectories = [], []
    for name, (state, effort, cost) in policies.items():
        rows.append(
            {
                "policy": name,
                "cost": cost,
                "infected_time": trapezoid_integral(state @ degree.pk, result.t),
            }
        )
        trajectories.append(
            pd.DataFrame(
                {
                    "policy": name,
                    "time": np.repeat(result.t, len(degree.k)),
                    "degree": np.tile(degree.k, len(result.t)),
                    "infected": state.ravel(),
                    "control": effort.ravel(),
                }
            )
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(out / "trajectories.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (state, _, _) in policies.items():
        ax.plot(result.t, state @ degree.pk, label=name)
    ax.set(xlabel="Time", ylabel="Population infected fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "infection.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j in np.unique(np.linspace(0, len(degree.k) - 1, min(4, len(degree.k))).astype(int)):
        ax.plot(result.t, control[:, j], label=f"k={int(degree.k[j])}")
    ax.set(xlabel="Time", ylabel="Degree-class control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "control.png", dpi=160)
    plt.close(fig)

    print(f"{source}: {graph.number_of_nodes()} nodes; mean degree {degree.kbar:.3g}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print(f"PMP residual: {result.value['final_stationarity_residual']:.3g}; files: {out}")
    if not result.value["converged"]:
        warnings.warn("The sweep reached its iteration limit; the saved policy is a candidate.")
    return comparison


def _package_smoke() -> dict[str, float | bool]:
    """Check all four solvers and the impulse reset independently of the tutorial."""
    k = np.array([1.0, 2.0, 4.0, 6.0])
    counts = np.array([12, 8, 5, 3])
    probability = counts / counts.sum()
    degree = DegreeData(k, counts, probability, float(k @ probability), np.repeat(k, counts))
    matrix = np.zeros((8, 8))
    for node in range(8):
        matrix[node, (node + 1) % 8] = matrix[(node + 1) % 8, node] = 0.5
    results = [
        solve_degree_control(degree, steps=12, iterations=32, tol=1e-4),
        solve_degree_game(degree, steps=12, iterations=32, tol=1e-4),
        solve_node_control(matrix, steps=12, iterations=32, tol=1e-4),
        solve_node_game(matrix, steps=12, iterations=32, tol=1e-4),
    ]
    for result in results:
        assert np.isfinite(result.x).all()
        assert np.all((result.x >= 0.0) & (result.x <= 1.0))
        assert result.value["converged"]
    initial = np.array([0.78, 0.17, 0.05])
    state, _ = rk4_integrate(
        lambda x, t: controlled_sir_rhs(x, 0.08, 0.12, MalwareParams()),
        initial,
        t0=0.0,
        dt=0.5,
        substeps=8,
        project=project_simplex3,
    )
    reset = isolation_jump(state, 0.5)
    return {
        "all_fbs_converged": all(bool(result.value["converged"]) for result in results),
        "max_fbs_final_delta": max(float(result.value["final_delta"]) for result in results),
        "impulse_mass_error": abs(float(reset.sum()) - 1.0),
        "impulse_infected_before": float(state[1]),
        "impulse_infected_after": float(reset[1]),
    }


def _source_root() -> Path:
    if not (ROOT / "docs" / "source").is_dir():
        raise RuntimeError(
            "This extension needs the repository checkout, not only its installed package."
        )
    return ROOT


def _build_docs() -> None:
    source = _source_root() / "docs" / "source"
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "network_control_foundations.tex",
        ],
        cwd=source,
        check=True,
    )
    target = source / "network_control_foundations.pdf"
    (source.parent / target.name).write_bytes(target.read_bytes())


def _study(args: argparse.Namespace) -> None:
    root = _source_root()
    subprocess.run(
        [
            sys.executable,
            "examples/foundations/code/network_control_examples.py",
            "--steps",
            str(args.steps),
            "--iterations",
            str(args.iterations),
            "--tolerance",
            str(args.tolerance),
            "--output-dir",
            str(args.output_dir),
        ],
        cwd=root,
        check=True,
    )
    if args.sizes:
        subprocess.run(
            [
                sys.executable,
                "examples/foundations/code/scalability_analysis.py",
                "compare",
                "--sizes",
                args.sizes,
                "--steps",
                str(args.steps),
                "--iterations",
                str(args.iterations),
                "--repeats",
                "1",
                "--output-dir",
                str(args.output_dir / "scalability"),
            ],
            cwd=root,
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Network -> degree classes -> optimal control.")
    parser.add_argument(
        "command", nargs="?", choices=["example", "check", "study", "docs"], default="example"
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--edge-list")
    inputs.add_argument("--adjacency-csv")
    inputs.add_argument("--graph-file")
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--source-col", default=0)
    parser.add_argument("--target-col", default=1)
    parser.add_argument("--weight-col", default=None)
    parser.add_argument("--demo-nodes", type=int, default=40)
    parser.add_argument("--demo-m", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--beta", type=float, default=0.12)
    parser.add_argument("--recovery", type=float, default=0.18)
    parser.add_argument("--control-cost", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sizes", default="", help="Optional network sizes for the study command.")
    parser.set_defaults(directed=False, adjacency_convention="graph", graph_backend="networkx")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Existing lecture commands continue to work; students need only the default command.
    aliases = {"smoke": "check", "figures": "example", "medium": "study"}
    if argv:
        argv[0] = aliases.get(argv[0], argv[0])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 2 or args.iterations < 1 or args.tolerance <= 0:
        parser.error("Use at least two steps, one iteration, and a positive tolerance.")
    if args.command == "check":
        print(_package_smoke())
    elif args.command == "docs":
        _build_docs()
    elif args.command == "study":
        _study(args)
    else:
        run_example(args)


if __name__ == "__main__":
    main()
