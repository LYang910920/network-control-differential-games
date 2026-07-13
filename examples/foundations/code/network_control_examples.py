# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Runnable foundation examples built from installable cybercontrol APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from cybercontrol.continuous_impulse import simulate_continuous_impulse
from cybercontrol.fbs import (
    TimeSeries,
    solve_degree_control,
    solve_degree_game,
    solve_node_control,
    solve_node_game,
)
from cybercontrol.foundation_evaluation import save_matched_mean_comparison
from cybercontrol.graph_data import (
    degree_distribution_from_graph,
    load_network,
    print_degree_distribution,
)

try:
    from .foundation_reporting import (
        plot_continuous_impulse,
        plot_degree_control,
        plot_degree_game,
        plot_node_control,
        plot_node_game,
        save_baseline_comparison,
        save_degree_distribution,
        save_fbs_convergence,
    )
    from .foundation_parameters import write_parameter_summary
except ImportError:
    from foundation_reporting import (
        plot_continuous_impulse,
        plot_degree_control,
        plot_degree_game,
        plot_node_control,
        plot_node_game,
        save_baseline_comparison,
        save_degree_distribution,
        save_fbs_convergence,
    )
    from foundation_parameters import write_parameter_summary


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = load_network(args)
    D = degree_distribution_from_graph(net.full_graph, mode=args.degree_mode)

    print(f"Loaded network: {net.source}")
    print(
        f"full input nodes={net.full_graph.number_of_nodes()}, edges={net.full_graph.number_of_edges()}, directed={net.full_graph.is_directed()}"
    )
    print(
        f"node-level matrix size={net.A.shape[0]} x {net.A.shape[1]} (after --max-nodes reduction if used)"
    )
    print_degree_distribution(D)
    save_degree_distribution(D, out_dir)
    write_parameter_summary(out_dir, args, D, net.A)

    results: dict[str, TimeSeries] = {}
    if args.examples in {"all", "degree"}:
        results["degree_control"] = solve_degree_control(
            D, steps=args.steps, iterations=args.iterations, tol=args.tolerance
        )
        results["degree_game"] = solve_degree_game(
            D, steps=args.steps, iterations=args.iterations, tol=args.tolerance
        )
        plot_degree_control(results["degree_control"], D, out_dir)
        plot_degree_game(results["degree_game"], D, out_dir)

    if args.examples in {"all", "node"}:
        node_steps = max(24, args.steps // 3)
        results["node_control"] = solve_node_control(
            net.A, steps=node_steps, iterations=args.iterations, tol=args.tolerance
        )
        results["node_game"] = solve_node_game(
            net.A, steps=node_steps, iterations=args.iterations, tol=args.tolerance
        )
        plot_node_control(results["node_control"], out_dir)
        plot_node_game(results["node_game"], out_dir)

    if args.examples in {"all", "continuous-impulse"}:
        plot_continuous_impulse(simulate_continuous_impulse(net.A), out_dir)

    save_fbs_convergence(results, out_dir)
    save_baseline_comparison(results, D, net.A, out_dir)
    save_matched_mean_comparison(results, D, net.A, out_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Thin orchestrator for degree/node control, games, and impulse examples."
    )
    p.add_argument("--edge-list", type=str, help="Path to source-target edge list.")
    p.add_argument("--adjacency-csv", type=str, help="Path to square adjacency matrix CSV.")
    p.add_argument("--graph-file", type=str, help="Path to GraphML/GEXF/GML/Pajek network file.")
    p.add_argument(
        "--adjacency-convention",
        choices=["graph", "model"],
        default="graph",
        help="For directed CSVs: graph means M[source,target]; model means A[i,j]=j influences i.",
    )
    p.add_argument(
        "--delimiter", type=str, default=None, help="Edge-list delimiter; default is whitespace."
    )
    p.add_argument(
        "--has-header", action="store_true", help="Set if edge-list file has a header row."
    )
    p.add_argument("--source-col", default=0, help="Source column index/name in edge list.")
    p.add_argument("--target-col", default=1, help="Target column index/name in edge list.")
    p.add_argument("--weight-col", default=None, help="Optional weight column index/name.")
    p.add_argument("--directed", action="store_true", help="Treat input as directed.")
    p.add_argument("--graph-backend", choices=["networkx", "igraph"], default="networkx")
    p.add_argument(
        "--max-nodes",
        type=int,
        default=8,
        help="Keep top weighted-degree nodes; use 0 to keep all.",
    )
    p.add_argument(
        "--normalize", choices=["none", "max-degree", "row", "spectral"], default="max-degree"
    )
    p.add_argument(
        "--degree-mode", choices=["in", "out", "total", "undirected"], default="undirected"
    )
    p.add_argument(
        "--examples", choices=["all", "degree", "node", "continuous-impulse"], default="all"
    )
    p.add_argument(
        "--steps", type=int, default=45, help="Time grid size for degree-level examples."
    )
    p.add_argument(
        "--iterations", type=int, default=45, help="Maximum FBS fixed-point evaluations."
    )
    p.add_argument(
        "--tolerance", type=float, default=1e-4, help="Projected stationarity-residual tolerance."
    )
    p.add_argument("--demo-nodes", type=int, default=14)
    p.add_argument("--demo-m", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/foundation_models"))
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
