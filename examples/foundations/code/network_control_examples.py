# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Runnable foundation examples built from installable cybercontrol APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cybercontrol.fbs import (  # noqa: F401 - compatibility exports
    TimeSeries,
    degree_control_model,
    degree_game_jacobian,
    degree_game_model,
    degree_game_rhs,
    degree_jacobian,
    degree_rhs,
    node_control_model,
    node_game_jacobian,
    node_game_model,
    node_game_rhs,
    node_jacobian,
    node_rhs,
    solve_degree_control,
    solve_degree_game,
    solve_node_control,
    solve_node_game,
)
from cybercontrol.foundation_evaluation import (  # noqa: F401 - compatibility exports
    evaluate_degree_control_policy,
    evaluate_degree_game_strategy,
    evaluate_node_control_policy,
    evaluate_node_game_strategy,
    save_matched_mean_comparison,
)
from cybercontrol.foundation_profiles import (  # noqa: F401 - compatibility exports
    CONTINUOUS_IMPULSE_PROFILE,
    NODE_CONTROL_PROFILE,
    NODE_GAME_PROFILE,
    ContinuousImpulseProfile,
)
from cybercontrol.graph_data import (  # noqa: F401 - compatibility exports
    DegreeData,
    NetworkData,
    degree_distribution_from_graph,
    degree_initial_state,
    graph_to_model_matrix,
    high_degree_nodes,
    load_network,
    node_initial_state,
    normalize_adjacency,
    prepare_model_matrix,
    print_degree_distribution,
)
from cybercontrol.numerics import (
    project_box as clip,
    solve_ode_grid as solve_grid,
    trapezoid_integral as integral,  # noqa: F401 - compatibility export
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
        write_parameter_summary,
    )
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
        write_parameter_summary,
    )


def simulate_continuous_impulse(A: np.ndarray, profile: ContinuousImpulseProfile = CONTINUOUS_IMPULSE_PROFILE) -> TimeSeries:
    """Time-varying continuous node control plus state jumps on high-degree nodes.

    This is a transparent simulation template for continuous-impulse dynamics.
    It does not solve the full impulse PMP; it shows how to combine solve_ivp segments with
    impulse maps x(tau+) = G(x(tau-), z_tau).
    """
    T = profile.horizon
    beta, delta, impulse_fraction = profile.beta, profile.delta, profile.impulse_fraction
    n = A.shape[0]
    controlled = high_degree_nodes(A, max(1, int(np.ceil(profile.controlled_node_fraction * n))))

    def continuous_u_level(tau: float) -> float:
        phase = tau / max(T, 1e-12)
        level = profile.continuous_base + profile.continuous_ramp * phase
        level += profile.continuous_wave * np.sin(2.0 * np.pi * phase)
        return float(clip(level, profile.continuous_lower, profile.continuous_upper))

    def rhs_with_time_varying_control(tau: float, y: np.ndarray) -> np.ndarray:
        u = np.zeros(n)
        u[controlled] = continuous_u_level(tau)
        return node_rhs(y, u, A, beta, delta)

    x = np.full(n, 0.03)
    x[high_degree_nodes(A, 2)] = 0.15

    impulse_times_arr = np.asarray(profile.impulse_times, dtype=float)
    impulse_times_arr = np.unique(np.sort(impulse_times_arr[(impulse_times_arr > 0.0) & (impulse_times_arr < T)]))
    segment_bounds = [0.0, *impulse_times_arr.tolist(), T]
    all_t, all_x = [], []
    for t0, t1 in zip(segment_bounds[:-1], segment_bounds[1:]):
        grid = np.linspace(t0, t1, max(2, int(30 * (t1 - t0)) + 1))
        segment = clip(solve_grid(rhs_with_time_varying_control, x, grid))
        start = 0 if not all_t else 1
        all_t.extend(grid[start:])
        all_x.extend(segment[start:])
        x = segment[-1].copy()
        if np.any(np.isclose(t1, impulse_times_arr)):
            # Store the post-jump state at the same time to make the jump visible.
            x[controlled] *= 1.0 - impulse_fraction
            all_t.append(t1)
            all_x.append(x.copy())

    t = np.asarray(all_t, dtype=float)
    impulse_heights = np.full(len(impulse_times_arr), impulse_fraction, dtype=float)
    continuous_control = np.array([continuous_u_level(tau) for tau in t], dtype=float)
    return TimeSeries(
        t=t,
        x=np.vstack(all_x),
        controls={
            "controlled_nodes": controlled,
            "continuous_control": continuous_control,
            "impulse_times": impulse_times_arr,
            "impulse_heights": impulse_heights,
        },
        value={
            "continuous_control_min": float(np.min(continuous_control)),
            "continuous_control_max": float(np.max(continuous_control)),
            "impulse_fraction": impulse_fraction,
            "controlled_count": float(len(controlled)),
        },
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "degree_control.png",
        "degree_game.png",
        "node_control.png",
        "node_game.png",
        "continuous_impulse.png",
        "baseline_comparison.png",
    ):
        legacy_path = out_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    net = load_network(args)
    D = degree_distribution_from_graph(net.full_graph, mode=args.degree_mode)

    print(f"Loaded network: {net.source}")
    print(f"full input nodes={net.full_graph.number_of_nodes()}, edges={net.full_graph.number_of_edges()}, directed={net.full_graph.is_directed()}")
    print(f"node-level matrix size={net.A.shape[0]} x {net.A.shape[1]} (after --max-nodes reduction if used)")
    print_degree_distribution(D)
    save_degree_distribution(D, out_dir)
    write_parameter_summary(out_dir, args, D, net.A)

    results: dict[str, TimeSeries] = {}
    if args.examples in {"all", "degree"}:
        results["degree_control"] = solve_degree_control(D, steps=args.steps)
        results["degree_game"] = solve_degree_game(D, steps=args.steps)
        plot_degree_control(results["degree_control"], D, out_dir)
        plot_degree_game(results["degree_game"], D, out_dir)

    if args.examples in {"all", "node"}:
        node_steps = max(24, args.steps // 3)
        results["node_control"] = solve_node_control(net.A, steps=node_steps)
        results["node_game"] = solve_node_game(net.A, steps=node_steps)
        plot_node_control(results["node_control"], out_dir)
        plot_node_game(results["node_game"], out_dir)

    if args.examples in {"all", "continuous-impulse"}:
        plot_continuous_impulse(simulate_continuous_impulse(net.A), out_dir)

    save_fbs_convergence(results, out_dir)
    save_baseline_comparison(results, D, net.A, out_dir)
    save_matched_mean_comparison(results, D, net.A, out_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-file extension examples with degree-k aggregation.")
    p.add_argument("--edge-list", type=str, help="Path to source-target edge list.")
    p.add_argument("--adjacency-csv", type=str, help="Path to square adjacency matrix CSV.")
    p.add_argument("--graph-file", type=str, help="Path to GraphML/GEXF/GML/Pajek network file.")
    p.add_argument("--adjacency-convention", choices=["graph", "model"], default="graph",
                   help="For directed CSVs: graph means M[source,target]; model means A[i,j]=j influences i.")
    p.add_argument("--delimiter", type=str, default=None, help="Edge-list delimiter; default is whitespace.")
    p.add_argument("--has-header", action="store_true", help="Set if edge-list file has a header row.")
    p.add_argument("--source-col", default=0, help="Source column index/name in edge list.")
    p.add_argument("--target-col", default=1, help="Target column index/name in edge list.")
    p.add_argument("--weight-col", default=None, help="Optional weight column index/name.")
    p.add_argument("--directed", action="store_true", help="Treat input as directed.")
    p.add_argument("--graph-backend", choices=["networkx", "igraph"], default="networkx")
    p.add_argument("--max-nodes", type=int, default=8, help="Keep top weighted-degree nodes; use 0 to keep all.")
    p.add_argument("--normalize", choices=["none", "max-degree", "row", "spectral"], default="max-degree")
    p.add_argument("--degree-mode", choices=["in", "out", "total", "undirected"], default="undirected")
    p.add_argument("--examples", choices=["all", "degree", "node", "continuous-impulse"], default="all")
    p.add_argument("--steps", type=int, default=45, help="Time grid size for degree-level examples.")
    p.add_argument("--demo-nodes", type=int, default=14)
    p.add_argument("--demo-m", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="example_outputs")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
