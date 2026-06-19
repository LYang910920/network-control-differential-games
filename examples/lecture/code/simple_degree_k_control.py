# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""
Simple degree-k network optimal-control example.

This is the short teaching script to read first.  It does four things:

1. load a network from an edge list, an adjacency CSV, or a built-in demo graph;
2. compute the empirical degree distribution N_k, P(k), and average degree <k>;
3. solve a degree-k SIS optimal-control problem by forward-backward sweep;
4. save a degree-distribution figure and a control/result figure.

The state is indexed by degree class k, not by S/I/R compartments:
    x[j, t] = infected fraction among nodes with degree k[j].

Install:
    pip install numpy scipy pandas matplotlib networkx

Run:
    python simple_degree_k_control.py
    python simple_degree_k_control.py --edge-list sample_data/sample_edges.csv --delimiter , --has-header
    python simple_degree_k_control.py --adjacency-csv sample_data/sample_adjacency.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from model_profiles import DegreeControlProfile, SIMPLE_DEGREE_CONTROL


EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from common_diagnostics import (  # noqa: E402
    RANDOM_BASELINE_COUNT,
    RANDOM_BASELINE_SEED,
    control_baseline_rows,
    save_control_baseline_plot,
    write_baseline_table,
)


# -----------------------------------------------------------------------------
# Small numerical helpers
# -----------------------------------------------------------------------------


def clip(z: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.minimum(np.maximum(z, lo), hi)


def trapz(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.trapezoid(y, t) if hasattr(np, "trapezoid") else np.trapz(y, t))


def as_time_function(t: np.ndarray, values: np.ndarray) -> Callable[[float], np.ndarray]:
    return interp1d(t, values, axis=0, bounds_error=False, fill_value="extrapolate")


def solve_ode_on_grid(rhs, y0: np.ndarray, t: np.ndarray, *, backward: bool = False) -> np.ndarray:
    """Solve an ODE on grid t.  For backward=True, y0 is the terminal value."""
    span = (t[-1], t[0]) if backward else (t[0], t[-1])
    grid = t[::-1] if backward else t
    sol = solve_ivp(rhs, span, y0, t_eval=grid, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)
    out = sol.y.T
    return out[::-1] if backward else out


# -----------------------------------------------------------------------------
# Network loading and degree distribution
# -----------------------------------------------------------------------------


def _resolve_column(df: pd.DataFrame, col_ref) -> str | int:
    """Allow a column name such as 'source' or a zero-based index such as 0."""
    if col_ref in df.columns:
        return col_ref
    return df.columns[int(col_ref)]


def load_graph(args: argparse.Namespace) -> nx.Graph:
    """Return a NetworkX graph from edge list, adjacency CSV, or demo graph."""
    if args.edge_list:
        sep = r"\s+" if args.delimiter is None else args.delimiter
        header = 0 if args.has_header else None
        df = pd.read_csv(args.edge_list, sep=sep, header=header, comment="#", engine="python")
        src = _resolve_column(df, args.source_col)
        dst = _resolve_column(df, args.target_col)
        df = df[df[src] != df[dst]].copy()
        G = nx.from_pandas_edgelist(df, src, dst, create_using=nx.Graph())
    elif args.adjacency_csv:
        A = pd.read_csv(args.adjacency_csv, header=None).to_numpy(float)
        A = np.asarray(A, dtype=float)
        A[~np.isfinite(A)] = 0.0
        A[A < 0] = 0.0
        np.fill_diagonal(A, 0.0)
        G = nx.from_numpy_array((A + A.T) > 0)
    else:
        G = nx.barabasi_albert_graph(args.demo_nodes, args.demo_m, seed=args.seed)

    G = nx.Graph(G)              # this simple example uses undirected degree classes
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def degree_distribution(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute observed degree classes k, counts N_k, probabilities P(k), and <k>."""
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    k, counts = np.unique(degrees, return_counts=True)
    p = counts / counts.sum()
    kbar = float(k @ p)
    return k.astype(float), counts.astype(int), p.astype(float), kbar


def print_degree_table(k: np.ndarray, counts: np.ndarray, p: np.ndarray, kbar: float) -> None:
    print("\nEmpirical degree distribution")
    print(f"average degree <k> = {kbar:.4f}")
    print("   k      N_k      P(k)")
    for kk, nn, pp in zip(k.astype(int), counts, p):
        print(f"{kk:4d} {nn:8d} {pp:9.4f}")


# -----------------------------------------------------------------------------
# Degree-k SIS optimal-control model
# -----------------------------------------------------------------------------


def theta(x: np.ndarray, k: np.ndarray, p: np.ndarray, kbar: float) -> float:
    """Mean-field infection pressure: Theta = sum_k k P(k) x_k / <k>."""
    return float((k * p) @ x / max(kbar, 1e-12))


def state_rhs(x: np.ndarray, u: np.ndarray, k: np.ndarray, p: np.ndarray, kbar: float,
              beta: float, delta: float) -> np.ndarray:
    """Degree-k SIS dynamics with degree-specific control u_k."""
    return beta * k * (1.0 - x) * theta(x, k, p, kbar) - (delta + u) * x


def state_jacobian(x: np.ndarray, u: np.ndarray, k: np.ndarray, p: np.ndarray, kbar: float,
                   beta: float, delta: float) -> np.ndarray:
    """Jacobian of state_rhs with respect to x."""
    dtheta_dx = k * p / max(kbar, 1e-12)
    J = beta * np.outer(k * (1.0 - x), dtheta_dx)
    J[np.diag_indices_from(J)] += -beta * k * theta(x, k, p, kbar) - (delta + u)
    return J


def initial_degree_state(k: np.ndarray, profile: DegreeControlProfile = SIMPLE_DEGREE_CONTROL) -> np.ndarray:
    """Use a small degree-dependent initial infection for visual contrast."""
    return clip(
        profile.initial_low + profile.initial_degree_scale * k / max(k.max(), 1.0),
        0.0,
        profile.initial_cap,
    )


def solve_degree_k_control(
    k: np.ndarray,
    p: np.ndarray,
    kbar: float,
    *,
    horizon: float = SIMPLE_DEGREE_CONTROL.horizon,
    steps: int = 220,
    iterations: int = SIMPLE_DEGREE_CONTROL.max_iterations,
    beta: float = SIMPLE_DEGREE_CONTROL.beta,
    delta: float = SIMPLE_DEGREE_CONTROL.delta,
    infection_weight: float = SIMPLE_DEGREE_CONTROL.infection_weight,
    control_weight: float = SIMPLE_DEGREE_CONTROL.control_weight,
    u_max: float = SIMPLE_DEGREE_CONTROL.u_max,
    damping: float = SIMPLE_DEGREE_CONTROL.damping,
    tol: float = SIMPLE_DEGREE_CONTROL.tolerance,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool]:
    """Forward-backward sweep for degree-k optimal control.

    Cost to minimize:
        integral [ q sum_k P(k) x_k + 0.5 r sum_k P(k) u_k^2 ] dt.
    """
    t = np.linspace(0.0, horizon, steps + 1)
    m = len(k)
    U = np.zeros((len(t), m))
    X0 = initial_degree_state(k)
    delta_history: list[float] = []

    for it in range(iterations):
        U_old = U.copy()
        U_of_t = as_time_function(t, U)

        X = clip(solve_ode_on_grid(
            lambda tau, y: state_rhs(y, U_of_t(tau), k, p, kbar, beta, delta),
            X0,
            t,
        ))
        X_of_t = as_time_function(t, X)

        def adjoint_rhs(tau: float, lam: np.ndarray) -> np.ndarray:
            x_now = X_of_t(tau)
            u_now = U_of_t(tau)
            J = state_jacobian(x_now, u_now, k, p, kbar, beta, delta)
            return -infection_weight * p - J.T @ lam

        Lambda = solve_ode_on_grid(adjoint_rhs, np.zeros(m), t, backward=True)

        # Stationarity of the Hamiltonian gives r P(k) u_k - lambda_k x_k = 0.
        candidate = Lambda * X / (control_weight * np.maximum(p, 1e-12))
        U = damping * clip(candidate, 0.0, u_max) + (1.0 - damping) * U_old

        delta_u = float(np.max(np.abs(U - U_old)))
        delta_history.append(delta_u)
        if delta_u < tol:
            print(f"forward-backward sweep converged after {it + 1} iterations")
            break

    X, cost = evaluate_degree_k_control(k, p, kbar, t, U, beta, delta, infection_weight, control_weight)
    converged = bool(delta_history and delta_history[-1] < tol)
    if not converged:
        print(f"forward-backward sweep warning: final control change = {delta_history[-1]:.3e}")
    return t, X, U, cost, np.asarray(delta_history, dtype=float), converged


def evaluate_degree_k_control(
    k: np.ndarray,
    p: np.ndarray,
    kbar: float,
    t: np.ndarray,
    U: np.ndarray,
    beta: float = SIMPLE_DEGREE_CONTROL.beta,
    delta: float = SIMPLE_DEGREE_CONTROL.delta,
    infection_weight: float = SIMPLE_DEGREE_CONTROL.infection_weight,
    control_weight: float = SIMPLE_DEGREE_CONTROL.control_weight,
) -> tuple[np.ndarray, float]:
    """Evaluate the degree-k objective for a fixed control curve."""
    X0 = initial_degree_state(k)
    U_of_t = as_time_function(t, U)
    X = clip(solve_ode_on_grid(
        lambda tau, y: state_rhs(y, U_of_t(tau), k, p, kbar, beta, delta),
        X0,
        t,
    ))
    cost = trapz(infection_weight * (X @ p) + 0.5 * control_weight * ((U * U) @ p), t)
    return X, cost


# -----------------------------------------------------------------------------
# Plotting and command line
# -----------------------------------------------------------------------------


def save_results(out_dir: Path, k: np.ndarray, counts: np.ndarray, p: np.ndarray, kbar: float,
                 t: np.ndarray, X: np.ndarray, U: np.ndarray, cost: float,
                 delta_history: np.ndarray, converged: bool, tolerance: float = 1e-4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("degree_k_control.png", "baseline_comparison.png"):
        legacy_path = out_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    pd.DataFrame({"k": k.astype(int), "N_k": counts, "P(k)": p}).to_csv(out_dir / "degree_distribution.csv", index=False)
    pd.DataFrame(
        [
            {"parameter": "model_level", "value": "degree-k", "meaning": "State is the infected fraction for each observed degree class."},
            {"parameter": "horizon", "value": float(t[-1]), "meaning": "Total simulated time."},
            {"parameter": "time_grid_steps", "value": len(t) - 1, "meaning": "Number of intervals used by the ODE/FBS time grid."},
            {"parameter": "beta", "value": SIMPLE_DEGREE_CONTROL.beta, "meaning": "Infection/contact rate in the SIS dynamics."},
            {"parameter": "delta", "value": SIMPLE_DEGREE_CONTROL.delta, "meaning": "Natural recovery/removal rate before control."},
            {"parameter": "u_max", "value": SIMPLE_DEGREE_CONTROL.u_max, "meaning": "Upper bound for the continuous healing/control rate u_k(t)."},
            {"parameter": "infection_weight", "value": SIMPLE_DEGREE_CONTROL.infection_weight, "meaning": "Objective weight on infected fraction."},
            {"parameter": "control_weight", "value": SIMPLE_DEGREE_CONTROL.control_weight, "meaning": "Quadratic cost weight on control effort."},
            {"parameter": "initial_infection_range", "value": f"{float(initial_degree_state(k).min()):.3f}-{float(initial_degree_state(k).max()):.3f}", "meaning": "Degree-dependent initial infected fraction range."},
            {"parameter": "fbs_tolerance", "value": tolerance, "meaning": "Convergence threshold for max control change."},
            {"parameter": "random_baselines", "value": RANDOM_BASELINE_COUNT, "meaning": "Number of random smooth controls used in the baseline comparison."},
        ]
    ).to_csv(out_dir / "parameter_summary.csv", index=False)

    plt.figure(figsize=(6.8, 4.0))
    plt.bar(k, p)
    plt.xlabel("degree k")
    plt.ylabel("P(k)")
    plt.title(f"Empirical degree distribution; average degree = {kbar:.2f}")
    plt.tight_layout()
    plt.savefig(out_dir / "degree_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(t, X @ p, label="state: population-weighted degree-class mean infection", linewidth=2.0, linestyle="-")
    plt.plot(t, U @ p, label="continuous control: population-weighted degree-class mean", linewidth=2.0, linestyle="-.")
    for j in sorted(set([0, len(k) // 2, len(k) - 1])):
        plt.plot(t, X[:, j], "--", linewidth=1.5, label=f"state: degree k={int(k[j])}")
    plt.xlabel("time")
    plt.ylabel("state / control")
    plt.title(f"Degree-k continuous optimal control; cost={cost:.2f}")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "degree_control_trajectory.png", dpi=180)
    plt.close()

    pd.DataFrame(
        {
            "iteration": np.arange(1, len(delta_history) + 1),
            "max_control_change": delta_history,
            "tolerance": np.full(len(delta_history), tolerance),
            "below_tolerance": delta_history < tolerance,
            "run_converged": np.full(len(delta_history), converged),
        }
    ).to_csv(out_dir / "fbs_convergence.csv", index=False)

    plt.figure(figsize=(7.2, 4.0))
    plt.semilogy(np.arange(1, len(delta_history) + 1), delta_history, marker="o", linewidth=2.0)
    plt.axhline(tolerance, color="0.45", linestyle=":", linewidth=1.4, label=f"tolerance={tolerance:.0e}")
    plt.xlabel("FBS iteration")
    plt.ylabel("max control change")
    plt.title(f"Forward-backward sweep convergence; converged={converged}")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fbs_convergence.png", dpi=180)
    plt.close()

    baseline_rows = control_baseline_rows(
        U,
        cost,
        lambda candidate: evaluate_degree_k_control(k, p, kbar, t, candidate)[1],
        random_upper=SIMPLE_DEGREE_CONTROL.u_max,
        seed=RANDOM_BASELINE_SEED,
    )
    write_baseline_table(baseline_rows, out_dir / "baseline_summary.csv")
    save_control_baseline_plot(
        baseline_rows,
        out_dir / "degree_control_baseline_comparison.png",
        title="Degree-k continuous control vs baselines",
        ylabel="cost (lower is better)",
        random_label=f"{RANDOM_BASELINE_COUNT} random smooth controls",
    )

    print(f"saved outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal degree-k network optimal-control example.")
    parser.add_argument("--edge-list", type=str, help="Edge-list file: source target, or CSV with --delimiter ,")
    parser.add_argument("--adjacency-csv", type=str, help="Square adjacency matrix CSV.")
    parser.add_argument("--delimiter", type=str, default=None, help="Edge-list delimiter; default is whitespace.")
    parser.add_argument("--has-header", action="store_true", help="Set if edge-list has a header row.")
    parser.add_argument("--source-col", default=0, help="Source column name/index.")
    parser.add_argument("--target-col", default=1, help="Target column name/index.")
    parser.add_argument("--demo-nodes", type=int, default=30, help="Number of demo nodes.")
    parser.add_argument("--demo-m", type=int, default=2, help="BA demo graph attachment parameter.")
    parser.add_argument("--steps", type=int, default=220, help="Time grid size on horizon T=14.0.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="simple_outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph = load_graph(args)
    k, counts, p, kbar = degree_distribution(graph)
    print(f"loaded graph with n={graph.number_of_nodes()}, m={graph.number_of_edges()}")
    print_degree_table(k, counts, p, kbar)
    t, X, U, cost, delta_history, converged = solve_degree_k_control(k, p, kbar, steps=args.steps)
    save_results(Path(args.output_dir), k, counts, p, kbar, t, X, U, cost, delta_history, converged)


if __name__ == "__main__":
    main()
