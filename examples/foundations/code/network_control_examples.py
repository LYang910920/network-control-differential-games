# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""
Single-file extension examples for network optimal control and differential games.

This main tutorial script can be read after the simple degree-k example.
It keeps the research-code-style examples in ONE Python file rather than many
modules: graph loading, degree distribution, ODE helpers, degree-k models,
node-level models, hybrid/impulse simulation, plotting, and CLI options.

It uses standard packages and the shared package for the heavy lifting:
    networkx / pandas   realistic network datasets and adjacency conversion
    cybercontrol        ODE-grid, projection, and quadrature helpers

Recommended installation
------------------------
    pip install numpy scipy pandas matplotlib networkx

Optional graph backend
----------------------
    pip install igraph

Run the built-in demo network
-----------------------------
    python network_control_examples.py

Run with a realistic edge-list dataset
--------------------------------------
    python network_control_examples.py --edge-list data/edges.txt
    python network_control_examples.py --edge-list data/edges.csv \
        --delimiter , --has-header --weight-col 2 --max-nodes 50

Run with an adjacency matrix CSV
-------------------------------
    # default: CSV uses ordinary graph convention M[source, target]
    python network_control_examples.py --adjacency-csv data/A.csv

    # use this if the CSV is already in the model convention A[i,j]=j influences i
    python network_control_examples.py --adjacency-csv data/A.csv --adjacency-convention model

Core convention
---------------
The ODE models use pressure = A @ x, so A[i,j] means "node j influences node i".
For a directed edge-list source -> target, source influences target, hence the
usual graph adjacency matrix is transposed before being used in the ODEs.

Degree-level models are indexed by observed degree classes k, not compartments:
    N_k, P(k)=N_k/n, <k>=sum_k kP(k),
    Theta(t)=sum_k kP(k)x_k(t)/<k>.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
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
from cybercontrol.heterogeneity import (
    DegreeSISParams,
    NodeSISParams,
    degree_correlated_node_sis_params,
    degree_correlated_sis_params,
    degree_sis_jacobian,
    degree_sis_rhs,
    node_sis_force,
    node_sis_jacobian,
    node_sis_rhs,
)
from cybercontrol.numerics import (
    as_time_function as as_function,
    project_box as clip,
    solve_ode_grid as solve_grid,
    trapezoid_integral as integral,
)
from cybercontrol.plotting import apply_clean_axes, plot_time_series, save_publication_figure
try:
    from .model_profiles import (
        DEGREE_CONTROL_PROFILE,
        DEGREE_GAME_PROFILE,
        HYBRID_PROFILE,
        HybridImpulseProfile,
        NODE_CONTROL_PROFILE,
        NODE_GAME_PROFILE,
    )
except ImportError:
    from model_profiles import (
        DEGREE_CONTROL_PROFILE,
        DEGREE_GAME_PROFILE,
        HYBRID_PROFILE,
        HybridImpulseProfile,
        NODE_CONTROL_PROFILE,
        NODE_GAME_PROFILE,
    )


LINE_WIDTH = 2.0
FIGSIZE_SINGLE = (7.4, 4.2)
FIGSIZE_STACKED = (7.4, 5.8)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class NetworkData:
    """Full input graph plus reduced node-level matrix A.

    full_graph is used to compute the empirical degree distribution P(k).
    graph and A may be a smaller high-degree induced subgraph, used only for
    node-level PMP/game examples where dense matrices are expensive.
    """

    full_graph: nx.Graph
    graph: nx.Graph
    A: np.ndarray
    labels: list[str]
    source: str


@dataclass
class DegreeData:
    """Empirical degree distribution used by the degree-k models."""

    k: np.ndarray          # observed degree classes
    Nk: np.ndarray         # number of nodes with degree k
    pk: np.ndarray         # P(k) = Nk / n
    kbar: float            # average degree <k>
    node_degree: np.ndarray


@dataclass
class TimeSeries:
    """Generic container for trajectories."""

    t: np.ndarray
    x: np.ndarray
    controls: dict[str, np.ndarray]
    value: dict[str, float]


def relax(new: np.ndarray, old: np.ndarray, weight: float = 0.35) -> np.ndarray:
    """Damped update for forward-backward sweep iterations."""
    return weight * new + (1.0 - weight) * old


def convergence_values(delta_history: list[float], tol: float) -> dict[str, float]:
    """Small numeric summary for forward-backward sweep diagnostics."""
    final_delta = float(delta_history[-1]) if delta_history else float("nan")
    return {
        "iterations": float(len(delta_history)),
        "final_delta": final_delta,
        "converged": float(final_delta < tol),
        "tolerance": float(tol),
    }


def savefig(path: Path) -> None:
    """Save the current figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    save_publication_figure(plt.gcf(), path, metadata={"source": "foundation example"})
    plt.close()
    print(f"saved {path}")


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


# ---------------------------------------------------------------------------
# Network loading: demo graph, edge list, adjacency matrix
# ---------------------------------------------------------------------------


def read_edge_table(args: argparse.Namespace) -> pd.DataFrame:
    """Read whitespace or CSV-like edge-list data with pandas."""
    sep = r"\s+" if args.delimiter is None else args.delimiter
    header = 0 if args.has_header else None
    return pd.read_csv(args.edge_list, sep=sep, header=header, comment="#", engine="python")


def resolve_column(df: pd.DataFrame, ref) -> str | int:
    """Resolve either a column name or a zero-based integer column index."""
    if ref in df.columns:
        return ref
    index = int(ref)
    return df.columns[index]


def edge_list_to_graph_networkx(args: argparse.Namespace) -> nx.Graph:
    """Build a NetworkX graph from an edge-list table."""
    df = read_edge_table(args)
    src = resolve_column(df, args.source_col)
    dst = resolve_column(df, args.target_col)
    graph_type = nx.DiGraph() if args.directed else nx.Graph()

    df = df[df[src] != df[dst]].copy()
    if args.weight_col is None:
        df["weight"] = 1.0
    else:
        w = resolve_column(df, args.weight_col)
        df["weight"] = pd.to_numeric(df[w], errors="coerce")
        df = df.dropna(subset=["weight"])
        df = df[df["weight"] > 0]

    # Multiple rows for the same edge are common in real datasets.
    # We aggregate them into one weighted edge.
    df = df.groupby([src, dst], as_index=False)["weight"].sum()
    return nx.from_pandas_edgelist(df, src, dst, edge_attr="weight", create_using=graph_type)


def edge_list_to_graph_igraph(args: argparse.Namespace) -> nx.Graph:
    """Optional igraph reader; returns NetworkX graph for the rest of the code."""
    try:
        import igraph as ig  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install igraph first: pip install igraph") from exc

    df = read_edge_table(args)
    src = resolve_column(df, args.source_col)
    dst = resolve_column(df, args.target_col)
    w = resolve_column(df, args.weight_col) if args.weight_col is not None else None
    triples = []
    for _, row in df.iterrows():
        weight = float(row[w]) if w is not None else 1.0
        if weight > 0 and row[src] != row[dst]:
            triples.append((str(row[src]), str(row[dst]), weight))

    ig_graph = ig.Graph.TupleList(triples, directed=args.directed, weights=True)
    G = nx.DiGraph() if args.directed else nx.Graph()
    for e in ig_graph.es:
        u = ig_graph.vs[e.source]["name"]
        v = ig_graph.vs[e.target]["name"]
        G.add_edge(u, v, weight=float(e["weight"]))
    return G


def graph_file_to_graph(path: Path, directed: bool) -> nx.Graph:
    """Load common graph formats with NetworkX: GraphML, GEXF, GML, Pajek/NET."""
    suffix = path.suffix.lower()
    if suffix == ".graphml":
        G = nx.read_graphml(path)
    elif suffix == ".gexf":
        G = nx.read_gexf(path)
    elif suffix == ".gml":
        G = nx.read_gml(path)
    elif suffix in {".net", ".pajek"}:
        G = nx.read_pajek(path)
    else:
        raise ValueError("--graph-file supports .graphml, .gexf, .gml, .net, and .pajek")
    return nx.DiGraph(G) if directed else nx.Graph(G)


def keep_top_degree_nodes(G: nx.Graph, max_nodes: int) -> nx.Graph:
    """Induced subgraph on highest weighted-degree nodes, for readable demos."""
    if max_nodes <= 0 or G.number_of_nodes() <= max_nodes:
        return G.copy()
    degree = dict(G.degree(weight="weight"))
    keep = sorted(degree, key=degree.get, reverse=True)[:max_nodes]
    return G.subgraph(keep).copy()


def normalize_adjacency(A: np.ndarray, method: str) -> np.ndarray:
    """Scale A for numerical stability."""
    A = np.asarray(A, dtype=float).copy()
    if method == "none":
        return A
    if method == "max-degree":
        scale = A.sum(axis=1).max(initial=0.0)
        return A / scale if scale > 0 else A
    if method == "row":
        row_sum = A.sum(axis=1)
        B = A.copy()
        positive = row_sum > 0
        B[positive, :] = B[positive, :] / row_sum[positive, None]
        return B
    if method == "spectral":
        radius = float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0
        return A / radius if radius > 0 else A
    raise ValueError("normalization must be: none, max-degree, row, spectral")


def prepare_model_matrix(A: np.ndarray, normalize: str) -> np.ndarray:
    """Clean and normalize a matrix already in model convention A[i,j]=j influences i."""
    A = np.asarray(A, dtype=float).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    A[~np.isfinite(A)] = 0.0
    A[A < 0] = 0.0
    np.fill_diagonal(A, 0.0)
    return normalize_adjacency(A, normalize)


def graph_to_model_matrix(G: nx.Graph, directed: bool, normalize: str) -> tuple[np.ndarray, list[str]]:
    """Convert graph to A where A[i,j] means j influences i."""
    labels = [str(v) for v in G.nodes()]
    M = nx.to_numpy_array(G, nodelist=list(G.nodes()), weight="weight", dtype=float)
    A = M.T if directed else np.maximum(M, M.T)
    return prepare_model_matrix(A, normalize), labels


def load_network(args: argparse.Namespace) -> NetworkData:
    """Load a graph and produce its ODE-ready influence matrix.

    The full graph is kept for degree-distribution computation.  If --max-nodes
    is used, only the node-level matrix A is reduced. This keeps the degree-k
    model faithful to the input dataset while keeping dense node-level examples
    small enough for a laptop.
    """
    if args.adjacency_csv:
        M = pd.read_csv(args.adjacency_csv, header=None).to_numpy(float)
        if args.adjacency_convention == "model":
            full_A = prepare_model_matrix(M, args.normalize)
            full_G = nx.from_numpy_array(full_A.T if args.directed else full_A, create_using=nx.DiGraph() if args.directed else nx.Graph())
        else:  # ordinary graph convention: M[source, target]
            graph_A = prepare_model_matrix(M, args.normalize)
            full_G = nx.from_numpy_array(graph_A, create_using=nx.DiGraph() if args.directed else nx.Graph())
        source = args.adjacency_csv
    elif args.edge_list:
        full_G = edge_list_to_graph_igraph(args) if args.graph_backend == "igraph" else edge_list_to_graph_networkx(args)
        source = args.edge_list
    elif args.graph_file:
        full_G = graph_file_to_graph(Path(args.graph_file), args.directed)
        nx.set_edge_attributes(full_G, {e: full_G.edges[e].get("weight", 1.0) for e in full_G.edges}, "weight")
        source = args.graph_file
    else:
        full_G = nx.barabasi_albert_graph(args.demo_nodes, args.demo_m, seed=args.seed)
        nx.set_edge_attributes(full_G, 1.0, "weight")
        source = f"Barabasi-Albert demo graph, n={args.demo_nodes}"

    # Remove self-loops in both the full graph and reduced graph.
    full_G = full_G.copy()
    full_G.remove_edges_from(nx.selfloop_edges(full_G))

    G = keep_top_degree_nodes(full_G, args.max_nodes)
    A, labels = graph_to_model_matrix(G, args.directed, args.normalize)
    return NetworkData(full_graph=full_G, graph=G, A=A, labels=labels, source=source)


# ---------------------------------------------------------------------------
# Degree distribution
# ---------------------------------------------------------------------------


def degree_distribution_from_graph(G: nx.Graph, mode: str = "undirected") -> DegreeData:
    """Compute N_k, P(k), and <k> from the FULL input graph.

    For undirected/contact networks, use mode="undirected".  For directed
    influence networks, choose in, out, or total depending on the modelling role
    of degree.
    """
    if (not G.is_directed()) or mode == "undirected":
        deg = np.array([d for _, d in nx.Graph(G).degree()], dtype=int)
    elif mode == "in":
        deg = np.array([d for _, d in G.in_degree()], dtype=int)   # type: ignore[attr-defined]
    elif mode == "out":
        deg = np.array([d for _, d in G.out_degree()], dtype=int)  # type: ignore[attr-defined]
    elif mode == "total":
        deg = np.array([G.in_degree(v) + G.out_degree(v) for v in G.nodes()], dtype=int)  # type: ignore[attr-defined]
    else:
        raise ValueError("degree mode must be: in, out, total, undirected")

    k, Nk = np.unique(deg, return_counts=True)
    pk = Nk / Nk.sum()
    return DegreeData(k=k.astype(float), Nk=Nk.astype(int), pk=pk.astype(float), kbar=float(k @ pk), node_degree=deg)


def print_degree_distribution(D: DegreeData) -> None:
    """Print a compact degree table."""
    print("\nEmpirical degree distribution used by degree-k models")
    print(f"average degree <k> = {D.kbar:.4f}; number of classes = {len(D.k)}")
    print("   k      N_k      P(k)")
    for k, count, prob in zip(D.k.astype(int), D.Nk, D.pk):
        print(f"{k:4d} {count:8d} {prob:9.4f}")


def save_degree_distribution(D: DegreeData, out_dir: Path) -> None:
    """Save degree distribution as CSV and figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"k": D.k.astype(int), "N_k": D.Nk, "P(k)": D.pk}).to_csv(out_dir / "degree_distribution.csv", index=False)
    plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.gca()
    ax.bar(D.k, D.pk, color="tab:blue", alpha=0.85)
    apply_clean_axes(ax, xlabel="degree k", ylabel="P(k)",
                     title=f"Empirical degree distribution; average degree={D.kbar:.2f}")
    savefig(out_dir / "degree_distribution")


def high_degree_nodes(A: np.ndarray, m: int) -> np.ndarray:
    """Indices of the m highest undirected-degree nodes."""
    degree = np.logical_or(A > 0, A.T > 0).sum(axis=1)
    return np.argsort(-degree)[: min(m, len(degree))]


def degree_initial_state(D: DegreeData) -> np.ndarray:
    """Small degree-dependent infected fraction used by degree-level examples."""
    return clip(0.02 + 0.08 * D.k / max(D.k.max(), 1.0), 0, 0.18)


def node_initial_state(A: np.ndarray) -> np.ndarray:
    """Seed the two highest-degree nodes so node-level trajectories are visible."""
    x0 = np.full(A.shape[0], 0.02)
    x0[high_degree_nodes(A, 2)] = 0.12
    return x0


# ---------------------------------------------------------------------------
# Degree-k optimal control and degree-k game
# ---------------------------------------------------------------------------


def theta_degree(x: np.ndarray, D: DegreeData) -> float:
    """Degree-level infection pressure Theta(t)."""
    return float((D.k * D.pk) @ x / max(D.kbar, 1e-12))


def degree_rhs(x: np.ndarray, u: np.ndarray, D: DegreeData, beta: float, delta: float) -> np.ndarray:
    """Degree-k SIS dynamics with scalar inputs broadcast through shared arrays."""
    model = DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk)
    return degree_sis_rhs(x, u, D.k, model)


def degree_jacobian(x: np.ndarray, u: np.ndarray, D: DegreeData, beta: float, delta: float) -> np.ndarray:
    """Jacobian of degree_rhs with respect to x."""
    model = DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk)
    return degree_sis_jacobian(x, u, D.k, model)


def solve_degree_control(D: DegreeData, steps: int = 45, iterations: int = 40, tol: float = 1e-4) -> TimeSeries:
    """PMP forward-backward sweep for u_k(t)."""
    params = DEGREE_CONTROL_PROFILE
    model = degree_correlated_sis_params(
        D.k,
        D.pk,
        strength=0.28,
        base=DegreeSISParams(
            susceptibility=params.beta,
            recovery=params.delta,
            state_weight=params.state_weight,
            control_weight=params.control_weight,
            control_bound=params.control_max,
        ),
    )
    t = np.linspace(0, params.horizon, steps + 1)
    u = np.zeros((len(t), len(D.k)))
    x0 = degree_initial_state(D)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_u = u.copy()
        u_fun = as_function(t, u)
        x = clip(solve_grid(lambda tau, y: degree_sis_rhs(y, u_fun(tau), D.k, model), x0, t))
        x_fun = as_function(t, x)

        def adjoint(tau, lam):
            x_now, u_now = x_fun(tau), u_fun(tau)
            return -model.state_weight * D.pk - degree_sis_jacobian(x_now, u_now, D.k, model).T @ lam

        lam = solve_grid(adjoint, np.zeros(len(D.k)), t, backward=True)
        u = relax(clip(lam * x / (model.control_weight * np.maximum(D.pk, 1e-12)), 0, model.control_bound), old_u)
        delta_u = float(np.max(abs(u - old_u)))
        delta_history.append(delta_u)
        if delta_u < tol:
            break

    u_fun = as_function(t, u)
    x = clip(solve_grid(lambda tau, y: degree_sis_rhs(y, u_fun(tau), D.k, model), x0, t))
    cost = integral((x * model.state_weight) @ D.pk + 0.5 * ((u * u * model.control_weight) @ D.pk), t)
    return TimeSeries(
        t=t,
        x=x,
        controls={"control": u, "fbs_delta": np.asarray(delta_history, dtype=float)},
        value={"cost": cost, **convergence_values(delta_history, tol)},
    )


def degree_game_rhs(x, attack, defend, D, beta, delta, model=None):
    """Degree-k attacker-defender dynamics."""
    model = DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk) if model is None else model
    pressure = model.mixing @ (model.infectivity * x)
    force = model.susceptibility * (1.0 + attack) * D.k * pressure
    return (1.0 - x) * force - (model.recovery + defend) * x


def degree_game_jacobian(x, attack, defend, D, beta, delta, model=None):
    """Jacobian of degree_game_rhs with respect to x."""
    model = DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk) if model is None else model
    pressure = model.mixing @ (model.infectivity * x)
    B = model.susceptibility[:, None] * (1.0 + attack)[:, None] * D.k[:, None] * model.mixing * model.infectivity[None, :]
    J = (1.0 - x)[:, None] * B
    J[np.diag_indices_from(J)] -= model.susceptibility * (1.0 + attack) * D.k * pressure + model.recovery + defend
    return J


def solve_degree_game(D: DegreeData, steps: int = 45, iterations: int = 45, tol: float = 1e-4) -> TimeSeries:
    """Open-loop Nash forward-backward sweep by degree class k."""
    params = DEGREE_GAME_PROFILE
    beta, delta = params.beta, params.delta
    model = degree_correlated_sis_params(
        D.k,
        D.pk,
        strength=0.22,
        base=DegreeSISParams(
            susceptibility=beta,
            recovery=delta,
            attack_reward=params.reward_attacker,
            defense_loss=params.loss_defender,
            attack_cost=params.cost_attack,
            defense_cost=params.cost_defense,
            attack_bound=params.attack_max,
            defense_bound=params.defense_max,
        ),
    )
    t = np.linspace(0, params.horizon, steps + 1)
    attack = np.zeros((len(t), len(D.k)))
    defend = np.zeros_like(attack)
    x0 = degree_initial_state(D)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_a, old_d = attack.copy(), defend.copy()
        a_fun, d_fun = as_function(t, attack), as_function(t, defend)
        x = clip(solve_grid(lambda tau, y: degree_game_rhs(y, a_fun(tau), d_fun(tau), D, beta, delta, model), x0, t))
        x_fun = as_function(t, x)

        def adjoint_attacker(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = degree_game_jacobian(x_now, a_now, d_now, D, beta, delta, model)
            return -model.attack_reward * D.pk - J.T @ lam

        def adjoint_defender(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = degree_game_jacobian(x_now, a_now, d_now, D, beta, delta, model)
            return model.defense_loss * D.pk - J.T @ lam

        lam_A = solve_grid(adjoint_attacker, np.zeros(len(D.k)), t, backward=True)
        lam_D = solve_grid(adjoint_defender, np.zeros(len(D.k)), t, backward=True)
        pressure = np.vstack([model.mixing @ (model.infectivity * row) for row in x])
        marginal_attack = model.susceptibility[None, :] * D.k[None, :] * (1.0 - x) * pressure
        attack = relax(clip(lam_A * marginal_attack / (model.attack_cost * np.maximum(D.pk, 1e-12)), 0, model.attack_bound), old_a)
        defend = relax(clip(-lam_D * x / (model.defense_cost * np.maximum(D.pk, 1e-12)), 0, model.defense_bound), old_d)
        delta_controls = float(max(np.max(abs(attack - old_a)), np.max(abs(defend - old_d))))
        delta_history.append(delta_controls)
        if delta_controls < tol:
            break

    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(solve_grid(lambda tau, y: degree_game_rhs(y, a_fun(tau), d_fun(tau), D, beta, delta, model), x0, t))
    JA = integral((model.attack_reward * x) @ D.pk - 0.5 * ((attack * attack * model.attack_cost) @ D.pk), t)
    JD = integral(-(model.defense_loss * x) @ D.pk - 0.5 * ((defend * defend * model.defense_cost) @ D.pk), t)
    return TimeSeries(
        t=t,
        x=x,
        controls={"attack": attack, "defense": defend, "fbs_delta": np.asarray(delta_history, dtype=float)},
        value={"JA": JA, "JD": JD, **convergence_values(delta_history, tol)},
    )


# ---------------------------------------------------------------------------
# Node-level control and node-level game
# ---------------------------------------------------------------------------


def node_rhs(x: np.ndarray, u: np.ndarray, A: np.ndarray, beta: float, delta: float) -> np.ndarray:
    """Node-level SIS dynamics."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0])
    return node_sis_rhs(x, u, A, model)


def node_jacobian(x: np.ndarray, u: np.ndarray, A: np.ndarray, beta: float, delta: float) -> np.ndarray:
    """Jacobian of node_rhs with respect to x."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0])
    return node_sis_jacobian(x, u, A, model)


def solve_node_control(A: np.ndarray, steps: int = 30, iterations: int = 35, tol: float = 1e-4) -> TimeSeries:
    """PMP forward-backward sweep for node controls u_i(t)."""
    params = NODE_CONTROL_PROFILE
    model = degree_correlated_node_sis_params(A, strength=0.28)
    t, n = np.linspace(0, params.horizon, steps + 1), A.shape[0]
    u = np.zeros((len(t), n))
    x0 = node_initial_state(A)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_u = u.copy()
        u_fun = as_function(t, u)
        x = clip(solve_grid(lambda tau, y: node_sis_rhs(y, u_fun(tau), A, model), x0, t))
        x_fun = as_function(t, x)

        def adjoint(tau, lam):
            x_now, u_now = x_fun(tau), u_fun(tau)
            return -model.state_weight - node_sis_jacobian(x_now, u_now, A, model).T @ lam

        lam = solve_grid(adjoint, np.zeros(n), t, backward=True)
        u = relax(clip(lam * x / model.control_weight, 0, model.control_bound), old_u)
        delta_u = float(np.max(abs(u - old_u)))
        delta_history.append(delta_u)
        if delta_u < tol:
            break

    u_fun = as_function(t, u)
    x = clip(solve_grid(lambda tau, y: node_sis_rhs(y, u_fun(tau), A, model), x0, t))
    cost = integral((x * model.state_weight).sum(axis=1) + 0.5 * (u * u * model.control_weight).sum(axis=1), t)
    return TimeSeries(
        t=t,
        x=x,
        controls={"control": u, "fbs_delta": np.asarray(delta_history, dtype=float)},
        value={"cost": cost, **convergence_values(delta_history, tol)},
    )


def node_game_rhs(x, attack, defend, A, beta, delta):
    """Node-level attacker-defender dynamics."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0])
    force = (1.0 + attack) * node_sis_force(x, A, model)
    return (1.0 - x) * force - (model.recovery + defend) * x


def node_game_jacobian(x, attack, defend, A, beta, delta):
    """Jacobian of node_game_rhs with respect to x."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0])
    pressure = A @ (model.infectivity * x)
    B = model.beta[:, None] * model.susceptibility[:, None] * (1.0 + attack)[:, None] * A * model.infectivity[None, :]
    J = (1.0 - x)[:, None] * B
    J[np.diag_indices_from(J)] -= model.beta * model.susceptibility * (1.0 + attack) * pressure + model.recovery + defend
    return J


def solve_node_game(A: np.ndarray, steps: int = 30, iterations: int = 40, tol: float = 1e-4) -> TimeSeries:
    """Open-loop Nash forward-backward sweep for node-level attack/defense."""
    params = NODE_GAME_PROFILE
    model = degree_correlated_node_sis_params(A, strength=0.22)
    t, n = np.linspace(0, params.horizon, steps + 1), A.shape[0]
    attack = np.zeros((len(t), n))
    defend = np.zeros_like(attack)
    x0 = node_initial_state(A)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_a, old_d = attack.copy(), defend.copy()
        a_fun, d_fun = as_function(t, attack), as_function(t, defend)
        x = clip(solve_grid(lambda tau, y: (1.0 - y) * (1.0 + a_fun(tau)) * node_sis_force(y, A, model) - (model.recovery + d_fun(tau)) * y, x0, t))
        x_fun = as_function(t, x)

        def adjoint_A(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            pressure = A @ (model.infectivity * x_now)
            B = model.beta[:, None] * model.susceptibility[:, None] * (1.0 + a_now)[:, None] * A * model.infectivity[None, :]
            J = (1.0 - x_now)[:, None] * B
            J[np.diag_indices_from(J)] -= model.beta * model.susceptibility * (1.0 + a_now) * pressure + model.recovery + d_now
            return -model.attack_reward - J.T @ lam

        def adjoint_D(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            pressure = A @ (model.infectivity * x_now)
            B = model.beta[:, None] * model.susceptibility[:, None] * (1.0 + a_now)[:, None] * A * model.infectivity[None, :]
            J = (1.0 - x_now)[:, None] * B
            J[np.diag_indices_from(J)] -= model.beta * model.susceptibility * (1.0 + a_now) * pressure + model.recovery + d_now
            return model.defense_loss - J.T @ lam

        lam_A = solve_grid(adjoint_A, np.zeros(n), t, backward=True)
        lam_D = solve_grid(adjoint_D, np.zeros(n), t, backward=True)
        pressure = np.vstack([A @ (model.infectivity * row) for row in x])
        marginal_attack = model.beta[None, :] * model.susceptibility[None, :] * (1.0 - x) * pressure
        attack = relax(clip(lam_A * marginal_attack / model.attack_cost, 0, model.attack_bound), old_a)
        defend = relax(clip(-lam_D * x / model.defense_cost, 0, model.defense_bound), old_d)
        delta_controls = float(max(np.max(abs(attack - old_a)), np.max(abs(defend - old_d))))
        delta_history.append(delta_controls)
        if delta_controls < tol:
            break

    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(solve_grid(lambda tau, y: (1.0 - y) * (1.0 + a_fun(tau)) * node_sis_force(y, A, model) - (model.recovery + d_fun(tau)) * y, x0, t))
    JA = integral((model.attack_reward * x).sum(axis=1) - 0.5 * (attack * attack * model.attack_cost).sum(axis=1), t)
    JD = integral(-(model.defense_loss * x).sum(axis=1) - 0.5 * (defend * defend * model.defense_cost).sum(axis=1), t)
    return TimeSeries(
        t=t,
        x=x,
        controls={"attack": attack, "defense": defend, "fbs_delta": np.asarray(delta_history, dtype=float)},
        value={"JA": JA, "JD": JD, **convergence_values(delta_history, tol)},
    )


# ---------------------------------------------------------------------------
# Hybrid / impulsive simulation on the network
# ---------------------------------------------------------------------------


def simulate_hybrid_impulse(A: np.ndarray, profile: HybridImpulseProfile = HYBRID_PROFILE) -> TimeSeries:
    """Time-varying continuous node control plus state jumps on high-degree nodes.

    This is a transparent simulation template for hybrid dynamics.  It does not
    solve the full hybrid PMP; it shows how to combine solve_ivp segments with
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


# ---------------------------------------------------------------------------
# Baseline and convergence diagnostics
# ---------------------------------------------------------------------------


def evaluate_degree_control_policy(D: DegreeData, t: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, float]:
    """Evaluate the degree-k control objective for a fixed admissible control."""
    params = DEGREE_CONTROL_PROFILE
    beta, delta = params.beta, params.delta
    q, r = params.state_weight, params.control_weight
    x0 = degree_initial_state(D)
    u_fun = as_function(t, u)
    x = clip(solve_grid(lambda tau, y: degree_rhs(y, u_fun(tau), D, beta, delta), x0, t))
    cost = integral(q * (x @ D.pk) + 0.5 * r * ((u * u) @ D.pk), t)
    return x, cost


def evaluate_node_control_policy(A: np.ndarray, t: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, float]:
    """Evaluate the node-level control objective for a fixed admissible control."""
    params = NODE_CONTROL_PROFILE
    beta, delta = params.beta, params.delta
    q, r = params.state_weight, params.control_weight
    x0 = node_initial_state(A)
    u_fun = as_function(t, u)
    x = clip(solve_grid(lambda tau, y: node_rhs(y, u_fun(tau), A, beta, delta), x0, t))
    cost = integral(q * x.sum(axis=1) + 0.5 * r * (u * u).sum(axis=1), t)
    return x, cost


def evaluate_degree_game_strategy(
    D: DegreeData,
    t: np.ndarray,
    attack: np.ndarray,
    defend: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Evaluate degree-level game payoffs for fixed open-loop strategies."""
    params = DEGREE_GAME_PROFILE
    beta, delta = params.beta, params.delta
    reward_A, loss_D = params.reward_attacker, params.loss_defender
    cost_A, cost_D = params.cost_attack, params.cost_defense
    x0 = degree_initial_state(D)
    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(solve_grid(lambda tau, y: degree_game_rhs(y, a_fun(tau), d_fun(tau), D, beta, delta), x0, t))
    JA = integral(reward_A * (x @ D.pk) - 0.5 * cost_A * ((attack * attack) @ D.pk), t)
    JD = integral(-loss_D * (x @ D.pk) - 0.5 * cost_D * ((defend * defend) @ D.pk), t)
    return x, JA, JD


def evaluate_node_game_strategy(
    A: np.ndarray,
    t: np.ndarray,
    attack: np.ndarray,
    defend: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Evaluate node-level game payoffs for fixed open-loop strategies."""
    params = NODE_GAME_PROFILE
    beta, delta = params.beta, params.delta
    reward_A, loss_D = params.reward_attacker, params.loss_defender
    cost_A, cost_D = params.cost_attack, params.cost_defense
    x0 = node_initial_state(A)
    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(solve_grid(lambda tau, y: node_game_rhs(y, a_fun(tau), d_fun(tau), A, beta, delta), x0, t))
    JA = integral(reward_A * x.sum(axis=1) - 0.5 * cost_A * (attack * attack).sum(axis=1), t)
    JD = integral(-loss_D * x.sum(axis=1) - 0.5 * cost_D * (defend * defend).sum(axis=1), t)
    return x, JA, JD


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
        rows = control_baseline_rows(
            result.controls["control"],
            float(result.value["cost"]),
            lambda u: evaluate_degree_control_policy(D, result.t, u)[1],
            random_upper=DEGREE_CONTROL_PROFILE.control_max,
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
        rows = control_baseline_rows(
            result.controls["control"],
            float(result.value["cost"]),
            lambda u: evaluate_node_control_policy(A, result.t, u)[1],
            random_upper=NODE_CONTROL_PROFILE.control_max,
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
        rows = game_baseline_rows(
            result.controls["attack"],
            result.controls["defense"],
            lambda attack, defense: evaluate_degree_game_strategy(D, result.t, attack, defense)[1:],
            attack_upper=DEGREE_GAME_PROFILE.attack_max,
            defense_upper=DEGREE_GAME_PROFILE.defense_max,
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
        rows = game_baseline_rows(
            result.controls["attack"],
            result.controls["defense"],
            lambda attack, defense: evaluate_node_game_strategy(A, result.t, attack, defense)[1:],
            attack_upper=NODE_GAME_PROFILE.attack_max,
            defense_upper=NODE_GAME_PROFILE.defense_max,
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
        print(f"saved model-specific baseline comparison figures to {out_dir}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


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
    apply_clean_axes(ax, xlabel="time", ylabel="state / control",
                     title=f"Degree-k continuous optimal control, cost={result.value['cost']:.2f}")
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
    apply_clean_axes(ax, xlabel="time", ylabel="state / control",
                     title=f"Degree-k continuous differential game, JA={result.value['JA']:.2f}, JD={result.value['JD']:.2f}")
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
    apply_clean_axes(ax, xlabel="time", ylabel="aggregate value",
                     title=f"Node-level continuous optimal control, cost={result.value['cost']:.2f}")
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
    apply_clean_axes(ax, xlabel="time", ylabel="aggregate value",
                     title=f"Node-level continuous differential game, JA={result.value['JA']:.2f}, JD={result.value['JD']:.2f}")
    ax.legend(frameon=False, fontsize=8)
    savefig(out_dir / "node_game_trajectory")


def plot_hybrid(result: TimeSeries, out_dir: Path) -> None:
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
        ylabel="state",
        title=f"Hybrid control: continuous + impulse on {controlled_count} high-degree nodes",
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
        ylabel="control",
        title=f"continuous range={continuous_min:.2f}-{continuous_max:.2f}; impulse fraction={impulse_fraction:.2f}",
    )
    ax_control.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save_publication_figure(
        fig,
        out_dir / "hybrid_impulse_trajectory.png",
        metadata={
            "model": "hybrid impulse example",
            "control_type": "continuous plus impulse control",
        },
    )
    plt.close(fig)
    print(f"saved {out_dir / 'hybrid_impulse_trajectory.png'}")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------


def write_parameter_summary(out_dir: Path, args: argparse.Namespace, D: DegreeData, A: np.ndarray) -> None:
    """Write a compact parameter map so readers do not need to hunt literals."""
    rows: list[dict[str, object]] = []

    def add(model: str, parameter: str, value: object, meaning: str) -> None:
        rows.append({"model": model, "parameter": parameter, "value": value, "meaning": meaning})

    def add_array(model: str, parameter: str, values: np.ndarray, meaning: str) -> None:
        add(model, parameter, np.array2string(np.asarray(values), precision=4, separator=";"), meaning)

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
    degree_control_model = degree_correlated_sis_params(
        D.k,
        D.pk,
        strength=0.28,
        base=DegreeSISParams(
            susceptibility=DEGREE_CONTROL_PROFILE.beta,
            recovery=DEGREE_CONTROL_PROFILE.delta,
            state_weight=DEGREE_CONTROL_PROFILE.state_weight,
            control_weight=DEGREE_CONTROL_PROFILE.control_weight,
            control_bound=DEGREE_CONTROL_PROFILE.control_max,
        ),
    )
    for name in ("susceptibility", "infectivity", "recovery", "state_weight", "control_bound"):
        add_array(DEGREE_CONTROL_PROFILE.label, f"resolved_{name}_by_degree_class", getattr(degree_control_model, name), "Class-specific heterogeneous array used by the degree-level FBS solve.")

    add(DEGREE_GAME_PROFILE.label, "horizon", DEGREE_GAME_PROFILE.horizon, "Total simulated time.")
    add(DEGREE_GAME_PROFILE.label, "time_grid_steps", args.steps, "Number of intervals in the degree-level game time grid.")
    add(DEGREE_GAME_PROFILE.label, "beta", DEGREE_GAME_PROFILE.beta, "Infection/contact rate under attack and defense.")
    add(DEGREE_GAME_PROFILE.label, "delta", DEGREE_GAME_PROFILE.delta, "Natural recovery/removal rate before defense.")
    add(DEGREE_GAME_PROFILE.label, "attack/defense bounds", f"{DEGREE_GAME_PROFILE.attack_max}/{DEGREE_GAME_PROFILE.defense_max}", "Upper bounds for continuous attack and defense strategies.")
    add(DEGREE_GAME_PROFILE.label, "payoff weights", f"{DEGREE_GAME_PROFILE.reward_attacker}/{DEGREE_GAME_PROFILE.loss_defender}", "Attacker reward and defender loss weights on infection.")

    node_steps = max(24, args.steps // 3)
    add(NODE_CONTROL_PROFILE.label, "horizon", NODE_CONTROL_PROFILE.horizon, "Total simulated time.")
    add(NODE_CONTROL_PROFILE.label, "time_grid_steps", node_steps, "Number of intervals in the node-level FBS time grid.")
    add(NODE_CONTROL_PROFILE.label, "beta", NODE_CONTROL_PROFILE.beta, "Node-level infection/contact rate.")
    add(NODE_CONTROL_PROFILE.label, "delta", NODE_CONTROL_PROFILE.delta, "Node-level natural recovery/removal rate before control.")
    add(NODE_CONTROL_PROFILE.label, "u_max", NODE_CONTROL_PROFILE.control_max, "Upper bound for continuous node control u_i(t).")
    add(NODE_CONTROL_PROFILE.label, "initial infected nodes", 2, "The two highest-degree nodes start with infection 0.12; others start at 0.02.")
    node_control_model = degree_correlated_node_sis_params(A, strength=0.28)
    for name in ("susceptibility", "infectivity", "recovery", "state_weight", "control_bound"):
        add_array(NODE_CONTROL_PROFILE.label, f"resolved_{name}_by_node", getattr(node_control_model, name), "Node-specific heterogeneous array used by the node-level FBS solve.")

    add(NODE_GAME_PROFILE.label, "horizon", NODE_GAME_PROFILE.horizon, "Total simulated time.")
    add(NODE_GAME_PROFILE.label, "time_grid_steps", node_steps, "Number of intervals in the node-level game time grid.")
    add(NODE_GAME_PROFILE.label, "beta", NODE_GAME_PROFILE.beta, "Node-level infection/contact rate under attack and defense.")
    add(NODE_GAME_PROFILE.label, "delta", NODE_GAME_PROFILE.delta, "Node-level natural recovery/removal rate before defense.")
    add(NODE_GAME_PROFILE.label, "attack/defense bounds", f"{NODE_GAME_PROFILE.attack_max}/{NODE_GAME_PROFILE.defense_max}", "Upper bounds for continuous attack and defense strategies.")

    add("hybrid continuous + impulse simulation", "horizon", HYBRID_PROFILE.horizon, "Total simulated time.")
    add("hybrid continuous + impulse simulation", "beta", HYBRID_PROFILE.beta, "Node-level infection/contact rate between impulse events.")
    add("hybrid continuous + impulse simulation", "delta", HYBRID_PROFILE.delta, "Natural recovery/removal rate between impulse events.")
    add("hybrid continuous + impulse simulation", "continuous_control_range", f"{HYBRID_PROFILE.continuous_lower}-{HYBRID_PROFILE.continuous_upper}", "Bounds for the time-varying continuous control level.")
    add("hybrid continuous + impulse simulation", "impulse_times", ", ".join(str(tau) for tau in HYBRID_PROFILE.impulse_times), "Discrete event times where the state is jumped.")
    add("hybrid continuous + impulse simulation", "impulse_fraction", HYBRID_PROFILE.impulse_fraction, "Fraction by which controlled high-degree node states are reduced at each impulse.")
    add("baselines", "random_smooth_controls", RANDOM_BASELINE_COUNT, "Random continuous controls/strategies used in each baseline comparison.")

    pd.DataFrame(rows).to_csv(out_dir / "parameter_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "degree_control.png",
        "degree_game.png",
        "node_control.png",
        "node_game.png",
        "hybrid_impulse.png",
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

    if args.examples in {"all", "hybrid"}:
        plot_hybrid(simulate_hybrid_impulse(net.A), out_dir)

    save_fbs_convergence(results, out_dir)
    save_baseline_comparison(results, D, net.A, out_dir)


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
    p.add_argument("--examples", choices=["all", "degree", "node", "hybrid"], default="all")
    p.add_argument("--steps", type=int, default=45, help="Time grid size for degree-level examples.")
    p.add_argument("--demo-nodes", type=int, default=14)
    p.add_argument("--demo-m", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="example_outputs")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
