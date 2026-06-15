"""
Single-file extension examples for network optimal control and differential games.

This compact companion script can be read after the simple degree-k example.
It keeps the research-code-style examples in ONE Python file rather than many
modules: graph loading, degree distribution, ODE helpers, degree-k models,
node-level models, hybrid/impulse simulation, plotting, and CLI options.

It uses standard packages for the heavy lifting:
    networkx / pandas   realistic network datasets and adjacency conversion
    scipy.solve_ivp     ODE integration for state and adjoint equations

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
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


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


# ---------------------------------------------------------------------------
# General numerical helpers
# ---------------------------------------------------------------------------


def clip(x, lo=0.0, hi=1.0):
    """Project a scalar/array to [lo, hi]."""
    return np.minimum(np.maximum(x, lo), hi)


def relax(new: np.ndarray, old: np.ndarray, weight: float = 0.35) -> np.ndarray:
    """Damped update for forward-backward sweep iterations."""
    return weight * new + (1.0 - weight) * old


def integral(y: np.ndarray, t: np.ndarray) -> float:
    """Trapezoidal integration with NumPy-version compatibility."""
    return float(np.trapezoid(y, t) if hasattr(np, "trapezoid") else np.trapz(y, t))


def as_function(t: np.ndarray, Y: np.ndarray) -> Callable[[float], np.ndarray]:
    """Vector-valued interpolation Y(tau) on the simulation grid."""
    return interp1d(t, Y, axis=0, bounds_error=False, fill_value="extrapolate")


def solve_grid(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y_boundary: np.ndarray,
    t: np.ndarray,
    *,
    backward: bool = False,
) -> np.ndarray:
    """Solve an ODE on grid t using scipy.integrate.solve_ivp.

    If backward=True, y_boundary is the terminal value y(T).  The returned array
    is always ordered from t[0] to t[-1].
    """
    t_span = (t[-1], t[0]) if backward else (t[0], t[-1])
    t_eval = t[::-1] if backward else t
    sol = solve_ivp(rhs, t_span, y_boundary, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y.T
    return Y[::-1] if backward else Y


def savefig(path: Path) -> None:
    """Save the current figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"saved {path}")


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
    plt.figure(figsize=(6.8, 4.0))
    plt.bar(D.k, D.pk)
    plt.xlabel("degree k")
    plt.ylabel("P(k)")
    plt.title(f"Empirical degree distribution; average degree={D.kbar:.2f}")
    savefig(out_dir / "degree_distribution.png")


def high_degree_nodes(A: np.ndarray, m: int) -> np.ndarray:
    """Indices of the m highest undirected-degree nodes."""
    degree = np.logical_or(A > 0, A.T > 0).sum(axis=1)
    return np.argsort(-degree)[: min(m, len(degree))]


# ---------------------------------------------------------------------------
# Degree-k optimal control and degree-k game
# ---------------------------------------------------------------------------


def theta_degree(x: np.ndarray, D: DegreeData) -> float:
    """Degree-level infection pressure Theta(t)."""
    return float((D.k * D.pk) @ x / max(D.kbar, 1e-12))


def degree_rhs(x: np.ndarray, u: np.ndarray, D: DegreeData, beta: float, delta: float) -> np.ndarray:
    """Degree-k SIS dynamics with healing/control u_k."""
    return beta * D.k * (1 - x) * theta_degree(x, D) - (delta + u) * x


def degree_jacobian(x: np.ndarray, u: np.ndarray, D: DegreeData, beta: float, delta: float) -> np.ndarray:
    """Jacobian of degree_rhs with respect to x."""
    dtheta = D.k * D.pk / max(D.kbar, 1e-12)
    J = beta * np.outer(D.k * (1 - x), dtheta)
    J[np.diag_indices_from(J)] += -beta * D.k * theta_degree(x, D) - (delta + u)
    return J


def solve_degree_control(D: DegreeData, steps: int = 45, iterations: int = 15) -> TimeSeries:
    """PMP forward-backward sweep for u_k(t)."""
    beta, delta, q, r, u_max = 0.65, 0.18, 3.0, 2.5, 1.2
    t = np.linspace(0, 14.0, steps + 1)
    u = np.zeros((len(t), len(D.k)))
    x0 = clip(0.02 + 0.08 * D.k / max(D.k.max(), 1.0), 0, 0.18)

    for _ in range(iterations):
        old_u = u.copy()
        u_fun = as_function(t, u)
        x = clip(solve_grid(lambda tau, y: degree_rhs(y, u_fun(tau), D, beta, delta), x0, t))
        x_fun = as_function(t, x)

        def adjoint(tau, lam):
            x_now, u_now = x_fun(tau), u_fun(tau)
            return -q * D.pk - degree_jacobian(x_now, u_now, D, beta, delta).T @ lam

        lam = solve_grid(adjoint, np.zeros(len(D.k)), t, backward=True)
        u = relax(clip(lam * x / (r * np.maximum(D.pk, 1e-12)), 0, u_max), old_u)
        if np.max(abs(u - old_u)) < 1e-4:
            break

    cost = integral(q * (x @ D.pk) + 0.5 * r * ((u * u) @ D.pk), t)
    return TimeSeries(t=t, x=x, controls={"control": u}, value={"cost": cost})


def degree_game_rhs(x, attack, defend, D, beta, delta):
    """Degree-k attacker-defender dynamics."""
    return beta * (1 + attack) * D.k * (1 - x) * theta_degree(x, D) - (delta + defend) * x


def degree_game_jacobian(x, attack, defend, D, beta, delta):
    """Jacobian of degree_game_rhs with respect to x."""
    dtheta = D.k * D.pk / max(D.kbar, 1e-12)
    J = beta * np.outer((1 + attack) * D.k * (1 - x), dtheta)
    J[np.diag_indices_from(J)] += -beta * (1 + attack) * D.k * theta_degree(x, D) - (delta + defend)
    return J


def solve_degree_game(D: DegreeData, steps: int = 45, iterations: int = 15) -> TimeSeries:
    """Open-loop Nash forward-backward sweep by degree class k."""
    beta, delta = 0.60, 0.15
    reward_A, loss_D, cost_A, cost_D = 4.0, 5.0, 3.0, 4.0
    a_max, d_max = 1.2, 1.2
    t = np.linspace(0, 14.0, steps + 1)
    attack = np.zeros((len(t), len(D.k)))
    defend = np.zeros_like(attack)
    x0 = clip(0.02 + 0.08 * D.k / max(D.k.max(), 1.0), 0, 0.18)

    for _ in range(iterations):
        old_a, old_d = attack.copy(), defend.copy()
        a_fun, d_fun = as_function(t, attack), as_function(t, defend)
        x = clip(solve_grid(lambda tau, y: degree_game_rhs(y, a_fun(tau), d_fun(tau), D, beta, delta), x0, t))
        x_fun = as_function(t, x)

        def adjoint_attacker(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = degree_game_jacobian(x_now, a_now, d_now, D, beta, delta)
            return -reward_A * D.pk - J.T @ lam

        def adjoint_defender(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = degree_game_jacobian(x_now, a_now, d_now, D, beta, delta)
            return loss_D * D.pk - J.T @ lam

        lam_A = solve_grid(adjoint_attacker, np.zeros(len(D.k)), t, backward=True)
        lam_D = solve_grid(adjoint_defender, np.zeros(len(D.k)), t, backward=True)
        theta = np.array([theta_degree(row, D) for row in x])[:, None]
        marginal_attack = beta * D.k[None, :] * (1 - x) * theta
        attack = relax(clip(lam_A * marginal_attack / (cost_A * np.maximum(D.pk, 1e-12)), 0, a_max), old_a)
        defend = relax(clip(-lam_D * x / (cost_D * np.maximum(D.pk, 1e-12)), 0, d_max), old_d)
        if max(np.max(abs(attack - old_a)), np.max(abs(defend - old_d))) < 1e-4:
            break

    JA = integral(reward_A * (x @ D.pk) - 0.5 * cost_A * ((attack * attack) @ D.pk), t)
    JD = integral(-loss_D * (x @ D.pk) - 0.5 * cost_D * ((defend * defend) @ D.pk), t)
    return TimeSeries(t=t, x=x, controls={"attack": attack, "defense": defend}, value={"JA": JA, "JD": JD})


# ---------------------------------------------------------------------------
# Node-level control and node-level game
# ---------------------------------------------------------------------------


def node_rhs(x: np.ndarray, u: np.ndarray, A: np.ndarray, beta: float, delta: float) -> np.ndarray:
    """Node-level SIS dynamics."""
    pressure = A @ x
    return beta * (1 - x) * pressure - (delta + u) * x


def node_jacobian(x: np.ndarray, u: np.ndarray, A: np.ndarray, beta: float, delta: float) -> np.ndarray:
    """Jacobian of node_rhs with respect to x."""
    pressure = A @ x
    J = beta * ((1 - x)[:, None] * A)
    J[np.diag_indices_from(J)] += -beta * pressure - (delta + u)
    return J


def solve_node_control(A: np.ndarray, steps: int = 30, iterations: int = 8) -> TimeSeries:
    """PMP forward-backward sweep for node controls u_i(t)."""
    beta, delta, q, r, u_max = 0.90, 0.16, 3.0, 2.2, 1.2
    t, n = np.linspace(0, 12.0, steps + 1), A.shape[0]
    u = np.zeros((len(t), n))
    x0 = np.full(n, 0.02)
    x0[high_degree_nodes(A, 2)] = 0.12

    for _ in range(iterations):
        old_u = u.copy()
        u_fun = as_function(t, u)
        x = clip(solve_grid(lambda tau, y: node_rhs(y, u_fun(tau), A, beta, delta), x0, t))
        x_fun = as_function(t, x)

        def adjoint(tau, lam):
            x_now, u_now = x_fun(tau), u_fun(tau)
            return -q * np.ones(n) - node_jacobian(x_now, u_now, A, beta, delta).T @ lam

        lam = solve_grid(adjoint, np.zeros(n), t, backward=True)
        u = relax(clip(lam * x / r, 0, u_max), old_u)
        if np.max(abs(u - old_u)) < 1e-4:
            break

    cost = integral(q * x.sum(axis=1) + 0.5 * r * (u * u).sum(axis=1), t)
    return TimeSeries(t=t, x=x, controls={"control": u}, value={"cost": cost})


def node_game_rhs(x, attack, defend, A, beta, delta):
    """Node-level attacker-defender dynamics."""
    pressure = A @ x
    return beta * (1 + attack) * (1 - x) * pressure - (delta + defend) * x


def node_game_jacobian(x, attack, defend, A, beta, delta):
    """Jacobian of node_game_rhs with respect to x."""
    pressure = A @ x
    J = beta * ((1 + attack) * (1 - x))[:, None] * A
    J[np.diag_indices_from(J)] += -beta * (1 + attack) * pressure - (delta + defend)
    return J


def solve_node_game(A: np.ndarray, steps: int = 30, iterations: int = 6) -> TimeSeries:
    """Open-loop Nash forward-backward sweep for node-level attack/defense."""
    beta, delta = 0.95, 0.15
    reward_A, loss_D, cost_A, cost_D = 4.0, 5.0, 4.0, 4.5
    a_max, d_max = 1.2, 1.2
    t, n = np.linspace(0, 12.0, steps + 1), A.shape[0]
    attack = np.zeros((len(t), n))
    defend = np.zeros_like(attack)
    x0 = np.full(n, 0.02)
    x0[high_degree_nodes(A, 2)] = 0.12

    for _ in range(iterations):
        old_a, old_d = attack.copy(), defend.copy()
        a_fun, d_fun = as_function(t, attack), as_function(t, defend)
        x = clip(solve_grid(lambda tau, y: node_game_rhs(y, a_fun(tau), d_fun(tau), A, beta, delta), x0, t))
        x_fun = as_function(t, x)

        def adjoint_A(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            return -reward_A * np.ones(n) - node_game_jacobian(x_now, a_now, d_now, A, beta, delta).T @ lam

        def adjoint_D(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            return loss_D * np.ones(n) - node_game_jacobian(x_now, a_now, d_now, A, beta, delta).T @ lam

        lam_A = solve_grid(adjoint_A, np.zeros(n), t, backward=True)
        lam_D = solve_grid(adjoint_D, np.zeros(n), t, backward=True)
        pressure = np.vstack([A @ row for row in x])
        attack = relax(clip(lam_A * beta * (1 - x) * pressure / cost_A, 0, a_max), old_a)
        defend = relax(clip(-lam_D * x / cost_D, 0, d_max), old_d)
        if max(np.max(abs(attack - old_a)), np.max(abs(defend - old_d))) < 1e-4:
            break

    JA = integral(reward_A * x.sum(axis=1) - 0.5 * cost_A * (attack * attack).sum(axis=1), t)
    JD = integral(-loss_D * x.sum(axis=1) - 0.5 * cost_D * (defend * defend).sum(axis=1), t)
    return TimeSeries(t=t, x=x, controls={"attack": attack, "defense": defend}, value={"JA": JA, "JD": JD})


# ---------------------------------------------------------------------------
# Hybrid / impulsive simulation on the network
# ---------------------------------------------------------------------------


def simulate_hybrid_impulse(A: np.ndarray, T=12.0, impulse_times=(3.0, 6.0, 9.0)) -> TimeSeries:
    """Continuous node control plus state jumps on high-degree nodes.

    This is a transparent simulation template for hybrid dynamics.  It does not
    solve the full hybrid PMP; it shows how to combine solve_ivp segments with
    impulse maps x(tau+) = G(x(tau-), z_tau).
    """
    beta, delta, continuous_u, impulse_fraction = 0.95, 0.15, 0.35, 0.55
    n = A.shape[0]
    controlled = high_degree_nodes(A, max(1, n // 4))
    u = np.zeros(n)
    u[controlled] = continuous_u
    x = np.full(n, 0.03)
    x[high_degree_nodes(A, 2)] = 0.15

    all_t, all_x = [], []
    for t0, t1 in zip([0.0, *impulse_times], [*impulse_times, T]):
        grid = np.linspace(t0, t1, max(2, int(30 * (t1 - t0)) + 1))
        segment = clip(solve_grid(lambda _, y: node_rhs(y, u, A, beta, delta), x, grid))
        all_t.extend(grid[:-1])
        all_x.extend(segment[:-1])
        x = segment[-1].copy()
        if t1 in impulse_times:
            x[controlled] *= 1.0 - impulse_fraction

    all_t.append(T)
    all_x.append(x)
    return TimeSeries(t=np.asarray(all_t), x=np.vstack(all_x), controls={"controlled_nodes": controlled}, value={})


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_degree_control(result: TimeSeries, D: DegreeData, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4.1))
    plt.plot(result.t, result.x @ D.pk, label="mean infection")
    plt.plot(result.t, result.controls["control"] @ D.pk, label="mean control")
    for j in sorted(set([0, len(D.k) // 2, len(D.k) - 1])):
        plt.plot(result.t, result.x[:, j], "--", label=f"x_k, k={int(D.k[j])}")
    plt.xlabel("time")
    plt.ylabel("state / control")
    plt.title(f"Degree-k optimal control; cost={result.value['cost']:.2f}")
    plt.legend()
    savefig(out_dir / "degree_control.png")


def plot_degree_game(result: TimeSeries, D: DegreeData, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4.1))
    plt.plot(result.t, result.x @ D.pk, label="mean infection")
    plt.plot(result.t, result.controls["attack"] @ D.pk, label="mean attack")
    plt.plot(result.t, result.controls["defense"] @ D.pk, label="mean defense")
    high = int(np.argmax(D.k))
    plt.plot(result.t, result.x[:, high], "--", label=f"x_k, high k={int(D.k[high])}")
    plt.xlabel("time")
    plt.ylabel("state / control")
    plt.title(f"Degree-k differential game; JA={result.value['JA']:.2f}, JD={result.value['JD']:.2f}")
    plt.legend()
    savefig(out_dir / "degree_game.png")


def plot_node_control(result: TimeSeries, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4.1))
    plt.plot(result.t, result.x.mean(axis=1), label="mean infection")
    plt.plot(result.t, result.x.max(axis=1), label="max infection")
    plt.plot(result.t, result.controls["control"].mean(axis=1), label="mean control")
    plt.xlabel("time")
    plt.ylabel("aggregate value")
    plt.title(f"Node-level optimal control; cost={result.value['cost']:.2f}")
    plt.legend()
    savefig(out_dir / "node_control.png")


def plot_node_game(result: TimeSeries, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4.1))
    plt.plot(result.t, result.x.mean(axis=1), label="mean infection")
    plt.plot(result.t, result.controls["attack"].mean(axis=1), label="mean attack")
    plt.plot(result.t, result.controls["defense"].mean(axis=1), label="mean defense")
    plt.xlabel("time")
    plt.ylabel("aggregate value")
    plt.title(f"Node-level game; JA={result.value['JA']:.2f}, JD={result.value['JD']:.2f}")
    plt.legend()
    savefig(out_dir / "node_game.png")


def plot_hybrid(result: TimeSeries, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4.1))
    plt.plot(result.t, result.x.mean(axis=1), label="mean infection")
    plt.plot(result.t, result.x.max(axis=1), label="max infection")
    for tau in (3.0, 6.0, 9.0):
        plt.axvline(tau, linestyle="--", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("state")
    plt.title(f"Hybrid impulse control on {len(result.controls['controlled_nodes'])} high-degree nodes")
    plt.legend()
    savefig(out_dir / "hybrid_impulse.png")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    net = load_network(args)
    D = degree_distribution_from_graph(net.full_graph, mode=args.degree_mode)

    print(f"Loaded network: {net.source}")
    print(f"full input nodes={net.full_graph.number_of_nodes()}, edges={net.full_graph.number_of_edges()}, directed={net.full_graph.is_directed()}")
    print(f"node-level matrix size={net.A.shape[0]} x {net.A.shape[1]} (after --max-nodes reduction if used)")
    print_degree_distribution(D)
    save_degree_distribution(D, out_dir)

    if args.examples in {"all", "degree"}:
        plot_degree_control(solve_degree_control(D, steps=args.steps), D, out_dir)
        plot_degree_game(solve_degree_game(D, steps=args.steps), D, out_dir)

    if args.examples in {"all", "node"}:
        node_steps = max(24, args.steps // 3)
        plot_node_control(solve_node_control(net.A, steps=node_steps), out_dir)
        plot_node_game(solve_node_game(net.A, steps=node_steps), out_dir)

    if args.examples in {"all", "hybrid"}:
        plot_hybrid(simulate_hybrid_impulse(net.A), out_dir)


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
