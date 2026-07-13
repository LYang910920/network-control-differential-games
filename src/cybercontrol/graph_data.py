"""Graph input, influence-matrix, and degree-distribution data structures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .numerics import project_box as clip

LOGGER = logging.getLogger(__name__)


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

    k: np.ndarray  # observed degree classes
    Nk: np.ndarray  # number of nodes with degree k
    pk: np.ndarray  # P(k) = Nk / n
    kbar: float  # average degree <k>
    node_degree: np.ndarray


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


def normalize_influence_matrix(A: np.ndarray, method: str) -> np.ndarray:
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
    return normalize_influence_matrix(A, normalize)


def graph_to_model_matrix(
    G: nx.Graph, directed: bool, normalize: str
) -> tuple[np.ndarray, list[str]]:
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
        clean = prepare_model_matrix(M, "none")
        if args.adjacency_convention == "model":
            full_G = nx.from_numpy_array(
                clean.T if args.directed else clean,
                create_using=nx.DiGraph() if args.directed else nx.Graph(),
            )
        else:  # ordinary graph convention: M[source, target]
            full_G = nx.from_numpy_array(
                clean,
                create_using=nx.DiGraph() if args.directed else nx.Graph(),
            )
        source = args.adjacency_csv
    elif args.edge_list:
        full_G = (
            edge_list_to_graph_igraph(args)
            if args.graph_backend == "igraph"
            else edge_list_to_graph_networkx(args)
        )
        source = args.edge_list
    elif args.graph_file:
        full_G = graph_file_to_graph(Path(args.graph_file), args.directed)
        nx.set_edge_attributes(
            full_G, {e: full_G.edges[e].get("weight", 1.0) for e in full_G.edges}, "weight"
        )
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


def degree_distribution_from_graph(G: nx.Graph, mode: str = "undirected") -> DegreeData:
    """Compute N_k, P(k), and <k> from the FULL input graph.

    For undirected/contact networks, use mode="undirected".  For directed
    influence networks, choose in, out, or total depending on the modelling role
    of degree.
    """
    if (not G.is_directed()) or mode == "undirected":
        deg = np.array([d for _, d in nx.Graph(G).degree()], dtype=int)
    elif mode == "in":
        deg = np.array([d for _, d in G.in_degree()], dtype=int)  # type: ignore[attr-defined]
    elif mode == "out":
        deg = np.array([d for _, d in G.out_degree()], dtype=int)  # type: ignore[attr-defined]
    elif mode == "total":
        deg = np.array([G.in_degree(v) + G.out_degree(v) for v in G.nodes()], dtype=int)  # type: ignore[attr-defined]
    else:
        raise ValueError("degree mode must be: in, out, total, undirected")

    k, Nk = np.unique(deg, return_counts=True)
    pk = Nk / Nk.sum()
    return DegreeData(
        k=k.astype(float),
        Nk=Nk.astype(int),
        pk=pk.astype(float),
        kbar=float(k @ pk),
        node_degree=deg,
    )


def print_degree_distribution(D: DegreeData) -> None:
    """Print a compact degree table."""
    LOGGER.info("\nEmpirical degree distribution used by degree-k models")
    LOGGER.info(f"average degree <k> = {D.kbar:.4f}; number of classes = {len(D.k)}")
    LOGGER.info("   k      N_k      P(k)")
    for k, count, prob in zip(D.k.astype(int), D.Nk, D.pk):
        LOGGER.info(f"{k:4d} {count:8d} {prob:9.4f}")


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


# Compatibility name retained for existing notebooks and scalability code.
normalize_adjacency = normalize_influence_matrix
