"""Read a network once; derive degree classes or a node influence matrix as needed."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .numerics import project_box as clip

LOGGER = logging.getLogger(__name__)


@dataclass
class NetworkData:
    """Full graph for degree statistics, with an optional reduced node-level graph."""

    full_graph: nx.Graph
    graph: nx.Graph
    A: np.ndarray
    labels: list[str]
    source: str


@dataclass
class DegreeData:
    """Columns are observed degree classes k, including k=0 when isolates exist."""

    k: np.ndarray
    Nk: np.ndarray
    pk: np.ndarray
    kbar: float
    node_degree: np.ndarray


def read_edge_table(args: argparse.Namespace) -> pd.DataFrame:
    """Read node identifiers as text so labels such as 001 remain intact."""
    return pd.read_csv(
        args.edge_list,
        sep=r"\s+" if args.delimiter is None else args.delimiter,
        header=0 if args.has_header else None,
        comment="#",
        dtype=str,
        engine="python",
    )


def resolve_column(df: pd.DataFrame, ref) -> str | int:
    """Accept a column name or a zero-based column position."""
    if ref in df.columns:
        return ref
    try:
        return df.columns[int(ref)]
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"Column {ref!r} is not in {list(df.columns)!r}") from exc


def _edge_records(args: argparse.Namespace) -> list[tuple[str, str, float]]:
    table = read_edge_table(args)
    source = resolve_column(table, args.source_col)
    target = resolve_column(table, args.target_col)
    weights = (
        np.ones(len(table))
        if args.weight_col is None
        else pd.to_numeric(table[resolve_column(table, args.weight_col)], errors="raise").to_numpy()
    )
    if table[[source, target]].isna().any().any():
        raise ValueError("Every edge needs a source and a target.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Edge weights must be finite and nonnegative.")
    return list(zip(table[source], table[target], weights))


def edge_list_to_graph_networkx(args: argparse.Namespace) -> nx.Graph:
    """Sum repeated edges, including reversed rows of an undirected edge."""
    graph = nx.DiGraph() if args.directed else nx.Graph()
    for source, target, weight in _edge_records(args):
        graph.add_nodes_from((source, target))
        if source != target and weight > 0:
            previous = graph.get_edge_data(source, target, {}).get("weight", 0.0)
            graph.add_edge(source, target, weight=previous + float(weight))
    return graph


def edge_list_to_graph_igraph(args: argparse.Namespace) -> nx.Graph:
    """Use the same edge semantics with igraph's built-in simplification."""
    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError('Install the optional reader with: pip install ".[graph]"') from exc
    records = _edge_records(args)
    graph = ig.Graph.TupleList(records, directed=args.directed, weights=True)
    graph.simplify(multiple=True, loops=True, combine_edges={"weight": "sum"})
    result = nx.DiGraph() if args.directed else nx.Graph()
    result.add_nodes_from(graph.vs["name"] if graph.vcount() else [])
    for edge in graph.es:
        if edge["weight"] > 0:
            result.add_edge(
                graph.vs[edge.source]["name"],
                graph.vs[edge.target]["name"],
                weight=float(edge["weight"]),
            )
    return result


def graph_file_to_graph(path: Path, directed: bool) -> nx.Graph:
    """Read GraphML, GEXF, GML or Pajek, summing parallel edges."""
    readers = {
        ".graphml": nx.read_graphml,
        ".gexf": nx.read_gexf,
        ".gml": nx.read_gml,
        ".net": nx.read_pajek,
        ".pajek": nx.read_pajek,
    }
    if path.suffix.lower() not in readers:
        raise ValueError("Use .graphml, .gexf, .gml, .net or .pajek.")
    original = readers[path.suffix.lower()](path)
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(original.nodes())
    for source, target, data in original.edges(data=True):
        weight = float(data.get("weight", 1.0))
        previous = graph.get_edge_data(source, target, {}).get("weight", 0.0)
        graph.add_edge(source, target, weight=previous + weight)
        if directed and not original.is_directed() and source != target:
            graph.add_edge(target, source, weight=previous + weight)
    return graph


def keep_top_degree_nodes(G: nx.Graph, max_nodes: int) -> nx.Graph:
    """Choose a teaching subgraph; its dynamics differ from those of the full graph."""
    if max_nodes <= 0 or G.number_of_nodes() <= max_nodes:
        return G.copy()
    degree = dict(G.degree(weight="weight"))
    keep = sorted(degree, key=degree.get, reverse=True)[:max_nodes]
    return G.subgraph(keep).copy()


def normalize_influence_matrix(A: np.ndarray, method: str) -> np.ndarray:
    """Apply an explicit model scaling; normalization changes transmission rates."""
    A = np.asarray(A, dtype=float).copy()
    if method == "none":
        return A
    if method == "row":
        totals = A.sum(axis=1, keepdims=True)
        return np.divide(A, totals, out=np.zeros_like(A), where=totals > 0)
    if method == "max-degree":
        scale = A.sum(axis=1).max(initial=0.0)
    elif method == "spectral":
        scale = float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0
    else:
        raise ValueError("normalization must be: none, max-degree, row, spectral")
    return A / scale if scale > 0 else A


def prepare_model_matrix(A: np.ndarray, normalize: str) -> np.ndarray:
    """Validate A[i,j] = influence from j to i, then remove self-influence."""
    A = np.asarray(A, dtype=float).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if not np.isfinite(A).all() or (A < 0).any():
        raise ValueError("Adjacency entries must be finite and nonnegative.")
    np.fill_diagonal(A, 0.0)
    return normalize_influence_matrix(A, normalize)


def graph_to_model_matrix(
    G: nx.Graph, directed: bool, normalize: str
) -> tuple[np.ndarray, list[str]]:
    """Use a fixed node order and transpose NetworkX's source-target convention."""
    nodes = list(G.nodes())
    matrix = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=float)
    matrix = matrix.T if directed else np.maximum(matrix, matrix.T)
    return prepare_model_matrix(matrix, normalize), [str(node) for node in nodes]


def load_graph(args: argparse.Namespace) -> tuple[nx.Graph, str]:
    """Load the full graph without constructing an unnecessary dense matrix."""
    if args.adjacency_csv:
        matrix = pd.read_csv(args.adjacency_csv, header=None).to_numpy(float)
        matrix = prepare_model_matrix(matrix, "none")
        if not args.directed and not np.allclose(matrix, matrix.T):
            raise ValueError("An undirected adjacency matrix must be symmetric; use --directed.")
        if args.directed and args.adjacency_convention == "model":
            matrix = matrix.T
        graph = nx.from_numpy_array(
            matrix, create_using=nx.DiGraph() if args.directed else nx.Graph()
        )
        source = args.adjacency_csv
    elif args.edge_list:
        reader = (
            edge_list_to_graph_igraph
            if args.graph_backend == "igraph"
            else edge_list_to_graph_networkx
        )
        graph, source = reader(args), args.edge_list
    elif args.graph_file:
        graph = graph_file_to_graph(Path(args.graph_file), args.directed)
        source = args.graph_file
    else:
        graph = nx.barabasi_albert_graph(args.demo_nodes, args.demo_m, seed=args.seed)
        source = f"Barabasi-Albert graph ({args.demo_nodes} nodes, seed {args.seed})"
    for _, _, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Edge weights must be finite and nonnegative.")
        data["weight"] = weight
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d["weight"] == 0])
    if graph.number_of_nodes() == 0:
        raise ValueError("The input graph has no nodes.")
    return graph, str(source)


def load_network(args: argparse.Namespace) -> NetworkData:
    """Retain the full graph; reduce only the node-level matrix when requested."""
    full_graph, source = load_graph(args)
    graph = keep_top_degree_nodes(full_graph, args.max_nodes)
    matrix, labels = graph_to_model_matrix(graph, args.directed, args.normalize)
    return NetworkData(full_graph, graph, matrix, labels, source)


def degree_distribution_from_graph(G: nx.Graph, mode: str = "undirected") -> DegreeData:
    """Compute N_k, P(k), and mean degree from distinct neighbors, not edge strength."""
    if mode not in {"undirected", "in", "out", "total"}:
        raise ValueError("degree mode must be: in, out, total, undirected")
    if not G.number_of_nodes():
        raise ValueError("A degree distribution needs at least one node.")
    if not G.is_directed() or mode == "undirected":
        view = nx.Graph(G).degree()
    elif mode == "in":
        view = G.in_degree()
    elif mode == "out":
        view = G.out_degree()
    else:
        view = G.degree()
    degrees = np.fromiter((value for _, value in view), dtype=int)
    k, counts = np.unique(degrees, return_counts=True)
    probability = counts / counts.sum()
    return DegreeData(k.astype(float), counts, probability, float(k @ probability), degrees)


def print_degree_distribution(D: DegreeData) -> None:
    LOGGER.info("Degree classes: %d; mean degree: %.4g", len(D.k), D.kbar)
    for k, count, probability in zip(D.k.astype(int), D.Nk, D.pk):
        LOGGER.info("k=%d, N_k=%d, P(k)=%.4g", k, count, probability)


def high_degree_nodes(A: np.ndarray, m: int) -> np.ndarray:
    degree = np.logical_or(A > 0, A.T > 0).sum(axis=1)
    return np.argsort(-degree)[: min(m, len(degree))]


def degree_initial_state(D: DegreeData) -> np.ndarray:
    return clip(0.02 + 0.08 * D.k / max(D.k.max(), 1.0), 0, 0.18)


def node_initial_state(A: np.ndarray) -> np.ndarray:
    initial = np.full(A.shape[0], 0.02)
    initial[high_degree_nodes(A, 2)] = 0.12
    return initial


normalize_adjacency = normalize_influence_matrix
