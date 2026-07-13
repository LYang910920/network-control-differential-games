"""Scale-free graph adapters used by scalability runners."""

from __future__ import annotations

import random

import igraph as ig
import networkx as nx
import numpy as np
import scipy.sparse as sp

from .graph_data import DegreeData


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
    return degree_distribution_from_degree(degree)


def degree_distribution_from_degree(degree: np.ndarray) -> DegreeData:
    """Reduce a node-degree vector to the degree-k state used by the FBS solver."""
    k, Nk = np.unique(degree, return_counts=True)
    pk = Nk / Nk.sum()
    return DegreeData(
        k=k.astype(float),
        Nk=Nk.astype(int),
        pk=pk.astype(float),
        kbar=float(k @ pk),
        node_degree=degree,
    )


def igraph_to_networkx(graph: ig.Graph) -> nx.Graph:
    """Use igraph's converter only for the optional node-level matrix workflow."""
    return nx.Graph(graph.to_networkx())


def igraph_to_sparse_adjacency(graph: ig.Graph) -> sp.csr_matrix:
    """Sparse undirected adjacency matrix for paired degree/node FBS comparison."""
    n = graph.vcount()
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64)
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
    return A


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
