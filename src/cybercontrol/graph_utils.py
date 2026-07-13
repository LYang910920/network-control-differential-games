"""Deterministic graph construction and community pooling helpers."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .network_models import contiguous_community_index, normalize_adjacency


def community_graph(
    nodes: int,
    communities: int,
    *,
    mean_degree: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a normalized dense stochastic-block graph and labels."""

    if nodes <= 1:
        raise ValueError("nodes must exceed one")
    labels = contiguous_community_index(nodes, communities)
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((nodes, nodes), dtype=np.float64)
    p_in = min(0.65, mean_degree / max(1, nodes // communities))
    p_out = min(0.12, p_in / 5.0)
    for i in range(nodes):
        for j in range(i + 1, nodes):
            probability = p_in if labels[i] == labels[j] else p_out
            if rng.random() < probability:
                adjacency[i, j] = adjacency[j, i] = 1.0
    for i in range(nodes):
        if adjacency[i].sum() == 0.0:
            j = (i + 1) % nodes
            adjacency[i, j] = adjacency[j, i] = 1.0
    return normalize_adjacency(adjacency), labels


def sparse_scale_free_graph(
    nodes: int,
    *,
    mean_degree: float = 6.0,
    seed: int = 0,
):
    """Return a normalized sparse Barabasi-Albert graph and degree labels."""

    import networkx as nx

    attachment = max(1, min(nodes - 1, int(round(mean_degree / 2.0))))
    graph = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    adjacency = nx.to_scipy_sparse_array(graph, dtype=np.float64, format="csr")
    return normalize_adjacency(sparse.csr_matrix(adjacency))


def community_mean(
    values: np.ndarray, labels: np.ndarray, communities: int | None = None
) -> np.ndarray:
    """Average node values within each integer-labeled community."""

    data = np.asarray(values)
    labels = np.asarray(labels, dtype=int)
    if data.shape[0] != labels.shape[0]:
        raise ValueError("values and labels must share the node axis")
    count = int(labels.max()) + 1 if communities is None else int(communities)
    return np.asarray([data[labels == group].mean(axis=0) for group in range(count)])
