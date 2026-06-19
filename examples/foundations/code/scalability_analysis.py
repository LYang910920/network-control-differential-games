# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Scalability analysis for forward-backward sweep optimal control.

The default experiment uses synthetic Barabasi-Albert scale-free networks with
100, 1000, 10000, 100000, and 1000000 nodes. For each graph seed, it runs both
degree-level FBS and node-level FBS on the same normalized SIS epidemic-control
problem, records runtime/convergence diagnostics, and writes one paired
comparison plot.

Degree-level FBS has one state/control/costate per observed degree class.
Node-level FBS has one state/control/costate per graph node. The paired
node-level solver uses sparse matrix products so the same tutorial comparison
can reach million-node synthetic graphs without materializing dense adjacency
matrices.

The graph work is delegated to libraries: igraph generates the
scale-free networks and degree vectors, pandas aggregates timing summaries, and
Matplotlib renders the final diagnostic plot. The control solver receives only
the reduced mathematical objects it actually needs.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import igraph as ig
import matplotlib
from matplotlib.ticker import FixedLocator, FuncFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


CODE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CODE_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from network_control_examples import (  # noqa: E402
    DegreeData,
    graph_to_model_matrix,
    solve_degree_control,
    solve_node_control,
)


DEFAULT_COMPARE_SIZES = (100, 1000, 10000, 100000, 1000000)
DEFAULT_DEGREE_SIZES = tuple(range(100, 2001, 100))
DEFAULT_NODE_SIZES = tuple(range(1000, 10001, 1000))


COMPARISON_DEFAULTS = {
    "horizon": 8.0,
    "beta": 1.20,
    "delta": 0.35,
    "state_weight": 1.0,
    "control_weight": 3.0,
    "control_max": 1.0,
    "damping": 0.35,
}


NODE_SCALABILITY_DEFAULTS = {
    "horizon": 8.0,
    "beta": 0.55,
    "delta": 0.20,
    "state_weight": 1.0,
    "control_weight": 8.0,
    "control_max": 0.8,
    "damping": 0.25,
}


def parse_sizes(raw: str) -> list[int]:
    sizes = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be a comma-separated list of positive integers")
    return sizes


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
    return DegreeData(k=k.astype(float), Nk=Nk.astype(int), pk=pk.astype(float), kbar=float(k @ pk), node_degree=degree)


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


def comparison_node_initial_state(degree: np.ndarray) -> np.ndarray:
    """Use the same degree-based initial infection rule as the degree model."""
    max_degree = max(float(degree.max(initial=1)), 1.0)
    return np.clip(0.02 + 0.08 * degree / max_degree, 0.0, 0.18)


def comparison_degree_initial_state(D: DegreeData) -> np.ndarray:
    """Initial infected fraction for each degree class in the paired comparison."""
    max_degree = max(float(D.k.max(initial=1.0)), 1.0)
    return np.clip(0.02 + 0.08 * D.k / max_degree, 0.0, 0.18)


def comparison_degree_theta(x: np.ndarray, D: DegreeData) -> float:
    return float((D.k * D.pk) @ x / max(D.kbar, 1e-12))


def comparison_degree_scale(D: DegreeData) -> float:
    """Use the graph's maximum degree to keep large-SF contact pressure stable."""
    return max(float(D.k.max(initial=1.0)), 1.0)


def comparison_degree_rhs(x: np.ndarray, u: np.ndarray, D: DegreeData) -> np.ndarray:
    """Normalized degree-level SIS dynamics used only for paired scalability."""
    params = COMPARISON_DEFAULTS
    theta = comparison_degree_theta(x, D)
    contact = (D.k / comparison_degree_scale(D)) * theta
    return params["beta"] * (1.0 - x) * contact - (params["delta"] + u) * x


def comparison_degree_jacobian(x: np.ndarray, u: np.ndarray, D: DegreeData) -> np.ndarray:
    """Jacobian of comparison_degree_rhs with respect to degree-class state."""
    params = COMPARISON_DEFAULTS
    kbar = max(D.kbar, 1e-12)
    dtheta = D.k * D.pk / kbar
    contact_scale = D.k / comparison_degree_scale(D)
    theta = comparison_degree_theta(x, D)
    J = params["beta"] * np.outer(contact_scale * (1.0 - x), dtheta)
    J[np.diag_indices_from(J)] += -params["beta"] * contact_scale * theta - (params["delta"] + u)
    return J


def comparison_node_rhs(x: np.ndarray, u: np.ndarray, A: sp.csr_matrix, degree_scale: float) -> np.ndarray:
    """Normalized node-level SIS dynamics matched to comparison_degree_rhs."""
    params = COMPARISON_DEFAULTS
    pressure = (A @ x) / max(degree_scale, 1e-12)
    return params["beta"] * (1.0 - x) * pressure - (params["delta"] + u) * x


def comparison_node_adjoint_rhs(
    mu: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    pressure: np.ndarray,
    A: sp.csr_matrix,
    degree_scale: float,
) -> np.ndarray:
    """Scaled node-level costate RHS for the averaged node objective."""
    params = COMPARISON_DEFAULTS
    offdiag = params["beta"] * (A.T @ ((1.0 - x) * mu)) / max(degree_scale, 1e-12)
    diag = (-params["beta"] * pressure - params["delta"] - u) * mu
    return -params["state_weight"] - offdiag - diag


def integrate_comparison_degree_state(D: DegreeData, t: np.ndarray, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Forward RK4 integration for the paired degree-level comparison."""
    x = np.empty((len(t), len(D.k)), dtype=np.float64)
    x[0] = x0
    for idx in range(len(t) - 1):
        h = float(t[idx + 1] - t[idx])
        u0, u1 = u[idx], u[idx + 1]
        umid = 0.5 * (u0 + u1)
        y0 = x[idx]
        k1 = comparison_degree_rhs(y0, u0, D)
        k2 = comparison_degree_rhs(np.clip(y0 + 0.5 * h * k1, 0.0, 1.0), umid, D)
        k3 = comparison_degree_rhs(np.clip(y0 + 0.5 * h * k2, 0.0, 1.0), umid, D)
        k4 = comparison_degree_rhs(np.clip(y0 + h * k3, 0.0, 1.0), u1, D)
        x[idx + 1] = np.clip(y0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0, 1.0)
    return x


def integrate_comparison_degree_adjoint(D: DegreeData, t: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Backward RK4 integration for the paired degree-level costate."""
    params = COMPARISON_DEFAULTS
    lam = np.zeros_like(x)
    for idx in range(len(t) - 2, -1, -1):
        h = -float(t[idx + 1] - t[idx])
        l1 = lam[idx + 1]
        x1, x0 = x[idx + 1], x[idx]
        u1, u0 = u[idx + 1], u[idx]
        xmid, umid = 0.5 * (x0 + x1), 0.5 * (u0 + u1)

        def rhs(lam_now: np.ndarray, x_now: np.ndarray, u_now: np.ndarray) -> np.ndarray:
            return -params["state_weight"] * D.pk - comparison_degree_jacobian(x_now, u_now, D).T @ lam_now

        k1 = rhs(l1, x1, u1)
        k2 = rhs(l1 + 0.5 * h * k1, xmid, umid)
        k3 = rhs(l1 + 0.5 * h * k2, xmid, umid)
        k4 = rhs(l1 + h * k3, x0, u0)
        lam[idx] = l1 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return lam


def integrate_comparison_node_state(
    A: sp.csr_matrix,
    degree_scale: float,
    t: np.ndarray,
    x0: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward RK4 integration for the paired sparse node-level comparison."""
    x = np.empty((len(t), A.shape[0]), dtype=np.float64)
    pressure = np.empty_like(x)
    x[0] = x0
    pressure[0] = (A @ x0) / max(degree_scale, 1e-12)
    for idx in range(len(t) - 1):
        h = float(t[idx + 1] - t[idx])
        u0, u1 = u[idx], u[idx + 1]
        umid = 0.5 * (u0 + u1)
        y0 = x[idx]
        k1 = comparison_node_rhs(y0, u0, A, degree_scale)
        k2 = comparison_node_rhs(np.clip(y0 + 0.5 * h * k1, 0.0, 1.0), umid, A, degree_scale)
        k3 = comparison_node_rhs(np.clip(y0 + 0.5 * h * k2, 0.0, 1.0), umid, A, degree_scale)
        k4 = comparison_node_rhs(np.clip(y0 + h * k3, 0.0, 1.0), u1, A, degree_scale)
        x[idx + 1] = np.clip(y0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0, 1.0)
        pressure[idx + 1] = (A @ x[idx + 1]) / max(degree_scale, 1e-12)
    return x, pressure


def integrate_comparison_node_adjoint(
    A: sp.csr_matrix,
    degree_scale: float,
    t: np.ndarray,
    x: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Backward RK4 integration for the paired sparse node-level costate."""
    mu = np.zeros_like(x)
    for idx in range(len(t) - 2, -1, -1):
        h = -float(t[idx + 1] - t[idx])
        m1 = mu[idx + 1]
        x1, x0 = x[idx + 1], x[idx]
        p1, p0 = pressure[idx + 1], pressure[idx]
        u1, u0 = u[idx + 1], u[idx]
        xmid, pmid, umid = 0.5 * (x0 + x1), 0.5 * (p0 + p1), 0.5 * (u0 + u1)
        k1 = comparison_node_adjoint_rhs(m1, x1, u1, p1, A, degree_scale)
        k2 = comparison_node_adjoint_rhs(m1 + 0.5 * h * k1, xmid, umid, pmid, A, degree_scale)
        k3 = comparison_node_adjoint_rhs(m1 + 0.5 * h * k2, xmid, umid, pmid, A, degree_scale)
        k4 = comparison_node_adjoint_rhs(m1 + h * k3, x0, u0, p0, A, degree_scale)
        mu[idx] = m1 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return mu


def update_comparison_node_control(
    A: sp.csr_matrix,
    degree_scale: float,
    t: np.ndarray,
    x: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Backward sweep that updates node controls without storing all costates."""
    params = COMPARISON_DEFAULTS
    new_u = np.empty_like(u)
    mu_next = np.zeros(A.shape[0], dtype=np.float64)

    terminal_candidate = np.zeros(A.shape[0], dtype=np.float64)
    new_u[-1] = params["damping"] * terminal_candidate + (1.0 - params["damping"]) * u[-1]
    max_delta = float(np.max(np.abs(new_u[-1] - u[-1])))

    for idx in range(len(t) - 2, -1, -1):
        h = -float(t[idx + 1] - t[idx])
        x1, x0 = x[idx + 1], x[idx]
        p1, p0 = pressure[idx + 1], pressure[idx]
        u1, u0 = u[idx + 1], u[idx]
        xmid, pmid, umid = 0.5 * (x0 + x1), 0.5 * (p0 + p1), 0.5 * (u0 + u1)
        k1 = comparison_node_adjoint_rhs(mu_next, x1, u1, p1, A, degree_scale)
        k2 = comparison_node_adjoint_rhs(mu_next + 0.5 * h * k1, xmid, umid, pmid, A, degree_scale)
        k3 = comparison_node_adjoint_rhs(mu_next + 0.5 * h * k2, xmid, umid, pmid, A, degree_scale)
        k4 = comparison_node_adjoint_rhs(mu_next + h * k3, x0, u0, p0, A, degree_scale)
        mu_now = mu_next + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        candidate = np.clip(mu_now * x0 / params["control_weight"], 0.0, params["control_max"])
        new_u[idx] = params["damping"] * candidate + (1.0 - params["damping"]) * u0
        max_delta = max(max_delta, float(np.max(np.abs(new_u[idx] - u0))))
        mu_next = mu_now
    return new_u, max_delta


def run_comparison_degree_fbs(D: DegreeData, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    """Fixed-grid degree-level FBS for the paired epidemic comparison."""
    params = COMPARISON_DEFAULTS
    t = np.linspace(0.0, params["horizon"], steps + 1)
    u = np.zeros((len(t), len(D.k)), dtype=np.float64)
    x0 = comparison_degree_initial_state(D)
    delta_history: list[float] = []
    for _ in range(iterations):
        old_u = u.copy()
        x = integrate_comparison_degree_state(D, t, x0, u)
        lam = integrate_comparison_degree_adjoint(D, t, x, u)
        denom = params["control_weight"] * np.maximum(D.pk, 1e-12)
        candidate = np.clip(lam * x / denom, 0.0, params["control_max"])
        u = params["damping"] * candidate + (1.0 - params["damping"]) * old_u
        delta_u = float(np.max(np.abs(u - old_u)))
        delta_history.append(delta_u)
        if delta_u < tol:
            break
    x = integrate_comparison_degree_state(D, t, x0, u)
    running = params["state_weight"] * (x @ D.pk) + 0.5 * params["control_weight"] * ((u * u) @ D.pk)
    cost = float(np.trapezoid(running, t) if hasattr(np, "trapezoid") else np.trapz(running, t))
    final_delta = float(delta_history[-1]) if delta_history else float("nan")
    return {
        "solver_type": "paired_fixed_grid_rk4",
        "state_dimension": float(len(D.k)),
        "degree_classes": float(len(D.k)),
        "matrix_nonzeros": float("nan"),
        "fbs_iterations": float(len(delta_history)),
        "final_delta": final_delta,
        "converged": float(final_delta < tol),
        "cost": cost,
        "mean_initial_state": float(x0 @ D.pk),
        "mean_final_state": float(x[-1] @ D.pk),
        "mean_control": float(u.mean()),
        "max_control": float(u.max()),
    }


def run_comparison_node_fbs(A: sp.csr_matrix, degree: np.ndarray, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    """Sparse node-level FBS for the paired epidemic comparison."""
    params = COMPARISON_DEFAULTS
    degree_scale = max(float(degree.max(initial=1.0)), 1.0)
    t = np.linspace(0.0, params["horizon"], steps + 1)
    u = np.zeros((len(t), A.shape[0]), dtype=np.float64)
    x0 = comparison_node_initial_state(degree)
    delta_history: list[float] = []
    for _ in range(iterations):
        x, pressure = integrate_comparison_node_state(A, degree_scale, t, x0, u)
        u, delta_u = update_comparison_node_control(A, degree_scale, t, x, pressure, u)
        delta_history.append(delta_u)
        if delta_u < tol:
            break
    x, _ = integrate_comparison_node_state(A, degree_scale, t, x0, u)
    running = params["state_weight"] * x.mean(axis=1) + 0.5 * params["control_weight"] * np.mean(u * u, axis=1)
    cost = float(np.trapezoid(running, t) if hasattr(np, "trapezoid") else np.trapz(running, t))
    final_delta = float(delta_history[-1]) if delta_history else float("nan")
    return {
        "solver_type": "paired_sparse_fixed_grid_rk4",
        "state_dimension": float(A.shape[0]),
        "degree_classes": float("nan"),
        "matrix_nonzeros": float(A.nnz),
        "fbs_iterations": float(len(delta_history)),
        "final_delta": final_delta,
        "converged": float(final_delta < tol),
        "cost": cost,
        "mean_initial_state": float(x0.mean()),
        "mean_final_state": float(x[-1].mean()),
        "mean_control": float(u.mean()),
        "max_control": float(u.max()),
    }


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


def sparse_node_initial_state(A: sp.csr_matrix) -> np.ndarray:
    """Seed the highest-degree nodes without materializing a dense matrix."""
    n = A.shape[0]
    x0 = np.full(n, 0.015, dtype=np.float64)
    degree = np.asarray(A.getnnz(axis=1), dtype=np.int64)
    count = min(3, n)
    if count:
        hubs = np.argsort(-degree)[:count]
        x0[hubs] = 0.10
    return x0


def sparse_node_rhs(
    x: np.ndarray,
    u: np.ndarray,
    A: sp.csr_matrix,
    *,
    beta: float,
    delta: float,
) -> np.ndarray:
    pressure = A @ x
    return beta * (1.0 - x) * pressure - (delta + u) * x


def integrate_sparse_state(
    A: sp.csr_matrix,
    t: np.ndarray,
    x0: np.ndarray,
    u: np.ndarray,
    *,
    beta: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward RK4 integration for sparse node-level SIS dynamics."""
    x = np.empty((len(t), A.shape[0]), dtype=np.float64)
    pressure = np.empty_like(x)
    x[0] = x0
    pressure[0] = A @ x0
    for idx in range(len(t) - 1):
        h = float(t[idx + 1] - t[idx])
        u0 = u[idx]
        u1 = u[idx + 1]
        umid = 0.5 * (u0 + u1)
        y0 = x[idx]
        k1 = sparse_node_rhs(y0, u0, A, beta=beta, delta=delta)
        k2 = sparse_node_rhs(np.clip(y0 + 0.5 * h * k1, 0.0, 1.0), umid, A, beta=beta, delta=delta)
        k3 = sparse_node_rhs(np.clip(y0 + 0.5 * h * k2, 0.0, 1.0), umid, A, beta=beta, delta=delta)
        k4 = sparse_node_rhs(np.clip(y0 + h * k3, 0.0, 1.0), u1, A, beta=beta, delta=delta)
        x[idx + 1] = np.clip(y0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0, 1.0)
        pressure[idx + 1] = A @ x[idx + 1]
    return x, pressure


def sparse_node_adjoint_rhs(
    lam: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    pressure: np.ndarray,
    A: sp.csr_matrix,
    *,
    beta: float,
    delta: float,
    state_weight: float,
) -> np.ndarray:
    # The reported objective is averaged over nodes. Here lam is the scaled
    # costate n*lambda, which keeps the stationarity update in run_sparse_node_fbs
    # free of repeated 1/n factors.
    diag_part = (-beta * pressure - delta - u) * lam
    offdiag_part = beta * (A.T @ ((1.0 - x) * lam))
    return -state_weight - offdiag_part - diag_part


def integrate_sparse_adjoint(
    A: sp.csr_matrix,
    t: np.ndarray,
    x: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
    *,
    beta: float,
    delta: float,
    state_weight: float,
) -> np.ndarray:
    """Backward RK4 integration for the sparse node-level costate equation."""
    lam = np.zeros_like(x)
    for idx in range(len(t) - 2, -1, -1):
        h = -float(t[idx + 1] - t[idx])
        l1 = lam[idx + 1]
        x1 = x[idx + 1]
        x0 = x[idx]
        p1 = pressure[idx + 1]
        p0 = pressure[idx]
        u1 = u[idx + 1]
        u0 = u[idx]
        xmid = 0.5 * (x0 + x1)
        pmid = 0.5 * (p0 + p1)
        umid = 0.5 * (u0 + u1)

        k1 = sparse_node_adjoint_rhs(l1, x1, u1, p1, A, beta=beta, delta=delta, state_weight=state_weight)
        k2 = sparse_node_adjoint_rhs(l1 + 0.5 * h * k1, xmid, umid, pmid, A, beta=beta, delta=delta, state_weight=state_weight)
        k3 = sparse_node_adjoint_rhs(l1 + 0.5 * h * k2, xmid, umid, pmid, A, beta=beta, delta=delta, state_weight=state_weight)
        k4 = sparse_node_adjoint_rhs(l1 + h * k3, x0, u0, p0, A, beta=beta, delta=delta, state_weight=state_weight)
        lam[idx] = l1 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return lam


def run_sparse_node_fbs(A: sp.csr_matrix, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    """Sparse node-level PMP/FBS sweep for large synthetic graph experiments."""
    params = NODE_SCALABILITY_DEFAULTS
    t = np.linspace(0.0, params["horizon"], steps + 1)
    u = np.zeros((len(t), A.shape[0]), dtype=np.float64)
    x0 = sparse_node_initial_state(A)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_u = u.copy()
        x, pressure = integrate_sparse_state(A, t, x0, u, beta=params["beta"], delta=params["delta"])
        lam = integrate_sparse_adjoint(
            A,
            t,
            x,
            pressure,
            u,
            beta=params["beta"],
            delta=params["delta"],
            state_weight=params["state_weight"],
        )
        candidate = np.clip(lam * x / params["control_weight"], 0.0, params["control_max"])
        u = params["damping"] * candidate + (1.0 - params["damping"]) * old_u
        delta_u = float(np.max(np.abs(u - old_u)))
        delta_history.append(delta_u)
        if delta_u < tol:
            break

    x, _ = integrate_sparse_state(A, t, x0, u, beta=params["beta"], delta=params["delta"])
    running = params["state_weight"] * x.mean(axis=1) + 0.5 * params["control_weight"] * np.mean(u * u, axis=1)
    if hasattr(np, "trapezoid"):
        cost = float(np.trapezoid(running, t))
    else:
        cost = float(np.trapz(running, t))
    final_delta = float(delta_history[-1]) if delta_history else float("nan")
    return {
        "fbs_iterations": float(len(delta_history)),
        "final_delta": final_delta,
        "converged": float(final_delta < tol),
        "cost": cost,
        "mean_initial_state": float(x0.mean()),
        "mean_final_state": float(x[-1].mean()),
        "mean_control": float(u.mean()),
        "max_control": float(u.max()),
        "horizon": float(params["horizon"]),
        "beta": float(params["beta"]),
        "delta": float(params["delta"]),
        "state_weight": float(params["state_weight"]),
        "control_weight": float(params["control_weight"]),
        "control_max": float(params["control_max"]),
        "damping": float(params["damping"]),
    }


def run_degree_trial(graph: ig.Graph, *, steps: int, iterations: int, tol: float) -> dict[str, float]:
    prep_start = time.perf_counter()
    degree_data = degree_distribution_from_igraph(graph)
    prep_seconds = time.perf_counter() - prep_start

    solve_start = time.perf_counter()
    result = solve_degree_control(degree_data, steps=steps, iterations=iterations, tol=tol)
    solve_seconds = time.perf_counter() - solve_start

    return {
        "model_level": "degree",
        "solver_type": "adaptive_solve_ivp",
        "state_dimension": float(len(degree_data.k)),
        "degree_classes": float(len(degree_data.k)),
        "matrix_nonzeros": float("nan"),
        "prep_seconds": prep_seconds,
        "fbs_seconds": solve_seconds,
        "total_seconds": prep_seconds + solve_seconds,
        "fbs_iterations": float(result.value["iterations"]),
        "final_delta": float(result.value["final_delta"]),
        "converged": float(result.value["converged"]),
        "cost": float(result.value["cost"]),
        "horizon": float("nan"),
        "beta": float("nan"),
        "delta": float("nan"),
        "state_weight": float("nan"),
        "control_weight": float("nan"),
        "control_max": float("nan"),
        "damping": float("nan"),
        "mean_initial_state": float("nan"),
        "mean_final_state": float("nan"),
        "mean_control": float("nan"),
        "max_control": float("nan"),
    }


def run_node_trial(graph: ig.Graph, *, steps: int, iterations: int, tol: float, solver: str) -> dict[str, float]:
    prep_start = time.perf_counter()
    if solver == "dense":
        A, _ = graph_to_model_matrix(igraph_to_networkx(graph), directed=False, normalize="max-degree")
    else:
        A = igraph_to_sparse_model_matrix(graph)
    prep_seconds = time.perf_counter() - prep_start

    solve_start = time.perf_counter()
    if solver == "dense":
        result = solve_node_control(A, steps=steps, iterations=iterations, tol=tol)
        stats = {
            "fbs_iterations": float(result.value["iterations"]),
            "final_delta": float(result.value["final_delta"]),
            "converged": float(result.value["converged"]),
            "cost": float(result.value["cost"]),
            "horizon": float("nan"),
            "beta": float("nan"),
            "delta": float("nan"),
            "state_weight": float("nan"),
            "control_weight": float("nan"),
            "control_max": float("nan"),
            "damping": float("nan"),
            "mean_initial_state": float("nan"),
            "mean_final_state": float("nan"),
            "mean_control": float("nan"),
            "max_control": float("nan"),
        }
        state_dimension = float(A.shape[0])
        nonzeros = float(np.count_nonzero(A))
    else:
        stats = run_sparse_node_fbs(A, steps=steps, iterations=iterations, tol=tol)
        state_dimension = float(A.shape[0])
        nonzeros = float(A.nnz)
    solve_seconds = time.perf_counter() - solve_start

    return {
        "model_level": "node",
        "solver_type": solver,
        "state_dimension": state_dimension,
        "degree_classes": float("nan"),
        "matrix_nonzeros": nonzeros,
        "prep_seconds": prep_seconds,
        "fbs_seconds": solve_seconds,
        "total_seconds": prep_seconds + solve_seconds,
        **stats,
    }


def run_paired_comparison_trials(
    graph: ig.Graph,
    *,
    degree: np.ndarray,
    steps: int,
    iterations: int,
    tol: float,
) -> list[dict[str, float | str]]:
    """Run degree-level and sparse node-level FBS on the same epidemic-control instance."""
    degree_float = degree.astype(float, copy=False)
    degree_data = degree_distribution_from_degree(degree.astype(int, copy=False))
    A = igraph_to_sparse_adjacency(graph)
    shared = {
        "horizon": float(COMPARISON_DEFAULTS["horizon"]),
        "beta": float(COMPARISON_DEFAULTS["beta"]),
        "delta": float(COMPARISON_DEFAULTS["delta"]),
        "state_weight": float(COMPARISON_DEFAULTS["state_weight"]),
        "control_weight": float(COMPARISON_DEFAULTS["control_weight"]),
        "control_max": float(COMPARISON_DEFAULTS["control_max"]),
        "damping": float(COMPARISON_DEFAULTS["damping"]),
        "mean_degree": float(degree_float.mean()) if len(degree_float) else float("nan"),
        "max_degree": float(degree_float.max(initial=0.0)),
    }

    rows: list[dict[str, float | str]] = []
    for model_level, runner in (
        ("degree", lambda: run_comparison_degree_fbs(degree_data, steps=steps, iterations=iterations, tol=tol)),
        ("node", lambda: run_comparison_node_fbs(A, degree_float, steps=steps, iterations=iterations, tol=tol)),
    ):
        solve_start = time.perf_counter()
        stats = runner()
        solve_seconds = time.perf_counter() - solve_start
        rows.append(
            {
                "model_level": model_level,
                "comparison_type": "paired_degree_node_epidemic",
                "prep_seconds": 0.0,
                "fbs_seconds": solve_seconds,
                "total_seconds": solve_seconds,
                **shared,
                **stats,
            }
        )
    return rows


def run_scalability_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for nodes in args.sizes:
        for repeat in range(1, args.repeats + 1):
            # Each row is one reproducible graph/control solve; the summary plot
            # then separates median behavior from repeat-to-repeat variation.
            seed = args.seed + 1000 * repeat + nodes
            graph = synthetic_scale_free_graph(nodes, args.attachment_m, seed)
            degree = np.asarray(graph.degree(), dtype=np.int64)
            mean_degree = float(degree.mean()) if len(degree) else float("nan")
            max_degree = float(degree.max(initial=0))
            if args.model_level == "compare":
                trial_rows = run_paired_comparison_trials(
                    graph,
                    degree=degree,
                    steps=args.steps,
                    iterations=args.iterations,
                    tol=args.tolerance,
                )
            elif args.model_level == "degree":
                stats = run_degree_trial(graph, steps=args.steps, iterations=args.iterations, tol=args.tolerance)
                trial_rows = [stats]
            else:
                stats = run_node_trial(
                    graph,
                    steps=args.steps,
                    iterations=args.iterations,
                    tol=args.tolerance,
                    solver=args.node_solver,
                )
                trial_rows = [stats]

            for stats in trial_rows:
                row: dict[str, float | int | str] = {
                    "network_model": "Barabasi-Albert scale-free",
                    "nodes": nodes,
                    "edges": graph.ecount(),
                    "attachment_m": args.attachment_m,
                    "repeat": repeat,
                    "seed": seed,
                    "time_grid_steps": args.steps,
                    "max_fbs_iterations": args.iterations,
                    "tolerance": args.tolerance,
                    "mean_degree": mean_degree,
                    "max_degree": max_degree,
                }
                row.update(stats)
                rows.append(row)
                print(
                    f"n={nodes:4d} repeat={repeat} level={stats['model_level']} "
                    f"solver={stats.get('solver_type', 'degree')} "
                    f"fbs={float(stats['fbs_seconds']):.3f}s iterations={int(float(stats['fbs_iterations']))} "
                    f"converged={bool(stats['converged'])}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model_level", "solver_type", "nodes"], as_index=False).agg(
        fbs_seconds_median=("fbs_seconds", "median"),
        fbs_seconds_min=("fbs_seconds", "min"),
        fbs_seconds_max=("fbs_seconds", "max"),
        total_seconds_median=("total_seconds", "median"),
        fbs_iterations_median=("fbs_iterations", "median"),
        state_dimension_median=("state_dimension", "median"),
        degree_classes_median=("degree_classes", "median"),
        matrix_nonzeros_median=("matrix_nonzeros", "median"),
        mean_degree_median=("mean_degree", "median"),
        max_degree_median=("max_degree", "median"),
        max_degree_min=("max_degree", "min"),
        max_degree_max=("max_degree", "max"),
        converged_runs=("converged", "sum"),
        repeats=("repeat", "count"),
    )
    grouped["all_runs_converged"] = grouped["converged_runs"] == grouped["repeats"]
    return grouped


def plot_filename(model_level: str, summary: pd.DataFrame) -> str:
    node_min = int(summary["nodes"].min())
    node_max = int(summary["nodes"].max())
    if model_level == "compare":
        return f"degree_node_fbs_comparison_{node_min}_{node_max}.png"
    return f"{model_level}_control_scalability_{node_min}_{node_max}.png"


def compact_node_tick(value: float, _position: int | None = None) -> str:
    """Readable log-axis labels for network sizes."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{int(value)}"


def plot_paired_comparison(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> Path:
    """Plot the paired degree-level versus node-level epidemic FBS comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    styles = {
        "degree": {
            "color": "tab:blue",
            "marker": "o",
            "linestyle": "-",
            "label": "degree-level FBS (state = degree classes)",
        },
        "node": {
            "color": "tab:orange",
            "marker": "s",
            "linestyle": "--",
            "label": "node-level sparse FBS (state = graph nodes)",
        },
    }
    node_min = int(summary["nodes"].min())
    node_max = int(summary["nodes"].max())
    x_ticks = sorted(summary["nodes"].unique())
    x_formatter = FuncFormatter(compact_node_tick)

    ax = axes[0]
    for model_level, group in summary.groupby("model_level", sort=False):
        style = styles[str(model_level)]
        raw_group = raw[raw["model_level"] == model_level]
        ax.scatter(raw_group["nodes"], raw_group["fbs_seconds"], color=style["color"], alpha=0.25, s=28)
        ax.plot(
            group["nodes"],
            group["fbs_seconds_median"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            color=style["color"],
            label=style["label"],
        )
        ax.fill_between(
            group["nodes"].to_numpy(float),
            group["fbs_seconds_min"].to_numpy(float),
            group["fbs_seconds_max"].to_numpy(float),
            color=style["color"],
            alpha=0.10,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of nodes in the same synthetic SF network")
    ax.set_ylabel("FBS solve time (seconds)")
    ax.set_title("Runtime on the same SIS model and graph seeds")
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for model_level, group in summary.groupby("model_level", sort=False):
        style = styles[str(model_level)]
        ax.plot(
            group["nodes"],
            group["state_dimension_median"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            color=style["color"],
            label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of nodes in the same synthetic SF network")
    ax.set_ylabel("FBS state dimension")
    ax.set_title("State dimension: degree classes vs graph nodes")
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    graph_summary = (
        raw.drop_duplicates(["nodes", "repeat", "seed"])
        .groupby("nodes", as_index=False)
        .agg(
            max_degree_median=("max_degree", "median"),
            max_degree_min=("max_degree", "min"),
            max_degree_max=("max_degree", "max"),
        )
    )
    ax.plot(
        graph_summary["nodes"],
        graph_summary["max_degree_median"],
        marker="^",
        linestyle="-.",
        linewidth=2.0,
        color="tab:green",
        label="median max degree",
    )
    ax.fill_between(
        graph_summary["nodes"].to_numpy(float),
        graph_summary["max_degree_min"].to_numpy(float),
        graph_summary["max_degree_max"].to_numpy(float),
        color="tab:green",
        alpha=0.12,
        label="min-max over graph seeds",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of nodes in the same synthetic SF network")
    ax.set_ylabel("maximum degree")
    ax.set_title("Synthetic SF hubs grow with network size")
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Paired degree-level vs node-level FBS on synthetic SF graphs (log scale)")
    fig.tight_layout()
    path = out_dir / plot_filename("compare", summary)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scalability(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path, model_level: str) -> Path:
    if model_level == "compare":
        return plot_paired_comparison(summary, raw, out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    color = "tab:blue" if model_level == "degree" else "tab:orange"
    node_min = int(summary["nodes"].min())
    node_max = int(summary["nodes"].max())
    tick_candidates = (500, 1000, 1500, 2000, 4000, 6000, 8000, 10000)
    x_ticks = [node_min] + [tick for tick in tick_candidates if node_min < tick <= node_max]

    ax = axes[0]
    for _, row in raw.iterrows():
        ax.scatter(row["nodes"], row["fbs_seconds"], color=color, alpha=0.35, s=28)
    ax.plot(
        summary["nodes"],
        summary["fbs_seconds_median"],
        marker="o",
        linewidth=2.2,
        color=color,
        label="median FBS runtime",
    )
    ax.fill_between(
        summary["nodes"].to_numpy(float),
        summary["fbs_seconds_min"].to_numpy(float),
        summary["fbs_seconds_max"].to_numpy(float),
        color=color,
        alpha=0.15,
        label="min-max over repeats",
    )
    ax.set_xlabel("number of nodes in synthetic SF network")
    ax.set_ylabel("FBS solve time (seconds)")
    ax.set_title(f"{model_level.title()}-level FBS runtime")
    margin = max(25, int(0.025 * (node_max - node_min + 1)))
    ax.set_xlim(node_min - margin, node_max + margin)
    ax.set_xticks(x_ticks)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(
        summary["nodes"],
        summary["state_dimension_median"],
        marker="s",
        linewidth=2.0,
        color="tab:green",
        label="median state dimension",
    )
    ax2 = ax.twinx()
    ax2.plot(
        summary["nodes"],
        summary["fbs_iterations_median"],
        marker="^",
        linewidth=2.0,
        color="tab:purple",
        label="median FBS iterations",
    )
    ax.set_xlabel("number of nodes in synthetic SF network")
    ax.set_ylabel("state dimension", color="tab:green")
    ax2.set_ylabel("FBS iterations", color="tab:purple")
    ax.set_title("Why runtime changes")
    ax.set_xlim(node_min - margin, node_max + margin)
    ax.set_xticks(x_ticks)
    ax.grid(True, alpha=0.25)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(f"Scalability analysis on synthetic scale-free networks ({node_min}-{node_max} nodes)")
    fig.tight_layout()
    path = out_dir / plot_filename(model_level, summary)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path, model_level: str, plot_path: Path) -> None:
    if model_level == "compare":
        largest_node = int(summary["nodes"].max())
        largest = summary[summary["nodes"] == largest_node].sort_values("model_level")
        rows = "\n".join(
            f"| {row.model_level} | {row.solver_type} | {row.state_dimension_median:.0f} | "
            f"{row.fbs_iterations_median:.0f} | {row.fbs_seconds_median:.3f} | {bool(row.all_runs_converged)} |"
            for row in largest.itertuples(index=False)
        )
        graph_summary = (
            raw.drop_duplicates(["nodes", "repeat", "seed"])
            .groupby("nodes", as_index=False)
            .agg(
                edges_median=("edges", "median"),
                mean_degree_median=("mean_degree", "median"),
                max_degree_median=("max_degree", "median"),
                max_degree_min=("max_degree", "min"),
                max_degree_max=("max_degree", "max"),
            )
        )
        graph_rows = "\n".join(
            f"| {int(row.nodes)} | {int(row.edges_median)} | {row.mean_degree_median:.2f} | "
            f"{row.max_degree_median:.0f} | {row.max_degree_min:.0f}-{row.max_degree_max:.0f} |"
            for row in graph_summary.itertuples(index=False)
        )
        params = COMPARISON_DEFAULTS
        text = f"""# Paired Degree-Level vs Node-Level FBS Comparison

This run compares two FBS discretizations of the same normalized SIS epidemic-control problem on the same synthetic Barabasi-Albert scale-free graphs. For each network size and repeat, the degree-level row and node-level row use the exact same graph seed.

## What Was Measured

- Network sizes: {", ".join(str(int(n)) for n in sorted(raw["nodes"].unique()))} nodes.
- Repeats per size: {int(raw["repeat"].max())}.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m={int(raw["attachment_m"].iloc[0])}`.
- Epidemic/control model: normalized SIS, `T={params["horizon"]}`, `beta={params["beta"]}`, `delta={params["delta"]}`, state weight `{params["state_weight"]}`, control weight `{params["control_weight"]}`, `u_max={params["control_max"]}`.
- Contact-pressure normalization: each graph's maximum degree, shared by the degree-level and node-level rows.
- Numerical solver for both rows: fixed-grid RK4 inside forward-backward sweep.
- Node-level implementation: sparse adjacency matrix; state/control/costate are still node-indexed.
- Time grid: `{int(raw["time_grid_steps"].iloc[0])}` intervals over the control horizon.
- Maximum FBS iterations: `{int(raw["max_fbs_iterations"].iloc[0])}`.
- FBS tolerance: `{float(raw["tolerance"].iloc[0]):.0e}`.
- Runtime column: `fbs_seconds`, measuring only the FBS solve after graph generation.

## Main Output

| File | Meaning |
| --- | --- |
| `{plot_path.name}` | Paired runtime, state-dimension, and max-degree comparison. |
| `paired_fbs_comparison.csv` | One row per model level, size, and repeat. |
| `paired_fbs_comparison_summary.csv` | Median/min/max runtime and convergence summary by model level and size. |

## Quick Reading

At {largest_node} nodes:

| Model level | Solver label | FBS state dimension | FBS iterations | Median FBS seconds | All runs converged |
| --- | --- | ---: | ---: | ---: | --- |
{rows}

## Synthetic SF Degree Growth Check

| Nodes | Edges | Mean degree | Median max degree | Max-degree range |
| ---: | ---: | ---: | ---: | ---: |
{graph_rows}

This is the comparison plot to use when asking whether node-level FBS is more expensive than degree-level FBS. The degree-level row keeps one state/control/costate per observed degree class. The node-level row keeps one state/control/costate per graph node on the same epidemic model and the same graph seed. Sparse matrix products keep the million-node run practical, but the node-indexed FBS state is still much larger.
"""
        (out_dir / "scalability_summary.md").write_text(text)
        return

    largest = summary.sort_values("nodes").iloc[-1]
    solver_type = str(raw["solver_type"].iloc[0])
    text = f"""# Scalability Analysis

This run measures `{model_level}`-level forward-backward sweep (FBS) optimal control on synthetic Barabasi-Albert scale-free networks.

## What Was Measured

- Network sizes: {", ".join(str(int(n)) for n in sorted(raw["nodes"].unique()))} nodes.
- Repeats per size: {int(raw["repeat"].max())}.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m={int(raw["attachment_m"].iloc[0])}`.
- Numerical solver: `{solver_type}`.
- Time grid: `{int(raw["time_grid_steps"].iloc[0])}` intervals over the control horizon.
- Maximum FBS iterations: `{int(raw["max_fbs_iterations"].iloc[0])}`.
- FBS tolerance: `{float(raw["tolerance"].iloc[0]):.0e}`.
- Runtime column: `fbs_seconds`, measuring the FBS solve after graph generation and preprocessing.
- Convergence check: `final_delta < tolerance` for each run.

## Main Output

| File | Meaning |
| --- | --- |
| `{plot_path.name}` | Runtime and state-dimension/iteration trends as SF network size grows. |
| `{model_level}_control_scalability.csv` | One row per size and repeat. |
| `{model_level}_control_scalability_summary.csv` | Median/min/max runtime and convergence summary by size. |

## Quick Reading

At {int(largest["nodes"])} nodes, the median FBS solve time was {largest["fbs_seconds_median"]:.3f} seconds over {int(largest["repeats"])} repeat(s). All runs at that size converged: {bool(largest["all_runs_converged"])}.

Read runtime columns with the solver type. The default paired comparison uses `--model-level compare` and runs degree-level and sparse node-level FBS on the same SIS epidemic-control problem. Standalone `--model-level degree` and `--model-level node` runs are useful smoke/scaling diagnostics, but their wall-clock times are not the paired degree-vs-node comparison unless the solver, grid, graph seed, and model equations are matched explicitly.
"""
    (out_dir / "scalability_summary.md").write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SF-network scalability analysis for FBS optimal control.")
    parser.add_argument(
        "--model-level",
        choices=["compare", "degree", "node"],
        default="compare",
        help="Use 'compare' for paired degree/node FBS on the same SIS model; standalone modes are diagnostics.",
    )
    parser.add_argument("--sizes", type=parse_sizes, default=None, help="Comma-separated node counts.")
    parser.add_argument("--node-solver", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--attachment-m", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sizes is None:
        if args.model_level == "compare":
            args.sizes = list(DEFAULT_COMPARE_SIZES)
        elif args.model_level == "node":
            args.sizes = list(DEFAULT_NODE_SIZES)
        else:
            args.sizes = list(DEFAULT_DEGREE_SIZES)
    if args.output_dir is None:
        default_folder = (
            "scalability_degree_node_sf" if args.model_level == "compare" else f"scalability_{args.model_level}_sf"
        )
        args.output_dir = Path("results") / default_folder
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = run_scalability_experiment(args)
    summary = summarize(df)

    prefix = "paired_fbs_comparison" if args.model_level == "compare" else f"{args.model_level}_control_scalability"
    raw_path = args.output_dir / f"{prefix}.csv"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    legacy_plot = args.output_dir / f"{args.model_level}_control_scalability.png"
    if legacy_plot.exists():
        legacy_plot.unlink()
    plot_path = plot_scalability(summary, df, args.output_dir, args.model_level)
    write_report(summary, df, args.output_dir, args.model_level, plot_path)
    print(f"saved scalability outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
