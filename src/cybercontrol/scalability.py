"""Sparse degree/node FBS kernels used by scalability experiments."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .graph_data import DegreeData

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


def comparison_node_rhs(
    x: np.ndarray, u: np.ndarray, A: sp.csr_matrix, degree_scale: float
) -> np.ndarray:
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


def integrate_comparison_degree_state(
    D: DegreeData, t: np.ndarray, x0: np.ndarray, u: np.ndarray
) -> np.ndarray:
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


def integrate_comparison_degree_adjoint(
    D: DegreeData, t: np.ndarray, x: np.ndarray, u: np.ndarray
) -> np.ndarray:
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
            return (
                -params["state_weight"] * D.pk
                - comparison_degree_jacobian(x_now, u_now, D).T @ lam_now
            )

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


def run_comparison_degree_fbs(
    D: DegreeData, *, steps: int, iterations: int, tol: float
) -> dict[str, float]:
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
    running = params["state_weight"] * (x @ D.pk) + 0.5 * params["control_weight"] * (
        (u * u) @ D.pk
    )
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


def run_comparison_node_fbs(
    A: sp.csr_matrix, degree: np.ndarray, *, steps: int, iterations: int, tol: float
) -> dict[str, float]:
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
    running = params["state_weight"] * x.mean(axis=1) + 0.5 * params["control_weight"] * np.mean(
        u * u, axis=1
    )
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

        k1 = sparse_node_adjoint_rhs(
            l1, x1, u1, p1, A, beta=beta, delta=delta, state_weight=state_weight
        )
        k2 = sparse_node_adjoint_rhs(
            l1 + 0.5 * h * k1,
            xmid,
            umid,
            pmid,
            A,
            beta=beta,
            delta=delta,
            state_weight=state_weight,
        )
        k3 = sparse_node_adjoint_rhs(
            l1 + 0.5 * h * k2,
            xmid,
            umid,
            pmid,
            A,
            beta=beta,
            delta=delta,
            state_weight=state_weight,
        )
        k4 = sparse_node_adjoint_rhs(
            l1 + h * k3, x0, u0, p0, A, beta=beta, delta=delta, state_weight=state_weight
        )
        lam[idx] = l1 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return lam


def run_sparse_node_fbs(
    A: sp.csr_matrix, *, steps: int, iterations: int, tol: float
) -> dict[str, float]:
    """Sparse node-level PMP/FBS sweep for large synthetic graph experiments."""
    params = NODE_SCALABILITY_DEFAULTS
    t = np.linspace(0.0, params["horizon"], steps + 1)
    u = np.zeros((len(t), A.shape[0]), dtype=np.float64)
    x0 = sparse_node_initial_state(A)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_u = u.copy()
        x, pressure = integrate_sparse_state(
            A, t, x0, u, beta=params["beta"], delta=params["delta"]
        )
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
    running = params["state_weight"] * x.mean(axis=1) + 0.5 * params["control_weight"] * np.mean(
        u * u, axis=1
    )
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
