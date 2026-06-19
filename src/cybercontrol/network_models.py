"""Canonical node-level SIPS/SIPRS network propagation models.

Adjacency convention: ``A[i, j]`` means node ``j`` contributes infection
pressure to node ``i``.  Use :func:`normalize_adjacency` before simulations when
rows should represent weighted neighbor averages.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy import sparse as sp

from .numerics import project_compartments

Array = np.ndarray


@dataclass(frozen=True)
class NodeSIPRSParams:
    """Rates for node-level SIPS/SIPRS models."""

    beta: float = 0.8
    gamma: float = 0.12
    omega_p: float = 0.03
    omega_r: float = 0.02

    def to_dict(self) -> dict[str, float]:
        """Return a serializable parameter dictionary."""
        return asdict(self)


def normalize_adjacency(adjacency: Any, *, eps: float = 1e-12):
    """Return row-normalized adjacency with the same dense/sparse convention."""

    if sp.issparse(adjacency):
        A = adjacency.tocsr().astype(np.float64)
        row_sum = np.asarray(A.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sum)
        mask = row_sum > eps
        inv[mask] = 1.0 / row_sum[mask]
        return sp.diags(inv) @ A
    A = np.asarray(adjacency, dtype=np.float64)
    row_sum = A.sum(axis=1, keepdims=True)
    return np.divide(A, row_sum, out=np.zeros_like(A), where=row_sum > eps)


def graph_pressure_numpy(adjacency: Any, infected: Array) -> Array:
    """Return infection pressure ``A @ infected`` for dense or sparse ``A``."""

    return np.asarray(adjacency @ np.asarray(infected, dtype=np.float64)).reshape(-1)


def _node_vector(value: float | Array, nodes: int, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(nodes, float(arr), dtype=np.float64)
    if arr.shape == (nodes,):
        return arr
    raise ValueError(f"{name} must be scalar or shape ({nodes},), got {arr.shape}")


def node_siprs_transition_rates(
    x: Array,
    adjacency: Any,
    params: NodeSIPRSParams = NodeSIPRSParams(),
    *,
    patch: float | Array = 0.0,
    clean: float | Array = 0.0,
    beta_boost: float | Array = 0.0,
) -> dict[str, Array]:
    """Return node-wise transition rates for ``x=[S,I,P,R]``."""

    state = np.asarray(x, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(f"SIPRS state must have shape (nodes, 4), got {state.shape}")
    nodes = state.shape[0]
    boost = _node_vector(beta_boost, nodes, "beta_boost")
    patch_rate = _node_vector(patch, nodes, "patch")
    clean_rate = _node_vector(clean, nodes, "clean")
    pressure = graph_pressure_numpy(adjacency, state[:, 1])
    return {
        "infection": params.beta * (1.0 + boost) * pressure,
        "patch": patch_rate,
        "clean": clean_rate,
        "recovery": np.full(nodes, params.gamma, dtype=np.float64),
        "waning_p": np.full(nodes, params.omega_p, dtype=np.float64),
        "waning_r": np.full(nodes, params.omega_r, dtype=np.float64),
    }


def node_siprs_rhs_numpy(
    x: Array,
    adjacency: Any,
    params: NodeSIPRSParams = NodeSIPRSParams(),
    *,
    patch: float | Array = 0.0,
    clean: float | Array = 0.0,
    beta_boost: float | Array = 0.0,
) -> Array:
    """Node-level SIPRS ODE for ``x=[S,I,P,R]``.

    ``patch`` moves susceptible mass ``S -> P``.  Natural recovery and
    ``clean`` move infected mass ``I -> R``.  ``omega_p`` and ``omega_r`` return
    patched/recovered mass to susceptibility.
    """

    state = project_compartments(x, axis=-1)
    S, I, P, R = state.T
    rates = node_siprs_transition_rates(
        state,
        adjacency,
        params,
        patch=patch,
        clean=clean,
        beta_boost=beta_boost,
    )
    infection = rates["infection"]
    patch_rate = rates["patch"]
    clean_rate = rates["clean"]
    recovery = rates["recovery"]
    dS = -infection * S - patch_rate * S + rates["waning_p"] * P + rates["waning_r"] * R
    dI = infection * S - (recovery + clean_rate) * I
    dP = patch_rate * S - rates["waning_p"] * P
    dR = (recovery + clean_rate) * I - rates["waning_r"] * R
    return np.column_stack([dS, dI, dP, dR])


def node_sips_rhs_numpy(
    x: Array,
    adjacency: Any,
    params: NodeSIPRSParams = NodeSIPRSParams(),
    *,
    patch: float | Array = 0.0,
    clean: float | Array = 0.0,
    beta_boost: float | Array = 0.0,
) -> Array:
    """Node-level SIPS ODE for ``x=[S,I,P]`` with recovered mass merged into P."""

    state = project_compartments(x, axis=-1)
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError(f"SIPS state must have shape (nodes, 3), got {state.shape}")
    nodes = state.shape[0]
    S, I, P = state.T
    patch_rate = _node_vector(patch, nodes, "patch")
    clean_rate = _node_vector(clean, nodes, "clean")
    boost = _node_vector(beta_boost, nodes, "beta_boost")
    infection = params.beta * (1.0 + boost) * graph_pressure_numpy(adjacency, I)
    dS = -infection * S - patch_rate * S + params.omega_p * P
    dI = infection * S - (params.gamma + clean_rate) * I
    dP = patch_rate * S + (params.gamma + clean_rate) * I - params.omega_p * P
    return np.column_stack([dS, dI, dP])


def graph_pressure_torch(adjacency, infected):
    """Torch infection pressure for dense or sparse adjacency tensors."""

    import torch

    if getattr(adjacency, "is_sparse", False):
        return torch.sparse.mm(adjacency, infected.reshape(-1, 1)).reshape(-1)
    return adjacency @ infected


def _torch_node_vector(value, nodes: int, like, name: str):
    import torch

    if not torch.is_tensor(value):
        return torch.full((nodes,), float(value), dtype=like.dtype, device=like.device)
    value = value.to(dtype=like.dtype, device=like.device)
    if value.ndim == 0:
        return value.expand(nodes)
    if tuple(value.shape) == (nodes,):
        return value
    raise ValueError(f"{name} must be scalar or shape ({nodes},), got {tuple(value.shape)}")


def node_siprs_rhs_torch(
    x,
    adjacency,
    params: NodeSIPRSParams = NodeSIPRSParams(),
    *,
    patch=0.0,
    clean=0.0,
    beta_boost=0.0,
):
    """Torch version of :func:`node_siprs_rhs_numpy` for ``x=[S,I,P,R]``."""

    import torch

    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"SIPRS state must have shape (nodes, 4), got {tuple(x.shape)}")
    state = torch.clamp(x, min=0.0)
    state = state / state.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    nodes = state.shape[0]
    S, I, P, R = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    patch_rate = _torch_node_vector(patch, nodes, state, "patch")
    clean_rate = _torch_node_vector(clean, nodes, state, "clean")
    boost = _torch_node_vector(beta_boost, nodes, state, "beta_boost")
    infection = params.beta * (1.0 + boost) * graph_pressure_torch(adjacency, I)
    dS = -infection * S - patch_rate * S + params.omega_p * P + params.omega_r * R
    dI = infection * S - (params.gamma + clean_rate) * I
    dP = patch_rate * S - params.omega_p * P
    dR = (params.gamma + clean_rate) * I - params.omega_r * R
    return torch.stack([dS, dI, dP, dR], dim=-1)


def node_sips_rhs_torch(
    x,
    adjacency,
    params: NodeSIPRSParams = NodeSIPRSParams(),
    *,
    patch=0.0,
    clean=0.0,
    beta_boost=0.0,
):
    """Torch version of :func:`node_sips_rhs_numpy` for ``x=[S,I,P]``."""

    import torch

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"SIPS state must have shape (nodes, 3), got {tuple(x.shape)}")
    state = torch.clamp(x, min=0.0)
    state = state / state.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    nodes = state.shape[0]
    S, I, P = state[:, 0], state[:, 1], state[:, 2]
    patch_rate = _torch_node_vector(patch, nodes, state, "patch")
    clean_rate = _torch_node_vector(clean, nodes, state, "clean")
    boost = _torch_node_vector(beta_boost, nodes, state, "beta_boost")
    infection = params.beta * (1.0 + boost) * graph_pressure_torch(adjacency, I)
    dS = -infection * S - patch_rate * S + params.omega_p * P
    dI = infection * S - (params.gamma + clean_rate) * I
    dP = patch_rate * S + (params.gamma + clean_rate) * I - params.omega_p * P
    return torch.stack([dS, dI, dP], dim=-1)


def sample_node_siprs_step(
    states: Array,
    adjacency: Any,
    params: NodeSIPRSParams = NodeSIPRSParams(),
    *,
    dt: float,
    patch: float | Array = 0.0,
    clean: float | Array = 0.0,
    beta_boost: float | Array = 0.0,
    rng: np.random.Generator | None = None,
) -> Array:
    """Sample one categorical SIPRS step using the canonical transition rates.

    States are encoded as ``0=S, 1=I, 2=P, 3=R``.  The function is intentionally
    small and suited for smoke tests; use tau-leaping or Gillespie simulation
    for high-fidelity stochastic studies.
    """

    rng = np.random.default_rng() if rng is None else rng
    current = np.asarray(states, dtype=np.int64)
    if np.any((current < 0) | (current > 3)):
        raise ValueError("SIPRS categorical states must be in {0,1,2,3}")
    one_hot = np.eye(4, dtype=np.float64)[current]
    rates = node_siprs_transition_rates(
        one_hot,
        adjacency,
        params,
        patch=patch,
        clean=clean,
        beta_boost=beta_boost,
    )
    out = current.copy()
    draw = rng.random(len(current))
    p_inf = np.clip(dt * rates["infection"], 0.0, 1.0)
    p_patch = np.clip(dt * rates["patch"], 0.0, 1.0 - p_inf)
    s_mask = current == 0
    out[s_mask & (draw < p_inf)] = 1
    out[s_mask & (draw >= p_inf) & (draw < p_inf + p_patch)] = 2

    p_recover = np.clip(dt * (rates["recovery"] + rates["clean"]), 0.0, 1.0)
    i_mask = current == 1
    out[i_mask & (draw < p_recover)] = 3

    p_wane_p = np.clip(dt * rates["waning_p"], 0.0, 1.0)
    p_mask = current == 2
    out[p_mask & (draw < p_wane_p)] = 0

    p_wane_r = np.clip(dt * rates["waning_r"], 0.0, 1.0)
    r_mask = current == 3
    out[r_mask & (draw < p_wane_r)] = 0
    return out

