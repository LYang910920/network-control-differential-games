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

from .heterogeneity import (
    ResolvedNodeSIPRSParams,
    community_correlated_node_values,
    degree_correlated_node_values,
    resolve_vector,
    seeded_lognormal_node_values,
)
from .numerics import project_compartments

Array = np.ndarray


@dataclass(frozen=True)
class NodeSIPRSParams:
    """Scalar-or-array rates and costs for node-level SIPS/SIPRS models.

    ``beta`` preserves the old homogeneous API.  Heterogeneous infection pressure
    is resolved as ``beta_i * susceptibility_i * A @ (infectivity_j I_j)``.
    """

    beta: object = 0.8
    gamma: object = 0.12
    omega_p: object = 0.03
    omega_r: object = 0.02
    susceptibility: object = 1.0
    infectivity: object = 1.0
    criticality: object = 1.0
    patch_cost: object = 1.0
    clean_cost: object = 1.0
    patch_bound: object = 1e9
    clean_bound: object = 1e9
    patch_efficacy: object = 1.0
    clean_efficacy: object = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable parameter dictionary."""

        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            arr = np.asarray(value)
            out[key] = float(arr) if arr.ndim == 0 else arr.astype(float).tolist()
        return out

    def resolve(self, nodes: int) -> ResolvedNodeSIPRSParams:
        """Return validated per-node arrays."""

        return ResolvedNodeSIPRSParams(
            beta=resolve_vector(self.beta, nodes, "beta"),
            susceptibility=resolve_vector(self.susceptibility, nodes, "susceptibility"),
            infectivity=resolve_vector(self.infectivity, nodes, "infectivity"),
            gamma=resolve_vector(self.gamma, nodes, "gamma"),
            omega_p=resolve_vector(self.omega_p, nodes, "omega_p"),
            omega_r=resolve_vector(self.omega_r, nodes, "omega_r"),
            criticality=resolve_vector(self.criticality, nodes, "criticality"),
            patch_cost=resolve_vector(self.patch_cost, nodes, "patch_cost"),
            clean_cost=resolve_vector(self.clean_cost, nodes, "clean_cost"),
            patch_bound=resolve_vector(self.patch_bound, nodes, "patch_bound"),
            clean_bound=resolve_vector(self.clean_bound, nodes, "clean_bound"),
            patch_efficacy=resolve_vector(self.patch_efficacy, nodes, "patch_efficacy"),
            clean_efficacy=resolve_vector(self.clean_efficacy, nodes, "clean_efficacy"),
        )


def homogeneous_node_siprs_params(
    nodes: int,
    *,
    beta: float = 0.8,
    gamma: float = 0.12,
    omega_p: float = 0.03,
    omega_r: float = 0.02,
    criticality: float = 1.0,
) -> NodeSIPRSParams:
    """Return a scalar-compatible homogeneous node-SIPRS profile."""

    return NodeSIPRSParams(
        beta=beta,
        gamma=gamma,
        omega_p=omega_p,
        omega_r=omega_r,
        criticality=criticality,
    )


def community_correlated_node_siprs_params(
    communities: Array,
    *,
    strength: float = 0.35,
    beta: float = 0.8,
    gamma: float = 0.12,
    omega_p: float = 0.03,
    omega_r: float = 0.02,
) -> NodeSIPRSParams:
    """Factory with rates/costs correlated with known community labels."""

    group = np.asarray(communities, dtype=int)
    return NodeSIPRSParams(
        beta=beta,
        gamma=community_correlated_node_values(group, base=gamma, strength=-0.35 * strength),
        omega_p=omega_p,
        omega_r=omega_r,
        susceptibility=community_correlated_node_values(group, base=1.0, strength=strength),
        infectivity=community_correlated_node_values(group, base=1.0, strength=0.65 * strength),
        criticality=community_correlated_node_values(group, base=1.0, strength=0.85 * strength),
        patch_cost=community_correlated_node_values(group, base=1.0, strength=0.30 * strength),
        clean_cost=community_correlated_node_values(group, base=1.0, strength=0.25 * strength),
        patch_bound=community_correlated_node_values(group, base=0.45, strength=-0.20 * strength),
        clean_bound=community_correlated_node_values(group, base=0.55, strength=-0.20 * strength),
        patch_efficacy=community_correlated_node_values(group, base=1.0, strength=-0.15 * strength),
        clean_efficacy=community_correlated_node_values(group, base=1.0, strength=-0.10 * strength),
    )


def degree_correlated_node_siprs_params(adjacency: Any, *, strength: float = 0.35) -> NodeSIPRSParams:
    """Factory with node risk correlated with graph degree/weighted degree."""

    return NodeSIPRSParams(
        beta=0.8,
        gamma=degree_correlated_node_values(adjacency, base=0.12, strength=0.20 * strength, inverse=True),
        omega_p=0.03,
        omega_r=0.02,
        susceptibility=degree_correlated_node_values(adjacency, base=1.0, strength=strength),
        infectivity=degree_correlated_node_values(adjacency, base=1.0, strength=0.60 * strength),
        criticality=degree_correlated_node_values(adjacency, base=1.0, strength=0.75 * strength),
        patch_bound=degree_correlated_node_values(adjacency, base=0.45, strength=0.15 * strength, inverse=True),
        clean_bound=degree_correlated_node_values(adjacency, base=0.55, strength=0.15 * strength, inverse=True),
    )


def seeded_lognormal_node_siprs_params(nodes: int, *, seed: int = 0, cv: float = 0.25) -> NodeSIPRSParams:
    """Factory for seeded lognormal node heterogeneity."""

    return NodeSIPRSParams(
        beta=seeded_lognormal_node_values(nodes, base=0.8, seed=seed, cv=cv),
        gamma=seeded_lognormal_node_values(nodes, base=0.12, seed=seed + 1, cv=cv),
        omega_p=seeded_lognormal_node_values(nodes, base=0.03, seed=seed + 2, cv=0.5 * cv),
        omega_r=seeded_lognormal_node_values(nodes, base=0.02, seed=seed + 3, cv=0.5 * cv),
        susceptibility=seeded_lognormal_node_values(nodes, base=1.0, seed=seed + 4, cv=cv),
        infectivity=seeded_lognormal_node_values(nodes, base=1.0, seed=seed + 5, cv=cv),
        criticality=seeded_lognormal_node_values(nodes, base=1.0, seed=seed + 6, cv=cv),
    )


def _resolve_params(params: NodeSIPRSParams | ResolvedNodeSIPRSParams, nodes: int) -> ResolvedNodeSIPRSParams:
    if isinstance(params, ResolvedNodeSIPRSParams):
        if params.beta.shape != (nodes,):
            raise ValueError(f"resolved node params must have shape ({nodes},), got {params.beta.shape}")
        return params
    return params.resolve(nodes)


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
    params: NodeSIPRSParams | ResolvedNodeSIPRSParams = NodeSIPRSParams(),
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
    resolved = _resolve_params(params, nodes)
    boost = _node_vector(beta_boost, nodes, "beta_boost")
    patch_rate = np.minimum(_node_vector(patch, nodes, "patch") * resolved.patch_efficacy, resolved.patch_bound)
    clean_rate = np.minimum(_node_vector(clean, nodes, "clean") * resolved.clean_efficacy, resolved.clean_bound)
    pressure = graph_pressure_numpy(adjacency, resolved.infectivity * state[:, 1])
    return {
        "infection": resolved.beta * resolved.susceptibility * (1.0 + boost) * pressure,
        "patch": patch_rate,
        "clean": clean_rate,
        "recovery": resolved.gamma,
        "waning_p": resolved.omega_p,
        "waning_r": resolved.omega_r,
    }


def node_siprs_rhs_numpy(
    x: Array,
    adjacency: Any,
    params: NodeSIPRSParams | ResolvedNodeSIPRSParams = NodeSIPRSParams(),
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
    params: NodeSIPRSParams | ResolvedNodeSIPRSParams = NodeSIPRSParams(),
    *,
    patch: float | Array = 0.0,
    clean: float | Array = 0.0,
    beta_boost: float | Array = 0.0,
) -> Array:
    """Node-level SIPS ODE for ``x=[S,I,P]`` with recovered mass merged into P."""

    state = project_compartments(x, axis=-1)
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError(f"SIPS state must have shape (nodes, 3), got {state.shape}")
    S, I, P = state.T
    x4 = np.column_stack([S, I, P, np.zeros_like(S)])
    rates = node_siprs_transition_rates(x4, adjacency, params, patch=patch, clean=clean, beta_boost=beta_boost)
    infection = rates["infection"]
    patch_rate = rates["patch"]
    clean_rate = rates["clean"]
    dS = -infection * S - patch_rate * S + rates["waning_p"] * P
    dI = infection * S - (rates["recovery"] + clean_rate) * I
    dP = patch_rate * S + (rates["recovery"] + clean_rate) * I - rates["waning_p"] * P
    return np.column_stack([dS, dI, dP])


def graph_pressure_torch(adjacency, infected):
    """Torch infection pressure for dense or sparse adjacency tensors."""

    import torch

    if getattr(adjacency, "is_sparse", False):
        return torch.sparse.mm(adjacency, infected.reshape(-1, 1)).reshape(-1)
    return adjacency @ infected


def _torch_node_vector(value, nodes: int, like, name: str):
    import torch

    if torch.is_tensor(value):
        value = value.to(dtype=like.dtype, device=like.device)
    else:
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if value.ndim == 0:
        return value.expand(nodes)
    if tuple(value.shape) == (nodes,):
        return value
    raise ValueError(f"{name} must be scalar or shape ({nodes},), got {tuple(value.shape)}")


def node_siprs_rhs_torch(
    x,
    adjacency,
    params: NodeSIPRSParams | ResolvedNodeSIPRSParams = NodeSIPRSParams(),
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
    patch_raw = _torch_node_vector(patch, nodes, state, "patch")
    clean_raw = _torch_node_vector(clean, nodes, state, "clean")
    boost = _torch_node_vector(beta_boost, nodes, state, "beta_boost")
    beta = _torch_node_vector(params.beta, nodes, state, "beta")
    susceptibility = _torch_node_vector(params.susceptibility, nodes, state, "susceptibility")
    infectivity = _torch_node_vector(params.infectivity, nodes, state, "infectivity")
    gamma = _torch_node_vector(params.gamma, nodes, state, "gamma")
    omega_p = _torch_node_vector(params.omega_p, nodes, state, "omega_p")
    omega_r = _torch_node_vector(params.omega_r, nodes, state, "omega_r")
    patch_rate = torch.minimum(
        patch_raw * _torch_node_vector(params.patch_efficacy, nodes, state, "patch_efficacy"),
        _torch_node_vector(params.patch_bound, nodes, state, "patch_bound"),
    )
    clean_rate = torch.minimum(
        clean_raw * _torch_node_vector(params.clean_efficacy, nodes, state, "clean_efficacy"),
        _torch_node_vector(params.clean_bound, nodes, state, "clean_bound"),
    )
    infection = beta * susceptibility * (1.0 + boost) * graph_pressure_torch(adjacency, infectivity * I)
    dS = -infection * S - patch_rate * S + omega_p * P + omega_r * R
    dI = infection * S - (gamma + clean_rate) * I
    dP = patch_rate * S - omega_p * P
    dR = (gamma + clean_rate) * I - omega_r * R
    return torch.stack([dS, dI, dP, dR], dim=-1)


def node_sips_rhs_torch(
    x,
    adjacency,
    params: NodeSIPRSParams | ResolvedNodeSIPRSParams = NodeSIPRSParams(),
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
    S, I, P = state[:, 0], state[:, 1], state[:, 2]
    zeros = torch.zeros_like(S)
    rhs4 = node_siprs_rhs_torch(
        torch.stack([S, I, P, zeros], dim=-1),
        adjacency,
        params,
        patch=patch,
        clean=clean,
        beta_boost=beta_boost,
    )
    dS, dI = rhs4[:, 0], rhs4[:, 1]
    dP = rhs4[:, 2] + rhs4[:, 3]
    return torch.stack([dS, dI, dP], dim=-1)


def sample_node_siprs_step(
    states: Array,
    adjacency: Any,
    params: NodeSIPRSParams | ResolvedNodeSIPRSParams = NodeSIPRSParams(),
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
