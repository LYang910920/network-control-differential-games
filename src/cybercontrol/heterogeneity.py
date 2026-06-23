"""Heterogeneous parameter helpers for degree and node network models.

The public rule is simple: callers may pass scalars for homogeneous examples or
arrays for heterogeneous examples.  This module resolves those specifications to
immutable arrays, validates shape/range, and provides small factories used by the
foundation and companion repositories.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

Array = np.ndarray


def resolve_vector(value, size: int, name: str, *, min_value: float | None = 0.0) -> Array:
    """Resolve a scalar or length-``size`` array to a finite float vector."""

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(size, float(arr), dtype=np.float64)
    elif arr.shape == (size,):
        out = arr.astype(np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be scalar or shape ({size},), got {arr.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    if min_value is not None and np.any(out < min_value):
        raise ValueError(f"{name} must be >= {min_value}")
    return out


def summarize_vector(values, name: str) -> dict[str, float | str]:
    """Return compact summary statistics for logging resolved arrays."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "name": name,
        "size": float(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _row_normalize(matrix: Array, name: str) -> Array:
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {M.shape}")
    if not np.all(np.isfinite(M)) or np.any(M < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    rows = M.sum(axis=1, keepdims=True)
    if np.any(rows <= 0.0):
        raise ValueError(f"{name} rows must have positive mass")
    return M / rows


def neutral_degree_mixing(k: Array, p: Array) -> Array:
    """Neutral degree mixing with rows ``M[k,l] = l p_l / <k>``."""

    degree = np.asarray(k, dtype=np.float64)
    prob = np.asarray(p, dtype=np.float64)
    kbar = float(degree @ prob)
    row = degree * prob / max(kbar, 1e-12)
    return np.repeat(row[None, :], len(degree), axis=0)


def assortative_degree_mixing(k: Array, p: Array, strength: float = 0.45, width: float = 0.75) -> Array:
    """Blend neutral mixing with a same-degree preference kernel."""

    degree = np.asarray(k, dtype=np.float64)
    neutral = neutral_degree_mixing(degree, p)
    logk = np.log(np.maximum(degree, 1.0))
    kernel = np.exp(-0.5 * ((logk[:, None] - logk[None, :]) / max(width, 1e-6)) ** 2)
    kernel = _row_normalize(kernel * np.asarray(p, dtype=np.float64)[None, :], "assortative kernel")
    alpha = float(np.clip(strength, 0.0, 1.0))
    return _row_normalize((1.0 - alpha) * neutral + alpha * kernel, "degree mixing")


@dataclass(frozen=True)
class DegreeSISParams:
    """Scalar-or-array parameters for heterogeneous degree-level SIS models."""

    susceptibility: object = 0.65
    infectivity: object = 1.0
    recovery: object = 0.18
    state_weight: object = 3.0
    control_weight: object = 2.5
    control_bound: object = 1.2
    attack_reward: object = 4.0
    defense_loss: object = 5.0
    attack_cost: object = 3.0
    defense_cost: object = 4.0
    attack_bound: object = 1.2
    defense_bound: object = 1.2
    mixing: object | None = None

    def resolve(self, k: Array, p: Array) -> "ResolvedDegreeSISParams":
        return resolve_degree_sis_params(k, p, self)


@dataclass(frozen=True)
class ResolvedDegreeSISParams:
    """Validated arrays used by degree-level solvers."""

    susceptibility: Array
    infectivity: Array
    recovery: Array
    state_weight: Array
    control_weight: Array
    control_bound: Array
    attack_reward: Array
    defense_loss: Array
    attack_cost: Array
    defense_cost: Array
    attack_bound: Array
    defense_bound: Array
    mixing: Array

    def summaries(self) -> list[dict[str, float | str]]:
        fields = (
            "susceptibility",
            "infectivity",
            "recovery",
            "state_weight",
            "control_weight",
            "control_bound",
            "attack_reward",
            "defense_loss",
            "attack_cost",
            "defense_cost",
            "attack_bound",
            "defense_bound",
        )
        return [summarize_vector(getattr(self, field), field) for field in fields]

    def matched_mean(self) -> "ResolvedDegreeSISParams":
        """Return a homogeneous profile with the same arithmetic mean arrays."""

        data = {
            field: np.full_like(self.susceptibility, float(np.mean(getattr(self, field))))
            for field in (
                "susceptibility",
                "infectivity",
                "recovery",
                "state_weight",
                "control_weight",
                "control_bound",
                "attack_reward",
                "defense_loss",
                "attack_cost",
                "defense_cost",
                "attack_bound",
                "defense_bound",
            )
        }
        return replace(self, **data)


def resolve_degree_sis_params(k: Array, p: Array, params: DegreeSISParams | None = None) -> ResolvedDegreeSISParams:
    """Resolve scalar/array degree SIS parameters for the observed degree classes."""

    params = DegreeSISParams() if params is None else params
    m = len(np.asarray(k))
    mixing = neutral_degree_mixing(k, p) if params.mixing is None else _row_normalize(params.mixing, "degree mixing")
    if mixing.shape != (m, m):
        raise ValueError(f"degree mixing must have shape ({m}, {m}), got {mixing.shape}")
    return ResolvedDegreeSISParams(
        susceptibility=resolve_vector(params.susceptibility, m, "degree susceptibility"),
        infectivity=resolve_vector(params.infectivity, m, "degree infectivity"),
        recovery=resolve_vector(params.recovery, m, "degree recovery"),
        state_weight=resolve_vector(params.state_weight, m, "degree state_weight"),
        control_weight=resolve_vector(params.control_weight, m, "degree control_weight", min_value=1e-12),
        control_bound=resolve_vector(params.control_bound, m, "degree control_bound"),
        attack_reward=resolve_vector(params.attack_reward, m, "degree attack_reward"),
        defense_loss=resolve_vector(params.defense_loss, m, "degree defense_loss"),
        attack_cost=resolve_vector(params.attack_cost, m, "degree attack_cost", min_value=1e-12),
        defense_cost=resolve_vector(params.defense_cost, m, "degree defense_cost", min_value=1e-12),
        attack_bound=resolve_vector(params.attack_bound, m, "degree attack_bound"),
        defense_bound=resolve_vector(params.defense_bound, m, "degree defense_bound"),
        mixing=mixing,
    )


def degree_correlated_sis_params(k: Array, p: Array, *, strength: float = 0.35, base: DegreeSISParams | None = None) -> ResolvedDegreeSISParams:
    """Factory with higher-degree classes carrying higher physical/economic risk."""

    degree = np.asarray(k, dtype=np.float64)
    z = (degree - degree.mean()) / (degree.std() + 1e-12)
    factor = np.exp(strength * z)
    base = DegreeSISParams() if base is None else base
    spec = replace(
        base,
        susceptibility=float(np.mean(resolve_vector(base.susceptibility, len(degree), "susceptibility"))) * factor,
        infectivity=float(np.mean(resolve_vector(base.infectivity, len(degree), "infectivity"))) * np.exp(0.5 * strength * z),
        recovery=float(np.mean(resolve_vector(base.recovery, len(degree), "recovery"))) / np.exp(0.25 * strength * z),
        state_weight=float(np.mean(resolve_vector(base.state_weight, len(degree), "state_weight"))) * factor,
        control_bound=float(np.mean(resolve_vector(base.control_bound, len(degree), "control_bound"))) / np.sqrt(factor),
        mixing=assortative_degree_mixing(degree, p, strength=min(0.75, abs(strength))),
    )
    return spec.resolve(degree, p)


def seeded_lognormal_degree_params(k: Array, p: Array, *, seed: int = 0, cv: float = 0.25, base: DegreeSISParams | None = None) -> ResolvedDegreeSISParams:
    """Factory for seeded lognormal degree-class heterogeneity with controlled spread."""

    base = DegreeSISParams() if base is None else base
    rng = np.random.default_rng(seed)
    size = len(np.asarray(k))
    sigma = np.sqrt(np.log1p(cv * cv))

    def sample(mean):
        values = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=size)
        return float(mean) * values

    spec = replace(
        base,
        susceptibility=sample(np.mean(resolve_vector(base.susceptibility, size, "susceptibility"))),
        infectivity=sample(np.mean(resolve_vector(base.infectivity, size, "infectivity"))),
        recovery=sample(np.mean(resolve_vector(base.recovery, size, "recovery"))),
        state_weight=sample(np.mean(resolve_vector(base.state_weight, size, "state_weight"))),
    )
    return spec.resolve(k, p)


def degree_sis_force(x: Array, k: Array, params: ResolvedDegreeSISParams) -> Array:
    """Return heterogeneous force of infection lambda_k."""

    infected_pressure = params.mixing @ (params.infectivity * np.asarray(x, dtype=np.float64))
    return params.susceptibility * np.asarray(k, dtype=np.float64) * infected_pressure


def degree_sis_rhs(x: Array, u: Array, k: Array, params: ResolvedDegreeSISParams) -> Array:
    """Heterogeneous degree-level SIS RHS."""

    state = np.asarray(x, dtype=np.float64)
    control = np.asarray(u, dtype=np.float64)
    force = degree_sis_force(state, k, params)
    return (1.0 - state) * force - (params.recovery + control) * state


def degree_sis_jacobian(x: Array, u: Array, k: Array, params: ResolvedDegreeSISParams) -> Array:
    """Analytic Jacobian of :func:`degree_sis_rhs` with respect to ``x``."""

    state = np.asarray(x, dtype=np.float64)
    force = degree_sis_force(state, k, params)
    B = params.susceptibility[:, None] * np.asarray(k, dtype=np.float64)[:, None] * params.mixing * params.infectivity[None, :]
    J = (1.0 - state)[:, None] * B
    J[np.diag_indices_from(J)] -= force + params.recovery + np.asarray(u, dtype=np.float64)
    return J


def finite_difference_jacobian(rhs, x: Array, eps: float = 1e-6) -> Array:
    """Small central-difference Jacobian helper for tests and audits."""

    state = np.asarray(x, dtype=np.float64)
    J = np.zeros((state.size, state.size), dtype=np.float64)
    for col in range(state.size):
        step = np.zeros_like(state)
        step[col] = eps
        J[:, col] = (rhs(state + step) - rhs(state - step)) / (2.0 * eps)
    return J


@dataclass(frozen=True)
class NodeSISParams:
    """Scalar-or-array parameters for heterogeneous node-level SIS models."""

    beta: object = 0.8
    susceptibility: object = 1.0
    infectivity: object = 1.0
    recovery: object = 0.16
    state_weight: object = 3.0
    control_weight: object = 2.2
    control_bound: object = 1.2
    attack_reward: object = 4.0
    defense_loss: object = 5.0
    attack_cost: object = 4.0
    defense_cost: object = 4.5
    attack_bound: object = 1.2
    defense_bound: object = 1.2

    def resolve(self, nodes: int) -> "ResolvedNodeSISParams":
        return resolve_node_sis_params(nodes, self)


@dataclass(frozen=True)
class ResolvedNodeSISParams:
    """Validated arrays used by node-level SIS solvers."""

    beta: Array
    susceptibility: Array
    infectivity: Array
    recovery: Array
    state_weight: Array
    control_weight: Array
    control_bound: Array
    attack_reward: Array
    defense_loss: Array
    attack_cost: Array
    defense_cost: Array
    attack_bound: Array
    defense_bound: Array

    def summaries(self) -> list[dict[str, float | str]]:
        return [
            summarize_vector(getattr(self, field), field)
            for field in (
                "beta",
                "susceptibility",
                "infectivity",
                "recovery",
                "state_weight",
                "control_weight",
                "control_bound",
                "attack_reward",
                "defense_loss",
                "attack_cost",
                "defense_cost",
                "attack_bound",
                "defense_bound",
            )
        ]

    def matched_mean(self) -> "ResolvedNodeSISParams":
        return replace(
            self,
            **{field: np.full_like(self.beta, float(np.mean(getattr(self, field)))) for field in self.__dataclass_fields__},
        )


def resolve_node_sis_params(nodes: int, params: NodeSISParams | None = None) -> ResolvedNodeSISParams:
    """Resolve scalar/array node SIS parameters to arrays."""

    params = NodeSISParams() if params is None else params
    return ResolvedNodeSISParams(
        beta=resolve_vector(params.beta, nodes, "node beta"),
        susceptibility=resolve_vector(params.susceptibility, nodes, "node susceptibility"),
        infectivity=resolve_vector(params.infectivity, nodes, "node infectivity"),
        recovery=resolve_vector(params.recovery, nodes, "node recovery"),
        state_weight=resolve_vector(params.state_weight, nodes, "node state_weight"),
        control_weight=resolve_vector(params.control_weight, nodes, "node control_weight", min_value=1e-12),
        control_bound=resolve_vector(params.control_bound, nodes, "node control_bound"),
        attack_reward=resolve_vector(params.attack_reward, nodes, "node attack_reward"),
        defense_loss=resolve_vector(params.defense_loss, nodes, "node defense_loss"),
        attack_cost=resolve_vector(params.attack_cost, nodes, "node attack_cost", min_value=1e-12),
        defense_cost=resolve_vector(params.defense_cost, nodes, "node defense_cost", min_value=1e-12),
        attack_bound=resolve_vector(params.attack_bound, nodes, "node attack_bound"),
        defense_bound=resolve_vector(params.defense_bound, nodes, "node defense_bound"),
    )


def node_sis_force(x: Array, adjacency, params: ResolvedNodeSISParams) -> Array:
    """Return heterogeneous node force of infection lambda_i."""

    infected_pressure = np.asarray(adjacency @ (params.infectivity * np.asarray(x, dtype=np.float64))).reshape(-1)
    return params.beta * params.susceptibility * infected_pressure


def node_sis_rhs(x: Array, u: Array, adjacency, params: ResolvedNodeSISParams) -> Array:
    """Heterogeneous node-level SIS RHS."""

    state = np.asarray(x, dtype=np.float64)
    force = node_sis_force(state, adjacency, params)
    return (1.0 - state) * force - (params.recovery + np.asarray(u, dtype=np.float64)) * state


def node_sis_jacobian(x: Array, u: Array, adjacency, params: ResolvedNodeSISParams) -> Array:
    """Analytic Jacobian of :func:`node_sis_rhs` with respect to ``x``."""

    state = np.asarray(x, dtype=np.float64)
    A = np.asarray(adjacency.toarray() if hasattr(adjacency, "toarray") else adjacency, dtype=np.float64)
    force = node_sis_force(state, A, params)
    B = params.beta[:, None] * params.susceptibility[:, None] * A * params.infectivity[None, :]
    J = (1.0 - state)[:, None] * B
    J[np.diag_indices_from(J)] -= force + params.recovery + np.asarray(u, dtype=np.float64)
    return J


def degree_correlated_node_sis_params(adjacency, *, strength: float = 0.35) -> ResolvedNodeSISParams:
    """Factory with node SIS rates/costs correlated with graph degree."""

    nodes = np.asarray(adjacency.sum(axis=1)).reshape(-1).size if hasattr(adjacency, "sum") else np.asarray(adjacency).shape[0]
    spec = NodeSISParams(
        beta=0.85,
        susceptibility=degree_correlated_node_values(adjacency, base=1.0, strength=strength),
        infectivity=degree_correlated_node_values(adjacency, base=1.0, strength=0.60 * strength),
        recovery=degree_correlated_node_values(adjacency, base=0.16, strength=0.20 * strength, inverse=True),
        state_weight=degree_correlated_node_values(adjacency, base=3.0, strength=0.70 * strength),
        control_bound=degree_correlated_node_values(adjacency, base=1.2, strength=0.15 * strength, inverse=True),
        attack_reward=degree_correlated_node_values(adjacency, base=4.0, strength=0.65 * strength),
        defense_loss=degree_correlated_node_values(adjacency, base=5.0, strength=0.65 * strength),
    )
    return spec.resolve(nodes)


@dataclass(frozen=True)
class ResolvedNodeSIPSParams:
    """Validated per-node arrays for heterogeneous SIPS dynamics."""

    beta: Array
    susceptibility: Array
    infectivity: Array
    gamma: Array
    omega: Array
    criticality: Array
    patch_cost: Array
    clean_cost: Array
    patch_bound: Array
    clean_bound: Array
    patch_efficacy: Array
    clean_efficacy: Array

    def summaries(self) -> list[dict[str, float | str]]:
        return [
            summarize_vector(getattr(self, field), field)
            for field in (
                "beta",
                "susceptibility",
                "infectivity",
                "gamma",
                "omega",
                "criticality",
                "patch_cost",
                "clean_cost",
                "patch_bound",
                "clean_bound",
                "patch_efficacy",
                "clean_efficacy",
            )
        ]

    def risk_score(self) -> Array:
        """Node-level risk proxy for observations and baseline policies."""

        return self.criticality * self.beta * self.susceptibility * self.infectivity / np.maximum(self.gamma, 1e-12)

    def matched_mean(self):
        """Return a homogeneous parameter object with matched means."""

        return replace(
            self,
            beta=np.full_like(self.beta, float(np.mean(self.beta))),
            susceptibility=np.full_like(self.susceptibility, float(np.mean(self.susceptibility))),
            infectivity=np.full_like(self.infectivity, float(np.mean(self.infectivity))),
            gamma=np.full_like(self.gamma, float(np.mean(self.gamma))),
            omega=np.full_like(self.omega, float(np.mean(self.omega))),
            criticality=np.full_like(self.criticality, float(np.mean(self.criticality))),
            patch_cost=np.full_like(self.patch_cost, float(np.mean(self.patch_cost))),
            clean_cost=np.full_like(self.clean_cost, float(np.mean(self.clean_cost))),
            patch_bound=np.full_like(self.patch_bound, float(np.mean(self.patch_bound))),
            clean_bound=np.full_like(self.clean_bound, float(np.mean(self.clean_bound))),
            patch_efficacy=np.full_like(self.patch_efficacy, float(np.mean(self.patch_efficacy))),
            clean_efficacy=np.full_like(self.clean_efficacy, float(np.mean(self.clean_efficacy))),
        )


def node_heterogeneity_summary(params: ResolvedNodeSIPSParams, mask: Array | None = None) -> Array:
    """Observation-ready summary: risk, susceptibility, infectivity, recovery."""

    index = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    return np.asarray(
        [
            np.mean(params.risk_score()[index]),
            np.mean(params.susceptibility[index]),
            np.mean(params.infectivity[index]),
            np.mean(params.gamma[index]),
        ],
        dtype=np.float64,
    )


def community_correlated_node_values(
    communities: Array,
    *,
    base: float,
    strength: float,
    floor: float = 1e-6,
) -> Array:
    """Create deterministic community multipliers around a base value."""

    group = np.asarray(communities, dtype=int)
    labels = np.unique(group)
    if labels.size == 1:
        return np.full(group.shape, float(base), dtype=np.float64)
    offsets = np.linspace(-1.0, 1.0, labels.size)
    values = np.empty(group.shape, dtype=np.float64)
    for label, offset in zip(labels, offsets):
        values[group == label] = max(floor, float(base) * float(np.exp(strength * offset)))
    return values


def degree_correlated_node_values(adjacency, *, base: float, strength: float, inverse: bool = False) -> Array:
    """Create node values correlated with row degree/weighted degree."""

    if hasattr(adjacency, "sum"):
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    else:
        degree = np.asarray(adjacency, dtype=np.float64).sum(axis=1)
    z = (degree - degree.mean()) / (degree.std() + 1e-12)
    sign = -1.0 if inverse else 1.0
    return float(base) * np.exp(sign * strength * z)


def seeded_lognormal_node_values(nodes: int, *, base: float, seed: int, cv: float) -> Array:
    """Seeded positive node values with approximate coefficient of variation."""

    sigma = np.sqrt(np.log1p(cv * cv))
    rng = np.random.default_rng(seed)
    return float(base) * rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=nodes)


def node_params_from_csv(path: str | Path, nodes: int) -> Mapping[str, Array]:
    """Load node parameter columns from a CSV file.

    The file may contain any subset of the public node parameter names.  A
    ``node`` column is optional; if present, rows are sorted by that index.
    """

    table = pd.read_csv(path)
    if "node" in table.columns:
        table = table.sort_values("node")
    if len(table) != nodes:
        raise ValueError(f"CSV parameter file must have {nodes} rows, got {len(table)}")
    allowed = {
        "beta",
        "susceptibility",
        "infectivity",
        "gamma",
        "omega",
        "criticality",
        "patch_cost",
        "clean_cost",
        "patch_bound",
        "clean_bound",
        "patch_efficacy",
        "clean_efficacy",
    }
    return {name: table[name].to_numpy(dtype=np.float64) for name in allowed if name in table.columns}
