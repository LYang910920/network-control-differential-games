"""Forward-backward sweep solvers for heterogeneous degree and node models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .foundation_profiles import (
    DEGREE_CONTROL_PROFILE,
    DEGREE_GAME_PROFILE,
    NODE_CONTROL_PROFILE,
    NODE_GAME_PROFILE,
)
from .graph_data import DegreeData, degree_initial_state, node_initial_state
from .heterogeneity import (
    DegreeSISParams,
    NodeSISParams,
    degree_correlated_node_sis_params,
    degree_correlated_sis_params,
    degree_sis_jacobian,
    degree_sis_rhs,
    node_sis_force,
    node_sis_jacobian,
    node_sis_rhs,
)
from .numerics import (
    as_time_function as as_function,
    project_box as clip,
    solve_ode_grid as solve_grid,
    trapezoid_integral as integral,
)


@dataclass(frozen=True)
class FBSOptions:
    """Explicit solver-grid and convergence choices for reproducible runs."""

    steps: int = 45
    max_iterations: int = 40
    tolerance: float = 1e-4
    damping: float = 0.35


@dataclass
class TimeSeries:
    """Generic container for trajectories."""

    t: np.ndarray
    x: np.ndarray
    controls: dict[str, np.ndarray]
    value: dict[str, float]


def relax(new: np.ndarray, old: np.ndarray, weight: float = 0.35) -> np.ndarray:
    """Damped update for forward-backward sweep iterations."""
    return weight * new + (1.0 - weight) * old


def convergence_values(delta_history: list[float], tol: float) -> dict[str, float]:
    """Small numeric summary for forward-backward sweep diagnostics."""
    final_delta = float(delta_history[-1]) if delta_history else float("nan")
    return {
        "iterations": float(len(delta_history)),
        "final_delta": final_delta,
        "converged": float(final_delta < tol),
        "tolerance": float(tol),
    }


def theta_degree(x: np.ndarray, D: DegreeData) -> float:
    """Degree-level infection pressure Theta(t)."""
    return float((D.k * D.pk) @ x / max(D.kbar, 1e-12))


def degree_rhs(
    x: np.ndarray, u: np.ndarray, D: DegreeData, beta: float, delta: float
) -> np.ndarray:
    """Degree-k SIS dynamics with scalar inputs broadcast through shared arrays."""
    model = DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk)
    return degree_sis_rhs(x, u, D.k, model)


def degree_jacobian(
    x: np.ndarray, u: np.ndarray, D: DegreeData, beta: float, delta: float
) -> np.ndarray:
    """Jacobian of degree_rhs with respect to x."""
    model = DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk)
    return degree_sis_jacobian(x, u, D.k, model)


def degree_control_model(D: DegreeData):
    """Resolved heterogeneous degree-control parameters used by solver and baselines."""
    params = DEGREE_CONTROL_PROFILE
    return degree_correlated_sis_params(
        D.k,
        D.pk,
        strength=0.28,
        base=DegreeSISParams(
            susceptibility=params.beta,
            recovery=params.delta,
            state_weight=params.state_weight,
            control_weight=params.control_weight,
            control_bound=params.control_max,
        ),
    )


def degree_game_model(D: DegreeData):
    """Resolved heterogeneous degree-game parameters used by solver and baselines."""
    params = DEGREE_GAME_PROFILE
    return degree_correlated_sis_params(
        D.k,
        D.pk,
        strength=0.22,
        base=DegreeSISParams(
            susceptibility=params.beta,
            recovery=params.delta,
            attack_reward=params.reward_attacker,
            defense_loss=params.loss_defender,
            attack_cost=params.cost_attack,
            defense_cost=params.cost_defense,
            attack_bound=params.attack_max,
            defense_bound=params.defense_max,
        ),
    )


def solve_degree_control(
    D: DegreeData,
    steps: int = 45,
    iterations: int = 40,
    tol: float = 1e-4,
    model=None,
) -> TimeSeries:
    """PMP forward-backward sweep for u_k(t)."""
    params = DEGREE_CONTROL_PROFILE
    model = degree_control_model(D) if model is None else model
    t = np.linspace(0, params.horizon, steps + 1)
    u = np.zeros((len(t), len(D.k)))
    x0 = degree_initial_state(D)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_u = u.copy()
        u_fun = as_function(t, u)
        x = clip(solve_grid(lambda tau, y: degree_sis_rhs(y, u_fun(tau), D.k, model), x0, t))
        x_fun = as_function(t, x)

        def adjoint(tau, lam):
            x_now, u_now = x_fun(tau), u_fun(tau)
            return (
                -model.state_weight * D.pk - degree_sis_jacobian(x_now, u_now, D.k, model).T @ lam
            )

        lam = solve_grid(adjoint, np.zeros(len(D.k)), t, backward=True)
        u = relax(
            clip(
                lam * x / (model.control_weight * np.maximum(D.pk, 1e-12)), 0, model.control_bound
            ),
            old_u,
        )
        delta_u = float(np.max(abs(u - old_u)))
        delta_history.append(delta_u)
        if delta_u < tol:
            break

    u_fun = as_function(t, u)
    x = clip(solve_grid(lambda tau, y: degree_sis_rhs(y, u_fun(tau), D.k, model), x0, t))
    cost = integral(
        (x * model.state_weight) @ D.pk + 0.5 * ((u * u * model.control_weight) @ D.pk), t
    )
    return TimeSeries(
        t=t,
        x=x,
        controls={"control": u, "fbs_delta": np.asarray(delta_history, dtype=float)},
        value={"cost": cost, **convergence_values(delta_history, tol)},
    )


def degree_game_rhs(x, attack, defend, D, beta, delta, model=None):
    """Degree-k attacker-defender dynamics."""
    model = (
        DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk)
        if model is None
        else model
    )
    pressure = model.mixing @ (model.infectivity * x)
    force = model.susceptibility * (1.0 + attack) * D.k * pressure
    return (1.0 - x) * force - (model.recovery + defend) * x


def degree_game_jacobian(x, attack, defend, D, beta, delta, model=None):
    """Jacobian of degree_game_rhs with respect to x."""
    model = (
        DegreeSISParams(susceptibility=beta, recovery=delta).resolve(D.k, D.pk)
        if model is None
        else model
    )
    pressure = model.mixing @ (model.infectivity * x)
    B = (
        model.susceptibility[:, None]
        * (1.0 + attack)[:, None]
        * D.k[:, None]
        * model.mixing
        * model.infectivity[None, :]
    )
    J = (1.0 - x)[:, None] * B
    J[np.diag_indices_from(J)] -= (
        model.susceptibility * (1.0 + attack) * D.k * pressure + model.recovery + defend
    )
    return J


def solve_degree_game(
    D: DegreeData,
    steps: int = 45,
    iterations: int = 45,
    tol: float = 1e-4,
    model=None,
) -> TimeSeries:
    """Open-loop Nash forward-backward sweep by degree class k."""
    params = DEGREE_GAME_PROFILE
    beta, delta = params.beta, params.delta
    model = degree_game_model(D) if model is None else model
    t = np.linspace(0, params.horizon, steps + 1)
    attack = np.zeros((len(t), len(D.k)))
    defend = np.zeros_like(attack)
    x0 = degree_initial_state(D)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_a, old_d = attack.copy(), defend.copy()
        a_fun, d_fun = as_function(t, attack), as_function(t, defend)
        x = clip(
            solve_grid(
                lambda tau, y: degree_game_rhs(y, a_fun(tau), d_fun(tau), D, beta, delta, model),
                x0,
                t,
            )
        )
        x_fun = as_function(t, x)

        def adjoint_attacker(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = degree_game_jacobian(x_now, a_now, d_now, D, beta, delta, model)
            return -model.attack_reward * D.pk - J.T @ lam

        def adjoint_defender(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = degree_game_jacobian(x_now, a_now, d_now, D, beta, delta, model)
            return model.defense_loss * D.pk - J.T @ lam

        lam_A = solve_grid(adjoint_attacker, np.zeros(len(D.k)), t, backward=True)
        lam_D = solve_grid(adjoint_defender, np.zeros(len(D.k)), t, backward=True)
        pressure = np.vstack([model.mixing @ (model.infectivity * row) for row in x])
        marginal_attack = model.susceptibility[None, :] * D.k[None, :] * (1.0 - x) * pressure
        attack = relax(
            clip(
                lam_A * marginal_attack / (model.attack_cost * np.maximum(D.pk, 1e-12)),
                0,
                model.attack_bound,
            ),
            old_a,
        )
        defend = relax(
            clip(
                -lam_D * x / (model.defense_cost * np.maximum(D.pk, 1e-12)), 0, model.defense_bound
            ),
            old_d,
        )
        delta_controls = float(max(np.max(abs(attack - old_a)), np.max(abs(defend - old_d))))
        delta_history.append(delta_controls)
        if delta_controls < tol:
            break

    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(
        solve_grid(
            lambda tau, y: degree_game_rhs(y, a_fun(tau), d_fun(tau), D, beta, delta, model), x0, t
        )
    )
    JA = integral(
        (model.attack_reward * x) @ D.pk - 0.5 * ((attack * attack * model.attack_cost) @ D.pk), t
    )
    JD = integral(
        -(model.defense_loss * x) @ D.pk - 0.5 * ((defend * defend * model.defense_cost) @ D.pk), t
    )
    return TimeSeries(
        t=t,
        x=x,
        controls={
            "attack": attack,
            "defense": defend,
            "fbs_delta": np.asarray(delta_history, dtype=float),
        },
        value={"JA": JA, "JD": JD, **convergence_values(delta_history, tol)},
    )


def node_rhs(x: np.ndarray, u: np.ndarray, A: np.ndarray, beta: float, delta: float) -> np.ndarray:
    """Node-level SIS dynamics."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0])
    return node_sis_rhs(x, u, A, model)


def node_jacobian(
    x: np.ndarray, u: np.ndarray, A: np.ndarray, beta: float, delta: float
) -> np.ndarray:
    """Jacobian of node_rhs with respect to x."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0])
    return node_sis_jacobian(x, u, A, model)


def node_control_model(A: np.ndarray):
    """Resolved heterogeneous node-control parameters used by solver and baselines."""
    profile = NODE_CONTROL_PROFILE
    base = NodeSISParams(
        beta=profile.beta,
        recovery=profile.delta,
        state_weight=profile.state_weight,
        control_weight=profile.control_weight,
        control_bound=profile.control_max,
    )
    return degree_correlated_node_sis_params(A, strength=0.28, base=base)


def node_game_model(A: np.ndarray):
    """Resolved heterogeneous node-game parameters used by solver and baselines."""
    profile = NODE_GAME_PROFILE
    base = NodeSISParams(
        beta=profile.beta,
        recovery=profile.delta,
        attack_reward=profile.reward_attacker,
        defense_loss=profile.loss_defender,
        attack_cost=profile.cost_attack,
        defense_cost=profile.cost_defense,
        attack_bound=profile.attack_max,
        defense_bound=profile.defense_max,
    )
    return degree_correlated_node_sis_params(A, strength=0.22, base=base)


def solve_node_control(
    A: np.ndarray,
    steps: int = 30,
    iterations: int = 35,
    tol: float = 1e-4,
    model=None,
) -> TimeSeries:
    """PMP forward-backward sweep for node controls u_i(t)."""
    params = NODE_CONTROL_PROFILE
    model = node_control_model(A) if model is None else model
    t, n = np.linspace(0, params.horizon, steps + 1), A.shape[0]
    u = np.zeros((len(t), n))
    x0 = node_initial_state(A)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_u = u.copy()
        u_fun = as_function(t, u)
        x = clip(solve_grid(lambda tau, y: node_sis_rhs(y, u_fun(tau), A, model), x0, t))
        x_fun = as_function(t, x)

        def adjoint(tau, lam):
            x_now, u_now = x_fun(tau), u_fun(tau)
            return -model.state_weight - node_sis_jacobian(x_now, u_now, A, model).T @ lam

        lam = solve_grid(adjoint, np.zeros(n), t, backward=True)
        u = relax(clip(lam * x / model.control_weight, 0, model.control_bound), old_u)
        delta_u = float(np.max(abs(u - old_u)))
        delta_history.append(delta_u)
        if delta_u < tol:
            break

    u_fun = as_function(t, u)
    x = clip(solve_grid(lambda tau, y: node_sis_rhs(y, u_fun(tau), A, model), x0, t))
    cost = integral(
        (x * model.state_weight).sum(axis=1) + 0.5 * (u * u * model.control_weight).sum(axis=1), t
    )
    return TimeSeries(
        t=t,
        x=x,
        controls={"control": u, "fbs_delta": np.asarray(delta_history, dtype=float)},
        value={"cost": cost, **convergence_values(delta_history, tol)},
    )


def node_game_rhs(x, attack, defend, A, beta, delta, model=None):
    """Node-level attacker-defender dynamics."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0]) if model is None else model
    force = (1.0 + attack) * node_sis_force(x, A, model)
    return (1.0 - x) * force - (model.recovery + defend) * x


def node_game_jacobian(x, attack, defend, A, beta, delta, model=None):
    """Jacobian of node_game_rhs with respect to x."""
    model = NodeSISParams(beta=beta, recovery=delta).resolve(A.shape[0]) if model is None else model
    pressure = A @ (model.infectivity * x)
    B = (
        model.beta[:, None]
        * model.susceptibility[:, None]
        * (1.0 + attack)[:, None]
        * A
        * model.infectivity[None, :]
    )
    J = (1.0 - x)[:, None] * B
    J[np.diag_indices_from(J)] -= (
        model.beta * model.susceptibility * (1.0 + attack) * pressure + model.recovery + defend
    )
    return J


def solve_node_game(
    A: np.ndarray,
    steps: int = 30,
    iterations: int = 40,
    tol: float = 1e-4,
    model=None,
) -> TimeSeries:
    """Open-loop Nash forward-backward sweep for node-level attack/defense."""
    params = NODE_GAME_PROFILE
    model = node_game_model(A) if model is None else model
    t, n = np.linspace(0, params.horizon, steps + 1), A.shape[0]
    attack = np.zeros((len(t), n))
    defend = np.zeros_like(attack)
    x0 = node_initial_state(A)
    delta_history: list[float] = []

    for _ in range(iterations):
        old_a, old_d = attack.copy(), defend.copy()
        a_fun, d_fun = as_function(t, attack), as_function(t, defend)
        x = clip(
            solve_grid(
                lambda tau, y: (
                    (1.0 - y) * (1.0 + a_fun(tau)) * node_sis_force(y, A, model)
                    - (model.recovery + d_fun(tau)) * y
                ),
                x0,
                t,
            )
        )
        x_fun = as_function(t, x)

        def adjoint_A(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = node_game_jacobian(x_now, a_now, d_now, A, 0.0, 0.0, model)
            return -model.attack_reward - J.T @ lam

        def adjoint_D(tau, lam):
            x_now, a_now, d_now = x_fun(tau), a_fun(tau), d_fun(tau)
            J = node_game_jacobian(x_now, a_now, d_now, A, 0.0, 0.0, model)
            return model.defense_loss - J.T @ lam

        lam_A = solve_grid(adjoint_A, np.zeros(n), t, backward=True)
        lam_D = solve_grid(adjoint_D, np.zeros(n), t, backward=True)
        pressure = np.vstack([A @ (model.infectivity * row) for row in x])
        marginal_attack = model.beta[None, :] * model.susceptibility[None, :] * (1.0 - x) * pressure
        attack = relax(
            clip(lam_A * marginal_attack / model.attack_cost, 0, model.attack_bound), old_a
        )
        defend = relax(clip(-lam_D * x / model.defense_cost, 0, model.defense_bound), old_d)
        delta_controls = float(max(np.max(abs(attack - old_a)), np.max(abs(defend - old_d))))
        delta_history.append(delta_controls)
        if delta_controls < tol:
            break

    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(
        solve_grid(
            lambda tau, y: (
                (1.0 - y) * (1.0 + a_fun(tau)) * node_sis_force(y, A, model)
                - (model.recovery + d_fun(tau)) * y
            ),
            x0,
            t,
        )
    )
    JA = integral(
        (model.attack_reward * x).sum(axis=1)
        - 0.5 * (attack * attack * model.attack_cost).sum(axis=1),
        t,
    )
    JD = integral(
        -(model.defense_loss * x).sum(axis=1)
        - 0.5 * (defend * defend * model.defense_cost).sum(axis=1),
        t,
    )
    return TimeSeries(
        t=t,
        x=x,
        controls={
            "attack": attack,
            "defense": defend,
            "fbs_delta": np.asarray(delta_history, dtype=float),
        },
        value={"JA": JA, "JD": JD, **convergence_values(delta_history, tol)},
    )
