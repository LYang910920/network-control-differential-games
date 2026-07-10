"""Objective, payoff, and matched-mean evaluation for foundation solvers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fbs import (
    TimeSeries,
    degree_control_model,
    degree_game_model,
    degree_game_rhs,
    node_control_model,
    node_game_model,
    node_game_rhs,
    solve_degree_control,
    solve_degree_game,
    solve_node_control,
    solve_node_game,
)
from .graph_data import DegreeData, degree_initial_state, node_initial_state
from .heterogeneity import degree_sis_rhs, node_sis_rhs
from .numerics import (
    as_time_function as as_function,
    project_box as clip,
    solve_ode_grid as solve_grid,
    trapezoid_integral as integral,
)

LOGGER = logging.getLogger(__name__)


def evaluate_degree_control_policy(
    D: DegreeData, t: np.ndarray, u: np.ndarray
) -> tuple[np.ndarray, float]:
    """Evaluate the degree-k control objective for a fixed admissible control."""
    model = degree_control_model(D)
    control = clip(u, 0, model.control_bound)
    x0 = degree_initial_state(D)
    u_fun = as_function(t, control)
    x = clip(solve_grid(lambda tau, y: degree_sis_rhs(y, u_fun(tau), D.k, model), x0, t))
    cost = integral(
        (x * model.state_weight) @ D.pk + 0.5 * ((control * control * model.control_weight) @ D.pk),
        t,
    )
    return x, cost


def _matched_mean_rows(
    name: str,
    level: str,
    problem: str,
    heterogeneous: TimeSeries,
    matched: TimeSeries,
) -> list[dict[str, object]]:
    """Rows for heterogeneous-vs-matched-mean comparison tables."""

    rows: list[dict[str, object]] = []
    for profile, result in (
        ("heterogeneous", heterogeneous),
        ("matched_mean_homogeneous", matched),
    ):
        base = {
            "example": name,
            "model_level": level,
            "problem": problem,
            "profile": profile,
            "iterations": int(
                result.value.get("iterations", len(result.controls.get("fbs_delta", [])))
            ),
            "converged": bool(result.value.get("converged", False)),
            "final_delta": float(result.value.get("final_delta", np.nan)),
        }
        if problem == "optimal_control":
            rows.append({**base, "metric": "cost", "value": float(result.value["cost"])})
        else:
            rows.append({**base, "metric": "attacker_payoff", "value": float(result.value["JA"])})
            rows.append({**base, "metric": "defender_payoff", "value": float(result.value["JD"])})
    return rows


def save_matched_mean_comparison(
    results: dict[str, TimeSeries],
    D: DegreeData,
    A: np.ndarray,
    out_dir: Path,
) -> None:
    """Solve matched-mean homogeneous baselines with the same canonical solvers."""

    rows: list[dict[str, object]] = []
    if "degree_control" in results:
        matched = solve_degree_control(
            D,
            steps=len(results["degree_control"].t) - 1,
            model=degree_control_model(D).matched_mean(),
        )
        rows.extend(
            _matched_mean_rows(
                "degree_control", "degree", "optimal_control", results["degree_control"], matched
            )
        )
    if "degree_game" in results:
        matched = solve_degree_game(
            D,
            steps=len(results["degree_game"].t) - 1,
            model=degree_game_model(D).matched_mean(),
        )
        rows.extend(
            _matched_mean_rows(
                "degree_game", "degree", "attacker_defender_game", results["degree_game"], matched
            )
        )
    if "node_control" in results:
        matched = solve_node_control(
            A,
            steps=len(results["node_control"].t) - 1,
            model=node_control_model(A).matched_mean(),
        )
        rows.extend(
            _matched_mean_rows(
                "node_control", "node", "optimal_control", results["node_control"], matched
            )
        )
    if "node_game" in results:
        matched = solve_node_game(
            A,
            steps=len(results["node_game"].t) - 1,
            model=node_game_model(A).matched_mean(),
        )
        rows.extend(
            _matched_mean_rows(
                "node_game", "node", "attacker_defender_game", results["node_game"], matched
            )
        )
    if rows:
        path = out_dir / "heterogeneous_vs_matched_mean.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        LOGGER.info(f"saved heterogeneous vs matched-mean comparison to {path}")


def evaluate_node_control_policy(
    A: np.ndarray, t: np.ndarray, u: np.ndarray
) -> tuple[np.ndarray, float]:
    """Evaluate the node-level control objective for a fixed admissible control."""
    model = node_control_model(A)
    control = clip(u, 0, model.control_bound)
    x0 = node_initial_state(A)
    u_fun = as_function(t, control)
    x = clip(solve_grid(lambda tau, y: node_sis_rhs(y, u_fun(tau), A, model), x0, t))
    cost = integral(
        (x * model.state_weight).sum(axis=1)
        + 0.5 * (control * control * model.control_weight).sum(axis=1),
        t,
    )
    return x, cost


def evaluate_degree_game_strategy(
    D: DegreeData,
    t: np.ndarray,
    attack: np.ndarray,
    defend: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Evaluate degree-level game payoffs for fixed open-loop strategies."""
    model = degree_game_model(D)
    attack = clip(attack, 0, model.attack_bound)
    defend = clip(defend, 0, model.defense_bound)
    x0 = degree_initial_state(D)
    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(
        solve_grid(
            lambda tau, y: degree_game_rhs(y, a_fun(tau), d_fun(tau), D, 0.0, 0.0, model), x0, t
        )
    )
    JA = integral(
        (model.attack_reward * x) @ D.pk - 0.5 * ((attack * attack * model.attack_cost) @ D.pk), t
    )
    JD = integral(
        -(model.defense_loss * x) @ D.pk - 0.5 * ((defend * defend * model.defense_cost) @ D.pk), t
    )
    return x, JA, JD


def evaluate_node_game_strategy(
    A: np.ndarray,
    t: np.ndarray,
    attack: np.ndarray,
    defend: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Evaluate node-level game payoffs for fixed open-loop strategies."""
    model = node_game_model(A)
    attack = clip(attack, 0, model.attack_bound)
    defend = clip(defend, 0, model.defense_bound)
    x0 = node_initial_state(A)
    a_fun, d_fun = as_function(t, attack), as_function(t, defend)
    x = clip(
        solve_grid(
            lambda tau, y: node_game_rhs(y, a_fun(tau), d_fun(tau), A, 0.0, 0.0, model), x0, t
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
    return x, JA, JD
