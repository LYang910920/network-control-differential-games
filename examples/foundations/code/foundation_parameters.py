"""Resolved parameter tables for the foundation model examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cybercontrol.baseline_diagnostics import RANDOM_BASELINE_COUNT
from cybercontrol.fbs import (
    degree_control_model,
    degree_game_model,
    node_control_model,
    node_game_model,
)
from cybercontrol.foundation_profiles import (
    CONTINUOUS_IMPULSE_PROFILE,
    DEGREE_CONTROL_PROFILE,
    DEGREE_GAME_PROFILE,
    NODE_CONTROL_PROFILE,
    NODE_GAME_PROFILE,
)
from cybercontrol.graph_data import DegreeData


def write_parameter_summary(
    out_dir: Path, args: argparse.Namespace, D: DegreeData, A: np.ndarray
) -> None:
    """Write a compact parameter map so readers do not need to hunt literals."""
    rows: list[dict[str, object]] = []

    def add(model: str, parameter: str, value: object, meaning: str) -> None:
        rows.append({"model": model, "parameter": parameter, "value": value, "meaning": meaning})

    def add_values(model: str, values: list[tuple[str, object, str]]) -> None:
        for parameter, value, meaning in values:
            add(model, parameter, value, meaning)

    def add_array(
        model: str,
        parameter: str,
        values: np.ndarray,
        meaning: str,
        *,
        weights: np.ndarray | None = None,
    ) -> None:
        arr = np.asarray(values, dtype=float)
        matched_mean = float(arr.mean())
        if weights is not None:
            normalized = np.asarray(weights, dtype=float)
            normalized = normalized / normalized.sum()
            matched_mean = float(normalized @ arr)
        add(model, parameter, np.array2string(arr, precision=4, separator=";"), meaning)
        add(
            model,
            f"{parameter}_summary",
            f"mean={arr.mean():.4g}; min={arr.min():.4g}; max={arr.max():.4g}",
            "Summary statistics for the resolved heterogeneous array.",
        )
        add(
            model,
            f"matched_mean_homogeneous_{parameter}",
            f"{matched_mean:.6g}",
            "Scalar matched-mean value used by the canonical homogeneous baseline.",
        )

    node_steps = max(24, args.steps // 3)
    add_values(
        "input graph",
        [
            ("full_nodes", D.node_degree.size, "Nodes used for the empirical degree distribution."),
            ("degree_classes", len(D.k), "Observed degree classes in degree-level models."),
            ("node_level_state_dimension", A.shape[0], "Nodes retained by dense node examples."),
            ("normalization", args.normalize, "Adjacency scaling before node-level ODE solves."),
        ],
    )
    add_values(
        "FBS",
        [
            ("max_iterations", args.iterations, "Maximum projected fixed-point evaluations."),
            (
                "stationarity_tolerance",
                args.tolerance,
                "Tolerance for the projected fixed-point residual before damping.",
            ),
        ],
    )

    profile_specs = [
        (
            DEGREE_CONTROL_PROFILE,
            degree_control_model(D),
            args.steps,
            "degree_class",
            D.pk,
            (
                "susceptibility",
                "infectivity",
                "recovery",
                "state_weight",
                "control_weight",
                "control_bound",
            ),
            [
                ("u_max", DEGREE_CONTROL_PROFILE.control_max, "Continuous healing upper bound."),
                (
                    "state/control weights",
                    f"{DEGREE_CONTROL_PROFILE.state_weight}/{DEGREE_CONTROL_PROFILE.control_weight}",
                    "Infection-versus-effort objective weights.",
                ),
            ],
        ),
        (
            DEGREE_GAME_PROFILE,
            degree_game_model(D),
            args.steps,
            "degree_class",
            D.pk,
            (
                "susceptibility",
                "infectivity",
                "recovery",
                "attack_reward",
                "defense_loss",
                "attack_cost",
                "defense_cost",
                "attack_bound",
                "defense_bound",
            ),
            [
                (
                    "attack/defense bounds",
                    f"{DEGREE_GAME_PROFILE.attack_max}/{DEGREE_GAME_PROFILE.defense_max}",
                    "Continuous strategy upper bounds.",
                ),
                (
                    "payoff weights",
                    f"{DEGREE_GAME_PROFILE.reward_attacker}/{DEGREE_GAME_PROFILE.loss_defender}",
                    "Attacker reward and defender infection-loss weights.",
                ),
            ],
        ),
        (
            NODE_CONTROL_PROFILE,
            node_control_model(A),
            node_steps,
            "node",
            None,
            (
                "susceptibility",
                "infectivity",
                "recovery",
                "state_weight",
                "control_weight",
                "control_bound",
            ),
            [
                ("u_max", NODE_CONTROL_PROFILE.control_max, "Continuous node-control upper bound."),
                (
                    "initial infected nodes",
                    2,
                    "Two highest-degree nodes start at 0.12; others at 0.02.",
                ),
            ],
        ),
        (
            NODE_GAME_PROFILE,
            node_game_model(A),
            node_steps,
            "node",
            None,
            (
                "susceptibility",
                "infectivity",
                "recovery",
                "attack_reward",
                "defense_loss",
                "attack_cost",
                "defense_cost",
                "attack_bound",
                "defense_bound",
            ),
            [
                (
                    "attack/defense bounds",
                    f"{NODE_GAME_PROFILE.attack_max}/{NODE_GAME_PROFILE.defense_max}",
                    "Continuous strategy upper bounds.",
                )
            ],
        ),
    ]
    for profile, resolved, steps, index_name, weights, fields, extra in profile_specs:
        factory = (
            "degree_correlated_sis_params"
            if index_name == "degree_class"
            else "degree_correlated_node_sis_params"
        )
        add_values(
            profile.label,
            [
                ("horizon", profile.horizon, "Total simulated time."),
                ("time_grid_steps", steps, "Number of FBS time intervals."),
                ("beta", profile.beta, "Baseline infection/contact rate."),
                ("delta", profile.delta, "Baseline natural recovery/removal rate."),
                (
                    "heterogeneity_profile",
                    f"{factory}(strength={profile.heterogeneity_strength})",
                    "Deterministic degree-correlated physical and economic arrays.",
                ),
                *extra,
            ],
        )
        for field in fields:
            add_array(
                profile.label,
                f"resolved_{field}_by_{index_name}",
                getattr(resolved, field),
                f"Heterogeneous {index_name.replace('_', ' ')} array used by the solver.",
                weights=weights,
            )

    impulse = CONTINUOUS_IMPULSE_PROFILE
    add_values(
        "continuous-impulse simulation",
        [
            ("horizon", impulse.horizon, "Total simulated time."),
            ("beta", impulse.beta, "Contact rate between impulse events."),
            ("delta", impulse.delta, "Natural recovery rate between impulse events."),
            (
                "continuous_control_range",
                f"{impulse.continuous_lower}-{impulse.continuous_upper}",
                "Bounds for the time-varying flow control.",
            ),
            ("impulse_times", ", ".join(map(str, impulse.impulse_times)), "State-reset times."),
            ("impulse_fraction", impulse.impulse_fraction, "Fraction removed at each reset."),
        ],
    )
    add(
        "baselines",
        "random_smooth_controls",
        RANDOM_BASELINE_COUNT,
        "Random continuous controls or strategies in each comparison.",
    )

    pd.DataFrame(rows).to_csv(out_dir / "parameter_summary.csv", index=False)
