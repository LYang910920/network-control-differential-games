# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cybercontrol.plotting import panel_label, publication_style, save_publication_figure


RANDOM_BASELINE_COUNT = 75
RANDOM_BASELINE_SEED = 20260617


def _add_model_key(
    row: dict[str, object],
    *,
    model_field: str | None,
    model_name: str | None,
) -> dict[str, object]:
    if not model_field:
        return row
    return {model_field: model_name or "", **row}


def smooth_random_controls(
    shape: tuple[int, int],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    rng: np.random.Generator,
    knots: int = 6,
) -> np.ndarray:
    """Return a smooth random continuous strategy on the simulation grid."""
    time_count, width = shape
    knot_count = max(2, min(knots, time_count))
    knot_x = np.linspace(0, time_count - 1, knot_count)
    grid_x = np.arange(time_count)
    knot_values = rng.uniform(lower, upper, size=(knot_count, width))
    values = np.empty((time_count, width), dtype=float)
    for col in range(width):
        values[:, col] = np.interp(grid_x, knot_x, knot_values[:, col])
    return values


def random_impulse_series(
    length: int,
    event_indices: np.ndarray | list[int],
    *,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a sparse random impulse strategy with nonzero values only at events."""
    values = np.zeros(length, dtype=float)
    events = np.asarray(event_indices, dtype=int)
    events = events[(events >= 0) & (events < length)]
    if len(events):
        values[events] = rng.uniform(lower, upper, size=len(events))
    return values


def write_baseline_table(rows: list[dict[str, object]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def control_baseline_rows(
    computed_control: np.ndarray,
    computed_value: float,
    evaluate_cost: Callable[[np.ndarray], float],
    *,
    random_upper: float,
    seed: int,
    random_lower: float = 0.0,
    random_count: int = RANDOM_BASELINE_COUNT,
    model_field: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, object]]:
    """Build computed, deterministic, and random rows for one control model."""
    rows: list[dict[str, object]] = []

    def add(row: dict[str, object]) -> None:
        rows.append(_add_model_key(row, model_field=model_field, model_name=model_name))

    add(
        {
            "scenario": "computed FBS control",
            "sample_type": "computed",
            "sample_id": "computed",
            "metric": "cost",
            "value": float(computed_value),
            "better": "lower",
        }
    )

    deterministic = {
        "no control": np.zeros_like(computed_control),
        "constant mean control": np.full_like(computed_control, float(np.mean(computed_control))),
    }
    for scenario, candidate in deterministic.items():
        add(
            {
                "scenario": scenario,
                "sample_type": "deterministic",
                "sample_id": scenario,
                "metric": "cost",
                "value": float(evaluate_cost(candidate)),
                "better": "lower",
            }
        )

    rng = np.random.default_rng(seed)
    for idx in range(random_count):
        random_control = smooth_random_controls(computed_control.shape, lower=random_lower, upper=random_upper, rng=rng)
        add(
            {
                "scenario": "random smooth control",
                "sample_type": "random",
                "sample_id": f"random_{idx + 1:03d}",
                "metric": "cost",
                "value": float(evaluate_cost(random_control)),
                "better": "lower",
            }
        )
    return rows


def game_baseline_rows(
    attack: np.ndarray,
    defense: np.ndarray,
    evaluate_payoffs: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    *,
    attack_upper: float,
    defense_upper: float,
    seed: int,
    random_count: int = RANDOM_BASELINE_COUNT,
    model_field: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, object]]:
    """Build two fixed-opponent panels for one continuous differential game."""
    rows: list[dict[str, object]] = []

    def add(row: dict[str, object]) -> None:
        rows.append(_add_model_key(row, model_field=model_field, model_name=model_name))

    zero_attack, zero_defense = np.zeros_like(attack), np.zeros_like(defense)
    attacker_payoff, defender_payoff = evaluate_payoffs(attack, defense)
    zero_attack_payoff, _ = evaluate_payoffs(zero_attack, defense)
    _, zero_defense_payoff = evaluate_payoffs(attack, zero_defense)

    add(
        {
            "panel": "fixed computed attack; vary defense",
            "scenario": "computed defense",
            "sample_type": "computed",
            "sample_id": "computed_defense",
            "metric": "defender payoff",
            "value": float(defender_payoff),
            "better": "higher",
        }
    )
    add(
        {
            "panel": "fixed computed attack; vary defense",
            "scenario": "zero defense",
            "sample_type": "deterministic",
            "sample_id": "zero_defense",
            "metric": "defender payoff",
            "value": float(zero_defense_payoff),
            "better": "higher",
        }
    )
    add(
        {
            "panel": "fixed computed defense; vary attack",
            "scenario": "computed attack",
            "sample_type": "computed",
            "sample_id": "computed_attack",
            "metric": "attacker payoff",
            "value": float(attacker_payoff),
            "better": "higher",
        }
    )
    add(
        {
            "panel": "fixed computed defense; vary attack",
            "scenario": "zero attack",
            "sample_type": "deterministic",
            "sample_id": "zero_attack",
            "metric": "attacker payoff",
            "value": float(zero_attack_payoff),
            "better": "higher",
        }
    )

    rng = np.random.default_rng(seed)
    for idx in range(random_count):
        random_attack = smooth_random_controls(attack.shape, upper=attack_upper, rng=rng)
        random_defense = smooth_random_controls(defense.shape, upper=defense_upper, rng=rng)
        random_attack_payoff, _ = evaluate_payoffs(random_attack, defense)
        _, random_defense_payoff = evaluate_payoffs(attack, random_defense)
        add(
            {
                "panel": "fixed computed defense; vary attack",
                "scenario": "random attack",
                "sample_type": "random",
                "sample_id": f"random_attack_{idx + 1:03d}",
                "metric": "attacker payoff",
                "value": float(random_attack_payoff),
                "better": "higher",
            }
        )
        add(
            {
                "panel": "fixed computed attack; vary defense",
                "scenario": "random defense",
                "sample_type": "random",
                "sample_id": f"random_defense_{idx + 1:03d}",
                "metric": "defender payoff",
                "value": float(random_defense_payoff),
                "better": "higher",
            }
        )
    return rows


def save_control_baseline_plot(
    rows: list[dict[str, object]],
    path: Path,
    *,
    title: str,
    ylabel: str,
    random_label: str = "random strategies",
) -> None:
    """Plot one optimal-control model against deterministic and random baselines."""
    df = pd.DataFrame(rows)
    deterministic = df[df["sample_type"] != "random"].copy()
    random_values = df[df["sample_type"] == "random"]["value"].to_numpy(float)

    labels = deterministic["scenario"].tolist()
    x_pos = np.arange(len(labels) + (1 if len(random_values) else 0))
    with publication_style():
        fig, ax = plt.subplots(figsize=(7.16, 4.0))
    colors = ["tab:blue"] + ["0.65"] * max(0, len(deterministic) - 1)
    ax.bar(x_pos[: len(deterministic)], deterministic["value"], color=colors, width=0.62)

    if len(random_values):
        random_x = len(deterministic)
        ax.boxplot(
            random_values,
            positions=[random_x],
            widths=0.56,
            patch_artist=True,
            boxprops={"facecolor": "tab:green", "alpha": 0.26},
            medianprops={"color": "tab:green", "linewidth": 2.0},
        )
        jitter = np.linspace(-0.08, 0.08, min(len(random_values), 30))
        sample = np.quantile(random_values, np.linspace(0.05, 0.95, len(jitter)))
        ax.scatter(np.full(len(sample), random_x) + jitter, sample, color="tab:green", s=12, alpha=0.35)
        labels.append(random_label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel(ylabel)
    panel_label(ax, title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_publication_figure(
        fig,
        path,
        metadata={
            "figure_type": "same-model baseline comparison",
            "random_baseline_count": int(len(random_values)),
        },
    )
    plt.close(fig)


def save_game_baseline_plot(
    rows: list[dict[str, object]],
    path: Path,
    *,
    title: str,
    ylabel: str = "payoff (higher is better)",
) -> None:
    """Plot one game model with fixed-opponent unilateral baseline panels."""
    df = pd.DataFrame(rows)
    panels = list(dict.fromkeys(df["panel"].tolist()))
    with publication_style():
        fig, axes = plt.subplots(1, len(panels), figsize=(3.58 * len(panels), 4.0), squeeze=False)

    for ax, panel in zip(axes.ravel(), panels):
        panel_df = df[df["panel"] == panel]
        deterministic = panel_df[panel_df["sample_type"] != "random"].copy()
        random_values = panel_df[panel_df["sample_type"] == "random"]["value"].to_numpy(float)
        labels = deterministic["scenario"].tolist()
        x_pos = np.arange(len(labels) + (1 if len(random_values) else 0))

        ax.scatter(
            x_pos[: len(deterministic)],
            deterministic["value"],
            s=80,
            color=["tab:blue"] + ["0.45"] * max(0, len(deterministic) - 1),
            zorder=3,
        )
        for x, y in zip(x_pos[: len(deterministic)], deterministic["value"]):
            ax.hlines(float(y), x - 0.24, x + 0.24, color="0.25", linewidth=1.2)

        if len(random_values):
            random_x = len(deterministic)
            ax.boxplot(
                random_values,
                positions=[random_x],
                widths=0.56,
                patch_artist=True,
                boxprops={"facecolor": "tab:green", "alpha": 0.26},
                medianprops={"color": "tab:green", "linewidth": 2.0},
            )
            jitter = np.linspace(-0.08, 0.08, min(len(random_values), 30))
            sample = np.quantile(random_values, np.linspace(0.05, 0.95, len(jitter)))
            ax.scatter(np.full(len(sample), random_x) + jitter, sample, color="tab:green", s=12, alpha=0.35)
            labels.append("random strategies")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=16, ha="right")
        ax.set_ylabel(ylabel)
        panel_label(ax, panel)
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    save_publication_figure(
        fig,
        path,
        metadata={
            "figure_type": "same-game unilateral baseline comparison",
            "caption_hint": title,
        },
    )
    plt.close(fig)
