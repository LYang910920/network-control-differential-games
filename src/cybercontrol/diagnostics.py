"""Shared helpers for diagnostic plots and glossaries.

Different examples use different numerical loops, but reader-facing diagnostics
should use consistent vocabulary.  This module keeps that small common layer in
the foundation package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import textwrap
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DiagnosticTerm:
    """One glossary row for a plotted or logged diagnostic."""

    name: str
    meaning: str
    read_as: str


COMMON_DIAGNOSTIC_TERMS: tuple[DiagnosticTerm, ...] = (
    DiagnosticTerm("iteration", "One optimizer, FBSM, or residual-update step.", "Use for solver progress on the x-axis."),
    DiagnosticTerm("episode", "One complete sampled-data rollout used by a policy-learning loop.", "Use for policy or game-learning curves."),
    DiagnosticTerm("rollout", "A forward simulation under a fixed policy, control, or parameter set.", "Use for validation outside the training loss."),
    DiagnosticTerm("training return", "Cumulative reward collected during learning, often with exploration.", "Noisy; do not treat one point as policy quality."),
    DiagnosticTerm("evaluation return", "Cumulative reward from the current policy under a fixed evaluation setting.", "Use rolling trends and compare with cyber metrics."),
    DiagnosticTerm("loss", "The scalar objective minimized by an optimizer.", "Must be read with its component losses."),
    DiagnosticTerm("data loss", "Mismatch between a model prediction and observed data.", "Shows data fit; it can improve while dynamics get worse if residual terms are weak."),
    DiagnosticTerm("ODE residual loss", "Mismatch between a learned time derivative and the ODE right-hand side.", "An equation-consistency check at collocation points."),
    DiagnosticTerm("residual loss", "Mismatch between a neural derivative and the model right-hand side.", "Small values indicate equation consistency, not necessarily optimality."),
    DiagnosticTerm("initial-condition loss", "Mismatch between the neural state and the known initial state.", "Anchors the trajectory at t=0."),
    DiagnosticTerm("boundary loss", "Mismatch at required initial or terminal boundary conditions.", "Important when terminal state or costate conditions are enforced numerically."),
    DiagnosticTerm("costate loss", "Mismatch in the PMP costate differential equation.", "Read with state and stationarity losses; one low residual alone is not enough."),
    DiagnosticTerm("stationarity loss", "Hamiltonian first-order condition residual, such as H_u.", "Evidence for PMP consistency in interior-control regions."),
    DiagnosticTerm("objective", "The control or learning target being minimized, such as infected burden plus control cost.", "Lower is better only within the same model and metric."),
    DiagnosticTerm("rollout objective", "Objective recomputed after simulating the original dynamics under a learned or fixed control.", "Validation outside the training residual."),
    DiagnosticTerm("correction regularizer", "Penalty that keeps a learned correction term small or smooth.", "Prevents a correction model from replacing the known mechanism."),
    DiagnosticTerm("mean control", "Average control intensity over the training or validation time grid.", "Useful for checking whether an objective is won by excessive intervention."),
    DiagnosticTerm("collocation point", "A time or state-time point where a residual is enforced without requiring observed data.", "Controls where equation consistency is checked."),
    DiagnosticTerm("held-out metric", "Error on data, times, trajectories, or graph seeds not used in fitting.", "A generalization check rather than a training loss."),
    DiagnosticTerm("control-update change", "Maximum change between consecutive FBSM controls or strategies.", "A convergence diagnostic; should decay toward tolerance."),
    DiagnosticTerm("rolling mean", "Moving average of recent noisy values.", "Shows trend without hiding stochastic variability."),
    DiagnosticTerm("baseline comparison", "Same-model comparison with no-control, fixed, random, or simple learned policies.", "Use before making a stronger method claim."),
)


def rolling_mean(values: Sequence[float], window: int = 5) -> list[float]:
    """Return a finite-value moving average with stable NaN handling."""

    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = [float(x) for x in values[start : idx + 1] if not math.isnan(float(x))]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def diagnostic_terms_for(names: Iterable[str]) -> list[DiagnosticTerm]:
    """Return glossary rows in the order requested, skipping unknown names."""

    by_name = {term.name: term for term in COMMON_DIAGNOSTIC_TERMS}
    return [by_name[name] for name in names if name in by_name]


def write_diagnostic_glossary(path: Path | str, terms: Sequence[DiagnosticTerm], *, title: str) -> None:
    """Write a compact Markdown glossary next to training outputs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"| `{term.name}` | {term.meaning} | {term.read_as} |" for term in terms)
    text = f"""# {title}

Use this page while reading the training-diagnostic figures and CSV histories.

| Term | Meaning | How to read it |
|---|---|---|
{rows}
"""
    path.write_text(text, encoding="utf-8")


def add_caption(fig, caption: str, *, width: int = 145, y: float = 0.015, fontsize: float = 9.0) -> None:
    """Add a wrapped caption under a Matplotlib figure."""

    fig.text(
        0.5,
        y,
        textwrap.fill(caption, width),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#333333",
    )
