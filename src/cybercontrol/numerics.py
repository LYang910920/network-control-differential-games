"""Numerical utilities shared by the tutorial repositories."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

Array = np.ndarray


def project_simplex(values: Array, eps: float = 1e-12) -> Array:
    """Return a nonnegative vector normalized to unit mass.

    This helper is a practical renormalization for compartment models.  It is
    deliberately simple and readable; use a true Euclidean projection if your
    optimization algorithm requires exact projection geometry.
    """

    y = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(y.sum())
    if total <= eps:
        out = np.zeros_like(y)
        out[0] = 1.0
        return out
    return y / total


def project_simplex3(x: Array, eps: float = 1e-12) -> Array:
    """Project the first three entries of ``x`` to a nonnegative unit simplex.

    Extra components, such as a deception or awareness state, are clipped to the
    interval ``[0, 1]``.  This matches the SIR-plus-auxiliary models used in the
    tutorials.
    """

    y = np.asarray(x, dtype=np.float64).copy()
    y[:3] = project_simplex(y[:3], eps=eps)
    if y.shape[0] > 3:
        y[3:] = np.clip(y[3:], 0.0, 1.0)
    return y


def rk4_step(
    rhs: Callable[[Array, float], Array],
    x: Array,
    t: float,
    h: float,
    project: Callable[[Array], Array] | None = None,
) -> Array:
    """Take one fourth-order Runge-Kutta step for ``x' = rhs(x, t)``."""

    y = np.asarray(x, dtype=np.float64)
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * h * k1, t + 0.5 * h)
    k3 = rhs(y + 0.5 * h * k2, t + 0.5 * h)
    k4 = rhs(y + h * k3, t + h)
    out = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return project(out) if project is not None else out


def rk4_integrate(
    rhs: Callable[[Array, float], Array],
    x0: Array,
    t0: float,
    dt: float,
    substeps: int = 10,
    project: Callable[[Array], Array] | None = None,
) -> Tuple[Array, Array]:
    """Integrate an ODE over one decision interval by RK4 substeps.

    The substeps are numerical solver points.  In sampled-data RL/MARL examples
    they are not policy-decision epochs.
    """

    if substeps <= 0:
        raise ValueError("substeps must be positive")

    h = dt / float(substeps)
    t = float(t0)
    y = np.asarray(x0, dtype=np.float64).copy()
    path = [y.copy()]
    for _ in range(substeps):
        y = rk4_step(rhs, y, t, h, project=project)
        t += h
        path.append(y.copy())
    return y, np.asarray(path)


def trapezoid(y: Array, x: Array) -> float:
    """Compatibility wrapper for NumPy trapezoidal integration."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
