"""Numerical utilities shared by the tutorial repositories."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

Array = np.ndarray


def project_box(values: Array, lower: float | Array = 0.0, upper: float | Array = 1.0) -> Array:
    """Project a scalar or array onto box constraints.

    This is the canonical helper for bounded states, controls, and strategies
    in the repository family.  It uses NumPy broadcasting, so ``lower`` and
    ``upper`` may be scalars or arrays matching ``values``.
    """

    return np.minimum(np.maximum(np.asarray(values, dtype=np.float64), lower), upper)


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


def project_compartments(values: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Project compartment arrays to nonnegative unit mass along ``axis``.

    This is the generic form used by node-level models.  A state shaped
    ``[N, C]`` is projected node-by-node when ``axis=-1``; a batch shaped
    ``[B, N, C]`` follows the same convention.  If a compartment vector has no
    positive mass after clipping, the first compartment receives unit mass.
    """

    y = np.moveaxis(np.maximum(np.asarray(values, dtype=np.float64), 0.0), axis, -1)
    if y.ndim == 1:
        total = float(y.sum())
        if total <= eps:
            out = np.zeros_like(y)
            out[0] = 1.0
            return np.moveaxis(out, -1, axis)
        return np.moveaxis(y / total, -1, axis)
    total = y.sum(axis=-1, keepdims=True)
    out = y / np.where(total > eps, total, 1.0)
    empty = np.squeeze(total <= eps, axis=-1)
    if np.any(empty):
        out = np.array(out, copy=True)
        out[empty, :] = 0.0
        out[empty, 0] = 1.0
    return np.moveaxis(out, -1, axis)


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


def trapezoid_integral(y: Array, x: Array) -> float:
    """Compatibility wrapper for NumPy trapezoidal integration."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def trapezoid(y: Array, x: Array) -> float:
    """Backward-compatible alias for :func:`trapezoid_integral`."""

    return trapezoid_integral(y, x)


def as_time_function(t: Array, values: Array) -> Callable[[float], Array]:
    """Return a vector-valued interpolation function on a time grid."""

    return interp1d(t, values, axis=0, bounds_error=False, fill_value="extrapolate")


def solve_ode_grid(
    rhs: Callable[[float, Array], Array],
    y_boundary: Array,
    t: Array,
    *,
    backward: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Array:
    """Solve an ODE on grid ``t`` and return rows ordered from start to end.

    If ``backward=True``, ``y_boundary`` is interpreted as the terminal value at
    ``t[-1]``.  This helper is used by forward-backward sweep examples so the
    forward state and backward costate integrations share one implementation.
    """

    time_grid = np.asarray(t, dtype=np.float64)
    span = (time_grid[-1], time_grid[0]) if backward else (time_grid[0], time_grid[-1])
    eval_grid = time_grid[::-1] if backward else time_grid
    sol = solve_ivp(rhs, span, np.asarray(y_boundary, dtype=np.float64), t_eval=eval_grid, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    out = sol.y.T
    return out[::-1] if backward else out
