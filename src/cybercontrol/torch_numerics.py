"""Differentiable projections and fixed-step integration for Torch tensors."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Install the optional Torch dependency with `pip install -e .[torch]`."
    ) from exc


def project_compartments_torch(values, axis: int = -1, eps: float = 1e-12):
    """Project nonnegative values onto a compartment simplex."""

    clipped = torch.clamp(values, min=0.0)
    total = clipped.sum(dim=axis, keepdim=True)
    projected = clipped / total.clamp_min(eps)
    empty = total <= eps
    if torch.any(empty):
        projected = torch.where(empty.expand_as(projected), torch.zeros_like(projected), projected)
        index = [slice(None)] * projected.ndim
        index[axis % projected.ndim] = 0
        index = tuple(index)
        projected[index] = torch.where(
            empty.squeeze(axis),
            torch.ones_like(projected[index]),
            projected[index],
        )
    return projected


def rk4_step_torch(state, dt, rhs, project_simplex: bool = True):
    """Advance one differentiable RK4 step and optionally project compartments."""

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    result = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    if project_simplex:
        result = project_compartments_torch(result)
    return result
