"""Compatibility imports for the pre-0.2 Torch helper module.

New code should import neural blocks from :mod:`cybercontrol.nn`, PINN
operators from :mod:`cybercontrol.pinn`, and runtime configuration from
:mod:`cybercontrol.experiments`.
"""

from __future__ import annotations

from .experiments import RuntimeConfig, resolve_torch_runtime
from .nn import MLP
from .pinn import positive, time_derivative

__all__ = [
    "BoundedControlNet",
    "CompartmentStateNet",
    "MLP",
    "SimplexStateNet",
    "configure_torch",
    "positive",
    "project_compartments_torch",
    "require_torch",
    "rk4_step_torch",
    "time_derivative",
]


def require_torch():
    """Import PyTorch with a clear optional-dependency message."""

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install the optional Torch dependency with `pip install -e .[torch]`."
        ) from exc
    return torch, nn


def configure_torch(
    seed: int | None = None,
    device: str | None = None,
    dtype: str = "float32",
    threads: int | None = 1,
):
    """Compatibility adapter returning ``(torch, device, dtype)``."""

    runtime = RuntimeConfig(
        seed=0 if seed is None else int(seed),
        device="auto" if device is None else str(device),
        dtype=dtype,
        threads=threads,
    )
    return resolve_torch_runtime(runtime)


def project_compartments_torch(values, axis: int = -1, eps: float = 1e-12):
    """Project nonnegative Torch values onto a compartment simplex."""

    torch, _ = require_torch()
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
            empty.squeeze(axis), torch.ones_like(projected[index]), projected[index]
        )
    return projected


class CompartmentStateNet(require_torch()[1].Module):
    """Time-to-state network normalized over ``compartments`` outputs."""

    def __init__(self, compartments: int = 3, width: int = 64, depth: int = 2):
        super().__init__()
        self.raw = MLP(1, compartments, width=width, depth=depth)

    def forward(self, time):
        torch, _ = require_torch()
        return torch.softmax(self.raw(time), dim=-1)


class SimplexStateNet(CompartmentStateNet):
    """Three-compartment state network retained for SIR/SIPS examples."""

    def __init__(self, width: int = 64, depth: int = 2):
        super().__init__(compartments=3, width=width, depth=depth)


class BoundedControlNet(require_torch()[1].Module):
    """Time-to-control network with values in ``[0, umax]``."""

    def __init__(self, width: int = 64, depth: int = 2, umax: float = 1.0):
        super().__init__()
        self.raw = MLP(1, 1, width=width, depth=depth)
        self.umax = float(umax)

    def forward(self, time):
        torch, _ = require_torch()
        return self.umax * torch.sigmoid(self.raw(time))


def rk4_step_torch(state, dt, rhs, project_simplex: bool = True):
    """Torch RK4 step for differentiable trajectory generation."""

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    result = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    if project_simplex:
        result = project_compartments_torch(result)
    return result
