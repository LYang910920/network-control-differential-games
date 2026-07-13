"""Compatibility imports for the pre-0.2 Torch helper module.

New code should import neural blocks from :mod:`cybercontrol.nn`, PINN
operators from :mod:`cybercontrol.pinn`, and runtime configuration from
:mod:`cybercontrol.experiments`.
"""

from __future__ import annotations

from .experiments import configure_torch
from .nn import BoundedControlNet, CompartmentStateNet, MLP, SimplexStateNet
from .pinn import positive, time_derivative
from .torch_numerics import project_compartments_torch, rk4_step_torch

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
