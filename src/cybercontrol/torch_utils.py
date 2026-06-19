"""PyTorch helpers for RL, PINN, PIDL, and PMP-informed PINN examples."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np


def require_torch():
    """Import PyTorch with a clear error message for optional users."""

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "This example requires PyTorch. Install the optional dependency with "
            "`pip install -e .[torch]` in the foundation repo, or install `torch`."
        ) from exc
    return torch, nn


def configure_torch(seed: int | None = None, device: str | None = None, dtype: str = "float32", threads: int = 1):
    """Set seeds/thread count and return ``(torch, device, dtype)``."""

    torch, _ = require_torch()
    if threads is not None and threads > 0:
        torch.set_num_threads(int(threads))
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_dtype = getattr(torch, dtype)
    torch.set_default_dtype(resolved_dtype)
    return torch, resolved_device, resolved_dtype


class MLP(require_torch()[1].Module):
    """Small fully connected network used across RL/PINN examples."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 2,
        activation: Callable | None = None,
        final_activation: Callable | None = None,
    ):
        torch, nn = require_torch()
        super().__init__()
        if activation is None:
            activation = nn.Tanh
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, width), activation()]
            last = width
        layers += [nn.Linear(last, out_dim)]
        if final_activation is not None:
            layers += [final_activation()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimplexStateNet(require_torch()[1].Module):
    """Time-to-state network whose outputs live on the SIR simplex."""

    def __init__(self, width: int = 64, depth: int = 2):
        super().__init__()
        self.raw = MLP(1, 3, width=width, depth=depth)

    def forward(self, t):
        torch, _ = require_torch()
        return torch.softmax(self.raw(t), dim=-1)


class BoundedControlNet(require_torch()[1].Module):
    """Time-to-control network with output in ``[0, umax]``."""

    def __init__(self, width: int = 64, depth: int = 2, umax: float = 1.0):
        super().__init__()
        self.raw = MLP(1, 1, width=width, depth=depth)
        self.umax = umax

    def forward(self, t):
        torch, _ = require_torch()
        return self.umax * torch.sigmoid(self.raw(t))


def positive(raw, eps: float = 1e-6):
    """Positive parameter transform used for rates such as beta and gamma."""

    torch, _ = require_torch()
    return torch.nn.functional.softplus(raw) + eps


def time_derivative(y, t):
    """Return ``dy/dt`` for every output channel of a time network."""

    torch, _ = require_torch()
    return torch.cat(
        [torch.autograd.grad(y[:, j].sum(), t, create_graph=True)[0] for j in range(y.shape[1])],
        dim=1,
    )


def rk4_step_torch(x, dt, rhs, project_simplex: bool = True):
    """Torch RK4 step for synthetic PINN/PIDL trajectories."""

    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    y = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    if project_simplex:
        torch, _ = require_torch()
        y = torch.clamp(y, 0.0, 1.0)
        y = y / y.sum(dim=-1, keepdim=True)
    return y
