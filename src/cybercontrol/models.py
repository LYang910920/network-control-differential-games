"""Shared malware, propagation, sampled-flow, and impulse model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .numerics import project_simplex3

Array = np.ndarray


@dataclass
class MalwareParams:
    """Parameters for the basic controlled SIR malware model."""

    beta: float = 0.65
    gamma: float = 0.05
    omega: float = 0.01


def controlled_sir_rhs(x: Array, u_patch: float, u_clean: float, p: MalwareParams) -> Array:
    """Continuous-control SIR malware dynamics for ``x = [S, I, R]``.

    ``u_patch`` moves vulnerable devices into the protected/recovered class and
    ``u_clean`` cleans compromised devices into the same class.
    """

    S, I, R = x[:3]
    dS = -p.beta * S * I - u_patch * S + p.omega * R
    dI = p.beta * S * I - (p.gamma + u_clean) * I
    dR = u_patch * S + (p.gamma + u_clean) * I - p.omega * R
    return np.array([dS, dI, dR], dtype=np.float64)


@dataclass
class SampledSIRParams:
    """Parameters for sampled-data SIR flow with interval action effects.

    The state is ``x=[S,I,R]``.  Actions are decoded outside this function and
    held constant over one decision interval.  Deception is an action-dependent
    reduction of the effective infection rate during that interval; it is not a
    fourth state variable.
    """

    beta0: float = 0.65
    gamma: float = 0.05
    omega: float = 0.01
    chi: float = 0.70


def sampled_sir_flow_rhs(x: Array, dpar: Dict[str, float], apar: Dict[str, float], p: SampledSIRParams) -> Array:
    """Continuous flow for a sampled-data SIR malware model under ZOH.

    The discrete action has already been decoded into rates.  Instantaneous
    jumps, such as isolation, are applied before calling this RHS and are not
    hidden inside the flow.
    Deception is an action effect over the current interval:
    ``beta_eff = beta_attack * max(0, 1 - chi * deceive)``.
    """

    S, I, R = x[:3]
    beta = apar.get("beta", p.beta0)
    clean = dpar.get("clean", 0.0) * apar.get("stealth_factor", 1.0)
    patch = dpar.get("patch", 0.0)
    effective_beta = beta * max(0.0, 1.0 - p.chi * dpar.get("deceive", 0.0))
    dS = -effective_beta * S * I - patch * S + p.omega * R
    dI = effective_beta * S * I - (p.gamma + clean) * I
    dR = patch * S + (p.gamma + clean) * I - p.omega * R
    return np.array([dS, dI, dR], dtype=np.float64)


def isolation_jump(x: Array, isolation_rate: float) -> Array:
    """Apply an impulse that immediately moves infected mass ``I`` into ``R``."""

    y = np.asarray(x, dtype=np.float64).copy()
    removed = min(max(isolation_rate, 0.0) * y[1], y[1])
    y[1] -= removed
    y[2] += removed
    return project_simplex3(y)


def controlled_sir_rhs_torch(x, u_patch, beta, gamma, u_clean=0.0, omega=0.0):
    """Torch version of :func:`controlled_sir_rhs`.

    ``u_patch`` moves ``S -> R`` and ``u_clean`` moves ``I -> R``.  ``omega``
    returns recovered/protected mass to susceptibility.  The default
    ``u_clean=0`` and ``omega=0`` preserve older patch-only examples,
    but the mathematical semantics now match the NumPy function.

    Imports are intentionally local so the shared package remains usable without
    PyTorch for the pure ODE examples.
    """

    import torch

    S, I, R = x[..., 0:1], x[..., 1:2], x[..., 2:3]
    if not torch.is_tensor(u_clean):
        u_clean = torch.as_tensor(u_clean, dtype=x.dtype, device=x.device)
    if not torch.is_tensor(omega):
        omega = torch.as_tensor(omega, dtype=x.dtype, device=x.device)
    return torch.cat(
        [
            -beta * S * I - u_patch * S + omega * R,
            beta * S * I - (gamma + u_clean) * I,
            u_patch * S + (gamma + u_clean) * I - omega * R,
        ],
        dim=-1,
    )


def sir_patch_only_rhs_torch(x, u_patch, beta, gamma):
    """Explicit patch-only Torch SIR RHS retained for older examples."""

    return controlled_sir_rhs_torch(x, u_patch, beta, gamma, u_clean=0.0, omega=0.0)


def sir_rhs_torch(x, beta, gamma):
    """Torch SIR RHS without control for synthetic trajectory generation."""

    import torch

    S, I = x[..., 0], x[..., 1]
    return torch.stack([-beta * S * I, beta * S * I - gamma * I, gamma * I], dim=-1)
