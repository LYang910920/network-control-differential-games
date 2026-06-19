"""Shared utilities for the cyber-control tutorial repository family.

The package keeps reusable mechanics in one place while each repository keeps
its own teaching scripts and narrative.  Public modules are intentionally small:

- :mod:`cybercontrol.numerics` for projections and ODE integration.
- :mod:`cybercontrol.models` for malware/hybrid dynamics and jump maps.
- :mod:`cybercontrol.torch_utils` for PINN/PIDL neural-network helpers.
- :mod:`cybercontrol.io` for reproducible outputs.
- :mod:`cybercontrol.plotting` for lightweight figure helpers.
"""

from .numerics import project_simplex, project_simplex3, rk4_integrate, rk4_step, trapezoid
from .models import (
    HybridParams,
    MalwareParams,
    controlled_sir_rhs,
    hybrid_rhs,
    isolation_jump,
)

__all__ = [
    "HybridParams",
    "MalwareParams",
    "controlled_sir_rhs",
    "hybrid_rhs",
    "isolation_jump",
    "project_simplex",
    "project_simplex3",
    "rk4_integrate",
    "rk4_step",
    "trapezoid",
]
