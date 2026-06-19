"""Shared utilities for the cyber-control guide repository family.

The package keeps reusable mechanics in one place while each repository keeps
its own teaching scripts and narrative.  Public modules are intentionally small:

- :mod:`cybercontrol.numerics` for projections and ODE integration.
- :mod:`cybercontrol.models` for malware/hybrid dynamics and jump maps.
- :mod:`cybercontrol.torch_utils` for PINN/PIDL neural-network helpers.
- :mod:`cybercontrol.io` for reproducible outputs.
- :mod:`cybercontrol.plotting` for lightweight figure helpers.
"""

from .numerics import (
    as_time_function,
    project_box,
    project_simplex,
    project_simplex3,
    rk4_integrate,
    rk4_step,
    solve_ode_grid,
    trapezoid,
    trapezoid_integral,
)
from .models import (
    HybridParams,
    MalwareParams,
    controlled_sir_rhs,
    hybrid_rhs,
    isolation_jump,
)
from .io import require_outputs, set_seed, write_csv, write_json
from .plotting import apply_clean_axes, clean_axes, plot_time_series

__all__ = [
    "HybridParams",
    "MalwareParams",
    "controlled_sir_rhs",
    "hybrid_rhs",
    "isolation_jump",
    "apply_clean_axes",
    "as_time_function",
    "clean_axes",
    "plot_time_series",
    "project_box",
    "project_simplex",
    "project_simplex3",
    "require_outputs",
    "rk4_integrate",
    "rk4_step",
    "set_seed",
    "solve_ode_grid",
    "trapezoid",
    "trapezoid_integral",
    "write_csv",
    "write_json",
]
