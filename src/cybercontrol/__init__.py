"""Shared utilities for the cyber-control guide repository family.

The package keeps reusable mechanics in one place while each repository keeps
its own tutorial scripts and narrative.  Public modules stay small:

- :mod:`cybercontrol.numerics` for projections and ODE integration.
- :mod:`cybercontrol.models` for malware/hybrid dynamics and jump maps.
- :mod:`cybercontrol.network_models` for node-level SIPS/SIPRS graph models.
- :mod:`cybercontrol.torch_utils` for PINN/PIDL neural-network helpers.
- :mod:`cybercontrol.io` for reproducible outputs.
- :mod:`cybercontrol.plotting` for lightweight figure helpers.
- :mod:`cybercontrol.diagnostics` for training-diagnostic terms and captions.
"""

from .numerics import (
    as_time_function,
    project_box,
    project_compartments,
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
from .network_models import (
    NodeSIPRSParams,
    graph_pressure_numpy,
    graph_pressure_torch,
    node_sips_rhs_numpy,
    node_sips_rhs_torch,
    node_siprs_rhs_numpy,
    node_siprs_rhs_torch,
    node_siprs_transition_rates,
    normalize_adjacency,
    sample_node_siprs_step,
)
from .io import require_outputs, set_seed, write_csv, write_json
from .plotting import apply_clean_axes, clean_axes, plot_time_series
from .diagnostics import (
    COMMON_DIAGNOSTIC_TERMS,
    DiagnosticTerm,
    add_caption,
    diagnostic_terms_for,
    rolling_mean,
    write_diagnostic_glossary,
)

__all__ = [
    "HybridParams",
    "MalwareParams",
    "controlled_sir_rhs",
    "hybrid_rhs",
    "isolation_jump",
    "NodeSIPRSParams",
    "COMMON_DIAGNOSTIC_TERMS",
    "DiagnosticTerm",
    "add_caption",
    "apply_clean_axes",
    "as_time_function",
    "clean_axes",
    "diagnostic_terms_for",
    "plot_time_series",
    "project_box",
    "project_compartments",
    "project_simplex",
    "project_simplex3",
    "require_outputs",
    "rk4_integrate",
    "rk4_step",
    "rolling_mean",
    "set_seed",
    "graph_pressure_numpy",
    "graph_pressure_torch",
    "node_sips_rhs_numpy",
    "node_sips_rhs_torch",
    "node_siprs_rhs_numpy",
    "node_siprs_rhs_torch",
    "node_siprs_transition_rates",
    "normalize_adjacency",
    "solve_ode_grid",
    "sample_node_siprs_step",
    "trapezoid",
    "trapezoid_integral",
    "write_csv",
    "write_diagnostic_glossary",
    "write_json",
]
