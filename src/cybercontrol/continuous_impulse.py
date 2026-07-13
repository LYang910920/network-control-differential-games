# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Continuous-flow control combined with explicit node-state reset events."""

from __future__ import annotations

import numpy as np

from .fbs import TimeSeries, node_rhs
from .foundation_profiles import CONTINUOUS_IMPULSE_PROFILE, ContinuousImpulseProfile
from .graph_data import high_degree_nodes
from .numerics import project_box, solve_ode_grid


def simulate_continuous_impulse(
    adjacency: np.ndarray,
    profile: ContinuousImpulseProfile = CONTINUOUS_IMPULSE_PROFILE,
) -> TimeSeries:
    """Simulate time-varying flow control with resets at declared event times.

    The model integrates the node-SIS flow between events. At each
    ``profile.impulse_times`` value, the selected high-degree nodes jump by
    ``x_i(tau+) = (1 - impulse_fraction) x_i(tau-)``. Running control and reset
    magnitudes remain separate in the returned record. This is a transparent
    simulation template, not an impulse-PMP solver.
    """

    horizon = profile.horizon
    beta = profile.beta
    delta = profile.delta
    impulse_fraction = profile.impulse_fraction
    nodes = adjacency.shape[0]
    controlled = high_degree_nodes(
        adjacency,
        max(1, int(np.ceil(profile.controlled_node_fraction * nodes))),
    )

    def continuous_level(time: float) -> float:
        phase = time / max(horizon, 1e-12)
        level = profile.continuous_base + profile.continuous_ramp * phase
        level += profile.continuous_wave * np.sin(2.0 * np.pi * phase)
        return float(project_box(level, profile.continuous_lower, profile.continuous_upper))

    def controlled_rhs(time: float, state: np.ndarray) -> np.ndarray:
        control = np.zeros(nodes)
        control[controlled] = continuous_level(time)
        return node_rhs(state, control, adjacency, beta, delta)

    state = np.full(nodes, 0.03)
    state[high_degree_nodes(adjacency, 2)] = 0.15
    impulse_times = np.asarray(profile.impulse_times, dtype=float)
    impulse_times = np.unique(
        np.sort(impulse_times[(impulse_times > 0.0) & (impulse_times < horizon)])
    )

    all_times: list[float] = []
    all_states: list[np.ndarray] = []
    segment_bounds = [0.0, *impulse_times.tolist(), horizon]
    for start_time, end_time in zip(segment_bounds[:-1], segment_bounds[1:]):
        grid = np.linspace(
            start_time,
            end_time,
            max(2, int(30 * (end_time - start_time)) + 1),
        )
        segment = project_box(solve_ode_grid(controlled_rhs, state, grid))
        first = 0 if not all_times else 1
        all_times.extend(grid[first:])
        all_states.extend(segment[first:])
        state = segment[-1].copy()
        if np.any(np.isclose(end_time, impulse_times)):
            state[controlled] *= 1.0 - impulse_fraction
            all_times.append(end_time)
            all_states.append(state.copy())

    times = np.asarray(all_times, dtype=float)
    continuous_control = np.asarray([continuous_level(time) for time in times])
    return TimeSeries(
        t=times,
        x=np.vstack(all_states),
        controls={
            "controlled_nodes": controlled,
            "continuous_control": continuous_control,
            "impulse_times": impulse_times,
            "impulse_heights": np.full(len(impulse_times), impulse_fraction),
        },
        value={
            "continuous_control_min": float(np.min(continuous_control)),
            "continuous_control_max": float(np.max(continuous_control)),
            "impulse_fraction": impulse_fraction,
            "controlled_count": float(len(controlled)),
        },
    )
