# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Named smoke-run parameter profiles for the tutorial examples.

Keeping these defaults in one small module makes the numerical examples easier
to audit: time horizons, propagation rates, control bounds, and impulse settings
are visible without reading each solver loop.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DegreeControlProfile:
    """Readable defaults for the minimal degree-k SIS control example."""

    horizon: float = 14.0
    beta: float = 0.65
    delta: float = 0.18
    infection_weight: float = 3.0
    control_weight: float = 2.5
    u_max: float = 1.2
    initial_low: float = 0.02
    initial_degree_scale: float = 0.08
    initial_cap: float = 0.18
    damping: float = 0.35
    tolerance: float = 1e-4
    max_iterations: int = 50


@dataclass(frozen=True)
class ContinuousControlProfile:
    """Main knobs for one continuous optimal-control smoke model."""

    horizon: float
    beta: float
    delta: float
    state_weight: float
    control_weight: float
    control_max: float
    label: str


@dataclass(frozen=True)
class ContinuousGameProfile:
    """Main knobs for one continuous attacker-defender game smoke model."""

    horizon: float
    beta: float
    delta: float
    reward_attacker: float
    loss_defender: float
    cost_attack: float
    cost_defense: float
    attack_max: float
    defense_max: float
    label: str


@dataclass(frozen=True)
class HybridImpulseProfile:
    """Main knobs for the transparent hybrid continuous-plus-impulse example."""

    horizon: float = 12.0
    beta: float = 0.95
    delta: float = 0.15
    impulse_fraction: float = 0.55
    impulse_times: tuple[float, ...] = (3.0, 6.0, 9.0)
    controlled_node_fraction: float = 0.25
    continuous_base: float = 0.16
    continuous_ramp: float = 0.24
    continuous_wave: float = 0.08
    continuous_lower: float = 0.10
    continuous_upper: float = 0.52


SIMPLE_DEGREE_CONTROL = DegreeControlProfile()

DEGREE_CONTROL_PROFILE = ContinuousControlProfile(
    horizon=14.0,
    beta=0.65,
    delta=0.18,
    state_weight=3.0,
    control_weight=2.5,
    control_max=1.2,
    label="degree-k continuous optimal control",
)
NODE_CONTROL_PROFILE = ContinuousControlProfile(
    horizon=12.0,
    beta=0.90,
    delta=0.16,
    state_weight=3.0,
    control_weight=2.2,
    control_max=1.2,
    label="node-level continuous optimal control",
)
DEGREE_GAME_PROFILE = ContinuousGameProfile(
    horizon=14.0,
    beta=0.60,
    delta=0.15,
    reward_attacker=4.0,
    loss_defender=5.0,
    cost_attack=3.0,
    cost_defense=4.0,
    attack_max=1.2,
    defense_max=1.2,
    label="degree-k continuous differential game",
)
NODE_GAME_PROFILE = ContinuousGameProfile(
    horizon=12.0,
    beta=0.95,
    delta=0.15,
    reward_attacker=4.0,
    loss_defender=5.0,
    cost_attack=4.0,
    cost_defense=4.5,
    attack_max=1.2,
    defense_max=1.2,
    label="node-level continuous differential game",
)
HYBRID_PROFILE = HybridImpulseProfile()


def describe_profiles() -> list[dict[str, str]]:
    """Return a table-friendly view of the main tutorial parameters."""
    return [
        {
            "name": "simple-degree-control",
            "type": "continuous control",
            "horizon": str(SIMPLE_DEGREE_CONTROL.horizon),
            "rates": f"beta={SIMPLE_DEGREE_CONTROL.beta}, delta={SIMPLE_DEGREE_CONTROL.delta}",
            "bounds": f"0 <= u_k(t) <= {SIMPLE_DEGREE_CONTROL.u_max}",
            "notes": f"FBS damping={SIMPLE_DEGREE_CONTROL.damping}, tol={SIMPLE_DEGREE_CONTROL.tolerance}",
        },
        {
            "name": "degree-control",
            "type": "degree-level continuous control",
            "horizon": str(DEGREE_CONTROL_PROFILE.horizon),
            "rates": f"beta={DEGREE_CONTROL_PROFILE.beta}, delta={DEGREE_CONTROL_PROFILE.delta}",
            "bounds": f"0 <= u_k(t) <= {DEGREE_CONTROL_PROFILE.control_max}",
            "notes": "state labels are degree-weighted mean and selected k classes",
        },
        {
            "name": "node-control",
            "type": "node-level continuous control",
            "horizon": str(NODE_CONTROL_PROFILE.horizon),
            "rates": f"beta={NODE_CONTROL_PROFILE.beta}, delta={NODE_CONTROL_PROFILE.delta}",
            "bounds": f"0 <= u_i(t) <= {NODE_CONTROL_PROFILE.control_max}",
            "notes": "state labels are node mean and node max",
        },
        {
            "name": "degree-game",
            "type": "degree-level continuous differential game",
            "horizon": str(DEGREE_GAME_PROFILE.horizon),
            "rates": f"beta={DEGREE_GAME_PROFILE.beta}, delta={DEGREE_GAME_PROFILE.delta}",
            "bounds": f"attack, defense in [0, {DEGREE_GAME_PROFILE.attack_max}]",
            "notes": f"attack cost={DEGREE_GAME_PROFILE.cost_attack}, defense cost={DEGREE_GAME_PROFILE.cost_defense}",
        },
        {
            "name": "node-game",
            "type": "node-level continuous differential game",
            "horizon": str(NODE_GAME_PROFILE.horizon),
            "rates": f"beta={NODE_GAME_PROFILE.beta}, delta={NODE_GAME_PROFILE.delta}",
            "bounds": f"attack, defense in [0, {NODE_GAME_PROFILE.attack_max}]",
            "notes": f"attack cost={NODE_GAME_PROFILE.cost_attack}, defense cost={NODE_GAME_PROFILE.cost_defense}",
        },
        {
            "name": "hybrid-impulse",
            "type": "hybrid: continuous plus impulse control",
            "horizon": str(HYBRID_PROFILE.horizon),
            "rates": f"beta={HYBRID_PROFILE.beta}, delta={HYBRID_PROFILE.delta}",
            "bounds": f"continuous in [{HYBRID_PROFILE.continuous_lower}, {HYBRID_PROFILE.continuous_upper}]",
            "notes": f"impulses at {HYBRID_PROFILE.impulse_times}, fraction={HYBRID_PROFILE.impulse_fraction}",
        },
    ]


if __name__ == "__main__":
    for row in describe_profiles():
        print(
            f"{row['name']}: type={row['type']}; horizon={row['horizon']}; "
            f"{row['rates']}; {row['bounds']}; {row['notes']}"
        )
