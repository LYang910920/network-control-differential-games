# Reference Figure Guide

This guide explains how to read the figures produced by the reference repository smoke runner.

## Axis Types

| X-axis | Used in | Meaning |
| --- | --- | --- |
| iteration | payoff/profit panels and `*_payoff.csv` | Numerical optimization or game-strategy iteration. These curves are used as smoke-level convergence diagnostics. |
| time | state and control panels, `*_timeseries.csv` | Simulated model time. These curves show state evolution and computed intervention or game strategies. |

## Plotting Convention

- Curves represent continuous states, controls, or game strategies.
- Vertical lines represent discrete impulse or pulse interventions.
- In state panels, vertical markers show when impulses are applied; state trajectories may jump or change direction at those times.

## Figure Types

Each reference figure has three panels:

| Panel | X-axis | What it shows |
| --- | --- | --- |
| Payoff/profit | iteration | Whether the forward-backward or policy-update loop produces finite, trackable objective values. |
| State trajectories | time | How the modeled network state evolves under the computed control/game strategy. |
| Control/strategy trajectories | time | Continuous strategies are curves; impulse strategies are vertical event lines. |

## Repository-specific Interpretation

| Figure | Model class | What to look for |
| --- | --- | --- |
| `opinion_malware.png` | Node-level optimal impulse control | Payoff over iterations; mean malware and opinion states with impulse-time markers; `u1` and `u2` shown only as discrete impulse magnitudes. |
| `propaganda_war.png` | Degree-level hybrid/impulsive differential game | Red/blue payoffs; degree-level state trajectories with pulse markers; continuous `ur`/`ub` strategies plus discrete `vr`/`vb` impulses. |
| `propaganda_tcss.png` | Node-level optimal impulse control with awareness | Profit over policy iterations; awareness/unawareness/removed state averages with pulse markers; `ca` and `cu` shown only as discrete impulse magnitudes. |
| `reference_repo_contact_sheet.png` | Mixed overview | A visual index for comparing all three reference smoke runs. |

The reference smoke runs are not full paper-scale reproductions. They use small local graphs and short horizons to check that the reference code imports, runs, exports CSV summaries, and produces interpretable figures.
