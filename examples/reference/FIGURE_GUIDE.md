# Reference Figure Guide

This guide explains how to read the figures produced by the reference repository smoke runner.

## Axis Types

| X-axis | Used in | Meaning |
| --- | --- | --- |
| iteration | payoff/profit panels and `*_payoff.csv` | Numerical optimization or game-strategy iteration. These curves are used as smoke-level convergence diagnostics. |
| time | state and control panels, `*_timeseries.csv` | Simulated model time. These curves show state evolution and computed intervention or game strategies. |

## Figure Types

Each reference figure has three panels:

| Panel | X-axis | What it shows |
| --- | --- | --- |
| Payoff/profit | iteration | Whether the forward-backward or policy-update loop produces finite, trackable objective values. |
| State trajectories | time | How the modeled network state evolves under the computed control/game strategy. |
| Control/strategy trajectories | time | How the computed intervention, impulse, attack, or defense policy changes over time. |

## Repository-specific Interpretation

| Figure | What to look for |
| --- | --- |
| `opinion_malware.png` | Payoff over iterations; mean malware and opinion states over time; impulse-control schedules `u1(t)` and `u2(t)`. |
| `propaganda_war.png` | Red/blue game payoffs over iterations; red/blue degree-level states over time; red/blue strategy variables over time. |
| `propaganda_tcss.png` | Profit over policy iterations; awareness/unawareness/removed state averages over time; impulse-control schedules `ca(t)` and `cu(t)`. |
| `reference_repo_contact_sheet.png` | A visual index for comparing all three reference smoke runs. |

The reference smoke runs are not full paper-scale reproductions. They use small local graphs and short horizons to check that the reference code imports, runs, exports CSV summaries, and produces interpretable figures.
