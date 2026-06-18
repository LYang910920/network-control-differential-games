# Reference Figure Guide

This guide explains how to read the figures produced by the reference repository smoke runner.

## Axis Types

| X-axis | Used in | Meaning |
| --- | --- | --- |
| iteration | payoff/profit panels and `*_payoff.csv` | Numerical optimization or game-strategy iteration. These curves are used as smoke-level convergence diagnostics. |
| time | state and control panels, `*_timeseries.csv` | Simulated model time. These curves show state evolution and computed intervention or game strategies. |

## Plotting Convention

- Continuous controls or strategies are time-indexed curves sampled on the simulation grid. Projected strategies can include flat segments at bounds, but they still enter the ODE between event times.
- Impulse controls or strategies act only at discrete event times. They are drawn as vertical event lines, not connected curves.
- Hybrid control combines continuous and impulse interventions; `PropagandaWar_TIFS_2024_Code` is the hybrid/impulsive differential-game smoke run.
- State labels specify the aggregation level: node mean over all nodes or degree-weighted mean over degree classes. In state panels, vertical markers show when impulses are applied; state trajectories may jump or change direction at those times. The smoke-run parameters are chosen to make both continuous-strategy variation and impulse-driven state changes visible.

## Figure Types

Each reference figure has three panels:

| Panel | X-axis | What it shows |
| --- | --- | --- |
| Payoff/profit | iteration | Whether the forward-backward or policy-update loop produces finite, trackable objective values. |
| State trajectories | time | How the modeled network state evolves under the computed control/game strategy. |
| Control/strategy trajectories | time | Continuous strategies are curves; impulse strategies are vertical event lines; hybrid examples show both. |

## Repository-specific Interpretation

| Figure | Model class | What to look for |
| --- | --- | --- |
| `opinion_malware.png` | Node-level optimal impulse control | Payoff over iterations; node-mean malware and opinion states with impulse-time markers; `u1` and `u2` shown only as discrete impulse magnitudes. |
| `propaganda_war.png` | Degree-level hybrid/impulsive differential game | Red/blue payoffs; degree-weighted state means with pulse markers; continuous `ur`/`ub` strategies plus discrete `vr`/`vb` impulses. |
| `propaganda_tcss.png` | Node-level optimal impulse control with awareness | Profit over policy iterations; node-mean awareness/unawareness/removed states over all nodes with pulse markers; `ca` and `cu` shown only as discrete impulse magnitudes. The smoke-run parameters are chosen to make impulse-induced jumps visible. |
| `reference_convergence.png` | Mixed convergence diagnostics | Absolute payoff/profit or strategy changes across smoke-run iterations. These curves should decrease or settle before the final strategy is interpreted. |
| `opinion_malware_baseline_comparison.png` | Node-level impulse-control baseline comparison | Computed impulse policy is compared with no-impulse and 75 random impulse policies for the same model. |
| `propaganda_war_baseline_comparison.png` | Degree-level hybrid-game baseline comparison | Two unilateral panels: fixed computed red strategy with varied blue, and fixed computed blue strategy with varied red. Each panel includes zero and random strategies. |
| `propaganda_tcss_baseline_comparison.png` | Node-level impulse-control baseline comparison | Computed impulse policy is compared with no-impulse and 75 random impulse policies for the same model. |
| `parameter_summary.csv` | Smoke-run parameter table | Concrete values for time horizon, step size, propagation rates, impulse event indices, control bounds, and baseline counts. |
| `reference_repo_contact_sheet.png` | Mixed overview | A visual index for comparing all three reference smoke runs and their model-specific baseline comparisons. |

The reference smoke runs are not full paper-scale reproductions. They use small local graphs and short horizons to check that the reference code imports, runs, exports CSV summaries, and produces interpretable figures.
