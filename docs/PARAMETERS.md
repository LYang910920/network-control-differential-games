# Parameter Reference

Use this page before changing equations or solver loops. It makes the smoke-run parameters visible in one place.

## Quick Commands

| Need | Command |
|---|---|
| Print foundation model profiles | `python examples/foundations/code/model_profiles.py` |
| Run a fast foundation check | `python run_all.py --skip-reference --skip-scalability` |
| Run reference-only smoke tests | `python run_all.py --skip-foundations` |
| Find exact values after a run | open `parameter_summary.csv` in each output folder |

## Terms Used In This Repository

| Term | Meaning |
|---|---|
| `trajectory` | A time-indexed state/control/strategy path produced by solving or simulating one model. |
| `baseline` | A comparison method for the same model, such as no control, constant control, random smooth control, no impulse, or unilateral strategy variation. |
| `random baseline count` | Number of random policies or strategies sampled for a baseline panel. These are sanity checks, not exhaustive optimality certificates. |
| `nominal parameter` | A parameter value assumed when designing a controller or baseline; use it when comparing a design-time model with a changed simulation or stress-test setting. |
| `stress test` | A deliberately harder or larger run used to check behavior under scaling, parameter mismatch, or changed network size. It is separate from a formal theorem. |
| `rollout` | A forward simulation under a fixed control, strategy, or policy. In the foundation examples, the resulting path is usually called a trajectory. |
| `PMP` | Pontryagin's maximum principle: necessary conditions for a continuous-time optimal-control or differential-game candidate. |
| `FBSM` | Forward-backward sweep method: numerically alternate state integration, costate integration, and control/strategy update until the update change is small. |
| `degree-level` | One state per observed degree class. Use it when degree distribution is the main network summary. |
| `node-level` | One state per graph node. Use it when adjacency, local interventions, or node-specific outputs matter. |
| `paper-level extension` | A research-ready adaptation with explicit model equations, parameters, baselines, multiple seeds or sensitivity checks, figure captions, and license/data notes. |
| `diagnostic` | A plotted/logged check of numerical behavior, such as FBSM control-update change, state conservation, runtime scaling, or baseline comparison. Shared helper definitions live in `cybercontrol.diagnostics`. |

## Tutorial Model Parameters

| Model | Control/game type | Horizon | Propagation/recovery | Bounds and costs |
|---|---|---:|---|---|
| Simple degree-k control | continuous optimal control | `T=14` | `beta=0.65`, `delta=0.18` | `0 <= u_k(t) <= 1.2`, infection weight `3.0`, control weight `2.5` |
| Degree-level control | continuous optimal control | `T=14` | `beta=0.65`, `delta=0.18` | `0 <= u_k(t) <= 1.2`, state weight `3.0`, control weight `2.5` |
| Node-level control | continuous optimal control | `T=12` | `beta=0.90`, `delta=0.16` | `0 <= u_i(t) <= 1.2`, state weight `3.0`, control weight `2.2` |
| Degree-level game | continuous differential game | `T=14` | `beta=0.60`, `delta=0.15` | attack/defense in `[0, 1.2]`, attack cost `3.0`, defense cost `4.0` |
| Node-level game | continuous differential game | `T=12` | `beta=0.95`, `delta=0.15` | attack/defense in `[0, 1.2]`, attack cost `4.0`, defense cost `4.5` |
| Hybrid impulse model | continuous plus impulse control | `T=12` | `beta=0.95`, `delta=0.15` | continuous control in `[0.10, 0.52]`, impulses at `t=3,6,9`, impulse fraction `0.55` |

## Solver And Baseline Parameters

| Item | Value | Where to change |
|---|---|---|
| Simple FBS damping | `0.35` | `examples/foundations/code/model_profiles.py` |
| Simple FBS tolerance | `1e-4` | `examples/foundations/code/model_profiles.py` |
| Simple FBS max iterations | `50` | `examples/foundations/code/model_profiles.py` |
| Foundation runner time grid | `45` by default | `python run_all.py --foundation-steps N` |
| Random baseline count | `75` per model/panel | `examples/common_diagnostics.py` |
| Random baseline seed | `20260617` | `examples/common_diagnostics.py` |
| Paired degree/node scalability node sizes | `100, 1000, 10000, 100000, 1000000` | `--scalability-sizes` |
| Paired degree/node scalability repeats | `1` by default | `--scalability-repeats` |
| Paired degree/node scalability model | same normalized SIS epidemic-control problem for both levels; contact pressure is scaled by each graph's maximum degree | `examples/foundations/code/scalability_analysis.py` |
| Paired degree/node scalability solver/FBS | fixed-grid RK4, sparse node adjacency, `T=8`, `beta=1.20`, `delta=0.35`, `60` intervals, `max_iter=80`, `tol=1e-4` | `--scalability-steps`, `--scalability-iterations` |
| Sparse node-only scalability node sizes | `1000, 2000, ..., 10000` | `--include-node-scalability --node-scalability-sizes ...` |
| Sparse node-level scalability repeats | `1` by default | `--node-scalability-repeats` |
| Sparse node-level scalability time grid | `25` intervals over `T=8` | `--node-scalability-steps` |
| Sparse node-only scalability solver/FBS | sparse fixed-grid RK4, `max_iter=50`, `tol=1e-4`, damping `0.25` | `examples/foundations/code/scalability_analysis.py` |
| Sparse node-level scalability rates/costs | `beta=0.55`, `delta=0.20`, state weight `1.0`, control weight `8.0`, `u_max=0.8` | `NODE_SCALABILITY_DEFAULTS` |

## Neural Hyperparameters

This repository does not train neural networks. It focuses on optimal control, differential games, impulse/hybrid control, FBS convergence, scalability checks, and reference-code smoke runs. If you add a neural experiment later, record the network width/depth, optimizer, learning rate, batch or collocation size, random seed, and stopping rule next to that experiment.

## Reading State Labels

| Label type | Meaning |
|---|---|
| Degree-weighted mean | average over degree classes weighted by `k P(k)` |
| Selected degree class `k` | one degree group, not one node |
| Node mean | average over the reduced node set |
| Node max | maximum state over the reduced node set |
| Impulse marker | discrete event time, not a continuous control curve |
