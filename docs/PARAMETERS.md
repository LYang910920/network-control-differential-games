# Parameter Reference

Use this page before changing equations or solver loops. It makes the smoke-run parameters visible in one place.

## Quick Commands

| Need | Command |
|---|---|
| Print tutorial model profiles | `python examples/lecture/code/model_profiles.py` |
| Run a fast tutorial check | `python run_all.py --skip-reference --skip-scalability` |
| Run reference-only smoke tests | `python run_all.py --skip-tutorial` |
| Find exact values after a run | open `parameter_summary.csv` in each output folder |

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
| Simple FBS damping | `0.35` | `examples/lecture/code/model_profiles.py` |
| Simple FBS tolerance | `1e-4` | `examples/lecture/code/model_profiles.py` |
| Simple FBS max iterations | `50` | `examples/lecture/code/model_profiles.py` |
| Companion runner time grid | `45` by default | `python run_all.py --tutorial-steps N` |
| Random baseline count | `75` per model/panel | `examples/common_diagnostics.py` |
| Random baseline seed | `20260617` | `examples/common_diagnostics.py` |
| Scalability node sizes | `100, 200, ..., 2000` | `--scalability-sizes` |
| Scalability repeats | `3` by default | `--scalability-repeats` |

## Neural Hyperparameters

This repository does not train neural networks. It focuses on optimal control, differential games, impulse/hybrid control, FBS convergence, and reference-code smoke runs. Neural training hyperparameters are made explicit in the companion Note 1 and Note 2 repositories.

## Reading State Labels

| Label type | Meaning |
|---|---|
| Degree-weighted mean | average over degree classes weighted by `k P(k)` |
| Selected degree class `k` | one degree group, not one node |
| Node mean | average over the reduced node set |
| Node max | maximum state over the reduced node set |
| Impulse marker | discrete event time, not a continuous control curve |
