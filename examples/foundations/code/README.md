# Foundation Example Code

This folder contains the maintained Python code for the foundation examples.

The common idea is always the same: convert a graph into a model state, solve a
small optimal-control or game update, and plot both the trajectory and the
diagnostics needed to trust the numerical result.

| File | Role | Main output |
| --- | --- | --- |
| `simple_degree_k_control.py` | Minimal degree-k continuous optimal-control example | `simple_*/degree_control_trajectory.png` |
| `network_control_examples.py` | Degree-level, node-level, differential-game, and hybrid/impulse examples | `companion_*/*_trajectory.png` |
| `model_profiles.py` | Named smoke-run parameters shared by the foundation scripts | `parameter_summary.csv` |
| `scalability_analysis.py` | Degree-level FBS timing on synthetic scale-free networks from 100 to 2000 nodes | `scalability_degree_sf/degree_control_scalability_100_2000.png` |
| `run_foundation_examples.py` | Rebuilds all foundation figures, CSV diagnostics, and generated indexes | `results/` |

## Parameter Quick Reference

The examples use small, visible smoke-run settings. Each output folder also writes `parameter_summary.csv` with the exact values used in that run.

| Example | Time horizon | Propagation / recovery | Control or game bounds | What the state labels mean |
| --- | ---: | --- | --- | --- |
| Simple degree-k control | `T=14`, `220` grid steps by default | `beta=0.65`, `delta=0.18` | Continuous healing control `0 <= u_k(t) <= 1.2` | Degree-weighted mean plus selected degree classes `k`. |
| Companion degree control | `T=14`, runner default `45` grid steps | `beta=0.65`, `delta=0.18` | Continuous healing control `0 <= u_k(t) <= 1.2` | Degree-weighted mean plus selected degree classes `k`. |
| Companion degree game | `T=14`, runner default `45` grid steps | `beta=0.60`, `delta=0.15` | Continuous attack/defense in `[0, 1.2]` | Degree-weighted infection mean and high-degree class. |
| Companion node control | `T=12`, at least `24` grid steps | `beta=0.90`, `delta=0.16` | Continuous node control `0 <= u_i(t) <= 1.2` | Mean and max over the reduced node set. |
| Companion node game | `T=12`, at least `24` grid steps | `beta=0.95`, `delta=0.15` | Continuous attack/defense in `[0, 1.2]` | Mean over the reduced node set. |
| Hybrid simulation | `T=12` | `beta=0.95`, `delta=0.15` | Continuous control range `0.10-0.52`; impulses at `t=3,6,9` reduce selected high-degree node states by `55%` | Mean and max over all reduced nodes; impulse times are vertical markers. |
| Degree scalability analysis | default sizes `100,200,...,2000` | Uses the degree-control profile above | Adaptive ODE/FBS, tolerance `1e-4`, default `60` max iterations | State dimension is number of observed degree classes, not node count. |
| Sparse node scalability analysis | default sizes `1000,2000,...,10000` when enabled | `beta=0.55`, `delta=0.20`, `T=8` | Fixed-grid RK4 FBS, tolerance `1e-4`, default `50` max iterations | State, costate, and control are node-indexed; report solver type with runtime. |

The root runner is usually easiest:

```bash
python run_all.py --skip-reference
```

Use the folder-level runner when you want to write results into a specific foundation output folder:

```bash
python examples/foundations/code/run_foundation_examples.py
```

For a quick code check without the scalability timing:

```bash
python run_all.py --skip-reference --skip-scalability
```

## First Extension Step

Before modifying solver loops, open `model_profiles.py` or run `python examples/foundations/code/model_profiles.py` from the repository root. For the full parameter table, read `../../../docs/PARAMETERS.md` from this folder. Then read `../../../docs/EXTENDING.md` for the paper-model adaptation checklist.
