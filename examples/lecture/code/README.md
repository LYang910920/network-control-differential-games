# Lecture Code

This folder contains the maintained Python code for the lecture examples.

The common idea is always the same: convert a graph into a model state, solve a
small optimal-control or game update, and plot both the trajectory and the
diagnostics needed to trust the numerical result.

| File | Role | Main output |
| --- | --- | --- |
| `simple_degree_k_control.py` | Minimal degree-k continuous optimal-control example | `simple_*/degree_control_trajectory.png` |
| `network_control_examples.py` | Degree-level, node-level, differential-game, and hybrid/impulse examples | `companion_*/*_trajectory.png` |
| `scalability_analysis.py` | Degree-level FBS timing on synthetic scale-free networks from 100 to 2000 nodes | `scalability_degree_sf/degree_control_scalability_100_2000.png` |
| `run_all_lecture_examples.py` | Rebuilds all lecture figures, CSV diagnostics, and generated indexes | `results/` |

The root runner is usually easiest:

```bash
python run_all.py --skip-reference
```

Use the folder-level runner when you want to write results into a specific lecture output folder:

```bash
python examples/lecture/code/run_all_lecture_examples.py
```

For a quick code check without the scalability timing:

```bash
python run_all.py --skip-reference --skip-scalability
```
