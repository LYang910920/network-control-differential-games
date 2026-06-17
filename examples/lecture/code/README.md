# Lecture Code

This folder contains the maintained Python code for the lecture examples.

| File | Role | Main output |
| --- | --- | --- |
| `simple_degree_k_control.py` | Minimal degree-k continuous optimal-control example | `simple_*/degree_control_trajectory.png` |
| `network_control_examples.py` | Degree-level, node-level, differential-game, and hybrid/impulse examples | `companion_*/*_trajectory.png` |
| `scalability_analysis.py` | Degree-level FBS timing on synthetic scale-free networks from 100 to 2000 nodes | `scalability_degree_sf/degree_control_scalability_100_2000.png` |
| `run_all_lecture_examples.py` | Rebuilds all lecture figures, CSV diagnostics, and generated indexes | `results/` |

The folder-level runner is the preferred entry point:

```bash
python examples/lecture/code/run_all_lecture_examples.py
```

For repository-level validation, use:

```bash
python run_all.py --skip-reference
```
