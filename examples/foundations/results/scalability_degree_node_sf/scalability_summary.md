# Paired Degree-Level vs Node-Level FBS Comparison

This run compares two FBS discretizations of the same normalized SIS epidemic-control problem on the same synthetic Barabasi-Albert scale-free graphs.

## What Was Measured

- Network sizes: 100, 250, 500, 1000, 2000 nodes.
- Repeats per size: 2.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m=3`.
- Epidemic/control model: normalized SIS, `T=8.0`, `beta=1.2`, `delta=0.35`, state weight `1.0`, control weight `3.0`, `u_max=1.0`.
- Numerical solver for both rows: fixed-grid RK4 inside forward-backward sweep.
- Time grid: `80` intervals over the control horizon.
- Maximum FBS iterations: `80`.
- FBS tolerance: `1e-04`.
- Runtime column: `fbs_seconds`, measuring only the FBS solve after graph generation.

## Main Output

| File | Meaning |
| --- | --- |
| `degree_node_fbs_comparison_100_2000.png` | Paired runtime and state-dimension comparison. |
| `paired_fbs_comparison.csv` | One row per model level, size, and repeat. |
| `paired_fbs_comparison_summary.csv` | Median/min/max runtime and convergence summary by model level and size. |

## Quick Reading

At 2000 nodes:

| Model level | Solver label | FBS state dimension | FBS iterations | Median FBS seconds | All runs converged |
| --- | --- | ---: | ---: | ---: | --- |
| degree | paired_fixed_grid_rk4 | 43 | 23 | 0.286 | True |
| node | paired_dense_fixed_grid_rk4 | 2000 | 24 | 5.671 | True |

This is the comparison plot to use when asking whether node-level FBS is more expensive than degree-level FBS. The degree-level row keeps one state/control/costate per observed degree class. The node-level row keeps one state/control/costate per graph node on the same epidemic model, so it should take longer as node count grows.
