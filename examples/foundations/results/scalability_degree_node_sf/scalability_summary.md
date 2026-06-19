# Paired Degree-Level vs Node-Level FBS Comparison

This run compares two FBS discretizations of the same normalized SIS epidemic-control problem on the same synthetic Barabasi-Albert scale-free graphs. For each network size and repeat, the degree-level row and node-level row use the exact same graph seed.

## What Was Measured

- Network sizes: 100, 1000, 10000, 100000, 1000000 nodes.
- Repeats per size: 1.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m=3`.
- Epidemic/control model: normalized SIS, `T=8.0`, `beta=1.2`, `delta=0.35`, state weight `1.0`, control weight `3.0`, `u_max=1.0`.
- Contact-pressure normalization: each graph's maximum degree, shared by the degree-level and node-level rows.
- Numerical solver for both rows: fixed-grid RK4 inside forward-backward sweep.
- Node-level implementation: sparse adjacency matrix; state/control/costate are still node-indexed.
- Time grid: `60` intervals over the control horizon.
- Maximum FBS iterations: `80`.
- FBS tolerance: `1e-04`.
- Runtime column: `fbs_seconds`, measuring only the FBS solve after graph generation.

## Main Output

| File | Meaning |
| --- | --- |
| `degree_node_fbs_comparison_100_1000000.png` | Paired runtime, state-dimension, and max-degree comparison. |
| `paired_fbs_comparison.csv` | One row per model level, size, and repeat. |
| `paired_fbs_comparison_summary.csv` | Median/min/max runtime and convergence summary by model level and size. |

## Quick Reading

At 1000000 nodes:

| Model level | Solver label | FBS state dimension | FBS iterations | Median FBS seconds | All runs converged |
| --- | --- | ---: | ---: | ---: | --- |
| degree | paired_fixed_grid_rk4 | 305 | 14 | 0.563 | True |
| node | paired_sparse_fixed_grid_rk4 | 1000000 | 14 | 133.781 | True |

## Synthetic SF Degree Growth Check

| Nodes | Edges | Mean degree | Median max degree | Max-degree range |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 294 | 5.88 | 24 | 24-24 |
| 1000 | 2994 | 5.99 | 78 | 78-78 |
| 10000 | 29994 | 6.00 | 173 | 173-173 |
| 100000 | 299994 | 6.00 | 511 | 511-511 |
| 1000000 | 2999994 | 6.00 | 1513 | 1513-1513 |

This is the comparison plot to use when asking whether node-level FBS is more expensive than degree-level FBS. The degree-level row keeps one state/control/costate per observed degree class. The node-level row keeps one state/control/costate per graph node on the same epidemic model and the same graph seed. Sparse matrix products keep the million-node run practical, but the node-indexed FBS state is still much larger.
