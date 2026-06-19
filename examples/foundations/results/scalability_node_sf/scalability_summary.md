# Scalability Analysis

This run measures `node`-level forward-backward sweep (FBS) optimal control on synthetic Barabasi-Albert scale-free networks.

## What Was Measured

- Network sizes: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 nodes.
- Repeats per size: 1.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m=3`.
- Numerical solver: `sparse`.
- Time grid: `25` intervals over the control horizon.
- Maximum FBS iterations: `50`.
- FBS tolerance: `1e-04`.
- Runtime column: `fbs_seconds`, measuring the FBS solve after graph generation and preprocessing.
- Convergence check: `final_delta < tolerance` for each run.

## Main Output

| File | Meaning |
| --- | --- |
| `node_control_scalability_1000_10000.png` | Runtime and state-dimension/iteration trends as SF network size grows. |
| `node_control_scalability.csv` | One row per size and repeat. |
| `node_control_scalability_summary.csv` | Median/min/max runtime and convergence summary by size. |

## Quick Reading

At 10000 nodes, the median FBS solve time was 0.651 seconds over 1 repeat(s). All runs at that size converged: True.

Read runtime columns with the solver type. Degree-level runs use an adaptive ODE solve on the reduced degree-class system. Sparse node-level runs use fixed-grid RK4 and sparse matrix products on a node-indexed system. These are both useful smoke/scaling diagnostics, but their wall-clock times are not a direct solver-speed comparison.
