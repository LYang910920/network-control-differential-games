# Scalability Analysis

This run measures `degree`-level forward-backward sweep (FBS) optimal control on synthetic Barabasi-Albert scale-free networks.

## What Was Measured

- Network sizes: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 nodes.
- Repeats per size: 3.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m=3`.
- Numerical solver: `adaptive_solve_ivp`.
- Time grid: `35` intervals over the control horizon.
- Maximum FBS iterations: `60`.
- FBS tolerance: `1e-04`.
- Runtime column: `fbs_seconds`, measuring the FBS solve after graph generation and preprocessing.
- Convergence check: `final_delta < tolerance` for each run.

## Main Output

| File | Meaning |
| --- | --- |
| `degree_control_scalability_100_2000.png` | Runtime and state-dimension/iteration trends as SF network size grows. |
| `degree_control_scalability.csv` | One row per size and repeat. |
| `degree_control_scalability_summary.csv` | Median/min/max runtime and convergence summary by size. |

## Quick Reading

At 2000 nodes, the median FBS solve time was 3.935 seconds over 3 repeat(s). All runs at that size converged: True.

Read runtime columns with the solver type. Degree-level runs use an adaptive ODE solve on the reduced degree-class system. Sparse node-level runs use fixed-grid RK4 and sparse matrix products on a node-indexed system. These are both useful smoke/scaling diagnostics, but their wall-clock times are not a direct solver-speed comparison.
