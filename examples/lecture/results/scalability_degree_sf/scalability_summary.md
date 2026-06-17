# Scalability Analysis

This run measures `degree`-level forward-backward sweep (FBS) optimal control on synthetic Barabasi-Albert scale-free networks.

## What Was Measured

- Network sizes: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 nodes.
- Repeats per size: 3.
- Runtime column: `fbs_seconds`, measuring the FBS solve after graph generation and preprocessing.
- Convergence check: `final_delta < tolerance` for each run.

## Main Output

| File | Meaning |
| --- | --- |
| `degree_control_scalability_100_2000.png` | Runtime and state-dimension/iteration trends as SF network size grows. |
| `degree_control_scalability.csv` | One row per size and repeat. |
| `degree_control_scalability_summary.csv` | Median/min/max runtime and convergence summary by size. |

## Quick Reading

At 2000 nodes, the median FBS solve time was 3.725 seconds over 3 repeat(s). All runs at that size converged: True.

For degree-level models, the FBS state dimension is the number of observed degree classes, so it grows much more slowly than the number of nodes. This is why degree-level analysis is a useful scalability baseline before attempting full node-level FBS.
