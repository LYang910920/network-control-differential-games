# Scalability Analysis

This run measures `node`-level forward-backward sweep (FBS) optimal control on synthetic Barabasi-Albert scale-free networks.

## What Was Measured

- Network sizes: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 nodes.
- Repeats per size: 1.
- Synthetic network model: Barabasi-Albert scale-free graph with attachment parameter `m=3`.
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

At 10000 nodes, the median FBS solve time was 0.619 seconds over 1 repeat(s). All runs at that size converged: True.

For degree-level models, the FBS state dimension is the number of observed degree classes, so it grows much more slowly than the number of nodes. For node-level models, the state, costate, and control are indexed by node. The optional large-node run therefore uses a sparse adjacency matrix and reports convergence of a node-indexed FBS sweep rather than converting the graph to a dense teaching matrix.
