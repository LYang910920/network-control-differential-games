# Guide to the reference repositories

This guide explains how the reference research-code repositories connect to the tutorial note and the two companion Python scripts. The three repositories correspond to the author's co-authored cyber/network-control publications: two IEEE TIFS papers and one IEEE TCSS paper. The goal is to help the reader move from the small tutorial examples to paper-level code without losing the mathematical structure.

## Common computational pattern

All three repositories follow the same broad workflow:

1. Load a network from an edge list, Matrix Market file, or other graph format.
2. Convert the network into a degree distribution, an adjacency matrix, or both.
3. Simulate the forward state dynamics.
4. Solve the backward adjoint equations.
5. Update continuous controls or impulse magnitudes using PMP/Hamiltonian conditions.
6. Evaluate the payoff or cost functional.
7. Compare the computed strategy against random, constant, no-control, or ablation baselines.

The companion scripts in this package use the same workflow with shorter equations and clearer variable names.

## Classification map

| Reference repository | Modeling level | Control/game structure | First thing to inspect |
| --- | --- | --- | --- |
| `OpinionMalware_TIFS_2025_Code` | Node-level coupled malware-opinion model | Optimal impulse control | `network.py`, then `opinionMalware.py` |
| `PropagandaWar_TIFS_2024_Code` | Degree-level red/blue population model | Hybrid/impulsive differential game | `demo_network.py`, then `propWar.py` |
| `Propaganda_TCSS_2025_Code` | Node-level awareness-aware propagation model | Optimal impulse control | `prop_network.py`, then `prop_propaganda.py` |

Use this classification before reading individual files. In particular, the IEEE TIFS 2024 repository is the degree-level example: it aggregates graph structure into degree distributions and then computes red/blue strategies in a hybrid or impulsive differential game.

## 1. OpinionMalware_TIFS_2025_Code

Repository: [XiaojuanCheng/OpinionMalware_TIFS_2025_Code](https://github.com/XiaojuanCheng/OpinionMalware_TIFS_2025_Code)

Local snapshot: [`reference_repositories/OpinionMalware_TIFS_2025_Code`](reference_repositories/OpinionMalware_TIFS_2025_Code/)

Publication context: IEEE TIFS, 2025.

Classification: node-level coupled malware-opinion model with optimal impulse control.

Main files:

- `network.py`: network construction, connected-component extraction, scale-free layer construction, and adjacency normalization.
- `opinionMalware.py`: coupled malware/opinion dynamics, adjoint equations, impulse-control search, and payoff calculation.

How to read it with the tutorial note:

- `create_multiplex_network_with_connected_nodes_edges(...)` corresponds to graph preprocessing.
- `normalized_matrix(...)` corresponds to adjacency normalization before node-level dynamics.
- Forward malware and opinion routines correspond to state equations.
- Backward routines correspond to adjoint equations.
- `optimalStrategy(...)` corresponds to impulse-Hamiltonian search over admissible impulse intensities.
- `payoff(...)` corresponds to the objective functional.

## 2. PropagandaWar_TIFS_2024_Code

Repository: [XiaojuanCheng/PropagandaWar_TIFS_2024_Code](https://github.com/XiaojuanCheng/PropagandaWar_TIFS_2024_Code)

Local snapshot: [`reference_repositories/PropagandaWar_TIFS_2024_Code`](reference_repositories/PropagandaWar_TIFS_2024_Code/)

Publication context: IEEE TIFS, 2024.

Classification: degree-level red/blue population model with a hybrid/impulsive differential game.

Main files:

- `demo_network.py`: graph loading and degree-distribution computation.
- `propWar.py`: forward-backward Nash/hybrid strategy computation.
- `comparison.py`: empirical Nash-equilibrium checking through random unilateral deviations.

How to read it with the tutorial note:

- `graphs(...)`, `graphs_tmx(...)`, and `my_degree(...)` correspond to network loading and empirical degree distribution.
- Degree arrays such as `kr`, `pkr`, `kb`, and `pkb` are the key bridge from graph data to the degree-level model.
- Forward routines correspond to controlled state dynamics.
- Backward routines correspond to PMP adjoint equations.
- Optimal-strategy routines correspond to bounded Hamiltonian maximization or minimization.
- Payoff routines correspond to each player's payoff functional.
- `comparison.py` matches the numerical recommendation to test unilateral deviations.

## 3. Propaganda_TCSS_2025_Code

Repository: [XiaojuanCheng/Propaganda_TCSS_2025_Code](https://github.com/XiaojuanCheng/Propaganda_TCSS_2025_Code)

Local snapshot: [`reference_repositories/Propaganda_TCSS_2025_Code`](reference_repositories/Propaganda_TCSS_2025_Code/)

Publication context: IEEE TCSS, 2025.

Classification: node-level awareness-aware propagation model with optimal impulse control.

Main files:

- `prop_network.py`: graph loading from edge lists or Matrix Market files.
- `prop_propaganda.py`: main optimal impulse-control experiment.
- `prop_cntrlComparison.py`: random-strategy comparison.
- `prop_noAwareness.py` and `prop_noAwareness_plot.py`: no-awareness baseline and plotting.

How to read it with the tutorial note:

- Network-loading functions correspond to edge-list-to-adjacency conversion.
- Forward propagation functions correspond to state dynamics.
- Backward functions correspond to adjoint dynamics.
- `policy_sim(...)` corresponds to impulse policy search.
- `profit_sim(...)` corresponds to payoff/objective evaluation.
- No-awareness code corresponds to an ablation model.

## Suggested reading order

1. Run `code/simple_degree_k_control.py` to understand the degree-k PMP workflow.
2. Read `code/network_control_examples.py` section by section.
3. Refresh the upstream reference repositories, if needed, with `download_reference_repositories.sh` from this directory.
4. Compare each reference repository with the workflow above.
5. Replace the simplified dynamics in the companion scripts with paper-specific dynamics only after the graph preprocessing and PMP structure are clear.
