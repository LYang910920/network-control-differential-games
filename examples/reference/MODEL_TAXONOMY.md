# Reference Repository Model Taxonomy

This document classifies the three reference repositories by modeling level, control/game structure, and the main computational pattern. It is meant to make the paper-level code easier to navigate before reading individual source files.

## Quick Classification

| Repository | Short label |
| --- | --- |
| [`OpinionMalware_TIFS_2025_Code`](reference_repositories/OpinionMalware_TIFS_2025_Code/) | Node-level malware-opinion optimal impulse control |
| [`PropagandaWar_TIFS_2024_Code`](reference_repositories/PropagandaWar_TIFS_2024_Code/) | Degree-level hybrid/impulsive differential game |
| [`Propaganda_TCSS_2025_Code`](reference_repositories/Propaganda_TCSS_2025_Code/) | Node-level awareness-aware optimal impulse control |

## Main Distinction

The most important split is the modeling level:

- **Degree-level models** aggregate nodes by degree class `k`. The state is defined for each degree group rather than for each individual node. In this repo, the clearest paper-level example is `PropagandaWar_TIFS_2024_Code`.
- **Node-level models** keep one state vector entry per node, usually coupled through an adjacency matrix. In this repo, `OpinionMalware_TIFS_2025_Code` and `Propaganda_TCSS_2025_Code` are node-level examples.

The second split is the decision structure:

- **Optimal impulse control** computes intervention magnitudes or schedules for one decision maker.
- **Differential games** compute interacting strategies for multiple players. `PropagandaWar_TIFS_2024_Code` is the main hybrid/impulsive differential-game reference.

## Repository Notes

### OpinionMalware_TIFS_2025_Code

- **Modeling level:** node-level.
- **Decision structure:** optimal impulse control.
- **Network representation:** multiplex/social network adjacency matrices.
- **State variables:** malware and opinion states, summarized as mean malware `c(t)` and opinion `o(t)`.
- **Core code path:** `network.py` builds and normalizes networks; `opinionMalware.py` runs forward states, backward adjoints, impulse strategy search, and payoff evaluation.
- **Closest tutorial example:** node-level control + hybrid impulse examples.

### PropagandaWar_TIFS_2024_Code

- **Modeling level:** degree-level.
- **Decision structure:** hybrid/impulsive differential game.
- **Network representation:** empirical degree distributions for red and blue networks.
- **State variables:** red and blue population states indexed by degree classes, such as `kr`, `pkr`, `kb`, and `pkb`.
- **Core code path:** `demo_network.py` computes degree distributions; `propWar.py` runs the red/blue forward-backward game; `comparison.py` supports unilateral-deviation checks.
- **Closest tutorial example:** degree-level game example.

This is the reference repository to read first if the goal is to understand how a degree-level network model becomes an impulsive or hybrid differential game.

### Propaganda_TCSS_2025_Code

- **Modeling level:** node-level.
- **Decision structure:** optimal impulse control.
- **Network representation:** adjacency matrix / graph-derived node interactions.
- **State variables:** awareness-aware propagation states, summarized as `Sa(t)`, `Su(t)`, and `R(t)`.
- **Core code path:** `prop_network.py` loads graph data; `prop_propaganda.py` runs forward states, backward adjoints, impulse policy search, and profit evaluation.
- **Closest tutorial example:** node-level control + impulse examples.

## Suggested Reading Order By Goal

| Goal | Start with |
| --- | --- |
| Understand degree-level PMP/Nash workflows | `examples/lecture/code/simple_degree_k_control.py`, then `PropagandaWar_TIFS_2024_Code/propWar.py` |
| Understand hybrid/impulsive differential games | `examples/lecture/code/network_control_examples.py --examples degree`, then `PropagandaWar_TIFS_2024_Code` |
| Understand node-level optimal control | `examples/lecture/code/network_control_examples.py --examples node`, then `Propaganda_TCSS_2025_Code` |
| Understand malware-opinion coupled dynamics | `OpinionMalware_TIFS_2025_Code/network.py`, then `OpinionMalware_TIFS_2025_Code/opinionMalware.py` |
| Understand awareness-aware impulse control | `Propaganda_TCSS_2025_Code/prop_network.py`, then `Propaganda_TCSS_2025_Code/prop_propaganda.py` |
