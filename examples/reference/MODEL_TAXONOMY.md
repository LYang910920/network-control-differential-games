# Reference Repository Model Taxonomy

This document classifies the three reference repositories by modeling level, control/game structure, and the main computational pattern. It is meant to make the paper-level code easier to navigate before reading individual source files.

## Quick Classification

| Reference repository | Venue | Modeling level | Control/game type | Network representation | Best matching lecture example |
| --- | --- | --- | --- | --- | --- |
| [`OpinionMalware_TIFS_2025_Code`](reference_repositories/OpinionMalware_TIFS_2025_Code/) | IEEE TIFS 2025 | Node-level coupled malware-opinion model | Optimal impulse control | Multiplex/social network adjacency matrices | Node-level control + hybrid impulse examples |
| [`PropagandaWar_TIFS_2024_Code`](reference_repositories/PropagandaWar_TIFS_2024_Code/) | IEEE TIFS 2024 | Degree-level red/blue population model | Hybrid/impulsive differential game | Empirical degree distributions for red and blue networks | Degree-level game example |
| [`Propaganda_TCSS_2025_Code`](reference_repositories/Propaganda_TCSS_2025_Code/) | IEEE TCSS 2025 | Node-level awareness-aware propagation model | Optimal impulse control | Adjacency matrix / graph-derived node interactions | Node-level control + impulse examples |

## Main Distinction

The most important split is the modeling level:

- **Degree-level models** aggregate nodes by degree class `k`. The state is defined for each degree group rather than for each individual node. In this repo, the clearest paper-level example is `PropagandaWar_TIFS_2024_Code`.
- **Node-level models** keep one state vector entry per node, usually coupled through an adjacency matrix. In this repo, `OpinionMalware_TIFS_2025_Code` and `Propaganda_TCSS_2025_Code` are node-level examples.

The second split is the decision structure:

- **Optimal impulse control** computes intervention magnitudes or schedules for one decision maker.
- **Differential games** compute interacting strategies for multiple players. `PropagandaWar_TIFS_2024_Code` is the main hybrid/impulsive differential-game reference.

## Repository-by-Repository Notes

### OpinionMalware_TIFS_2025_Code

- **Modeling level:** node-level.
- **State variables:** malware and opinion states, represented in the smoke run as mean malware `c(t)` and opinion `o(t)`.
- **Decision structure:** optimal impulse control.
- **Core code path:** `network.py` builds and normalizes networks; `opinionMalware.py` runs forward states, backward adjoints, impulse strategy search, and payoff evaluation.
- **Smoke-run figure:** the left panel shows payoff over iterations, the middle panel shows malware/opinion state trajectories over time, and the right panel shows impulse-control schedules over time.

### PropagandaWar_TIFS_2024_Code

- **Modeling level:** degree-level.
- **State variables:** red and blue population states indexed by degree classes, such as `kr`, `pkr`, `kb`, and `pkb`.
- **Decision structure:** hybrid/impulsive differential game.
- **Core code path:** `demo_network.py` computes degree distributions; `propWar.py` runs the red/blue forward-backward game; `comparison.py` supports unilateral-deviation checks.
- **Smoke-run figure:** the left panel shows red/blue payoff updates across iterations, the middle panel shows degree-level state trajectories over time, and the right panel shows red/blue strategy variables over time.

This is the reference repository to read first if the goal is to understand how a degree-level network model becomes an impulsive or hybrid differential game.

### Propaganda_TCSS_2025_Code

- **Modeling level:** node-level.
- **State variables:** awareness-aware propagation states, summarized in the smoke run as `Sa(t)`, `Su(t)`, and `R(t)`.
- **Decision structure:** optimal impulse control.
- **Core code path:** `prop_network.py` loads graph data; `prop_propaganda.py` runs forward states, backward adjoints, impulse policy search, and profit evaluation.
- **Smoke-run figure:** the left panel shows profit over policy iterations, the middle panel shows state averages over time, and the right panel shows impulse-control schedules over time.

## Suggested Reading Order By Goal

| Goal | Start with |
| --- | --- |
| Understand degree-level PMP/Nash workflows | `examples/lecture/code/simple_degree_k_control.py`, then `PropagandaWar_TIFS_2024_Code/propWar.py` |
| Understand hybrid/impulsive differential games | `examples/lecture/code/network_control_examples.py --examples degree`, then `PropagandaWar_TIFS_2024_Code` |
| Understand node-level optimal control | `examples/lecture/code/network_control_examples.py --examples node`, then `Propaganda_TCSS_2025_Code` |
| Understand malware-opinion coupled dynamics | `OpinionMalware_TIFS_2025_Code/network.py`, then `OpinionMalware_TIFS_2025_Code/opinionMalware.py` |
| Understand awareness-aware impulse control | `Propaganda_TCSS_2025_Code/prop_network.py`, then `Propaganda_TCSS_2025_Code/prop_propaganda.py` |
