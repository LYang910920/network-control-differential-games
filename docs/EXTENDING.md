# Extending The Network Control Tutorial

This repository is meant to be a foundation before moving into larger paper-level code. Keep the same contracts while replacing the small tutorial models:

```text
network data -> state aggregation -> dynamics -> control/game update -> diagnostics -> baseline comparison
```

## First Extension Step

Start with the visible profile file:

```bash
python run_all.py --skip-reference --skip-scalability
```

Then open `examples/foundations/code/model_profiles.py`. It collects the time horizons, propagation/recovery rates, control bounds, impulse times, and labels used by the tutorial examples. Change one profile value at a time before editing solver code.

## Control Types

| Type | Meaning | First code hook | Figure/readout to check |
|---|---|---|---|
| Continuous control | control can vary along the time grid and changes ODE rates | `simple_degree_k_control.py`, `network_control_examples.py` | continuous control curves over time |
| Impulse control | action occurs only at selected event times and creates a state jump | `HybridImpulseProfile.impulse_times`, hybrid simulation block | vertical impulse markers and pre/post state change |
| Hybrid control | continuous flow control plus discrete impulse interventions | `HYBRID_PROFILE`, hybrid plotting block | separated continuous curve and impulse markers |
| Continuous differential game | attacker and defender strategies evolve on the time grid | degree/node game functions | two strategy curves plus unilateral baseline comparisons |

## Model-Level Choices

| Model level | State object | Use it when | First code hook |
|---|---|---|---|
| Degree-level | one state per observed degree class `k` | the paper model depends on degree distribution or mean-field contact pressure | `DegreeData`, degree-control/game functions |
| Node-level | one state per selected graph node | adjacency, centrality, local interventions, or node-specific control matters | `NetworkData.A`, node-control/game functions |
| Sparse node-level scalability | one state, costate, and control per graph node | you want to stress-test FBS convergence from 1000 to 10000 nodes | `scalability_analysis.py --model-level node --node-solver sparse` |
| Reference repo smoke run | paper-level source snapshot plus lightweight wrapper | you want to compare tutorial intuition with the author's IEEE TIFS/TCSS code | `examples/reference/run_reference_smoke.py` |

## From Tutorial Code To Paper Models

| Paper-model ingredient | First tutorial hook | What to preserve while extending |
|---|---|---|
| New graph data | edge-list or adjacency CLI options | clear convention for whether `A[i,j]` means `j` influences `i` |
| New state variables | `TimeSeries`, degree/node state arrays | labels showing degree class, node, or aggregate average |
| New propagation model | RHS functions inside `network_control_examples.py` | nonnegative states and visible parameter summary |
| New payoff/objective | cost and reward terms near the FBS/game update | separate state cost, control cost, attack reward, and terminal terms |
| New impulses | `HybridImpulseProfile` and hybrid simulation block | event times, jump magnitudes, and pre/post-jump diagnostics |
| Larger random baselines | `examples/common_diagnostics.py` | same model compared only with its own baselines |
| Paper-level repository | `examples/reference/MODEL_TAXONOMY.md` | upstream citation, license, and model classification |

## Reference Repository Bridge

Use the three reference repositories as paper-level examples, not as hidden black boxes:

| Reference code | Tutorial bridge | Adaptation question |
|---|---|---|
| `PropagandaWar_TIFS_2024_Code` | degree-level continuous game plus hybrid/impulse examples | How does degree-level control/game logic change when the model has impulsive interventions? |
| `OpinionMalware_TIFS_2025_Code` | network loading, node/degree aggregation, smoke-run diagnostics | How do opinion/malware states and graph pressure map onto the tutorial state convention? |
| `Propaganda_TCSS_2025_Code` | node-level control and hybrid impulse examples | Which state is node-specific, which output is averaged, and where do impulses enter? |

## Research-Grade Checklist

Before treating a run as evidence rather than a tutorial smoke run:

1. Run multiple random graph seeds and initial conditions.
2. Keep one baseline-comparison figure per model, not mixed across unrelated models.
3. Report whether each state curve is a selected node, selected degree class, or average.
4. For games, test unilateral deviations by fixing one player's computed strategy and varying the other player's strategy.
5. Log FBS convergence and change parameters if the forward-backward sweep fails to converge.
6. For degree-level versus node-level scale claims, use the default paired run `python run_all.py --skip-reference` and report convergence, state dimension, solver type, and runtime from `scalability_degree_node_sf/`. The optional `--include-node-scalability` run is a separate sparse node-only stress test; use it to check larger node-indexed systems, not as the degree-vs-node comparison figure.
7. Keep upstream licenses, paper citations, and dataset redistribution rights visible.
