# Research extensions

Work through the [degree-class example and model guide](../../docs/GUIDE.md) first. These three attributed source snapshots show how the same forward-state/backward-costate structure appears in research models with additional states and interventions. Their original code, READMEs and licenses are retained in `reference_repositories/`.

| Research implementation | Read first | What to trace |
|---|---|---|
| [OpinionMalware TIFS 2025](https://github.com/XiaojuanCheng/OpinionMalware_TIFS_2025_Code) | `network.py`, then `opinionMalware.py` | Two network layers, coupled awareness/infection dynamics, adjoints and impulse intensity selection |
| [PropagandaWar TIFS 2024](https://github.com/XiaojuanCheng/PropagandaWar_TIFS_2024_Code) | `demo_network.py`, then `propWar.py` | Empirical degree classes, two-player controls and hybrid decisions |
| [Propaganda TCSS 2025](https://github.com/XiaojuanCheng/Propaganda_TCSS_2025_Code) | `prop_network.py`, then `prop_propaganda.py` | Awareness-dependent propagation, impulse policies and payoff evaluation |

For each model, identify the state vector, information available to each player, control bounds, running/terminal costs and jump map before running a search. Check the maximization/minimization convention before copying an adjoint equation. `comparison.py` and the TCSS comparison scripts illustrate policy comparisons; sampled unilateral deviations do not prove Nash equilibrium.

The source snapshots are teaching extensions, not prerequisites for the main example. Full-paper datasets and potentially long parameter searches are not launched by the default command. The optional `run_reference_smoke.py` adapter retains the small reference-code execution checks; consult its `--help` only when working with those upstream implementations. The NumPy/NetworkX compatibility patch for the opinion-malware snapshot is recorded in `patches/`.

The [third-party notices](../../THIRD_PARTY_NOTICES.md) describe attribution and licensing. These models have different state meanings, time scales and intervention semantics; do not compare their raw payoff numbers as though they were the same optimization problem.
