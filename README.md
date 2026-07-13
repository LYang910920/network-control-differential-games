# Network Control and Differential Games

Foundation models, numerical methods, and reusable Python components for network
optimal control, attacker-defender differential games, and impulse interventions.
The `cybercontrol` package is the shared implementation used by the two companion
repositories.

This repository has mixed licensing. Read [LICENSE](LICENSE),
[COPYRIGHT_AND_LICENSE.md](COPYRIGHT_AND_LICENSE.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse or redistribution.

## Repository Family

| Repository | Purpose |
|---|---|
| **Network Control and Differential Games** | Shared equations, solvers, heterogeneous profiles, plots, and classical control/game examples. |
| [Cyber Control and Game Learning](https://github.com/LYang910920/note1-cyber-control-games) | Sampled environments, DDQN, CTDE, MAPPO, and game evaluation. |
| [Physics-Informed Cyber Control](https://github.com/LYang910920/note2-pinn-pidl-cyber-control) | Inverse PINN, PIDL, neural control, and PMP-informed learning. |

## Five-Minute Start

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[graph,torch,dev]"
python -m cybercontrol smoke
```

The smoke run executes heterogeneous degree/node control and game cases and writes
its outputs to `artifacts/smoke/`. Generated results are ignored by Git.

```bash
python -m cybercontrol medium --output-dir artifacts/medium
python -m cybercontrol figures --output-dir artifacts/figures
python -m cybercontrol docs
```

`medium` adds matched-mean homogeneous baselines and a paired degree/node scaling
study. `docs` requires `latexmk` and builds the current tutorial PDF.

## Code Map

| Topic | Canonical implementation |
|---|---|
| Scalar/array parameter resolution | `cybercontrol.heterogeneity` |
| Degree- and node-level SIS equations | `cybercontrol.heterogeneity` |
| Node-level SIPS equations | `cybercontrol.network_models` |
| RK4, projections, and quadrature | `cybercontrol.numerics` |
| FBS control and game solvers | `cybercontrol.fbs` |
| Flow plus reset simulation | `cybercontrol.continuous_impulse` |
| Graph loading and sparse operations | `cybercontrol.graph_data`, `cybercontrol.graph_utils` |
| Neural blocks, differentiable numerics, RL, and PINN operators | `cybercontrol.nn`, `cybercontrol.torch_numerics`, `cybercontrol.rl`, `cybercontrol.pinn` |
| Metrics, manifests, and plotting | `cybercontrol.foundation_evaluation`, `cybercontrol.experiments`, `cybercontrol.plotting` |

The runnable examples are thin callers of these modules:

| Example | Purpose |
|---|---|
| `examples/foundations/code/simple_degree_k_control.py` | Minimal heterogeneous degree-class control problem. |
| `examples/foundations/code/network_control_examples.py` | Degree/node control, games, matched baselines, and continuous-impulse simulation. |
| `examples/foundations/code/scalability_analysis.py` | Degree-HMF versus explicit sparse node FBS scaling. |
| `examples/reference/run_reference_smoke.py` | Bounded checks of preserved reference snapshots. |

See [Models and API](docs/MODELS_AND_API.md) for equations and notation-to-code
mapping.

## Maintained Experiments

The degree and node solvers use the same graph seed, scalar rates, objective
weights, time grid, and stopping rule. Their states have different dimensions:
degree-HMF tracks occupied degree classes; node FBS tracks every node.

![Degree- and node-level FBS runtime and state dimension](docs/assets/degree_node_fbs_scalability.png)

The model runner also compares heterogeneous parameters with matched-mean
homogeneous parameters and smooth random controls. These outputs are regenerated
under `artifacts/`, rather than stored as additional homepage graphics.

## Extending a Model

1. Write the state order, units, flow equations, reset map, and objective before editing code.
2. Start from a typed parameter profile in `cybercontrol.foundation_profiles`.
3. Reuse the canonical RHS, Jacobian, projection, and FBS routines.
4. Add an example-specific component only when it does not belong in the shared package.
5. Report convergence, invariant errors, parameter arrays, seeds, and matched baselines.

Detailed instructions are in [Reproducibility](docs/REPRODUCIBILITY.md) and
[From Model to Paper](docs/FROM_MODEL_TO_PAPER.md).

## Validation

```bash
python -m compileall -q src tests scripts examples/foundations/code
ruff check .
ruff format --check src tests scripts examples/foundations/code
pytest -q
python -m cybercontrol smoke
```

The reference repositories under `examples/reference/reference_repositories/` are
preserved snapshots. They are not part of the canonical package and retain their
own notices and assumptions.

## Tutorial and Citation

- [Tutorial PDF](docs/network_control_foundations.pdf)
- [Citation guidance](CITATION.md)

The tutorial reflects the author's research experience in optimal control,
differential games, and cyber/network security, including work published in IEEE
TIFS, TDSC, TSMC, TNSE, and TCSS.
