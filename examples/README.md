# Examples

The maintained examples call importable `cybercontrol` modules. Use the public
runner for complete workflows and the scripts below when changing one model.

## Foundation models

| Path | Purpose |
|---|---|
| `foundations/code/simple_degree_k_control.py` | Minimal heterogeneous degree-class optimal control |
| `foundations/code/network_control_examples.py` | Degree/node control, games and continuous-impulsive simulation |
| `foundations/code/scalability_analysis.py` | Degree-HMF versus explicit-adjacency FBS scaling |

From the repository root:

```bash
python -m cybercontrol smoke
python -m cybercontrol medium --output-dir artifacts/medium
```

The medium run writes resolved parameters, projected stationarity residuals,
matched-mean baselines, figures and a Git/hardware manifest under `artifacts/`.
The scalability rows share a graph seed, scalar rates, objective weights and
time grid; the degree-HMF and explicit-adjacency equations are related
resolutions, not identical dynamics.

## Reference snapshots

`reference/` contains preserved source snapshots from the three paper
repositories, their original notices, a model taxonomy and a bounded smoke
runner. Treat these files as third-party/reference material. They are not part
of the canonical `cybercontrol` implementation.

```bash
python examples/reference/run_reference_smoke.py --help
```
