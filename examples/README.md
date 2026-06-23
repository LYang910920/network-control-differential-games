# Examples

This directory has two tracks.

## Track 1: Tutorial Examples

Path:

```text
examples/foundations/
```

Use this track first. The scripts are small, self-contained, and written to show the modeling pattern clearly.

| File | Purpose | Best use |
| --- | --- | --- |
| `code/simple_degree_k_control.py` | Minimal degree-k SIS optimal-control example | First PMP/control run |
| `code/network_control_examples.py` | Degree games, node-level models, continuous-impulse simulation | Compare model variants |
| `code/scalability_analysis.py` | Paired degree/node runtime experiment | Compare degree-level and node-level FBS on the same SIS control problem |
| `code/run_foundation_examples.py` | Runs all foundation examples and saves figures | Reproduce the included plots |

Typical run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py --skip-reference
```

Fast foundation-only check without the scalability timing:

```bash
python run_all.py --skip-reference --skip-scalability
```

Short paired degree/node scalability timing while developing:

```bash
python run_all.py --skip-reference --scalability-sizes 100,200,300 --scalability-repeats 1
```

Heavier sparse node-level scalability run:

```bash
python run_all.py --skip-reference --include-node-scalability
```

The default scalability run writes `scalability_degree_node_sf/` and compares degree-level FBS with sparse node-level FBS on the same normalized SIS epidemic-control model and the same synthetic scale-free graph seeds at 100, 1000, 10000, 100000, and 1000000 nodes. The heavy run adds `scalability_node_sf/` with a separate node-indexed sparse FBS stress test using a different node-only parameter profile. Do not mix its wall-clock times with the paired comparison.

Outputs:

```text
artifacts/foundation_runs/rerun_YYYYMMDD_HHMMSS/
```

Each foundation result folder includes `parameter_summary.csv` so the time horizon, grid size, propagation/recovery rates, control bounds, impulse settings, and random-baseline count are visible without reading the source first.

## Track 2: Reference Smoke Runs

Path:

```text
examples/reference/
```

Use this track after the foundation examples. It keeps source snapshots from three upstream research repositories and runs small smoke tests with local sample data. The three reference repositories correspond to the author's co-authored cyber/network-control papers: two in IEEE TIFS and one in IEEE TCSS.

| File or folder | Purpose |
| --- | --- |
| `reference_repositories/` | Curated third-party source snapshots with upstream README/LICENSE files |
| `MODEL_TAXONOMY.md` | Classifies the three reference repositories by model level and control/game type |
| `run_reference_smoke.py` | Runs short checks across the three reference repositories |
| `reference_repository_guide.md` | Explains how each reference repo maps to the tutorial workflow |
| `sample_data/` | Small local graphs used instead of full paper datasets |
| `patches/` | Compatibility patch documentation for newer dependency versions |

Typical run from the repository root:

```bash
python run_all.py --skip-foundations
```

Outputs:

```text
artifacts/reference_runs/reference_repos_rerun/
```

The reference smoke runner also writes `parameter_summary.csv` with each paper-level smoke run's horizon, step size, rate parameters, impulse event indices, strategy bounds, and baseline count.

## How The Two Tracks Connect

| Concept | Tutorial examples | Reference repositories |
| --- | --- | --- |
| Network preprocessing | `sample_data/`, degree distribution helpers | graph loaders and dataset-specific preprocessing |
| Forward dynamics | compact ODE/state functions | paper-specific propagation dynamics |
| Backward adjoints | short PMP adjoint routines | longer model-specific adjoint systems |
| Control update | bounded Hamiltonian update | continuous or impulse strategy search |
| Validation | deterministic/random baseline plots plus paired degree/node scalability and optional sparse node-only stress testing | smoke tests, payoff summaries, and model-specific deterministic/random baseline checks |
