# Examples

This directory has two tracks.

## Track 1: Lecture Examples

Path:

```text
examples/lecture/
```

Use this track first. The scripts are small, self-contained, and written to show the modeling pattern clearly.

| File | Purpose | Best use |
| --- | --- | --- |
| `code/simple_degree_k_control.py` | Minimal degree-k SIS optimal-control example | First PMP/control run |
| `code/network_control_examples.py` | Degree games, node-level models, hybrid impulse simulation | Compare model variants |
| `run_all_lecture_examples.py` | Runs all lecture examples and saves figures | Reproduce the included plots |

Typical run from the repository root:

```bash
python3 run_all.py --skip-reference
```

Outputs:

```text
examples/lecture/results/rerun_YYYYMMDD_HHMMSS/
```

## Track 2: Reference Smoke Runs

Path:

```text
examples/reference/
```

Use this track after the lecture examples. It keeps source snapshots from three upstream research repositories and runs small smoke tests with local sample data. The three reference repositories correspond to the author's co-authored cyber/network-control papers: two in IEEE TIFS and one in IEEE TCSS.

| File or folder | Purpose |
| --- | --- |
| `reference_repositories/` | Curated third-party source snapshots with upstream README/LICENSE files |
| `run_reference_smoke.py` | Runs short checks across the three reference repositories |
| `reference_repository_guide.md` | Explains how each reference repo maps to the lecture workflow |
| `sample_data/` | Small local graphs used instead of full paper datasets |
| `patches/` | Compatibility patch documentation for newer dependency versions |

Typical run from the repository root:

```bash
python3 run_all.py --skip-lecture
```

Outputs:

```text
examples/reference/results/reference_repos_rerun/
```

## How The Two Tracks Connect

| Concept | Lecture examples | Reference repositories |
| --- | --- | --- |
| Network preprocessing | `sample_data/`, degree distribution helpers | graph loaders and dataset-specific preprocessing |
| Forward dynamics | compact ODE/state functions | paper-specific propagation dynamics |
| Backward adjoints | short PMP adjoint routines | longer model-specific adjoint systems |
| Control update | bounded Hamiltonian update | continuous or impulse strategy search |
| Validation | simple baselines and plots | smoke tests, payoff summaries, unilateral-deviation checks where available |
