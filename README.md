# Network Optimal Control and Differential Games

Teaching notes, runnable examples, and reference-code smoke tests for network optimal control, differential games, and hybrid or impulsive interventions.

This repository is public, but it does **not** grant a single blanket open-source license. Tutorial materials, generated examples, and third-party source snapshots have different copyright contexts. See [Copyright and License Notes](COPYRIGHT_AND_LICENSE.md) and [Third-party Notices](THIRD_PARTY_NOTICES.md).

## Author Note

This tutorial was created from my research experience in optimal control, differential games, hybrid/impulsive control, and cyber/network-security applications over the past few years. The perspective is informed by publications in venues including IEEE TIFS, TDSC, TSMC, TNSE, TCSS, and related journals.

The repository is meant to be an educational bridge: start from the mathematical conditions, run small teaching examples, and then inspect how similar ideas appear in paper-level research code.

## Start Here

| Goal | Where to go | What you get |
| --- | --- | --- |
| Read the math | [`docs/lecture_note.pdf`](docs/lecture_note.pdf) | Optimal control, differential games, hybrid control, network models |
| Learn how the code maps to the math | [`docs/code_walkthrough_and_model_adaptation_guide.pdf`](docs/code_walkthrough_and_model_adaptation_guide.pdf) | Run commands, model conventions, Jacobians, Hamiltonian updates, adaptation checklist |
| Run the clean teaching examples | [`examples/lecture/`](examples/lecture/) | Degree-k control, degree game, node-level control/game, hybrid impulse simulation |
| Inspect paper-level code patterns | [`examples/reference/`](examples/reference/) | Smoke runs for three reference repositories using small local sample data |
| Check copyright/citations | [`COPYRIGHT_AND_LICENSE.md`](COPYRIGHT_AND_LICENSE.md), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Public-repo license boundaries and upstream attribution |

## Quick Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py
```

If `python-igraph` is hard to install globally, install it locally for the reference runner:

```bash
python3 -m pip install --target examples/reference/pydeps python-igraph
python3 run_all.py
```

Run only the lecture examples:

```bash
python3 run_all.py --skip-reference
```

Run only the reference smoke tests:

```bash
python3 run_all.py --skip-lecture
```

## Recommended Learning Path

1. Open [`docs/lecture_note.pdf`](docs/lecture_note.pdf) for the mathematical setup.
2. Run the lecture examples first with `python3 run_all.py --skip-reference`.
3. Read [`docs/code_walkthrough_and_model_adaptation_guide.pdf`](docs/code_walkthrough_and_model_adaptation_guide.pdf) while comparing it with [`examples/lecture/code/`](examples/lecture/code/).
4. Run the reference smoke tests with `python3 run_all.py --skip-lecture`.
5. Use [`examples/reference/reference_repository_guide.md`](examples/reference/reference_repository_guide.md) to map the paper-level code back to the simplified examples.

## Output Preview

Lecture examples:

![Lecture companion examples](examples/lecture/results/examples_contact_sheet.png)

Reference-repository smoke runs:

![Reference smoke runs](examples/reference/results/reference_repos/reference_repo_contact_sheet.png)

After a fresh run, new outputs are written to timestamped or rerun folders:

| Command | Output location |
| --- | --- |
| `python3 run_all.py --skip-reference` | `examples/lecture/results/rerun_YYYYMMDD_HHMMSS/` |
| `python3 run_all.py --skip-lecture` | `examples/reference/results/reference_repos_rerun/` |
| `python3 run_all.py` | both locations above |

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── run_all.py
├── COPYRIGHT_AND_LICENSE.md
├── THIRD_PARTY_NOTICES.md
├── CITATION.md
├── docs/
│   ├── README.md
│   ├── lecture_note.pdf
│   ├── lecture_note.tex
│   ├── code_walkthrough_and_model_adaptation_guide.pdf
│   └── code_walkthrough_and_model_adaptation_guide.tex
└── examples/
    ├── README.md
    ├── lecture/
    │   ├── README.md
    │   ├── run_all_lecture_examples.py
    │   ├── code/
    │   ├── sample_data/
    │   └── results/
    └── reference/
        ├── README.md
        ├── run_reference_smoke.py
        ├── reference_repositories/
        ├── sample_data/
        ├── patches/
        └── results/
```

## What Is Included

For a code-first map of the examples, see [`examples/README.md`](examples/README.md).

### Lecture examples

The lecture examples are self-contained and should be the first code you run.

- `simple_degree_k_control.py`: a compact degree-k SIS optimal-control example.
- `network_control_examples.py`: degree-level games, node-level control/game models, and a hybrid impulse simulation.
- `sample_data/`: a small edge list and adjacency matrix.
- `results/`: precomputed figures and degree-distribution CSV files.

Go deeper in [examples/lecture/README.md](examples/lecture/README.md).

### Reference source snapshots

The reference folder includes source-code snapshots from three upstream research repositories:

- `OpinionMalware_TIFS_2025_Code`
- `PropagandaWar_TIFS_2024_Code`
- `Propaganda_TCSS_2025_Code`

Each snapshot keeps its upstream `README` and `LICENSE`. Full paper datasets are not included. The smoke runner uses small local sample data so the workflows can run without redistributing external datasets.

Go deeper in [examples/reference/README.md](examples/reference/README.md).

## Model Adaptation Checklist

When adapting the examples to a new model, work in this order:

1. Choose the modeling level: degree-level, node-level, or hybrid/impulse.
2. Replace the state equation `f(x, u)`.
3. Update the Jacobian `f_x`.
4. Update the objective or payoff.
5. Re-derive the Hamiltonian control update from `f_u`.
6. Check state and control constraints.
7. Run short-horizon tests first.
8. Add no-control, constant-control, random-control, or unilateral-deviation baselines.

For differential games, the computed controls are open-loop Nash candidates satisfying necessary conditions. Treat them as numerical candidates until unilateral-deviation checks support the interpretation.

## Troubleshooting

If a run fails with `ModuleNotFoundError`, install the root requirements in the Python environment you are actually using:

```bash
python3 -m pip install -r requirements.txt
python3 -c "import networkx, scipy, pandas, matplotlib; print('core dependencies ok')"
```

If `python-igraph` is the only difficult package, use the local install path:

```bash
python3 -m pip install --target examples/reference/pydeps python-igraph
python3 run_all.py --skip-lecture
```

## Public-repository Notes

- No project-wide license is granted by default.
- Tutorial PDFs and LaTeX sources are included as educational materials; confirm redistribution rights before reusing them elsewhere.
- Third-party source snapshots retain their upstream licenses and citations.
- Full paper datasets are intentionally not vendored.
- Generated figures and CSV files are included for quick inspection and reproducibility checks.
