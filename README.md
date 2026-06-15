# Network Optimal Control and Differential Games

This repository collects teaching materials and runnable examples for network optimal control, differential games, and hybrid or impulsive intervention models.

It is organized for two purposes:

1. Run the self-contained lecture examples quickly.
2. Use lightweight smoke runs to understand how the paper-level reference repositories map to the lecture code.

## Quick Start

Install the lecture-example dependencies:

```bash
cd examples/lecture
python3 -m pip install -r requirements.txt
python3 run_all_lecture_examples.py
```

Run the reference-repository smoke tests:

```bash
cd ../reference
python3 -m pip install -r requirements.txt
python3 run_reference_smoke.py
```

If `python-igraph` is difficult to install globally, install it locally:

```bash
cd examples/reference
python3 -m pip install --target pydeps python-igraph
python3 run_reference_smoke.py --pydeps pydeps
```

## Repository Layout

```text
.
├── README.md
├── COPYRIGHT_AND_LICENSE.md
├── THIRD_PARTY_NOTICES.md
├── docs/
│   ├── lecture_note.pdf
│   ├── lecture_note.tex
│   ├── code_walkthrough_and_model_adaptation_guide.pdf
│   └── code_walkthrough_and_model_adaptation_guide.tex
└── examples/
    ├── lecture/
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── run_all_lecture_examples.py
    │   ├── code/
    │   ├── sample_data/
    │   └── results/
    └── reference/
        ├── README.md
        ├── requirements.txt
        ├── download_reference_repositories.sh
        ├── run_reference_smoke.py
        ├── patches/
        ├── sample_data/
        └── results/
```

## Documents

- [Lecture note PDF](docs/lecture_note.pdf)
- [Lecture note LaTeX](docs/lecture_note.tex)
- [Code walkthrough and model adaptation guide PDF](docs/code_walkthrough_and_model_adaptation_guide.pdf)
- [Code walkthrough and model adaptation guide LaTeX](docs/code_walkthrough_and_model_adaptation_guide.tex)

The walkthrough guide is used as the organizing reference for this repository. In particular, the code and README structure follow this sequence:

1. Environment setup and basic execution.
2. Network data input: edge lists, adjacency CSV files, directed degree modes.
3. Core model conventions: degree-level arrays versus node-level arrays.
4. Simple degree-k optimal control.
5. Compact extension: degree games, node games, hybrid impulses.
6. Model adaptation checklist: RHS, Jacobian, objective, Hamiltonian update, constraints, baselines.
7. Mapping from teaching examples to reference research repositories.

## Lecture Examples

Go to:

```bash
cd examples/lecture
```

Run everything:

```bash
python3 run_all_lecture_examples.py
```

Useful outputs already included:

- [Simple-example overview](examples/lecture/results/simple_contact_sheet.png)
- [Companion-example overview](examples/lecture/results/examples_contact_sheet.png)

The lecture examples are the best place to start because they are self-contained and use small built-in or sample networks.

## Reference Repository Smoke Runs

Go to:

```bash
cd examples/reference
```

Run the lightweight smoke tests:

```bash
python3 run_reference_smoke.py
```

The reference source snapshots are already included under `reference_repositories/`. Use `download_reference_repositories.sh` only if you want to refresh them from upstream.

Useful outputs already included:

- [Reference repo smoke-run overview](examples/reference/results/reference_repos/reference_repo_contact_sheet.png)
- [Smoke-run summary CSV](examples/reference/results/reference_repos/smoke_run_summary.csv)

The smoke runner intentionally uses small networks and short horizons. It is meant to verify code paths and explain model mapping, not to reproduce full paper-scale experiments.

## Model Adaptation Checklist

When adapting the code to a new model, work in this order:

1. Decide degree-level, node-level, or hybrid/impulse modeling.
2. Replace the state equation `f(x, u)`.
3. Update the Jacobian `f_x`.
4. Update the objective or payoff.
5. Re-derive the Hamiltonian control update from `f_u`.
6. Check state and control constraints.
7. Run short-horizon tests first.
8. Add no-control, constant-control, random-control, or unilateral-deviation baselines.

For differential games, treat the computed controls as open-loop Nash candidates satisfying necessary conditions. They should be checked against unilateral deviations before being interpreted as numerically convincing Nash strategies.

## Copyright and Publication Notes

Read [COPYRIGHT_AND_LICENSE.md](COPYRIGHT_AND_LICENSE.md) before publishing this repository publicly.

Short version:

- This repo does not grant a blanket open-source license by default.
- The tutorial documents are included as user-supplied educational materials; confirm authorship and redistribution rights before making a public GitHub repo.
- The three reference repositories are third-party works. Source snapshots are included with their upstream README and LICENSE files preserved.
- Full paper datasets are not vendored; the smoke tests use small local sample data.

## Suggested GitHub Workflow

```bash
git init
git add .
git commit -m "Organize network control tutorial materials"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Before pushing to a public repository, review `docs/` and decide whether the PDFs/LaTeX should be public or kept in a private repo.
