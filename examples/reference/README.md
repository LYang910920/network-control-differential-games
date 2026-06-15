# Reference Repository Smoke Runs

This directory contains a lightweight runner for the three paper-level reference repositories discussed in the lecture material.

It is designed for orientation and code walkthroughs. It is not a full reproduction of the papers' large-scale experiments.

## Contents

```text
reference/
├── README.md
├── requirements.txt
├── download_reference_repositories.sh
├── run_reference_smoke.py
├── reference_repository_guide.md
├── sample_data/
│   └── sample_adjacency.csv
├── patches/
│   └── opinion_malware_numpy_networkx_compat.patch
└── results/
    └── reference_repos/
```

## Install

```bash
python3 -m pip install -r requirements.txt
```

If `python-igraph` is inconvenient to install globally, install it locally:

```bash
python3 -m pip install --target pydeps python-igraph
```

The runner will automatically look for `pydeps/`.

## Download upstream repositories

```bash
bash download_reference_repositories.sh
```

This creates:

```text
reference_repositories/
├── OpinionMalware_TIFS_2025_Code/
├── PropagandaWar_TIFS_2024_Code/
└── Propaganda_TCSS_2025_Code/
```

The upstream repositories are excluded from Git by default. This keeps third-party source code and datasets out of your own repository history unless you intentionally vendor them.

## Run the smoke tests

```bash
python3 run_reference_smoke.py
```

Outputs are written to:

```text
results/reference_repos_rerun/
```

Choose a destination:

```bash
python3 run_reference_smoke.py --output-dir results/my_run
```

If your downloaded repositories are somewhere else, pass the parent directory that contains `reference_repositories/`:

```bash
python3 run_reference_smoke.py \
  --package-dir /path/to/package_or_reference_workspace \
  --output-dir results/my_run
```

## Existing results

The current curated run is stored in:

```text
results/reference_repos/
```

Key files:

```text
results/reference_repos/reference_repo_contact_sheet.png
results/reference_repos/smoke_run_summary.csv
results/reference_repos/opinion_malware_timeseries.csv
results/reference_repos/propaganda_war_timeseries.csv
results/reference_repos/propaganda_tcss_timeseries.csv
```

## What each smoke run does

### OpinionMalware_TIFS_2025_Code

- Uses the upstream `email-univ` data after downloading the upstream repo.
- Samples 60 nodes.
- Runs the coupled malware state `c(t)`, opinion state `o(t)`, adjoint equations, impulse strategy update, and payoff calculation.
- Requires a small compatibility patch on newer NumPy/NetworkX versions because the original code expects `.A1` on matrix-like objects. `run_reference_smoke.py` applies this patch automatically to the downloaded local copy before importing it.

Patch:

```text
patches/opinion_malware_numpy_networkx_compat.patch
```

### PropagandaWar_TIFS_2024_Code

- The paper datasets are not redistributed in the upstream repository.
- The smoke run generates small synthetic graphs, converts them into empirical degree distributions, and runs the degree-level red/blue propaganda-war game.
- This verifies the forward equations, backward adjoint equations, strategy updates, and payoff reporting.

### Propaganda_TCSS_2025_Code

- The paper datasets are not redistributed in the upstream repository.
- The smoke run uses `sample_data/sample_adjacency.csv`.
- It runs the awareness-aware node-level propagation model, backward adjoints, impulse policy update, and profit calculation.

## Full paper-scale reproduction

To move from smoke runs to paper-scale reproduction:

1. Download the datasets named in each upstream README.
2. Place them in the exact `data/` layout expected by the upstream scripts.
3. Use the paper-specific parameter blocks in the upstream `__main__` sections.
4. Create the expected result directories before running.
5. Preserve upstream licenses and citations.
6. Expect the full-size runs to be much slower than these smoke tests.

## Copyright note

The upstream repositories retain their original licenses and citations. See:

```text
../../THIRD_PARTY_NOTICES.md
../../COPYRIGHT_AND_LICENSE.md
```
