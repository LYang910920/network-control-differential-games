# Reference Repository Smoke Runs

This directory contains a lightweight runner for the three paper-level reference repositories discussed in the tutorial material.

It is designed for orientation and code walkthroughs. It is not a full reproduction of the papers' large-scale experiments.

## Publication context

The three reference repositories correspond to the author's co-authored cyber/network-control publications: two IEEE TIFS papers and one IEEE TCSS paper.

| Repository | Venue | Model class |
| --- | --- | --- |
| [`OpinionMalware_TIFS_2025_Code`](reference_repositories/OpinionMalware_TIFS_2025_Code/) | IEEE TIFS 2025 | Node-level malware-opinion optimal impulse control |
| [`PropagandaWar_TIFS_2024_Code`](reference_repositories/PropagandaWar_TIFS_2024_Code/) | IEEE TIFS 2024 | Degree-level hybrid/impulsive differential game |
| [`Propaganda_TCSS_2025_Code`](reference_repositories/Propaganda_TCSS_2025_Code/) | IEEE TCSS 2025 | Node-level awareness-aware optimal impulse control |

For a more explicit classification, see [`MODEL_TAXONOMY.md`](MODEL_TAXONOMY.md).

Upstream links and licenses:

| Repository | Upstream | License |
| --- | --- | --- |
| `OpinionMalware_TIFS_2025_Code` | [GitHub](https://github.com/XiaojuanCheng/OpinionMalware_TIFS_2025_Code) | [Apache-2.0](reference_repositories/OpinionMalware_TIFS_2025_Code/LICENSE) |
| `PropagandaWar_TIFS_2024_Code` | [GitHub](https://github.com/XiaojuanCheng/PropagandaWar_TIFS_2024_Code) | [Apache-2.0](reference_repositories/PropagandaWar_TIFS_2024_Code/LICENSE) |
| `Propaganda_TCSS_2025_Code` | [GitHub](https://github.com/XiaojuanCheng/Propaganda_TCSS_2025_Code) | [MIT](reference_repositories/Propaganda_TCSS_2025_Code/LICENSE) |

## Contents

```text
reference/
├── README.md
├── MODEL_TAXONOMY.md
├── run_reference_smoke.py
├── reference_repository_guide.md
├── reference_repositories/
├── sample_data/
└── patches/
```

## Install

```bash
python3 -m venv ../../.venv
source ../../.venv/bin/activate
python -m pip install -r ../../requirements.txt
```

If `python-igraph` is inconvenient to install in the active environment, install it locally:

```bash
python -m pip install --target pydeps python-igraph
```

The runner will automatically look for `pydeps/`.

## Reference source code

This repository includes source snapshots in:

```text
reference_repositories/
├── OpinionMalware_TIFS_2025_Code/
├── PropagandaWar_TIFS_2024_Code/
└── Propaganda_TCSS_2025_Code/
```

The snapshots include source files plus upstream README and LICENSE files. Full paper datasets are not included. To refresh the snapshots from upstream, run:

```bash
bash download_reference_repositories.sh
```

By default this downloads full upstream clones to `reference_repositories_upstream/` so it does not overwrite the curated source snapshots in `reference_repositories/`.

## Run the smoke tests

```bash
python run_reference_smoke.py
```

Outputs are written to:

```text
../../artifacts/reference_runs/reference_repos_rerun/
```

Choose a destination:

```bash
python run_reference_smoke.py --output-dir ../../artifacts/reference_runs/my_run
```

If your downloaded repositories are somewhere else, pass the parent directory that contains `reference_repositories/`:

```bash
python run_reference_smoke.py \
  --package-dir /path/to/package_or_reference_workspace \
  --output-dir ../../artifacts/reference_runs/my_run
```

Fresh runs write `smoke_run_report.md`, `smoke_run_summary.csv`, per-model time-series CSVs, and baseline-comparison figures into the selected output directory. The README uses only the curated contact sheet in `docs/assets/`.

Plot convention: continuous controls and game strategies are time-indexed curves sampled on the simulation grid; projected continuous strategies may have flat bound segments but still enter the ODE between event times. Impulse controls act only at discrete event times and are drawn as vertical lines; hybrid control combines both. State labels indicate whether each trajectory is a node mean over all nodes or a degree-weighted mean over degree classes. For concrete values such as `T`, `h`, rate parameters, impulse intervals, event indices, and bounds, start with the generated `parameter_summary.csv`.

## What each smoke run does

### OpinionMalware_TIFS_2025_Code

- Classification: node-level coupled malware-opinion model with optimal impulse control.
- Uses the local `sample_data/opinion_malware_edges.edges` graph so this repository can run without redistributing upstream datasets.
- Uses a 20-node sample graph; the runner is configured to allow up to 60 nodes when a larger local sample is provided.
- Runs the coupled malware state `c(t)`, opinion state `o(t)`, adjoint equations, impulse strategy update, and payoff calculation.
- Requires a small compatibility patch on newer NumPy/NetworkX versions because the original code expects `.A1` on matrix-like objects. `run_reference_smoke.py` applies this patch automatically to the local source copy before importing it.

Patch:

```text
patches/opinion_malware_numpy_networkx_compat.patch
```

### PropagandaWar_TIFS_2024_Code

- Classification: degree-level red/blue population model with a hybrid/impulsive differential game.
- The paper datasets are not redistributed in the upstream repository.
- The smoke run generates small synthetic graphs, converts them into empirical degree distributions, and runs the degree-level red/blue propaganda-war game.
- This verifies the forward equations, backward adjoint equations, strategy updates, and payoff reporting.

### Propaganda_TCSS_2025_Code

- Classification: node-level awareness-aware propagation model with optimal impulse control.
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
