# Lecture Examples

This directory contains the self-contained teaching examples from the network optimal-control and differential-games lecture package.

## Contents

```text
lecture/
├── README.md
├── requirements.txt
├── run_all_lecture_examples.py   # compatibility wrapper
├── code/
│   ├── simple_degree_k_control.py
│   ├── network_control_examples.py
│   ├── scalability_analysis.py
│   └── run_all_lecture_examples.py
├── sample_data/
│   ├── sample_edges.csv
│   └── sample_adjacency.csv
└── results/
```

## Install

```bash
python3 -m venv ../../.venv
source ../../.venv/bin/activate
python -m pip install -r ../../requirements.txt
```

## Run all examples

```bash
python code/run_all_lecture_examples.py
```

By default, outputs are written to:

```text
results/rerun_YYYYMMDD_HHMMSS/
```

Specify a destination:

```bash
python code/run_all_lecture_examples.py --output-root results/my_run
```

The legacy command `python run_all_lecture_examples.py` still works; it calls the runner in `code/`.

## Run simple degree-k control only

Built-in demo graph:

```bash
python code/simple_degree_k_control.py --output-dir results/simple_builtin_sf_new
```

Sample edge list:

```bash
python code/simple_degree_k_control.py \
  --edge-list sample_data/sample_edges.csv \
  --delimiter , \
  --has-header \
  --source-col source \
  --target-col target \
  --output-dir results/simple_sample_edges_new
```

Sample adjacency matrix:

```bash
python code/simple_degree_k_control.py \
  --adjacency-csv sample_data/sample_adjacency.csv \
  --output-dir results/simple_sample_adjacency_new
```

## Run compact companion examples

All compact examples:

```bash
python code/network_control_examples.py --output-dir results/companion_builtin_sf_new
```

Only degree-level control/game:

```bash
python code/network_control_examples.py --examples degree --output-dir results/degree_only
```

Only node-level control/game:

```bash
python code/network_control_examples.py --examples node --output-dir results/node_only
```

Only hybrid impulse simulation:

```bash
python code/network_control_examples.py --examples hybrid --output-dir results/hybrid_only
```

## Run scalability analysis

Degree-level FBS timing on synthetic scale-free networks from 100 to 2000 nodes in steps of 100. The synthetic Barabasi-Albert graphs are generated with `python-igraph`; the rest of the model code uses NumPy/SciPy/Pandas/Matplotlib and NetworkX where those libraries make the code clearer.

```bash
python code/scalability_analysis.py --output-dir results/scalability_degree_sf_new
```

The default run uses sizes `100,200,...,2000` with three repeats per size. To change it:

```bash
python code/scalability_analysis.py --sizes 100,200,300,400,500 --repeats 3
```

## Existing results

Useful precomputed figures:

```text
results/simple_contact_sheet.png
results/companion_contact_sheet.png
results/README.md
results/companion_builtin_sf/fbs_convergence.png
results/companion_builtin_sf/degree_control_trajectory.png
results/companion_builtin_sf/degree_game_trajectory.png
results/companion_builtin_sf/node_control_trajectory.png
results/companion_builtin_sf/hybrid_impulse_trajectory.png
results/scalability_degree_sf/degree_control_scalability_100_2000.png
```

The detailed subdirectories are grouped by purpose:

| Folder pattern | Purpose |
| --- | --- |
| `simple_*` | Minimal degree-k continuous optimal-control smoke runs. |
| `companion_*` | Degree-level, node-level, game, and hybrid examples on the same input graph. |
| `scalability_degree_sf` | Degree-level FBS runtime from 100 to 2000 synthetic scale-free nodes. |

Control comparisons include 75 random smooth-control baselines. Game comparisons use two unilateral panels: fixed computed attack with varied defense, and fixed computed defense with varied attack.

For figure interpretation, see [`FIGURE_GUIDE.md`](FIGURE_GUIDE.md). Each fresh run also writes a generated `README.md` into its output directory.

## What to learn here

1. How to compute empirical degree classes `k`, counts `N_k`, probabilities `P(k)`, and average degree `<k>`.
2. How degree-level arrays differ from node-level arrays.
3. How a forward-backward sweep solves the PMP state/adjoint/control update loop.
4. How attacker-defender differential games extend the same workflow.
5. How hybrid impulse simulations combine continuous ODE segments with jump updates.
6. How degree-level aggregation changes the runtime profile as synthetic networks grow.
