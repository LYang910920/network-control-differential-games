# Foundation Examples

This directory contains the self-contained foundation examples from the network optimal-control and differential-games tutorial package.

## Contents

```text
foundations/
├── README.md
├── requirements.txt
├── run_foundation_examples.py   # convenience wrapper
├── code/
│   ├── simple_degree_k_control.py
│   ├── network_control_examples.py
│   ├── scalability_analysis.py
│   └── run_foundation_examples.py
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
python code/run_foundation_examples.py
```

By default, outputs are written to:

```text
results/rerun_YYYYMMDD_HHMMSS/
```

Specify a destination:

```bash
python code/run_foundation_examples.py --output-root results/my_run
```

The wrapper command `python run_foundation_examples.py` calls the runner in `code/`.

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

## Run paired degree/node scalability analysis

Paired FBS timing on synthetic scale-free networks. For each graph size, the script runs degree-level FBS and sparse node-level FBS on the same normalized SIS epidemic-control model, using the same graph seed, RK4 time grid, and FBS tolerance.

```bash
python code/scalability_analysis.py --output-dir results/scalability_degree_node_sf_new
```

The default run uses sizes `1000,2000,...,10000` with two repeats per size. To change it:

```bash
python code/scalability_analysis.py --sizes 1000,2000,3000 --repeats 2
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
results/scalability_degree_node_sf/degree_node_fbs_comparison_1000_10000.png
```

The detailed subdirectories are grouped by purpose:

| Folder pattern | Purpose |
| --- | --- |
| `simple_*` | Minimal degree-k continuous optimal-control smoke runs. |
| `companion_*` | Degree-level, node-level, game, and hybrid examples on the same input graph. |
| `scalability_degree_node_sf` | Paired degree-level versus sparse node-level FBS runtime on the same epidemic-control problem and graph seeds. |

Control comparisons include 75 random smooth-control baselines. Game comparisons use two unilateral panels: fixed computed attack with varied defense, and fixed computed defense with varied attack.

Each `simple_*` and `companion_*` result folder also includes `parameter_summary.csv`, which lists the concrete smoke-run settings: time horizon, grid size, infection/contact rate, recovery rate, control or strategy bounds, impulse times, and baseline count.

For figure interpretation, see [`FIGURE_GUIDE.md`](FIGURE_GUIDE.md). Each fresh run also writes a generated `README.md` into its output directory.

## What to learn here

1. How to compute empirical degree classes `k`, counts `N_k`, probabilities `P(k)`, and average degree `<k>`.
2. How degree-level arrays differ from node-level arrays.
3. How a forward-backward sweep solves the PMP state/adjoint/control update loop.
4. How attacker-defender differential games extend the same workflow.
5. How hybrid impulse simulations combine continuous ODE segments with jump updates.
6. How degree-level aggregation and node-indexed state dimension change the FBS runtime profile as synthetic networks grow.
