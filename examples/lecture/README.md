# Lecture Examples

This directory contains the self-contained teaching examples from the network optimal-control and differential-games lecture package.

## Contents

```text
lecture/
├── README.md
├── requirements.txt
├── run_all_lecture_examples.py
├── code/
│   ├── simple_degree_k_control.py
│   └── network_control_examples.py
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
python run_all_lecture_examples.py
```

By default, outputs are written to:

```text
results/rerun_YYYYMMDD_HHMMSS/
```

Specify a destination:

```bash
python run_all_lecture_examples.py --output-root results/my_run
```

## Run simple degree-k control only

Built-in demo graph:

```bash
python code/simple_degree_k_control.py --output-dir results/simple_demo_new
```

Sample edge list:

```bash
python code/simple_degree_k_control.py \
  --edge-list sample_data/sample_edges.csv \
  --delimiter , \
  --has-header \
  --source-col source \
  --target-col target \
  --output-dir results/simple_edges_new
```

Sample adjacency matrix:

```bash
python code/simple_degree_k_control.py \
  --adjacency-csv sample_data/sample_adjacency.csv \
  --output-dir results/simple_adjacency_new
```

## Run compact companion examples

All compact examples:

```bash
python code/network_control_examples.py --output-dir results/examples_demo_new
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

## Existing results

Useful precomputed figures:

```text
results/simple_contact_sheet.png
results/examples_contact_sheet.png
```

The detailed subdirectories contain individual PNG files and degree-distribution CSV files.

For figure interpretation, see [`FIGURE_GUIDE.md`](FIGURE_GUIDE.md). Each fresh run also writes `figure_explanations.md` into its output directory.

## What to learn here

1. How to compute empirical degree classes `k`, counts `N_k`, probabilities `P(k)`, and average degree `<k>`.
2. How degree-level arrays differ from node-level arrays.
3. How a forward-backward sweep solves the PMP state/adjoint/control update loop.
4. How attacker-defender differential games extend the same workflow.
5. How hybrid impulse simulations combine continuous ODE segments with jump updates.
