# Reproducibility

## Commands

```bash
python -m pip install -e ".[graph,torch,dev]"
python -m cybercontrol smoke
python -m cybercontrol medium --output-dir artifacts/medium
python -m cybercontrol figures --output-dir artifacts/figures
python -m cybercontrol docs
```

`smoke` checks execution and invariants with short grids. `medium` uses longer
grids, heterogeneous/matched-mean comparisons, and paired degree/node scaling at
100, 1,000, and 10,000 nodes by default. Change the bounded scaling study with
`--sizes`, `--steps`, `--iterations`, and `--tolerance`.

## Required Run Record

Keep the following with any reported result:

- Git commit and dirty state;
- Python/platform information and resolved device;
- graph source, size, normalization, and seed;
- horizon, grid size, FBS limit, tolerance, damping, and convergence status;
- full resolved parameter arrays and summary statistics;
- objective/payoff definition, control bounds, and baseline definitions;
- runtime and invariant errors.

The public runner writes JSON manifests and CSV parameter tables under the chosen
output directory. `artifacts/` is ignored so exploratory runs do not become
repository content.

## Numerical Checks

Before interpreting a control or game result, verify:

1. state values remain finite and within their admissible set;
2. SIPS mass error is near numerical precision;
3. analytic Jacobians agree with finite differences;
4. projected stationarity residual reaches the configured tolerance;
5. changing the grid or damping does not materially change the conclusion;
6. heterogeneous and matched-mean runs use the same graph, horizon, and solver.

For game results, also report unilateral strategy perturbations. The FBS fixed
point is numerical evidence for the configured open-loop game, not a general
equilibrium proof.

## PDF Build

```bash
cd docs/source
latexmk -pdf -interaction=nonstopmode network_control_foundations.tex
```

The public runner copies the built PDF to `docs/`. Auxiliary LaTeX files remain
ignored.

## Reference Snapshots

The three directories below `examples/reference/reference_repositories/` preserve
paper-code snapshots. Do not reformat, relicense, or merge their implementations
into `cybercontrol`. The smoke runner records which paths ran and which required
upstream assumptions or compatibility handling.
