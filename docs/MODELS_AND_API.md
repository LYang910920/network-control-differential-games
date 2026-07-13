# Models and API

This page maps the maintained mathematical models to the `cybercontrol` package.
All state arrays use floating-point population shares unless stated otherwise.

## Degree-Level SIS

For occupied degree classes `k`, the state is `x[k]`, the infected share of class
`k`. With degree mixing matrix `M`,

```text
lambda[k] = susceptibility[k] * k
            * sum_l M[k,l] * infectivity[l] * x[l]
dx[k]/dt  = (1 - x[k]) * lambda[k]
            - (recovery[k] + control[k]) * x[k]
```

`DegreeSISParams` accepts scalars or arrays. Scalars broadcast to all degree
classes, which preserves the homogeneous model. `neutral_degree_mixing` and
`assortative_degree_mixing` construct documented mixing matrices.

| Mathematics | Python |
|---|---|
| degree classes and weights | `DegreeData.k`, `DegreeData.pk` |
| resolved physical/economic arrays | `ResolvedDegreeSISParams` |
| force and RHS | `degree_sis_force`, `degree_sis_rhs` |
| analytic Jacobian | `degree_sis_jacobian` |
| control/game model assembly | `cybercontrol.fbs.degree_control_model`, `degree_game_model` |

## Node-Level SIS

For node infection probabilities `x[i]` and adjacency matrix `A`,

```text
lambda[i] = susceptibility[i]
            * sum_j A[i,j] * infectivity[j] * x[j]
dx[i]/dt  = (1 - x[i]) * lambda[i]
            - (recovery[i] + control[i]) * x[i]
```

The canonical functions are `node_sis_force`, `node_sis_rhs`, and
`node_sis_jacobian`. Dense and sparse adjacency paths use the same equation.

## Node-Level SIPS

Each node has state `[S, I, P]`; `S + I + P = 1`.

```text
dS/dt = -S * lambda - patch * S + omega * P
dI/dt =  S * lambda - (gamma + clean) * I
dP/dt =  patch * S + (gamma + clean) * I - omega * P
```

Here `lambda[i] = susceptibility[i] * sum_j A[i,j] * infectivity[j] * I[j]`.
`NodeSIPSParams` resolves per-node susceptibility, infectivity, recovery `gamma`,
protection waning `omega`, action costs, bounds, criticality, and efficacy.

Use `node_sips_rhs_numpy` for simulation and `node_sips_rhs_torch` for differentiable
models. Tests cover conservation, dense/sparse agreement, and NumPy/Torch parity.

## Parameter Profiles

The profile constructors accept homogeneous values, degree-correlated values,
community-correlated values, seeded lognormal values, and arrays loaded from CSV.
Every experiment should record the resolved arrays, summary statistics, seed,
heterogeneity strength, and matched-mean homogeneous baseline.

Foundation example constants are grouped in
`cybercontrol.foundation_profiles`. The generated `parameter_summary.csv` is the
authoritative record for a particular run.

## Control and Game Solvers

`cybercontrol.fbs` provides importable forward-backward sweep components for:

- degree-level optimal control;
- degree-level attacker-defender open-loop games;
- node-level optimal control;
- node-level attacker-defender open-loop games.

Controls are projected onto their class- or node-specific bounds. Convergence is
reported with the projected fixed-point residual; a small change in objective
alone is not treated as stationarity.

`cybercontrol.continuous_impulse.simulate_continuous_impulse` integrates flow
segments and applies reset maps only at declared impulse times. Sampled actions
without a reset remain flow controls; this distinction is tested in Note 1.

## Torch Modules

`cybercontrol.nn` owns neural architectures and bounded/simplex output networks.
`cybercontrol.torch_numerics` owns differentiable projection and RK4 operations;
`cybercontrol.pinn` owns automatic-differentiation residual operators. Runtime
device, thread, dtype, and seed choices enter through
`cybercontrol.experiments.configure_torch` at runner boundaries.

`cybercontrol.torch_utils` is a compatibility re-export for pre-0.2 imports. New
code should use the focused modules above.

## Output Shapes

| Quantity | Shape |
|---|---|
| degree state or control | `(time, degree_class)` |
| node SIS state or control | `(time, node)` |
| node SIPS state | `(time, node, 3)` |
| degree mixing matrix | `(degree_class, degree_class)` |
| node adjacency | `(node, node)` dense or sparse |

Use `cybercontrol.io.write_json` and `write_csv` at runner boundaries. Library
functions return arrays or typed results and do not print or change global device,
thread, or random-number state at import time.
