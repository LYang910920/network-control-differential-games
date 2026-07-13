# From Model to Paper

## 1. State the Mechanism

Define the state order, units, graph resolution, control timing, and any reset
map. Explain why degree classes or explicit nodes are appropriate. Distinguish a
continuous-time signal, a sampled zero-order-hold action, and an impulse.

## 2. Specify the Decision Problem

Write the admissible controls, bounds, running objective, terminal objective, and
jump costs. For a game, give each player's payoff and information structure.
State whether strategies are open-loop, feedback, centralized, or decentralized.

## 3. Establish a Numerical Baseline

Implement the uncontrolled model first. Check invariants and compare dense,
sparse, NumPy, and Torch forms where applicable. Then run the canonical FBS solver
and at least one independent baseline such as direct transcription, a rule-based
policy, or smooth random controls.

## 4. Design Experiments

Use matched graphs and matched-mean homogeneous parameters when studying
heterogeneity. Vary graph seeds, parameter seeds, heterogeneity strength, budget,
and time-grid resolution. Separate algorithm convergence plots from state and
control trajectories.

## 5. Report Evidence

At minimum report:

- cumulative and terminal objective/payoff;
- peak, final, and time-integrated adverse state;
- control or impulse cost and budget use;
- FBS residual and iteration count;
- mass, bound, and nonnegativity errors;
- runtime and state dimension;
- sensitivity and baseline comparisons.

Figure captions must say whether a curve is one node, one degree class, or an
average; identify continuous, sampled, impulse, or continuous-impulse control;
and define every axis.

## 6. Limit Claims

Synthetic experiments establish behavior for the stated model and parameter
range. They do not establish calibration, operational effectiveness, or a general
Nash equilibrium. Record identifiability, observability, solver, and data limits
where they affect interpretation.

## 7. Map the Manuscript to the Repository

| Manuscript section | Reproducible repository evidence |
|---|---|
| Model | equations, state order, resolved parameters, graph construction |
| Method | importable solver and typed configuration |
| Baselines | matched configuration and common metrics |
| Experiments | manifest, CSV metrics, and deterministic seeds |
| Ablation/sensitivity | one-factor and multi-seed comparisons |
| Figures | script, data source, axis definitions, vector output where needed |
| Limitations | failed runs, non-identifiability, scalability and modeling assumptions |
