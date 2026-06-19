# From Model To Paper

This workflow turns a tutorial example into a paper-ready study. It is written
for cyber propagation, defense, misinformation, malware, and network-control
models.

## 1. Formulate The Model

| Decision | What to write down |
| --- | --- |
| State | Node-level, degree-level, or aggregate compartments; define every unit. |
| Control type | Continuous control, impulse control, discrete sampled decisions, or hybrid control. |
| Players | Single controller, attacker-defender game, or multi-agent game. |
| Dynamics | ODE flow `x' = f(x,u,v)`, jump map `x(tau+) = G(x(tau-),a)`, and network coupling. |
| Constraints | Positivity, simplex conservation, control bounds, budgets, event times, and terminal conditions. |
| Objective/payoff | Running cost, terminal cost, impulse cost, attacker payoff, defender cost, and sign conventions. |

Keep `docs/NOTATION_TO_CODE.md` open while translating equations into Python.

## 2. Add Baselines First

| Problem type | Minimum baselines |
| --- | --- |
| Optimal control | No control, constant controls, random controls, PMP/FBSM candidate, sensitivity to initial state. |
| Differential game | Fixed attacker vs varied defenders, fixed defender vs varied attackers, random unilateral deviations, Nash-gap proxy. |
| Sampled-data policy method | Rule policy, random policy, no-defense policy, fixed-action policies, multiple random seeds. |
| Data-assisted model check | Interpolation, wrong-parameter dynamics, known-mechanism-only run, held-out trajectory, noisy-data run. |
| Impulse/hybrid | No impulse, fixed impulse schedule, randomized impulse times, continuous-only version, impulse-only version. |

Use "PMP/FBS candidate" unless unilateral-deviation or Nash-gap evidence
supports a stronger equilibrium statement.

## 3. Choose The Method Path

### PMP/FBSM

Define `f(x,u)`, running/terminal cost, Hamiltonian, state equations, costate
equations, stationarity/KKT condition, relaxation rule, and convergence metric.
Plot control update versus iteration and state/control curves versus time.

### Sampled-Data Policy Methods

Define observation, action, reward, transition order, horizon, and
termination/truncation. Separate ODE solver substeps from sampled decision
epochs. Compare against rule-based, fixed, and random policies before presenting
learned policies.

### Data-Assisted Model Checks

Split losses into data, ODE residual, boundary/initial condition,
conservation/positivity, and regularization terms. Keep known mechanisms
explicit when adding learned correction terms. Report parameter error, state
error, residual error, and sensitivity to data density.

### Hybrid And Impulse Games

Write continuous flow and jump maps separately. Keep impulse costs separate from
running costs. For games, create payoff plots with one strategy fixed and the
other side varied.

## 4. Figure Set

1. State evolution over time, labeled as node-level, degree-class, or aggregate mean.
2. Control/game strategy over time, with continuous controls as curves and impulses as markers.
3. Convergence plot against iteration.
4. Baseline comparison for the same model, including 50 to 100 random strategies when practical.
5. Game payoff comparison with two panels: fixed attack while defense varies, and fixed defense while attack varies.
6. Ablation plot for one important modeling or loss-design choice.

## 5. Manuscript Skeleton

1. Motivation and threat/control scenario.
2. Related work and gap.
3. Model formulation and assumptions.
4. Method: PMP conditions, game conditions, sampled-data policy formulation, data-assisted residual, or hybrid/impulse formulation.
5. Algorithm and implementation details.
6. Experimental setup, parameters, datasets, and baselines.
7. Results: convergence, state/control behavior, baseline comparison, ablation, sensitivity, and limitations.
8. Discussion of when the method is reliable and when it is only a numerical candidate.
9. Conclusion and future extensions.
