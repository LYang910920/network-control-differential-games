# Notation To Code Map

This file bridges tutorial notation and Python variables. Keep it open before
editing a model, cost, payoff, or figure label.

## Continuous-Time Core

| Mathematical notation | Python name | Meaning |
| --- | --- | --- |
| `T` | `T`, `args.T` | Time horizon. |
| `n` | `n`, `n_steps`, `steps` | Number of time intervals; state arrays usually have `n + 1` rows. |
| `t_i` | `t`, `time_grid` | Numerical time grid. |
| `x(t)` | `x`, `X`, `state` | State trajectory. |
| `u(t)` | `u`, `control` | Continuous control sampled on the time grid. |
| `lambda(t)` | `lam`, `costate` | PMP costate trajectory. |
| `f(x,u)` | `state_rhs`, `rhs`, `f_state` | ODE right-hand side. |
| `H` | `hamiltonian`, `H` | Hamiltonian for PMP optimal-control or differential-game conditions. |
| `J` | `objective`, `cost`, `payoff` | Objective/cost/payoff value. |

## Malware/SIR Variables

| Notation | Python name | Meaning |
| --- | --- | --- |
| `S(t)` | `x[:, 0]`, `S` | Susceptible or vulnerable share. |
| `I(t)` | `x[:, 1]`, `I` | Infected or compromised share. |
| `R(t)` | `x[:, 2]`, `R` | Recovered or protected share. |
| `beta` | `beta`, `params.beta` | Propagation or compromise rate. |
| `gamma` | `gamma`, `params.gamma` | Recovery or removal rate. |
| `omega` | `omega`, `params.omega` | Loss of immunity/protection rate. |
| `u_patch` | `u_patch`, `patch` | Continuous patching rate. |
| `u_clean` | `u_clean`, `clean` | Continuous cleaning/removal rate. |

## Degree-Level Network Variables

| Notation | Python name | Meaning |
| --- | --- | --- |
| `k` | `k`, `degrees` | Degree class. |
| `N_k` | `Nk` | Number of nodes in degree class `k`. |
| `P(k)` | `p`, `pk` | Population probability of degree class `k`. |
| `<k>` | `kbar` | Mean degree. |
| `Theta(t)` | `theta`, `Theta` | Degree-weighted infection pressure, typically `sum(k * P(k) * x_k) / kbar`. |
| `sum_k P(k)x_k` | `x @ p`, `result.x @ D.pk` | Population-weighted degree-class mean. Do not label this as degree-weighted unless `k P(k)` is used. |
| `M_{k\ell}` | `ResolvedDegreeSISParams.mixing` | Row-stochastic degree mixing matrix. Neutral mixing uses `ell * P(ell) / <k>`. |
| `s_k,\iota_k,\delta_k` | `susceptibility`, `infectivity`, `recovery` | Class-specific physical parameters; scalars broadcast to all degree classes. |
| `q_k,r_k,\bar u_k` | `state_weight`, `control_weight`, `control_bound` | Class-specific objective weight and control bound. |
| `\lambda_k=s_k k sum_\ell M_{k\ell}\iota_\ell x_\ell` | `degree_sis_force(x, k, params)` | Heterogeneous degree-class infection pressure. |
| `\dot x_k=(1-x_k)\lambda_k-(\delta_k+u_k)x_k` | `degree_sis_rhs(x, u, k, params)` | Heterogeneous degree-level SIS dynamics. |

## Control Types

| Control type | Code pattern | Figure label rule |
| --- | --- | --- |
| Continuous control | A curve `u(t)` on the ODE grid. | Label as `continuous control u(t)` and use a line plot. |
| Impulse control | A jump map `x(tau_j+) = G(x(tau_j-), a_j)`. | Label as `impulse at tau_j` and use vertical markers or stems. |
| Discrete sampled action | An action chosen at `t_k` for an MDP or Markov game. | Label as `action epoch k` or `sampled decision t_k`. |
| Hybrid control | Continuous flow plus discrete/impulse modes. | Show flow with lines and jumps/actions with markers or step plots. |

## Shared Package

| Module | Use it for |
| --- | --- |
| `cybercontrol.numerics` | RK4 integration, simplex projection, trapezoidal integration. |
| `cybercontrol.models` | Malware SIR RHS, hybrid flow, isolation jump maps, Torch RHS helpers. |
| `cybercontrol.heterogeneity` | Scalar-or-array degree/node parameters, profile factories, SIS RHS/Jacobians, and finite-difference checks. |
| `cybercontrol.network_models` | Node-level SIPS/SIPRS graph pressure, NumPy/Torch RHS functions, stochastic SIPRS transition helper. |
| `cybercontrol.torch_utils` | Optional PyTorch helpers: MLPs, simplex state networks, bounded control networks, positive transforms, and autograd time derivatives. |
| `cybercontrol.io` | CSV/JSON writing and reproducible seeding. |
| `cybercontrol.plotting` | Shared plotting boxes/arrows and lightweight axis cleanup. |

## Node SIPS/SIPRS Variables

| Mathematical symbol | Python variable | Meaning |
| --- | --- | --- |
| `x_i=[S_i,I_i,P_i]` | `x[i, :]` with shape `(nodes, 3)` | SIPS node state. |
| `x_i=[S_i,I_i,P_i,R_i]` | `x[i, :]` with shape `(nodes, 4)` | SIPRS node state. |
| `\widetilde A_{ij}` | `adjacency[i, j]` | Node `j` contributes infection pressure to node `i`. |
| `u_i^p` | `patch[i]` | Preventive patching rate, `S -> P`. |
| `u_i^c` | `clean[i]` | Cleaning/remediation rate, `I -> R`. |
| `\omega_p,\omega_r` | `NodeSIPRSParams.omega_p`, `.omega_r` | Waning rates `P -> S` and `R -> S`. |
| `s_i,\iota_i,\gamma_i` | `NodeSIPRSParams.susceptibility`, `.infectivity`, `.gamma` | Per-node susceptibility, infectivity and recovery; scalars broadcast to all nodes. |
| `c_i,\bar u_i,\eta_i` | `criticality`, `patch/clean_cost`, `patch/clean_bound`, `patch/clean_efficacy` | Per-node economic weight, action cost, bound and efficacy. |
| `\lambda_i=s_i sum_j A_{ij}\iota_j I_j` | `node_siprs_transition_rates(...)[\"infection\"]` | Heterogeneous infection pressure used by NumPy and Torch RHS functions. |
