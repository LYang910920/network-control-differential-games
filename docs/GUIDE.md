# From a network to a control policy

## 1. What the first experiment answers

A network contains devices that can become infected and recover. Intervention increases the recovery rate, but intervention itself has a cost. The question is how much effort to allocate to each degree class over time.

Run `python -m cybercontrol` from the repository after installation. Read `src/cybercontrol/cli.py` from `run_example` downward: load the graph, compute degree classes, choose model parameters, solve, compare policies, and plot. There is no second implementation of PMP in this entry point; the equations are in `heterogeneity.py` and the sweep is in `fbs.py`.

The default graph is a seeded 40-node Barabasi-Albert teaching network. Its size is chosen to make the example easy to inspect, not to represent a realistic experiment. The numerical rates are illustrative, not fitted empirical estimates. Real-network studies should report the dataset, preprocessing, rate units and parameter sources.

## 2. Network data and degree classes

For a graph with n nodes, `degree_distribution_from_graph` computes the observed degrees k, counts N_k, probabilities P(k)=N_k/n, and mean degree sum_k k P(k). The k=0 class is retained when the graph contains isolates. A column of the state array is a **degree class**, not a compartment:

```python
degree = degree_distribution_from_graph(graph)
# result.x.shape == (number_of_times, number_of_observed_degree_classes)
# result.x[:, j] is x_k(t), with k = degree.k[j]
```

An edge list cannot describe a node absent from every row. Use GraphML or an adjacency matrix when isolated nodes must be retained. An adjacency CSV must contain numbers only, without row labels or a header; its row and column order identifies the nodes. In a CSV edge list, node identifiers are preserved as text. Repeated edges are summed, including reversed records of an undirected edge. A zero-weight edge contributes no contact; negative or nonfinite weights are rejected. Degree means the number of distinct neighbors, not weighted strength.

```bash
python -m cybercontrol --edge-list data/edges.csv --delimiter , --has-header --source-col source --target-col target --weight-col weight
python -m cybercontrol --graph-file data/network.graphml
python -m cybercontrol --adjacency-csv data/adjacency.csv
```

The introductory closure assumes an undirected, uncorrelated contact network. Edge weights are stored by the reader but are not used in this unweighted degree closure. A directed in-degree distribution alone does not specify a directed spreading model. Use the node equations, or formulate an appropriate joint-degree/mixing model, when direction, edge strength or degree correlations matter.

For a node-level model, obtain its influence matrix only when needed:

```python
import networkx as nx

nodes = list(graph.nodes())
A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr").T
# A[i, j] is the influence of node j on node i.
# A @ x is the incoming infection pressure.
```

The transpose is essential for a directed source-target graph. Save `nodes` with node-level results. The dense classroom solvers use `graph_to_model_matrix`; sparse large-network implementations are in `scalability.py` and `network_models.py`. An induced high-degree subgraph is a different network, not an exact reduction of the full dynamics. Matrix normalization also changes the effective transmission model and must be stated explicitly.

## 3. Equations and code

The first example uses

\[
\Theta(t)=\frac{\sum_k kP(k)x_k(t)}{\langle k\rangle},\qquad
\dot x_k=\beta k(1-x_k)\Theta-(\delta+u_k)x_k.
\]

It minimizes

\[
J(u)=\int_0^T\sum_k P(k)\left[3x_k(t)+\frac{r}{2}u_k(t)^2\right]dt,
\qquad 0\leq u_k\leq1.2.
\]

Here beta is the per-contact transmission rate, delta is natural recovery, and r is the intervention penalty. Time units must be consistent with all three rates. The default horizon is 14 time units. `DegreeSISParams(...).resolve(k, pk)` expands scalars into aligned class-wise arrays. Its mixing matrix has rows M_kl=lP(l)/<k> for the uncorrelated example.

With the minimization Hamiltonian H=L+lambda^T f, the code solves the state equation forward and

\[
\dot\lambda=-L_x-f_x^T\lambda,\qquad \lambda(T)=0,
\qquad
u_k=\operatorname{clip}\!\left(\frac{\lambda_kx_k}{rP(k)},0,1.2\right)
\]

backward. `fbs.py` then relaxes the control toward u_k. The convergence quantity is the **undamped projected fixed-point residual**, not merely the small damped step. State trajectories are replayed under the returned controls before costs are evaluated.

A converged sweep gives a numerical candidate satisfying the discretized necessary conditions. It does not by itself establish global optimality. The comparisons in the first experiment use the same dynamics, initial state, horizon, quadrature and objective. The constant baseline is the best of nine tested uniform intensities, not the globally best time-independent vector.

## 4. Read the results

`infection.png` shows the population-weighted infection fraction. `control.png` shows selected degree classes. `degree_distribution.csv` records the graph statistics; `trajectories.csv` stores the states and controls with explicit time and degree labels. `comparison.csv` reports total cost and infected time for each policy.

A policy may reduce infection yet have a higher total cost because it uses more effort. Read both measures. Compare solver refinements with `--steps 160` and a tighter `--tolerance`; changing the time grid is a numerical check, whereas changing beta or the network defines a different problem.

## 5. Modify a model without breaking its mathematics

Start with parameters, then change equations. The default coefficients are visible together in `run_example`. For class-dependent susceptibility, recovery, costs or bounds, pass arrays to `DegreeSISParams` with one value for every entry in `degree.k`. To add assortativity, provide a row-stochastic degree mixing matrix and explain its source.

For a new state equation, modify these pieces together:

| Mathematical object | Code location |
|---|---|
| State RHS f | `degree_sis_rhs` or `node_sis_rhs` in `heterogeneity.py` |
| State Jacobian f_x | The corresponding Jacobian function |
| Running and terminal objective | The solver and its matching evaluator |
| Adjoint equation and terminal value | The backward part of the solver |
| Hamiltonian control minimizer | The projected update in the solver |

For example, replacing recovery control by transmission reduction changes f to beta(1-u_k)k(1-x_k)Theta-delta*x_k. Both f_x and f_u change. Its quadratic-cost minimizer is clip(lambda_k*beta*k*(1-x_k)*Theta/(r*P(k)),0,u_max), with u_max no larger than one for this interpretation. Reusing the old lambda_k*x_k update would solve a different problem.

With terminal cost Phi(x(T)), add Phi to every objective evaluation and use lambda(T)=Phi_x. With a shared budget, componentwise clipping is generally insufficient: solve the coupled Hamiltonian minimization subject to that budget. With several states per class, define one packing order and use it for the RHS, Jacobian, adjoints and output labels.

Check a hand-derived Jacobian against centered finite differences on interior states. Check invariance of the physical state set, replay the final controls, and compare numerical grids before interpreting a result. A baseline comparison is evidence, not a sufficiency theorem.

## 6. Extend to games and hybrid control

`solve_degree_game` and `solve_node_game` use a common forward state and one costate per player. Each Hamiltonian update holds the opponent's strategy fixed. Test unilateral deviations against a frozen opponent; random deviations give a lower bound on exploitable gain and cannot prove Nash equilibrium. An open-loop time schedule is not a feedback policy.

`simulate_continuous_impulse` separates continuous segments from the reset map at an intervention time. It demonstrates a specified hybrid policy. Optimizing jump magnitudes or times additionally requires their costs, admissible sets and jump/timing optimality conditions. Avoid calling a fixed impulse schedule an optimal hybrid solution.

The broader worked examples remain available through:

```bash
python -m cybercontrol study --steps 60 --iterations 50 --output-dir results/extensions
```

This is an optional extension with more outputs and a longer run than the first example. The research implementations in `examples/reference/reference_repositories` are attributed separately and have their own model assumptions. Read their equations before transferring a policy or parameter.

## 7. Continue with learning and inference

**Note 1** converts continuous dynamics into a sampled environment: observe, act, apply any reset, integrate the flow, and receive an interval reward. Its learned policies use observations at decision times. Solver substeps are not extra decisions. Evaluate learning against an untrained policy and meaningful fixed policies on held-out initial conditions before undertaking larger MARL studies.

**Note 2** estimates dynamics or latent mechanisms from observations. Data fit, physical residual, held-out prediction and parameter recovery answer different questions. A small residual does not establish identifiability. Keep an ODE-fitting baseline when the mechanism is already known.

When editing the common foundation and a note together, use a separate environment, install this repository with `python -m pip install -e ".[torch,dev]"`, then install the note with `python -m pip install --no-deps -e ../note1-cyber-control-games` (or Note 2). This keeps the editable foundation in place. Standalone note installations instead use the tested Git revision declared in their `pyproject.toml`.

## 8. Tests and lecture PDF

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m cybercontrol check
python -m cybercontrol docs
```

Only the last command requires a LaTeX installation with `latexmk`. The normal example does not. The `smoke`, `figures` and `medium` command names used in existing teaching materials map to `check`, `example` and `study` respectively. Tests cover the numerical contracts independently of the scientific interpretation of any one plot.

## Research papers behind the examples

The lecture connects the examples to Yang and collaborators’ work on APT evaluation, patch-injection and repair games, backup/recovery, conflicting opinions, rumor refutation, impulsive defense and outsourcing, hybrid propaganda competition, awareness-dependent interventions, coupled malware-opinion dynamics, and feedback-loop malware control. Its Research examples section explains each connection and gives the complete formal references, including the three associated research repositories. Read the paper-specific state definitions before adapting a tutorial model.
