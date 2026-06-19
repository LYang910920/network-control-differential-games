#!/usr/bin/env node
"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { FileBlob, PresentationFile } = require("@oai/artifact-tool");

const ROOT = path.resolve(__dirname, "../../..");
const WORKSPACE = path.dirname(ROOT);
const OUT = __dirname;
const RENDER_DIR = path.join(OUT, "rendered");
const SLIDE_DIR = path.join(RENDER_DIR, "slides");
const ASSET_DIR = path.join(OUT, "assets", "repo_figures");
const QA_DIR = path.join(OUT, "qa");

const COLORS = {
  teal: "#007D98",
  lightTeal: "#58BCAF",
  magenta: "#C74298",
  yellow: "#FFD923",
  orange: "#FF9300",
  grey: "#E1E1E1",
  charcoal: "#222222",
  text: "#17202A",
  muted: "#59656F",
  bg: "#FFFFFF",
  pale: "#F5FAFB",
};

const FIGURES = {
  foundation_contact: {
    src: path.join(ROOT, "examples/foundations/results/companion_contact_sheet.png"),
    file: "foundation_contact_sheet.png",
    alt: "Contact sheet of foundation optimal-control and differential-game figures.",
  },
  scalability: {
    src: path.join(ROOT, "examples/foundations/results/scalability_degree_node_sf/degree_node_fbs_comparison_100_1000000.png"),
    file: "degree_node_fbs_scalability.png",
    alt: "Degree-level and node-level FBS runtime comparison on scale-free graphs.",
  },
  fbs_convergence: {
    src: path.join(ROOT, "examples/foundations/results/companion_builtin_sf/fbs_convergence.png"),
    file: "foundation_fbs_convergence.png",
    alt: "Foundation FBS convergence histories for control and game examples.",
  },
  hybrid: {
    src: path.join(ROOT, "examples/foundations/results/companion_builtin_sf/hybrid_impulse_trajectory.png"),
    file: "foundation_hybrid_impulse.png",
    alt: "Hybrid control trajectory with continuous flow and impulse event markers.",
  },
  reference: {
    src: path.join(ROOT, "examples/reference/results/reference_repos/reference_repo_contact_sheet.png"),
    file: "reference_repo_contact_sheet.png",
    alt: "Reference repository smoke-run contact sheet.",
  },
  note1_arch: {
    src: path.join(WORKSPACE, "note1-cyber-control-games/figures/neural_architectures.png"),
    file: "note1_neural_architectures.png",
    alt: "DDQN, CTDE, and MAPPO architecture overview.",
  },
  note1_timing: {
    src: path.join(WORKSPACE, "note1-cyber-control-games/figures/timing_semantics.png"),
    file: "note1_timing_semantics.png",
    alt: "Timing diagram distinguishing action epochs, solver substeps, and impulse events.",
  },
  note1_fbsm: {
    src: path.join(WORKSPACE, "note1-cyber-control-games/figures/fbsm_malware_control.png"),
    file: "note1_fbsm_malware_control.png",
    alt: "FBSM malware-control baseline with state and control curves.",
  },
  note1_hybrid: {
    src: path.join(WORKSPACE, "note1-cyber-control-games/figures/hybrid_policy_comparison.png"),
    file: "note1_hybrid_policy_comparison.png",
    alt: "Hybrid policy comparison across no-defense, fixed, rule, and learned policies.",
  },
  note1_training: {
    src: path.join(WORKSPACE, "note1-cyber-control-games/figures/training_iteration_diagnostics.png"),
    file: "note1_training_iteration_diagnostics.png",
    alt: "Training diagnostics for FBSM, DDQN, and compact CTDE runs.",
  },
  note1_matrix: {
    src: path.join(WORKSPACE, "note1-cyber-control-games/figures/game_response_matrix.png"),
    file: "note1_game_response_matrix.png",
    alt: "Attacker-defender response matrix with cumulative compromised exposure.",
  },
  note1_robust: {
    src: path.join(WORKSPACE, "note1-cyber-control-games/figures/node_level_learning_advantage.png"),
    file: "note1_node_level_learning_advantage.png",
    alt: "Node-level robustness comparison under nominal and shifted infection pressure.",
  },
  note2_arch: {
    src: path.join(WORKSPACE, "note2-pinn-pidl-cyber-control/figures/neural_architectures.png"),
    file: "note2_neural_architectures.png",
    alt: "PINN, PIDL, neural control, and PMP-informed network blocks.",
  },
  note2_inverse: {
    src: path.join(WORKSPACE, "note2-pinn-pidl-cyber-control/figures/inverse_pinn_sparse_data.png"),
    file: "note2_inverse_pinn_sparse_data.png",
    alt: "Inverse PINN sparse-observation setup.",
  },
  note2_pidl: {
    src: path.join(WORKSPACE, "note2-pinn-pidl-cyber-control/figures/pidl_missing_mechanism.png"),
    file: "note2_pidl_missing_mechanism.png",
    alt: "PIDL known mechanism plus learned correction example.",
  },
  note2_training: {
    src: path.join(WORKSPACE, "note2-pinn-pidl-cyber-control/figures/training_iteration_diagnostics.png"),
    file: "note2_training_iteration_diagnostics.png",
    alt: "PINN and PIDL training diagnostics with separated loss terms.",
  },
  note2_baseline: {
    src: path.join(WORKSPACE, "note2-pinn-pidl-cyber-control/figures/baseline_comparison.png"),
    file: "note2_baseline_comparison.png",
    alt: "Method-specific baseline comparison for learned PINN and PIDL methods.",
  },
};

const slides = [
  s("S01", "Orientation", "From Network Models to Intelligent Cyber Control", "Use the three repositories as one learning path: model, solve, learn, validate, and write.", "title", ["Foundation, game learning, and PINN/PIDL companions", "Deakin-style student onboarding deck", "Built from the current public repositories and generated figures"], { visual: "pipeline", steps: ["Threat", "Model", "Solver", "Evidence", "Paper"] }, "Open with the idea that the repos are not separate demos. They are a staged route from equations to paper-level experiments."),
  s("S02", "Orientation", "Learning outcomes", "After this deck, students should know where to start and what to change next.", "bullets", ["Run the first smoke tests from a clean checkout.", "Map each equation term to a Python variable and output figure.", "Choose between PMP/FBSM, RL/MARL, PINN/PIDL, and hybrid games.", "Design baselines, ablations, and paper-ready figures."], null, "Emphasize practical fluency: run, inspect, modify, validate."),
  s("S03", "Orientation", "The project pipeline", "Every example follows the same research pattern.", "diagram", ["State the cyber process and control surface.", "Pick a solver or learner that matches timing and information.", "Compare with baselines and write the evidence trail."], { visual: "pipeline", steps: ["Threat scenario", "Mathematical model", "Method", "Experiment", "Manuscript"] }, "Use this as the mental map for all later sections."),
  s("S04", "Orientation", "Three repositories at a glance", "The family is staged: foundation first, then game learning, then PINN/PIDL.", "diagram", ["Foundation: notation, shared package, continuous/impulse/hybrid examples.", "Note 1: ODE-to-MDP, DDQN, compact CTDE, node-SIPRS MAPPO.", "Note 2: inverse PINN, PIDL, neural control, PMP-informed residuals."], { visual: "repo-family" }, "Make clear that Note 1 and Note 2 build on the foundation package instead of redefining the basic cyber dynamics."),
  s("S05", "Orientation", "Recommended learning sequence", "Students should run before extending.", "timeline", ["First 15 minutes: install and run smoke examples.", "First week: trace one equation to code and change one parameter.", "Second week: add a baseline and explain the output figure.", "Project stage: extend states, agents, losses, or graph settings."], { visual: "timeline", steps: ["15 min", "1 week", "2 weeks", "Project"] }, "This slide sets expectations for students who may otherwise jump straight to a complex paper model."),
  s("S06", "Orientation", "Prerequisites and environment", "The examples are intentionally small, but the workflow is research-shaped.", "bullets", ["Python 3.10+ with NumPy, SciPy, Matplotlib, pandas, and PyTorch for neural examples.", "Optional python-igraph for scale-free graph generation and scalability figures.", "Device setting: auto picks MPS, CUDA, then CPU when supported.", "Use a sibling-folder workspace for all three repos."], null, "Point students to the README quick starts. Hardware acceleration helps for neural diagnostics, but smoke tests are CPU-friendly."),
  s("S07", "Orientation", "Workspace and install", "Keep the foundation package installed once and reused by both companion notes.", "code", ["git clone https://github.com/LYang910920/network-control-differential-games", "git clone https://github.com/LYang910920/note1-cyber-control-games", "git clone https://github.com/LYang910920/note2-pinn-pidl-cyber-control", "cd note1-cyber-control-games", "pip install -e ../network-control-differential-games", "pip install -r requirements.txt"], { visual: "code" }, "Explain that editable install makes shared fixes immediately visible to Note 1 and Note 2."),
  s("S08", "Orientation", "The first 15-minute success path", "A fast smoke run gives students a known-good baseline.", "code", ["cd network-control-differential-games", "pip install -e .", "python run_all.py --skip-reference --skip-scalability", "cd ../note1-cyber-control-games", "bash scripts/run_smoke_tests.sh", "cd ../note2-pinn-pidl-cyber-control", "bash scripts/run_smoke_tests.sh"], { visual: "code" }, "The point is not full convergence; it is to confirm the environment and locate the first generated figures."),
  s("S09", "Orientation", "Read equations and code together", "Every mathematical object should have a concrete code location.", "two-column", ["Equation: state, control, parameter, payoff, constraint.", "Code: RHS function, solver step, update rule, metric, plot label.", "Output: CSV row, figure caption, and README interpretation."], { visual: "mapping", pairs: [["x(t)", "state array"], ["u(t)", "control profile"], ["f(x,u)", "rhs(...)"], ["J", "metric/payoff"]] }, "This sets up the notation-to-code habit students need before modifying models."),
  s("S10", "Orientation", "Shared notation and state shapes", "Most mistakes are shape mistakes, so name the shape first.", "table", ["Aggregate: one population vector.", "Degree level: one state per degree class k.", "Node level: one state vector per node i.", "Graph pressure: A[i,j] contributes from j to i."], { visual: "shape-table" }, "Reinforce that labels and captions must say selected node, selected degree, or aggregate mean."),

  s("S11", "Common foundation", "Model granularity", "Aggregate, degree-level, and node-level models answer different questions.", "diagram", ["Aggregate: compact and transparent.", "Degree level: keeps heterogeneity by k while staying low-dimensional.", "Node level: keeps topology and local interventions at higher cost."], { visual: "model-levels" }, "Connect this to the earlier scalability work: node-level FBS is expensive because the state dimension grows with nodes."),
  s("S12", "Common foundation", "SIR, SIPS, and SIPRS", "The state diagram tells students what each compartment can and cannot do.", "diagram", ["SIR: susceptible, infected, recovered.", "SIPS: susceptible, infected, patched, susceptible again.", "SIPRS: adds recovered or aware state and waning transitions."], { visual: "state-diagram" }, "Explain that P is a patch/protected state, not the same as R recovery/awareness."),
  s("S13", "Common foundation", "Network pressure convention", "The code uses one explicit adjacency direction.", "bullets", ["A[i,j] is the influence from node j into node i.", "Pressure at node i is a weighted sum over neighbor infection states.", "Normalize pressure when comparing graph sizes.", "Sparse and dense paths should match on small tests."], null, "This prevents transposed adjacency bugs when students move to paper-level graph data."),
  s("S14", "Common foundation", "Continuous control", "A continuous control is a time-indexed curve, not a single scalar.", "diagram", ["The solver samples u(t) on a mesh.", "The true interpretation is a continuously adjustable intervention.", "Plots should show the curve against time, with units or normalized intensity."], { visual: "continuous-control" }, "Contrast this with the sampled learning action later; sampled decisions are held or applied at epochs."),
  s("S15", "Common foundation", "Sampled decisions versus solver substeps", "RL decisions happen at t_k; RK4 substeps only integrate the flow.", "diagram", ["Observe x(t_k).", "Choose action a_k.", "Integrate ODE with internal substeps.", "Return x(t_{k+1})."], { visual: "timing" }, "This distinction matters for DDQN, CTDE, and MAPPO examples."),
  s("S16", "Common foundation", "Impulse and hybrid control", "Impulse control changes the state at event times; hybrid control combines jumps and flow.", "diagram", ["Continuous: u(t) changes rates over time.", "Impulse: x(tau_j+) = G(x(tau_j-), q_j).", "Hybrid: flow between impulses plus jumps at selected times."], { visual: "hybrid" }, "Mention that captions and labels should explicitly say continuous, impulse, or hybrid."),
  s("S17", "Common foundation", "Single-controller objective", "Optimal-control examples balance cyber harm against intervention cost.", "bullets", ["State exposure: infected, compromised, or propaganda-active mass.", "Control cost: patching, filtering, isolation, or awareness effort.", "Terminal cost: final risk or cleanup burden.", "Constraints: state simplex and bounded control."], null, "Use this to introduce why baselines matter: reducing infection by spending infinite control is not a useful method."),
  s("S18", "Common foundation", "Differential-game payoff", "A game adds strategic response to the same dynamical system.", "bullets", ["Defender action reduces harm but costs resources.", "Attacker action increases pressure but also costs effort.", "Open-loop candidates need unilateral-deviation checks before strong claims.", "Response matrices are clearer than isolated payoff numbers."], null, "Keep the claim discipline explicit. Avoid saying Nash unless the evidence supports it."),
  s("S19", "Common foundation", "Open-loop and feedback strategies", "Open-loop plans depend on time; feedback policies also observe the state.", "two-column", ["Open-loop: u(t) is computed before deployment.", "Feedback: pi(x,t) reacts to observed state.", "FBSM often returns open-loop schedules.", "RL/MARL examples learn sampled feedback policies."], { visual: "two-lanes", lanes: ["Open-loop schedule", "Feedback policy"] }, "This prepares the comparison between nominal-beta FBSM and feedback DDQN robustness."),
  s("S20", "Common foundation", "Method-selection map", "Choose the method from data, control timing, and information structure.", "diagram", ["Known equations + continuous control -> PMP/FBSM.", "Sampled decision-making -> DDQN or CTDE/MAPPO.", "Sparse data or missing mechanism -> PINN/PIDL.", "Jump interventions -> impulse or hybrid games."], { visual: "method-map" }, "Students should be able to justify method choice before coding."),

  s("S21", "Foundation repository", "Foundation repo code map", "The foundation repo owns the canonical shared pieces.", "diagram", ["src/cybercontrol: reusable numerics, dynamics, diagnostics, Torch helpers.", "examples/foundations: beginner-readable continuous, game, node, and hybrid examples.", "examples/reference: small smoke runs for paper-level code snapshots."], { visual: "repo-code-map", repo: "foundation" }, "Point out that Note 1 and Note 2 import these utilities to keep semantics aligned."),
  s("S22", "Foundation repository", "PMP intuition", "Costates measure how future cost changes when the state is perturbed.", "bullets", ["Forward: simulate the state under a candidate control.", "Backward: propagate marginal value information.", "Update: improve control from the Hamiltonian condition.", "Repeat until the control update is small."], null, "Do not over-formalize. This is intuition before equations."),
  s("S23", "Foundation repository", "PMP optimality system", "The numerical solver follows the state, costate, and stationarity equations.", "code", ["state:      x_dot = f(x, u, theta)", "costate:   lambda_dot = -H_x(x, u, lambda)", "control:   H_u(x, u, lambda) = 0", "bounds:    u_min <= u(t) <= u_max", "diagnose:  ||u_new - u_old|| over iterations"], { visual: "code" }, "Tie each line to code: RHS, adjoint, projection, and convergence plot."),
  s("S24", "Foundation repository", "FBSM loop", "FBSM alternates forward simulation, backward adjoints, and relaxed updates.", "diagram", ["Initialize control.", "Forward integrate state.", "Backward integrate costate.", "Update and project control.", "Check convergence."], { visual: "loop" }, "The iteration axis in convergence figures is the algorithm loop, not physical time."),
  s("S25", "Foundation repository", "FBSM code-to-math walkthrough", "The same model terms appear in RHS, adjoint, update, metric, and plot label.", "code", ["for it in range(max_iter):", "    x = forward_state(u)", "    lam = backward_costate(x, u)", "    u_candidate = hamiltonian_update(x, lam)", "    u = relax_and_project(u, u_candidate)", "    history.append(norm(u - u_old))"], { visual: "code" }, "This is pseudocode, not a verbatim source listing. It shows where students should inspect the real functions."),
  s("S26", "Foundation repository", "Degree versus node-level scalability", "The same epidemic-control problem becomes much larger at node level.", "image", ["Same synthetic scale-free graph seed for each size.", "Degree-level state uses observed degree classes.", "Node-level state uses one entry per graph node."], { visual: "image", asset: "scalability" }, "Use the actual checked-in scalability figure. Stress that log scaling helps compare 100 to 1,000,000 nodes."),
  s("S27", "Foundation repository", "Hybrid/impulse workflow", "Hybrid examples must label both continuous flow and discrete jumps.", "image", ["Continuous dynamics evolve between event times.", "Impulse events are shown as vertical markers.", "Captions say whether curves are node, degree-class, or aggregate averages."], { visual: "image", asset: "hybrid" }, "This reinforces the user's prior requirement about clear labels and captions."),
  s("S28", "Foundation repository", "Foundation hands-on checkpoint", "Students should leave the foundation repo able to change one model responsibly.", "bullets", ["Run python run_all.py --skip-reference --skip-scalability.", "Open docs/NOTATION_TO_CODE.md and trace one term.", "Change beta or cost weight and predict the figure change.", "Add one baseline and update the caption."], null, "This checkpoint is intentionally concrete."),

  s("S29", "Game-learning companion", "Why convert the ODE to an MDP?", "Learning methods need observations, actions, rewards, and transitions.", "diagram", ["Continuous model gives the flow.", "Environment wrapper selects decision epochs.", "Reward summarizes cyber harm and action cost.", "Policy learns from sampled rollouts."], { visual: "ode-mdp" }, "This is the conceptual bridge from optimal control to RL."),
  s("S30", "Game-learning companion", "One sampled transition", "A learning step wraps a continuous-time simulation interval.", "diagram", ["observe x(t_k-)", "choose action", "apply jump if any", "flow to t_{k+1}", "return reward and next state"], { visual: "pipeline" }, "Use t_k for decision epochs and tau_j for exogenous or designed impulse points."),
  s("S31", "Game-learning companion", "State, observation, action, reward", "Good RL code states exactly what the agent can see and change.", "table", ["State: simulator variables used by the environment.", "Observation: features exposed to the policy.", "Action: discrete or continuous intervention choice.", "Reward: signed scalar aligned with cyber objective."], { visual: "shape-table" }, "This is where terms like nominal and unknown proxy must be clear in docs."),
  s("S32", "Game-learning companion", "DDQN architecture", "DDQN is the beginner-friendly bridge from policy rules to learned feedback.", "image", ["Replay buffer stabilizes updates.", "Target network slows the moving target.", "Evaluation uses deterministic rollouts separate from training noise."], { visual: "image", asset: "note1_arch" }, "Use the existing Note 1 architecture figure."),
  s("S33", "Game-learning companion", "DDQN run command", "The smoke path is short; the longer diagnostics are optional.", "code", ["cd note1-cyber-control-games", "bash scripts/run_smoke_tests.sh", "python scripts/generate_figures.py", "python scripts/run_training_iterations.py --profile quick", "python scripts/run_training_iterations.py --profile gpu --device auto"], { visual: "code" }, "Explain that smoke tests confirm code paths; longer profiles show learning behavior."),
  s("S34", "Game-learning companion", "From one defender to multiple agents", "Multi-agent control starts by deciding what is local and what is shared.", "bullets", ["Agent: community, region, node group, or player.", "Local observation: what each agent sees.", "Centralized critic input: global training context.", "Evaluation: decentralized policies acting from local observations."], null, "This slide introduces CTDE without calling every multi-agent baseline MAPPO."),
  s("S35", "Game-learning companion", "Markov game and CTDE", "A Markov game adds multiple policies and coupled rewards.", "diagram", ["Each agent chooses an action at the same decision epoch.", "The environment applies joint action and continuous flow.", "Training may use centralized information.", "Deployment should respect the observation design."], { visual: "ctde" }, "Stress the distinction between concept and algorithm."),
  s("S36", "Game-learning companion", "Compact CTDE versus full MAPPO", "Do not call a compact baseline MAPPO unless it has the MAPPO machinery.", "two-column", ["Compact CTDE: small actor/critic demonstration, short logs, limited claims.", "Full MAPPO: rollout buffer, GAE, clipping, entropy/value losses, minibatch epochs, masks."], { visual: "compare", left: "Compact CTDE", right: "Full MAPPO" }, "This responds directly to the prior audit requirement."),
  s("S37", "Game-learning companion", "MAPPO rollout and update", "MAPPO separates data collection from clipped policy updates.", "diagram", ["Collect rollout.", "Compute returns and GAE.", "Shuffle minibatches.", "Update actor and critic with clipping.", "Evaluate on fixed seeds."], { visual: "loop" }, "Define rollout as a complete sequence under a policy, not just one transition."),
  s("S38", "Game-learning companion", "Node-SIPRS community environment", "The node-level SIPRS wrapper is the bridge to paper-level graph MARL.", "bullets", ["Nodes carry S, I, P, and R shares or probabilities.", "Communities map node groups to defender agents.", "Actions allocate patching, filtering, or recovery effort.", "Rewards combine infected exposure and intervention costs."], null, "Keep this aligned with the foundation network_models.py specification."),
  s("S39", "Game-learning companion", "Cooperative community defense", "The cooperative setting asks regions to reduce global exposure under local observations.", "image", ["The smoke baseline is small enough to inspect.", "Use multiple seeds before making learning claims.", "Report infection exposure, cost, peak, final state, and constraint errors."], { visual: "image", asset: "note1_training" }, "Use the training diagnostics as a visual anchor while warning about stochastic learning curves."),
  s("S40", "Game-learning companion", "Attacker-defender response example", "Game outputs should compare responses, not only one learned curve.", "image", ["Fix attacker and vary defender policies.", "Fix defender and vary attacker policies.", "Use payoff or exposure metrics with clear directionality."], { visual: "image", asset: "note1_matrix" }, "This matches the user's request for baseline comparison framing."),
  s("S41", "Game-learning companion", "Evaluation terms", "Use plain definitions before showing training curves.", "bullets", ["Nominal: parameter setting used for training or planning.", "Robustness: lower exposure under specified mismatch, not a guarantee.", "Unknown proxy: observable feature used when the true hidden count is unavailable.", "Rollout: simulator trajectory under a fixed policy or parameter set."], null, "These terms were previously flagged as unclear; the deck repeats the clarified definitions."),
  s("S42", "Game-learning companion", "Game-learning hands-on checkpoint", "Students should be able to add one policy and one metric.", "bullets", ["Run bash scripts/run_smoke_tests.sh.", "Open docs/PARAMETERS.md before changing hyperparameters.", "Add a rule-based baseline in the policy comparison table.", "Explain whether the curve is training, evaluation, or rollout behavior."], null, "Give a concrete end state for Note 1."),

  s("S43", "PINN/PIDL companion", "Why physics-informed learning?", "PINN/PIDL methods use equations when data are sparse, noisy, or partial.", "bullets", ["Data loss fits observations.", "Residual loss penalizes violation of known dynamics.", "Boundary or initial losses anchor the trajectory.", "Control or PMP losses encode decision structure."], null, "This orients students before the neural examples."),
  s("S44", "PINN/PIDL companion", "PINN as constrained regression", "The network is trained to fit data and satisfy differential constraints.", "diagram", ["Neural state function.", "Autograd derivative.", "ODE residual.", "Data and boundary loss.", "Validation rollout."], { visual: "pinn" }, "Emphasize that residual consistency is not the same as real-world validation."),
  s("S45", "PINN/PIDL companion", "Forward, inverse, control, PMP-informed tasks", "The same building blocks support several learning questions.", "table", ["Forward PINN: learn state trajectory with known parameters.", "Inverse PINN: estimate hidden parameters from sparse data.", "Control PINN: learn control and state networks together.", "PMP-informed PINN: add costate and stationarity residuals."], { visual: "shape-table" }, "Students should name the task before choosing a loss."),
  s("S46", "PINN/PIDL companion", "Inverse PINN sparse-observation example", "Sparse observations become usable when the residual supplies structure.", "image", ["Observed points are not a full trajectory.", "The residual links hidden states and parameters.", "Held-out state error checks whether the reconstruction is meaningful."], { visual: "image", asset: "note2_inverse" }, "Use the actual Note 2 figure."),
  s("S47", "PINN/PIDL companion", "PIDL: known mechanism plus correction", "PIDL is useful when part of the mechanism is trusted and part is missing.", "image", ["Keep known SIR/SIPRS terms explicit.", "Train a correction network for the missing term.", "Evaluate learned correction and rollout behavior separately."], { visual: "image", asset: "note2_pidl" }, "Make the known-plus-unknown split concrete."),
  s("S48", "PINN/PIDL companion", "Direct neural control", "Direct control PINNs optimize a neural policy against dynamics and objective terms.", "bullets", ["State network approximates trajectory.", "Control network outputs bounded intervention intensity.", "Objective loss represents exposure plus effort.", "Constraint penalties keep state and control feasible."], null, "Direct control is not automatically PMP-consistent; that comes in the next slide."),
  s("S49", "PINN/PIDL companion", "PMP-informed residuals", "PMP-informed PINNs add optimality conditions to the training objective.", "code", ["loss = data_loss", "loss += w_ode * ||x_t - f(x,u)||^2", "loss += w_adj * ||lambda_t + H_x||^2", "loss += w_stat * ||H_u||^2", "loss += w_bc * boundary_loss"], { visual: "code" }, "These are schematic terms. Students should inspect the actual loss names in docs/PARAMETERS.md."),
  s("S50", "PINN/PIDL companion", "Node/graph SIPRS extension", "The graph inverse PINN bridge reuses the same node-SIPRS equations.", "bullets", ["Generate synthetic truth from the shared graph simulator.", "Observe sparse nodes, states, or times.", "Train against data, graph residual, and constraint losses.", "Report state, parameter, residual, and held-out graph metrics."], null, "This is the route from toy inverse PINN to paper-level graph experiments."),
  s("S51", "PINN/PIDL companion", "Evaluation with ground truth", "Synthetic truth lets students measure errors directly.", "image", ["Use held-out time points and nodes.", "Track separate losses rather than one aggregate loss.", "Compare to interpolation, wrong-parameter rollout, and known-only baselines."], { visual: "image", asset: "note2_training" }, "This slide encourages honest metrics."),
  s("S52", "PINN/PIDL companion", "Evaluation without ground truth", "When real truth is unavailable, use residual checks and predictive validation.", "image", ["Held-out observations: does the model predict unseen data?", "Residual consistency: does it obey known equations?", "Rollout stability: does the learned mechanism produce plausible trajectories?"], { visual: "image", asset: "note2_baseline" }, "Avoid claiming identifiability without stronger evidence."),
  s("S53", "PINN/PIDL companion", "PINN/PIDL hands-on checkpoint", "Students should be able to add one loss term and one validation metric.", "bullets", ["Run bash scripts/run_smoke_tests.sh.", "Open docs/PARAMETERS.md for loss weights and network widths.", "Remove or change one loss term and predict the failure mode.", "Add a held-out metric before increasing model complexity."], null, "Give students a safe extension route."),

  s("S54", "Integration", "One cross-repo research question", "Paper-level work needs model, method, baselines, and evidence.", "diagram", ["Foundation: define the node-SIPRS model.", "Note 1: learn feedback/game policies.", "Note 2: infer unknown graph mechanisms.", "Paper: compare under clear limits."], { visual: "research-workflow" }, "This is the cross-repo synthesis slide."),
  s("S55", "Integration", "From model to paper", "The final deliverable is a disciplined evidence chain, not just a trained model.", "timeline", ["Question", "Equations", "Baselines", "Method", "Ablations", "Validation", "Figures", "Claims"], { visual: "timeline", steps: ["Question", "Model", "Method", "Evidence", "Paper"] }, "Close the core deck by sending students to docs/from_model_to_paper.md and companion PAPER_WORKFLOW.md files."),

  s("A01", "Appendix", "Command cheat sheet", "Most workflows use a small set of repeatable commands.", "code", ["python run_all.py --skip-reference --skip-scalability", "bash scripts/run_smoke_tests.sh", "python scripts/generate_figures.py", "python scripts/run_training_iterations.py --profile quick", "pytest", "python -m compileall ."], { visual: "code" }, "Use this as the live-demo slide if time is short."),
  s("A02", "Appendix", "File and function index", "The fastest way to extend is to start from the right file.", "table", ["Foundation: src/cybercontrol/network_models.py.", "Foundation examples: examples/foundations/code/.", "Note 1 profiles: src/scenario_profiles.py.", "Note 2 profiles: src/experiment_profiles.py."], { visual: "shape-table" }, "Students should not start by editing generated outputs."),
  s("A03", "Appendix", "Hyperparameter starting points", "Keep hyperparameters visible and small before scaling.", "bullets", ["Read docs/PARAMETERS.md first in each repo.", "Smoke runs use short horizons and few iterations by design.", "Neural diagnostics expose learning rate, width, epochs, seeds, and device.", "Scale one axis at a time: graph size, horizon, seeds, or network width."], null, "This aligns with the user's request that parameters and hyperparameters be easy to see."),
  s("A04", "Appendix", "Common errors and debugging", "Most failures are environment, shape, or timing mismatches.", "bullets", ["Import error: install the foundation package in editable mode.", "Shape error: check aggregate, degree, or node-level state dimensions.", "Flat control curve: check cost weights and bounds.", "Unclear plot: fix label before changing the method."], null, "Frame debugging as a reproducibility habit."),
  s("A05", "Appendix", "Glossary", "Use terms consistently across notes and figures.", "bullets", ["Nominal: parameter setting used for planning or training.", "Rollout: simulator trajectory under fixed policy/parameters.", "Robustness: performance under a named shift, not a proof.", "Hybrid: continuous flow plus discrete jumps."], null, "This reinforces the terminology cleanup in the repositories."),
  s("A06", "Appendix", "Repository URLs and references", "The public repos are the authoritative source for commands and figures.", "bullets", ["Foundation URL: see the README.", "Note 1 and Note 2 URLs: see related repos.", "Reference snapshots keep upstream notices.", "Do not redistribute private templates without rights."], { visual: "image", asset: "reference" }, "End with the licensing reminder: do not redistribute private templates or third-party assets without rights."),
];

function s(id, section, title, takeaway, layout, body, visual, notes) {
  return {
    id,
    section,
    title,
    takeaway,
    layout,
    body,
    visual: visual || null,
    notes,
    citation: "Repository files and generated figures from the three-repo tutorial family.",
    demoCommand: demoFor(section, title),
  };
}

function demoFor(section, title) {
  if (title.includes("Workspace")) return "pip install -e ../network-control-differential-games";
  if (title.includes("15-minute")) return "bash scripts/run_smoke_tests.sh";
  if (section === "Foundation repository") return "python run_all.py --skip-reference --skip-scalability";
  if (section === "Game-learning companion") return "bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py";
  if (section === "PINN/PIDL companion") return "bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py";
  return "";
}

async function main() {
  const template = resolveTemplate();
  await resetDirs();
  const copiedAssets = await copyFigureAssets();
  const presentation = await PresentationFile.importPptx(await FileBlob.load(template));
  while (presentation.slides.items.length > 0) {
    presentation.slides.getItem(0).delete();
  }
  applyTheme(presentation);

  const manifest = [];
  for (const [idx, item] of slides.entries()) {
    const slide = presentation.slides.add({ layout: layoutFor(item) });
    slide.background.fill = item.layout === "title" || item.layout === "section" ? COLORS.teal : COLORS.bg;
    buildSlide(slide, item, idx + 1, copiedAssets, manifest);
    slide.speakerNotes.textFrame.setText(notesFor(item));
    slide.speakerNotes.setVisible(true);
  }

  const deckContentPath = path.join(OUT, "deck_content.json");
  const notesPath = path.join(OUT, "speaker_notes.md");
  const manifestPath = path.join(OUT, "asset_manifest.csv");
  await fs.writeFile(deckContentPath, JSON.stringify(slides, null, 2) + "\n", "utf8");
  await fs.writeFile(notesPath, buildNotesMarkdown(), "utf8");
  await fs.writeFile(manifestPath, buildManifestCsv(manifest), "utf8");

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(SLIDE_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    await fs.writeFile(path.join(SLIDE_DIR, `${stem}.layout.json`), await (await slide.export({ format: "layout" })).text());
  }
  await writeBlob(path.join(RENDER_DIR, "contact_sheet.webp"), await presentation.export({ format: "webp", montage: { columns: 6, slideWidth: 240, gap: 12, padding: 16, background: "#FFFFFF" }, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(presentation);
  const pptxPath = path.join(OUT, "three_repo_student_onboarding_deakin.pptx");
  await pptx.save(pptxPath);

  await makePdfAndPngContactSheet();
  await writeStaticAudit();
  await writeReadme(template);
  console.log(`Built ${slides.length} slides`);
  console.log(pptxPath);
}

function resolveTemplate() {
  const argIndex = process.argv.indexOf("--template");
  const argTemplate = argIndex >= 0 ? process.argv[argIndex + 1] : "";
  const candidates = [
    argTemplate,
    process.env.DEAKIN_TEMPLATE,
    path.join(WORKSPACE, "_taskpacks/codex_three_repo_deakin_ppt_pack_v4/assets/Deakin_TEAL_16x9.potx"),
    path.join(process.env.HOME || "", "Downloads/codex_three_repo_deakin_ppt_pack_v4/assets/Deakin_TEAL_16x9.potx"),
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      require("node:fs").accessSync(candidate);
      return path.resolve(candidate);
    } catch {
      continue;
    }
  }
  throw new Error("Template not found. Pass --template /path/to/Deakin_TEAL_16x9.potx or set DEAKIN_TEMPLATE.");
}

async function resetDirs() {
  await fs.mkdir(OUT, { recursive: true });
  await fs.rm(RENDER_DIR, { recursive: true, force: true });
  await fs.rm(ASSET_DIR, { recursive: true, force: true });
  await fs.rm(QA_DIR, { recursive: true, force: true });
  await fs.mkdir(SLIDE_DIR, { recursive: true });
  await fs.mkdir(ASSET_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
}

async function copyFigureAssets() {
  const copied = {};
  for (const [key, info] of Object.entries(FIGURES)) {
    const dst = path.join(ASSET_DIR, info.file);
    try {
      await fs.copyFile(info.src, dst);
      copied[key] = { ...info, dst, exists: true };
    } catch {
      copied[key] = { ...info, dst, exists: false };
    }
  }
  return copied;
}

function applyTheme(presentation) {
  presentation.theme.colorScheme = {
    name: "Deakin TEAL",
    themeColors: {
      accent1: COLORS.teal,
      accent2: COLORS.lightTeal,
      accent3: COLORS.magenta,
      accent4: COLORS.yellow,
      accent5: COLORS.orange,
      accent6: COLORS.grey,
      bg1: COLORS.bg,
      bg2: COLORS.pale,
      tx1: COLORS.text,
      tx2: COLORS.muted,
      dk1: "#000000",
      dk2: COLORS.charcoal,
      lt1: "#FFFFFF",
      lt2: COLORS.grey,
      hlink: COLORS.teal,
      folHlink: COLORS.magenta,
    },
  };
}

function layoutFor(item) {
  if (item.layout === "title") return "Title Slide";
  if (item.layout === "section") return "Section Slide A";
  return "Title and Content A";
}

function buildSlide(slide, item, page, assets, manifest) {
  if (item.layout === "title") {
    buildTitleSlide(slide, item, page);
    return;
  }
  if (item.layout === "section") {
    buildSectionSlide(slide, item, page);
    return;
  }
  addChrome(slide, item, page);
  addTitle(slide, item.title, 70, 64, 760, 64, 31, COLORS.text);
  addText(slide, item.takeaway, 70, 126, 920, 42, 19, false, COLORS.teal);

  if (item.visual?.visual === "image") {
    addBullets(slide, item.body, 70, 184, 355, 300, 21);
    addImage(slide, item, assets, manifest, 455, 184, 745, 412);
  } else if (item.visual?.visual === "compare") {
    drawCompare(slide, item.visual.left, item.visual.right, 120, 205, 1040, 290);
  } else if (item.layout === "code") {
    addCodeBlock(slide, item.body, 82, 190, 760, 300);
    addSidebar(slide, ["Run it", "Use the full command in speaker notes or README.", "Then inspect the generated CSV and figure."], 900, 190, 290, 300);
  } else if (item.layout === "table") {
    drawSimpleTable(slide, item.body, 90, 190, 1080, 330);
  } else if (item.layout === "two-column") {
    drawTwoColumn(slide, item);
  } else if (item.layout === "timeline") {
    addBullets(slide, item.body, 76, 188, 440, 315, 21);
    drawTimeline(slide, item.visual?.steps || item.body, 555, 210, 600, 240);
  } else if (item.visual) {
    addBullets(slide, item.body, 74, 188, 390, 330, 21);
    drawVisual(slide, item, 505, 184, 670, 360);
  } else {
    addBullets(slide, item.body, 105, 205, 980, 320, 24);
  }
  addTakeawayBand(slide, item.takeaway);
}

function buildTitleSlide(slide, item, page) {
  addRect(slide, 0, 0, 1280, 720, COLORS.teal, "none");
  addRect(slide, 0, 540, 1280, 180, COLORS.charcoal, "none");
  addText(slide, "DEAKIN", 910, 46, 260, 46, 26, true, "#FFFFFF");
  addTitle(slide, item.title, 84, 115, 820, 160, 43, "#FFFFFF");
  addText(slide, item.body.join("\n"), 88, 310, 760, 110, 23, false, "#FFFFFF");
  drawFlow(slide, item.visual.steps, 105, 580, 760, 70, "#FFFFFF", COLORS.yellow);
  addText(slide, `Slide ${page} | Three-repo tutorial family`, 86, 675, 520, 26, 14, false, "#FFFFFF");
}

function buildSectionSlide(slide, item, page) {
  addRect(slide, 0, 0, 1280, 720, COLORS.teal, "none");
  addText(slide, item.section.toUpperCase(), 90, 86, 520, 34, 17, true, COLORS.yellow);
  addTitle(slide, item.title, 90, 150, 850, 98, 42, "#FFFFFF");
  addText(slide, item.takeaway, 92, 278, 800, 56, 23, false, "#FFFFFF");
  addBullets(slide, item.body, 105, 385, 760, 160, 24, "#FFFFFF");
  addText(slide, `DEAKIN | ${page}`, 1025, 664, 180, 26, 13, true, "#FFFFFF");
}

function addChrome(slide, item, page) {
  addRect(slide, 0, 0, 1280, 22, COLORS.teal, "none");
  addText(slide, item.section, 70, 28, 470, 24, 13, true, COLORS.teal);
  addText(slide, "DEAKIN", 1072, 28, 120, 24, 14, true, COLORS.teal);
  addRect(slide, 70, 620, 1060, 1, COLORS.grey, "none");
  addText(slide, "Network control tutorial family | Deakin University CRICOS Provider Code: 00113B", 70, 642, 800, 24, 12, false, COLORS.muted);
  addText(slide, String(page).padStart(2, "0"), 1140, 638, 58, 30, 16, true, COLORS.teal);
}

function addTakeawayBand(slide, text) {
  addRect(slide, 70, 548, 1060, 48, COLORS.pale, COLORS.lightTeal);
  addText(slide, text, 92, 560, 1015, 24, 15, false, COLORS.text);
}

function addText(slide, text, left, top, width, height, fontSize, bold, color) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { fontSize: toPptPx(fontSize), bold: !!bold, color, fontFace: "Calibri" };
  return shape;
}

function addTitle(slide, text, left, top, width, height, fontSize, color) {
  try {
    const title = slide.placeholders.getItem("title");
    title.text = text;
    title.text.style = { fontSize: toPptPx(fontSize), bold: true, color, fontFace: "Calibri" };
    title.position = { left, top, width, height };
    return title;
  } catch {
    return addText(slide, text, left, top, width, height, fontSize, true, color);
  }
}

function toPptPx(fontSizePt) {
  return Math.max(24, Math.round(fontSizePt * 4 / 3));
}

function addBullets(slide, bullets, left, top, width, height, fontSize, color = COLORS.text) {
  const text = bullets.map((b) => `- ${b}`).join("\n");
  return addText(slide, text, left, top, width, height, fontSize, false, color);
}

function addCodeBlock(slide, lines, left, top, width, height) {
  addRect(slide, left, top, width, height, "#F4F6F7", "#B8C4CC");
  addText(slide, lines.join("\n"), left + 24, top + 24, width - 48, height - 48, 18, false, COLORS.charcoal);
}

function addSidebar(slide, lines, left, top, width, height) {
  addRect(slide, left, top, width, height, COLORS.pale, COLORS.lightTeal);
  addText(slide, lines[0], left + 22, top + 24, width - 44, 28, 20, true, COLORS.teal);
  addText(slide, lines.slice(1).join("\n\n"), left + 22, top + 72, width - 44, height - 92, 18, false, COLORS.text);
}

function addRect(slide, left, top, width, height, fill, lineFill = COLORS.grey, radius = "rounded-sm") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1 },
    borderRadius: radius,
  });
}

function drawVisual(slide, item, left, top, width, height) {
  const type = item.visual.visual;
  if (type === "pipeline") return drawFlow(slide, item.visual.steps || item.body, left, top + 60, width, 130, COLORS.teal, COLORS.yellow);
  if (type === "repo-family") return drawRepoFamily(slide, left, top, width, height);
  if (type === "model-levels") return drawModelLevels(slide, left, top, width, height);
  if (type === "state-diagram") return drawStateDiagram(slide, left, top, width, height);
  if (type === "continuous-control") return drawContinuousControl(slide, left, top, width, height);
  if (type === "timing") return drawTiming(slide, left, top, width, height);
  if (type === "hybrid") return drawHybrid(slide, left, top, width, height);
  if (type === "loop") return drawLoop(slide, item.visual.steps || item.body, left, top, width, height);
  if (type === "method-map") return drawMethodMap(slide, left, top, width, height);
  if (type === "ode-mdp") return drawFlow(slide, ["ODE flow", "Env wrapper", "Reward", "Policy"], left, top + 80, width, 110, COLORS.teal, COLORS.orange);
  if (type === "ctde") return drawCTDE(slide, left, top, width, height);
  if (type === "pinn") return drawFlow(slide, ["Network", "Autograd", "Residual", "Loss", "Validation"], left, top + 80, width, 110, COLORS.magenta, COLORS.yellow);
  if (type === "research-workflow") return drawRepoFamily(slide, left, top, width, height, true);
  if (type === "mapping") return drawMapping(slide, item.visual.pairs, left, top, width, height);
  if (type === "two-lanes") return drawLanes(slide, item.visual.lanes, left, top, width, height);
  if (type === "compare") return drawCompare(slide, item.visual.left, item.visual.right, left, top, width, height);
  if (type === "shape-table") return drawSimpleTable(slide, item.body, left, top, width, height);
  if (type === "repo-code-map") return drawRepoCodeMap(slide, left, top, width, height);
}

function drawFlow(slide, steps, left, top, width, height, fill, accent) {
  const gap = 18;
  const boxW = (width - gap * (steps.length - 1)) / steps.length;
  steps.forEach((step, i) => {
    const x = left + i * (boxW + gap);
    addRect(slide, x, top, boxW, height, i % 2 === 0 ? fill : COLORS.lightTeal, "none");
    addText(slide, step, x + 12, top + 26, boxW - 24, height - 36, 18, true, "#FFFFFF");
    if (i < steps.length - 1) {
      addText(slide, ">", x + boxW + 3, top + height / 2 - 13, 16, 22, 18, true, accent);
    }
  });
}

function drawRepoFamily(slide, left, top, width, height, research = false) {
  const labels = research
    ? [["Foundation", "model and simulator"], ["Note 1", "feedback/game learning"], ["Note 2", "inverse/missing terms"]]
    : [["Foundation", "math, package, examples"], ["Note 1", "RL, CTDE, MAPPO"], ["Note 2", "PINN, PIDL, PMP residuals"]];
  labels.forEach((row, i) => {
    const y = top + 25 + i * 102;
    addRect(slide, left + i * 28, y, width - i * 56, 72, i === 0 ? COLORS.teal : i === 1 ? COLORS.magenta : COLORS.orange, "none");
    addText(slide, row[0], left + 24 + i * 28, y + 12, 180, 26, 22, true, "#FFFFFF");
    addText(slide, row[1], left + 220 + i * 28, y + 16, width - 280, 24, 20, false, "#FFFFFF");
  });
}

function drawModelLevels(slide, left, top, width, height) {
  const rows = [["Aggregate", "few states"], ["Degree k", "degree-class states"], ["Node i", "graph-size states"]];
  rows.forEach((r, i) => {
    const x = left + i * (width / 3);
    addRect(slide, x, top + 75, width / 3 - 26, 150, i === 0 ? COLORS.lightTeal : i === 1 ? COLORS.teal : COLORS.magenta, "none");
    addText(slide, r[0], x + 18, top + 108, width / 3 - 62, 28, 24, true, "#FFFFFF");
    addText(slide, r[1], x + 18, top + 154, width / 3 - 62, 52, 19, false, "#FFFFFF");
  });
}

function drawStateDiagram(slide, left, top, width, height) {
  const nodes = [["S", left + 30, top + 120, COLORS.lightTeal], ["I", left + 225, top + 120, COLORS.magenta], ["P", left + 420, top + 50, COLORS.teal], ["R", left + 420, top + 190, COLORS.orange]];
  nodes.forEach(([label, x, y, color]) => {
    addRect(slide, x, y, 96, 70, color, "none");
    addText(slide, label, x + 30, y + 15, 36, 34, 30, true, "#FFFFFF");
  });
  [["infection", 145, 132], ["patch", 330, 72], ["recover", 330, 212], ["wane", 535, 140]].forEach(([txt, dx, dy]) => {
    addText(slide, `${txt} >`, left + dx, top + dy, 110, 24, 15, true, COLORS.charcoal);
  });
}

function drawContinuousControl(slide, left, top, width, height) {
  addRect(slide, left, top + 45, width, 250, "#FFFFFF", COLORS.grey);
  addText(slide, "u(t)", left + 25, top + 50, 60, 24, 16, true, COLORS.teal);
  addText(slide, "time", left + width - 80, top + 270, 60, 22, 14, false, COLORS.muted);
  for (let i = 0; i < 9; i++) {
    const x = left + 70 + i * 55;
    const y = top + 220 - Math.sin(i / 1.2) * 44 - i * 5;
    addRect(slide, x, y, 42, 5, COLORS.magenta, "none");
  }
  addText(slide, "sampled mesh represents a continuous intervention curve", left + 85, top + 320, width - 100, 30, 17, false, COLORS.text);
}

function drawTiming(slide, left, top, width, height) {
  addText(slide, "Decision epochs t_k", left, top + 40, 220, 24, 18, true, COLORS.teal);
  addRect(slide, left + 20, top + 110, width - 80, 4, COLORS.teal, "none");
  [0, 1, 2, 3, 4].forEach((i) => {
    const x = left + 45 + i * 120;
    addRect(slide, x, top + 96, 8, 32, COLORS.teal, "none");
    addText(slide, `t_${i}`, x - 8, top + 135, 36, 22, 13, false, COLORS.text);
  });
  addText(slide, "RK4 substeps are internal", left + 110, top + 175, 300, 24, 17, false, COLORS.muted);
}

function drawHybrid(slide, left, top, width, height) {
  drawTiming(slide, left, top, width, height);
  [left + 178, left + 418].forEach((x) => {
    addRect(slide, x, top + 56, 4, 185, COLORS.orange, "none");
    addText(slide, "jump", x - 18, top + 30, 48, 22, 13, true, COLORS.orange);
  });
}

function drawLoop(slide, steps, left, top, width, height) {
  const positions = [[left + 210, top + 20], [left + 430, top + 130], [left + 300, top + 260], [left + 70, top + 235], [left + 40, top + 80]];
  steps.slice(0, 5).forEach((step, i) => {
    const [x, y] = positions[i];
    addRect(slide, x, y, 190, 62, i % 2 ? COLORS.lightTeal : COLORS.teal, "none");
    addText(slide, step, x + 14, y + 15, 162, 30, 17, true, "#FFFFFF");
  });
  addText(slide, "repeat until convergence", left + 215, top + 350, 260, 28, 18, true, COLORS.magenta);
}

function drawMethodMap(slide, left, top, width, height) {
  const rows = [["Known ODE", "PMP/FBSM"], ["Sampled actions", "DDQN / CTDE / MAPPO"], ["Sparse data", "PINN / PIDL"], ["Jumps", "Impulse / hybrid game"]];
  drawSimpleTable(slide, rows.map((r) => `${r[0]} -> ${r[1]}`), left, top + 25, width, 275);
}

function drawMapping(slide, pairs, left, top, width, height) {
  pairs.forEach(([a, b], i) => {
    const y = top + 40 + i * 62;
    addRect(slide, left, y, 180, 44, COLORS.teal, "none");
    addText(slide, a, left + 16, y + 10, 145, 22, 18, true, "#FFFFFF");
    addText(slide, ">", left + 205, y + 10, 24, 22, 18, true, COLORS.orange);
    addRect(slide, left + 250, y, 250, 44, COLORS.pale, COLORS.lightTeal);
    addText(slide, b, left + 268, y + 10, 215, 22, 18, false, COLORS.text);
  });
}

function drawLanes(slide, lanes, left, top, width, height) {
  lanes.forEach((lane, i) => {
    addRect(slide, left + i * (width / 2), top + 70, width / 2 - 28, 210, i === 0 ? COLORS.teal : COLORS.magenta, "none");
    addText(slide, lane, left + 24 + i * (width / 2), top + 96, width / 2 - 78, 38, 25, true, "#FFFFFF");
    addText(slide, i === 0 ? "time-indexed plan" : "state-aware policy", left + 24 + i * (width / 2), top + 158, width / 2 - 78, 30, 19, false, "#FFFFFF");
  });
}

function drawCompare(slide, leftLabel, rightLabel, left, top, width, height) {
  const copy = [["Short tutorial baseline", "Full algorithmic machinery"], ["Limited claims", "Multi-seed evaluation"], ["Readable first", "Research-ready next"]];
  [leftLabel, rightLabel].forEach((label, i) => {
    const x = left + i * (width / 2);
    addRect(slide, x, top + 40, width / 2 - 32, 260, i === 0 ? COLORS.lightTeal : COLORS.teal, "none");
    addText(slide, label, x + 22, top + 64, width / 2 - 76, 34, 24, true, "#FFFFFF");
    addBullets(slide, copy.map((r) => r[i]), x + 28, top + 120, width / 2 - 90, 120, 18, "#FFFFFF");
  });
}

function drawCTDE(slide, left, top, width, height) {
  addRect(slide, left + 40, top + 70, 170, 70, COLORS.lightTeal, "none");
  addRect(slide, left + 40, top + 200, 170, 70, COLORS.lightTeal, "none");
  addRect(slide, left + 320, top + 130, 230, 92, COLORS.teal, "none");
  addText(slide, "Actor 1", left + 85, top + 92, 80, 24, 18, true, "#FFFFFF");
  addText(slide, "Actor 2", left + 85, top + 222, 80, 24, 18, true, "#FFFFFF");
  addText(slide, "Centralized critic", left + 348, top + 160, 175, 28, 20, true, "#FFFFFF");
  addText(slide, "local observations", left + 20, top + 300, 220, 24, 16, false, COLORS.muted);
  addText(slide, "global training signal", left + 340, top + 250, 220, 24, 16, false, COLORS.muted);
}

function drawRepoCodeMap(slide, left, top, width, height) {
  const rows = [
    ["src/cybercontrol", "shared reusable package"],
    ["examples/foundations", "clean beginner examples"],
    ["examples/reference", "paper-level smoke runs"],
  ];
  rows.forEach((r, i) => {
    const y = top + 55 + i * 80;
    addRect(slide, left, y, 260, 52, COLORS.teal, "none");
    addText(slide, r[0], left + 18, y + 14, 225, 22, 18, true, "#FFFFFF");
    addText(slide, r[1], left + 295, y + 14, 320, 22, 18, false, COLORS.text);
  });
}

function drawTwoColumn(slide, item) {
  const mid = Math.ceil(item.body.length / 2);
  addBullets(slide, item.body.slice(0, mid), 90, 205, 480, 260, 22);
  addBullets(slide, item.body.slice(mid), 645, 205, 480, 260, 22);
  addRect(slide, 610, 205, 2, 250, COLORS.lightTeal, "none");
  if (item.visual) drawVisual(slide, item, 645, 250, 430, 210);
}

function drawSimpleTable(slide, rows, left, top, width, height) {
  const rowH = Math.min(62, height / Math.max(rows.length, 1));
  rows.forEach((row, i) => {
    const y = top + i * rowH;
    addRect(slide, left, y, width, rowH - 8, i % 2 === 0 ? COLORS.pale : "#FFFFFF", COLORS.grey);
    addText(slide, row, left + 22, y + 14, width - 44, rowH - 22, 19, i === 0 && rows.length <= 4, COLORS.text);
  });
}

function drawTimeline(slide, steps, left, top, width, height) {
  addRect(slide, left + 25, top + 118, width - 70, 5, COLORS.teal, "none");
  steps.slice(0, 5).forEach((step, i) => {
    const x = left + 35 + i * ((width - 110) / Math.max(steps.length - 1, 1));
    addRect(slide, x, top + 92, 54, 54, i % 2 ? COLORS.magenta : COLORS.teal, "none");
    addText(slide, String(i + 1), x + 18, top + 106, 18, 22, 18, true, "#FFFFFF");
    addText(slide, step, x - 28, top + 160, 110, 42, 15, false, COLORS.text);
  });
}

function addImage(slide, item, assets, manifest, left, top, width, height) {
  const key = item.visual.asset;
  const asset = assets[key];
  if (!asset || !asset.exists) {
    addRect(slide, left, top, width, height, COLORS.pale, COLORS.lightTeal);
    addText(slide, "Figure missing at build time", left + 32, top + 150, width - 64, 40, 24, true, COLORS.teal);
    return;
  }
  const bytes = require("node:fs").readFileSync(asset.dst);
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  slide.images.add({
    blob: arrayBuffer,
    contentType: "image/png",
    alt: asset.alt,
    fit: "contain",
    position: { left, top, width, height },
  });
  manifest.push({
    slide: item.id,
    title: item.title,
    asset: path.relative(OUT, asset.dst),
    source: path.relative(WORKSPACE, asset.src),
    license: "Generated from owned tutorial repository outputs; see repository license and notices.",
    alt: asset.alt,
  });
}

function notesFor(item) {
  return [
    `Purpose: ${item.takeaway}`,
    `Explain: ${item.notes}`,
    item.demoCommand ? `Demo cue: ${item.demoCommand}` : "Demo cue: connect this slide to the relevant README section.",
    "Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.",
  ].join("\n");
}

function buildNotesMarkdown() {
  return slides.map((item, idx) => [
    `## ${String(idx + 1).padStart(2, "0")}. ${item.title}`,
    "",
    `Section: ${item.section}`,
    "",
    notesFor(item),
    "",
  ].join("\n")).join("\n");
}

function buildManifestCsv(rows) {
  const header = ["slide_id", "slide_title", "asset_path", "source_path", "license_or_ownership", "alt_text"];
  const allRows = [header, ...rows.map((r) => [r.slide, r.title, r.asset, r.source, r.license, r.alt])];
  return allRows.map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}

function csvCell(value) {
  const s = String(value ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}

async function makePdfAndPngContactSheet() {
  const py = `
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import sys, math
slide_dir = Path(sys.argv[1])
pdf_path = Path(sys.argv[2])
sheet_path = Path(sys.argv[3])
pngs = sorted(slide_dir.glob("slide-*.png"))
images = [Image.open(p).convert("RGB") for p in pngs]
if images:
    images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=96)
thumb_w = 240
thumbs = []
for i, img in enumerate(images, 1):
    thumb = ImageOps.contain(img, (thumb_w, 135))
    canvas = Image.new("RGB", (thumb_w, 165), "white")
    canvas.paste(thumb, (0, 0))
    ImageDraw.Draw(canvas).text((8, 142), f"{i:02d}", fill=(0, 125, 152))
    thumbs.append(canvas)
cols = 6
rows = math.ceil(len(thumbs) / cols) if thumbs else 1
sheet = Image.new("RGB", (cols * thumb_w, rows * 165), "white")
for i, thumb in enumerate(thumbs):
    sheet.paste(thumb, ((i % cols) * thumb_w, (i // cols) * 165))
sheet.save(sheet_path)
`;
  execFileSync("python3", ["-c", py, SLIDE_DIR, path.join(OUT, "three_repo_student_onboarding_deakin.pdf"), path.join(RENDER_DIR, "contact_sheet.png")], { stdio: "inherit" });
}

async function writeStaticAudit() {
  const audit = {
    slide_count: slides.length,
    core_slide_count: slides.filter((x) => !x.id.startsWith("A")).length,
    appendix_slide_count: slides.filter((x) => x.id.startsWith("A")).length,
    duplicate_ids: duplicates(slides.map((x) => x.id)),
    missing_titles: slides.filter((x) => !x.title).map((x) => x.id),
    dense_body_slides: slides.filter((x) => x.body.join(" ").length > 650).map((x) => x.id),
    code_line_limit_issues: slides.filter((x) => x.layout === "code" && x.body.length > 12).map((x) => x.id),
    generated_outputs: [
      "three_repo_student_onboarding_deakin.pptx",
      "three_repo_student_onboarding_deakin.pdf",
      "deck_content.json",
      "speaker_notes.md",
      "asset_manifest.csv",
      "rendered/contact_sheet.png",
      "rendered/slides/slide-XX.png",
    ],
  };
  await fs.writeFile(path.join(QA_DIR, "deck_static_audit.json"), JSON.stringify(audit, null, 2) + "\n", "utf8");
}

function duplicates(values) {
  const seen = new Set();
  const dupes = new Set();
  for (const value of values) {
    if (seen.has(value)) dupes.add(value);
    seen.add(value);
  }
  return [...dupes];
}

async function writeReadme(template) {
  const readme = `# Three-Repo Student Onboarding Deck

This folder contains a Deakin-style onboarding deck for the three-repository tutorial family:

- network-control-differential-games
- note1-cyber-control-games
- note2-pinn-pidl-cyber-control

The deck is built from the current repository text and generated figures. The private Deakin POTX template is used as a local input and is not committed.

## Outputs

- \`three_repo_student_onboarding_deakin.pptx\`: editable PowerPoint deck.
- \`three_repo_student_onboarding_deakin.pdf\`: image-based PDF generated from rendered slides.
- \`deck_content.json\`: slide IDs, titles, takeaways, notes, commands, and visual metadata.
- \`speaker_notes.md\`: presenter notes for every slide.
- \`asset_manifest.csv\`: figure source, ownership/license note, slide usage, and alt text.
- \`rendered/contact_sheet.png\`: visual QA contact sheet.
- \`rendered/slides/\`: one PNG render and layout JSON per slide.
- \`qa/deck_static_audit.json\`: lightweight content audit.

## Rebuild

From this folder:

\`\`\`bash
NODE_PATH=/Users/ylx910920/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \\
/Users/ylx910920/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node build_deck.cjs \\
  --template /path/to/Deakin_TEAL_16x9.potx
\`\`\`

The Deakin template is intentionally passed as a local file path. Do not commit the private POTX unless redistribution rights are confirmed. If the sibling Note 1 or Note 2 repositories are absent, the deck still builds, but slides that depend on their generated figures show a missing-figure placeholder.

## Teaching Use

Use the core slides for a 60-90 minute onboarding session. The appendix gives command, file, hyperparameter, debugging, glossary, and reference slides for live support.
`;
  await fs.writeFile(path.join(OUT, "README.md"), readme, "utf8");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
