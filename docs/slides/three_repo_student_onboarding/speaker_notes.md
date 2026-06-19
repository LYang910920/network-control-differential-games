## 01. From Network Models to Intelligent Cyber Control

Section: Orientation

Purpose: Use the three repositories as one learning path: model, solve, learn, validate, and write.
Explain: Open with the idea that the repos are not separate demos. They are a staged route from equations to paper-level experiments.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 02. Learning outcomes

Section: Orientation

Purpose: After this deck, students should know where to start and what to change next.
Explain: Emphasize practical fluency: run, inspect, modify, validate.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 03. The project pipeline

Section: Orientation

Purpose: Every example follows the same research pattern.
Explain: Use this as the mental map for all later sections.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 04. Three repositories at a glance

Section: Orientation

Purpose: The family is staged: foundation first, then game learning, then PINN/PIDL.
Explain: Make clear that Note 1 and Note 2 build on the foundation package instead of redefining the basic cyber dynamics.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 05. Recommended learning sequence

Section: Orientation

Purpose: Students should run before extending.
Explain: This slide sets expectations for students who may otherwise jump straight to a complex paper model.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 06. Prerequisites and environment

Section: Orientation

Purpose: The examples are intentionally small, but the workflow is research-shaped.
Explain: Point students to the README quick starts. Hardware acceleration helps for neural diagnostics, but smoke tests are CPU-friendly.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 07. Workspace and install

Section: Orientation

Purpose: Keep the foundation package installed once and reused by both companion notes.
Explain: Explain that editable install makes shared fixes immediately visible to Note 1 and Note 2.
Demo cue: pip install -e ../network-control-differential-games
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 08. The first 15-minute success path

Section: Orientation

Purpose: A fast smoke run gives students a known-good baseline.
Explain: The point is not full convergence; it is to confirm the environment and locate the first generated figures.
Demo cue: bash scripts/run_smoke_tests.sh
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 09. Read equations and code together

Section: Orientation

Purpose: Every mathematical object should have a concrete code location.
Explain: This sets up the notation-to-code habit students need before modifying models.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 10. Shared notation and state shapes

Section: Orientation

Purpose: Most mistakes are shape mistakes, so name the shape first.
Explain: Reinforce that labels and captions must say selected node, selected degree, or aggregate mean.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 11. Model granularity

Section: Common foundation

Purpose: Aggregate, degree-level, and node-level models answer different questions.
Explain: Connect this to the earlier scalability work: node-level FBS is expensive because the state dimension grows with nodes.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 12. SIR, SIPS, and SIPRS

Section: Common foundation

Purpose: The state diagram tells students what each compartment can and cannot do.
Explain: Explain that P is a patch/protected state, not the same as R recovery/awareness.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 13. Network pressure convention

Section: Common foundation

Purpose: The code uses one explicit adjacency direction.
Explain: This prevents transposed adjacency bugs when students move to paper-level graph data.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 14. Continuous control

Section: Common foundation

Purpose: A continuous control is a time-indexed curve, not a single scalar.
Explain: Contrast this with the sampled learning action later; sampled decisions are held or applied at epochs.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 15. Sampled decisions versus solver substeps

Section: Common foundation

Purpose: RL decisions happen at t_k; RK4 substeps only integrate the flow.
Explain: This distinction matters for DDQN, CTDE, and MAPPO examples.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 16. Impulse and hybrid control

Section: Common foundation

Purpose: Impulse control changes the state at event times; hybrid control combines jumps and flow.
Explain: Mention that captions and labels should explicitly say continuous, impulse, or hybrid.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 17. Single-controller objective

Section: Common foundation

Purpose: Optimal-control examples balance cyber harm against intervention cost.
Explain: Use this to introduce why baselines matter: reducing infection by spending infinite control is not a useful method.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 18. Differential-game payoff

Section: Common foundation

Purpose: A game adds strategic response to the same dynamical system.
Explain: Keep the claim discipline explicit. Avoid saying Nash unless the evidence supports it.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 19. Open-loop and feedback strategies

Section: Common foundation

Purpose: Open-loop plans depend on time; feedback policies also observe the state.
Explain: This prepares the comparison between nominal-beta FBSM and feedback DDQN robustness.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 20. Method-selection map

Section: Common foundation

Purpose: Choose the method from data, control timing, and information structure.
Explain: Students should be able to justify method choice before coding.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 21. Foundation repo code map

Section: Foundation repository

Purpose: The foundation repo owns the canonical shared pieces.
Explain: Point out that Note 1 and Note 2 import these utilities to keep semantics aligned.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 22. PMP intuition

Section: Foundation repository

Purpose: Costates measure how future cost changes when the state is perturbed.
Explain: Do not over-formalize. This is intuition before equations.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 23. PMP optimality system

Section: Foundation repository

Purpose: The numerical solver follows the state, costate, and stationarity equations.
Explain: Tie each line to code: RHS, adjoint, projection, and convergence plot.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 24. FBSM loop

Section: Foundation repository

Purpose: FBSM alternates forward simulation, backward adjoints, and relaxed updates.
Explain: The iteration axis in convergence figures is the algorithm loop, not physical time.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 25. FBSM code-to-math walkthrough

Section: Foundation repository

Purpose: The same model terms appear in RHS, adjoint, update, metric, and plot label.
Explain: This is pseudocode, not a verbatim source listing. It shows where students should inspect the real functions.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 26. Degree versus node-level scalability

Section: Foundation repository

Purpose: The same epidemic-control problem becomes much larger at node level.
Explain: Use the actual checked-in scalability figure. Stress that log scaling helps compare 100 to 1,000,000 nodes.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 27. Hybrid/impulse workflow

Section: Foundation repository

Purpose: Hybrid examples must label both continuous flow and discrete jumps.
Explain: This reinforces the user's prior requirement about clear labels and captions.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 28. Foundation hands-on checkpoint

Section: Foundation repository

Purpose: Students should leave the foundation repo able to change one model responsibly.
Explain: This checkpoint is intentionally concrete.
Demo cue: python run_all.py --skip-reference --skip-scalability
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 29. Why convert the ODE to an MDP?

Section: Game-learning companion

Purpose: Learning methods need observations, actions, rewards, and transitions.
Explain: This is the conceptual bridge from optimal control to RL.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 30. One sampled transition

Section: Game-learning companion

Purpose: A learning step wraps a continuous-time simulation interval.
Explain: Use t_k for decision epochs and tau_j for exogenous or designed impulse points.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 31. State, observation, action, reward

Section: Game-learning companion

Purpose: Good RL code states exactly what the agent can see and change.
Explain: This is where terms like nominal and unknown proxy must be clear in docs.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 32. DDQN architecture

Section: Game-learning companion

Purpose: DDQN is the beginner-friendly bridge from policy rules to learned feedback.
Explain: Use the existing Note 1 architecture figure.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 33. DDQN run command

Section: Game-learning companion

Purpose: The smoke path is short; the longer diagnostics are optional.
Explain: Explain that smoke tests confirm code paths; longer profiles show learning behavior.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 34. From one defender to multiple agents

Section: Game-learning companion

Purpose: Multi-agent control starts by deciding what is local and what is shared.
Explain: This slide introduces CTDE without calling every multi-agent baseline MAPPO.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 35. Markov game and CTDE

Section: Game-learning companion

Purpose: A Markov game adds multiple policies and coupled rewards.
Explain: Stress the distinction between concept and algorithm.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 36. Compact CTDE versus full MAPPO

Section: Game-learning companion

Purpose: Do not call a compact baseline MAPPO unless it has the MAPPO machinery.
Explain: This responds directly to the prior audit requirement.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 37. MAPPO rollout and update

Section: Game-learning companion

Purpose: MAPPO separates data collection from clipped policy updates.
Explain: Define rollout as a complete sequence under a policy, not just one transition.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 38. Node-SIPRS community environment

Section: Game-learning companion

Purpose: The node-level SIPRS wrapper is the bridge to paper-level graph MARL.
Explain: Keep this aligned with the foundation network_models.py specification.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 39. Cooperative community defense

Section: Game-learning companion

Purpose: The cooperative setting asks regions to reduce global exposure under local observations.
Explain: Use the training diagnostics as a visual anchor while warning about stochastic learning curves.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 40. Attacker-defender response example

Section: Game-learning companion

Purpose: Game outputs should compare responses, not only one learned curve.
Explain: This matches the user's request for baseline comparison framing.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 41. Evaluation terms

Section: Game-learning companion

Purpose: Use plain definitions before showing training curves.
Explain: These terms were previously flagged as unclear; the deck repeats the clarified definitions.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 42. Game-learning hands-on checkpoint

Section: Game-learning companion

Purpose: Students should be able to add one policy and one metric.
Explain: Give a concrete end state for Note 1.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 43. Why physics-informed learning?

Section: PINN/PIDL companion

Purpose: PINN/PIDL methods use equations when data are sparse, noisy, or partial.
Explain: This orients students before the neural examples.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 44. PINN as constrained regression

Section: PINN/PIDL companion

Purpose: The network is trained to fit data and satisfy differential constraints.
Explain: Emphasize that residual consistency is not the same as real-world validation.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 45. Forward, inverse, control, PMP-informed tasks

Section: PINN/PIDL companion

Purpose: The same building blocks support several learning questions.
Explain: Students should name the task before choosing a loss.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 46. Inverse PINN sparse-observation example

Section: PINN/PIDL companion

Purpose: Sparse observations become usable when the residual supplies structure.
Explain: Use the actual Note 2 figure.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 47. PIDL: known mechanism plus correction

Section: PINN/PIDL companion

Purpose: PIDL is useful when part of the mechanism is trusted and part is missing.
Explain: Make the known-plus-unknown split concrete.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 48. Direct neural control

Section: PINN/PIDL companion

Purpose: Direct control PINNs optimize a neural policy against dynamics and objective terms.
Explain: Direct control is not automatically PMP-consistent; that comes in the next slide.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 49. PMP-informed residuals

Section: PINN/PIDL companion

Purpose: PMP-informed PINNs add optimality conditions to the training objective.
Explain: These are schematic terms. Students should inspect the actual loss names in docs/PARAMETERS.md.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 50. Node/graph SIPRS extension

Section: PINN/PIDL companion

Purpose: The graph inverse PINN bridge reuses the same node-SIPRS equations.
Explain: This is the route from toy inverse PINN to paper-level graph experiments.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 51. Evaluation with ground truth

Section: PINN/PIDL companion

Purpose: Synthetic truth lets students measure errors directly.
Explain: This slide encourages honest metrics.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 52. Evaluation without ground truth

Section: PINN/PIDL companion

Purpose: When real truth is unavailable, use residual checks and predictive validation.
Explain: Avoid claiming identifiability without stronger evidence.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 53. PINN/PIDL hands-on checkpoint

Section: PINN/PIDL companion

Purpose: Students should be able to add one loss term and one validation metric.
Explain: Give students a safe extension route.
Demo cue: bash scripts/run_smoke_tests.sh && python scripts/generate_figures.py
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 54. One cross-repo research question

Section: Integration

Purpose: Paper-level work needs model, method, baselines, and evidence.
Explain: This is the cross-repo synthesis slide.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 55. From model to paper

Section: Integration

Purpose: The final deliverable is a disciplined evidence chain, not just a trained model.
Explain: Close the core deck by sending students to docs/from_model_to_paper.md and companion PAPER_WORKFLOW.md files.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 56. Command cheat sheet

Section: Appendix

Purpose: Most workflows use a small set of repeatable commands.
Explain: Use this as the live-demo slide if time is short.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 57. File and function index

Section: Appendix

Purpose: The fastest way to extend is to start from the right file.
Explain: Students should not start by editing generated outputs.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 58. Hyperparameter starting points

Section: Appendix

Purpose: Keep hyperparameters visible and small before scaling.
Explain: This aligns with the user's request that parameters and hyperparameters be easy to see.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 59. Common errors and debugging

Section: Appendix

Purpose: Most failures are environment, shape, or timing mismatches.
Explain: Frame debugging as a reproducibility habit.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 60. Glossary

Section: Appendix

Purpose: Use terms consistently across notes and figures.
Explain: This reinforces the terminology cleanup in the repositories.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.

## 61. Repository URLs and references

Section: Appendix

Purpose: The public repos are the authoritative source for commands and figures.
Explain: End with the licensing reminder: do not redistribute private templates or third-party assets without rights.
Demo cue: connect this slide to the relevant README section.
Common misconception: a plotted curve or training loss is evidence to interpret, not an automatic proof of optimality or generalization.
