# Start Here

This page is the compact map for the tutorial. You can ignore most files at first.

Some historical file and folder names still contain `lecture` for compatibility. They refer to the tutorial note, tutorial examples, and generated tutorial results.

## Big Picture

```text
tutorial note
  -> degree-level and node-level network models
  -> continuous control, differential games, impulse control, hybrid control
  -> runnable tutorial examples
  -> paper-level reference smoke runs
```

## Five-Minute Path

1. Open `docs/lecture_note.pdf` for the tutorial narrative and mathematical setup.
2. Run `python run_all.py --skip-reference --skip-scalability` to verify the tutorial examples quickly.
3. Open `examples/lecture/results/README.md` to see what each generated figure means.
4. Open `docs/PARAMETERS.md` before changing rates, horizons, bounds, or impulse settings.
5. Run `python run_all.py --skip-tutorial` to check the reference-code smoke runs.
6. Read `examples/reference/MODEL_TAXONOMY.md` before inspecting the paper-level repositories.
7. Read `docs/EXTENDING.md` before adapting the code to a paper-specific model.

## Folder Map

| Path | Purpose |
| --- | --- |
| `docs/` | tutorial note, code walkthrough, model-adaptation guide |
| `docs/PARAMETERS.md` | main parameters, solver settings, baseline counts, and state-label meanings |
| `examples/lecture/` | self-contained tutorial examples and generated figures |
| `examples/reference/` | reference repository snapshots and smoke-run wrapper |
| `examples/reference/MODEL_TAXONOMY.md` | degree-level/node-level and continuous/impulse/hybrid classification |
| `docs/EXTENDING.md` | student-facing path from tutorial code to paper models |

## Code Reading Order

1. `examples/lecture/code/simple_degree_k_control.py`: minimal degree-k continuous optimal control.
2. `examples/lecture/code/model_profiles.py`: smoke-run parameters such as horizon, rates, bounds, and impulse times.
3. `examples/lecture/code/network_control_examples.py`: degree-level, node-level, game, and hybrid examples.
4. `examples/lecture/code/scalability_analysis.py`: synthetic scale-free timing experiment using `python-igraph`.
5. `examples/reference/run_reference_smoke.py`: paper-level smoke-run wrapper for the three reference repositories.

## Parameter First

Each tutorial output folder writes `parameter_summary.csv`. Start there when you want concrete values such as `T`, grid size, infection/contact rates, recovery rates, control bounds, impulse times, and random-baseline count.
