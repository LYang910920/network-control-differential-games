# Network control and differential games

Learn how a network becomes a dynamical model, how Pontryagin's principle produces a control, and how the same model extends to games and instantaneous interventions.

**Start here:** run one degree-class control example, inspect its state and control curves, then follow [the walkthrough](docs/GUIDE.md). The [lecture notes](docs/network_control_foundations.pdf) develop PMP, HJB, differential games and hybrid jump conditions.

## Run the example

Use Python 3.11 or 3.12. From this repository:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1

python -m pip install -e .
python -m cybercontrol
```

The example runs on a CPU and creates `results/`: two figures, the empirical degree distribution, the degree-class trajectories and a policy comparison table. It compares a PMP candidate with no intervention and the best of nine constant intervention levels. No external dataset, GPU or LaTeX installation is needed.

```bash
python -m cybercontrol --beta 0.15 --control-cost 4 --output-dir results/another_case
python -m cybercontrol --edge-list examples/foundations/sample_data/sample_edges.csv --delimiter , --has-header --source-col source --target-col target --weight-col weight
```

For whitespace-separated `source target` data, use `--edge-list data/edges.txt` without the CSV options. The degree distribution uses the **whole input graph** and does not require a dense adjacency matrix.

## Read and extend

| Purpose | Read |
|---|---|
| Run, interpret and adapt the example | [Walkthrough](docs/GUIDE.md) |
| Follow the complete small experiment | [`cli.py`](src/cybercontrol/cli.py), `run_example` |
| Trace the mathematics into the solver | [`heterogeneity.py`](src/cybercontrol/heterogeneity.py) and [`fbs.py`](src/cybercontrol/fbs.py) |
| Explore node models, games and jumps | The extension section of the walkthrough |
| Read the three research implementations | [Reference examples](examples/reference/README.md) |

This is the common numerical library for [Note 1: learning control and games](https://github.com/LYang910920/note1-cyber-control-games) and [Note 2: PINN and PIDL](https://github.com/LYang910920/note2-pinn-pidl-cyber-control). Start with the classical model here; use Note 1 for learned feedback policies and Note 2 for inference from sparse observations.

For development, install `python -m pip install -e ".[dev]"` and run `pytest -q`. Numerical tests are kept in `tests/`, separate from the student experiment. Licensing and upstream attribution are described in [COPYRIGHT_AND_LICENSE.md](COPYRIGHT_AND_LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
