# Foundation Examples

Install the repository once, then use the shared runner:

```bash
python -m pip install -e ".[graph,torch]"
python -m cybercontrol smoke
python -m cybercontrol medium --output-dir artifacts/medium
```

For a focused edit, run one thin script from the repository root:

```bash
python examples/foundations/code/simple_degree_k_control.py \
  --steps 48 --output-dir artifacts/simple_degree

python examples/foundations/code/network_control_examples.py \
  --steps 45 --iterations 45 --tolerance 1e-4 \
  --output-dir artifacts/foundation_models

python examples/foundations/code/scalability_analysis.py \
  --model-level compare --sizes 100,1000,10000 \
  --steps 32 --iterations 45 --output-dir artifacts/scalability
```

`simple_degree_k_control.py` is the minimal heterogeneous control example.
`network_control_examples.py` is the advanced orchestrator; its algorithms live
in `cybercontrol.fbs`, `cybercontrol.heterogeneity` and
`cybercontrol.foundation_evaluation`. Generated CSVs and figures belong under
ignored `artifacts/`.
