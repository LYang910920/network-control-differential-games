# Impulse Strategies for Suppressing Cyber Propaganda With Awareness

This repository contains the source code accompanying our paper:

> Xiaojuan Cheng, Lu-Xing Yang, Qingyi Zhu, Chenquan Gan, Gang Li.
> **"Impulse Strategies for Suppressing Cyber Propaganda With Awareness."**
> *IEEE Transactions on Computational Social Systems*, vol. 12, no. 5, pp. 2685–2695, 2025.
> DOI: [10.1109/TCSS.2024.3522889](https://doi.org/10.1109/TCSS.2024.3522889)
> IEEE Xplore: <https://ieeexplore.ieee.org/document/10835850>

## Overview

We propose an **optimal impulse control (OIC)** framework for suppressing the spread of cyber propaganda on online social networks. The model captures how user **awareness** and **network topology** jointly shape propagation dynamics, and derives impulse strategies that minimize a cost functional combining propaganda prevalence and intervention effort.

The code implements:

- The forward propagation model on a given network.
- The backward adjoint system for the co-states.
- The iterative search over impulse policies (`policy_sim`) and the corresponding payoff (`profit_sim`).
- A no-awareness baseline used for the comparison experiments in the paper.
- A random-strategy baseline.

## Files

| File | Purpose |
| --- | --- |
| `prop_network.py` | Loads a network from a `.csv` edgelist or a `.mtx` Matrix Market file and returns its adjacency matrix. |
| `prop_propaganda.py` | Main algorithm — forward / backward / policy iteration and the profit functional. Entry point for the OIC experiments. |
| `prop_cntrlComparison.py` | Comparison between the optimal impulse strategy and random impulse strategies. |
| `prop_noAwareness.py` | Baseline model without the awareness component, used in the ablation experiment. |
| `prop_noAwareness_plot.py` | Plotting utilities for the no-awareness baseline results. |

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `python-igraph`
- `scipy`

Install:

```bash
pip install numpy pandas matplotlib python-igraph scipy
```

## Datasets

The experiments in the paper use three public social-network datasets. They are **not redistributed** here — please download them from the original sources and place the files under `./data/network_data/`:

- **Facebook Politician Pages** (`politician_edges.csv`) — SNAP / GEMSEC dataset.
  <https://snap.stanford.edu/data/gemsec-Facebook.html>
- **Twitter Retweet (rt-retweet-crawl)** (`rt-retweet-crawl.mtx`) — Network Repository.
  <https://networkrepository.com/rt-retweet-crawl.php>
- **YouTube social network (soc-youtube)** (`soc-youtube.mtx`) — Network Repository.
  <https://networkrepository.com/soc-youtube.php>

The expected directory layout (relative to the repository root) is:

```
./data/network_data/politician_edges.csv
./data/network_data/rt-retweet-crawl.mtx
./data/network_data/soc-youtube.mtx
./data/exp_data/...           # generated at runtime
./data/exp1-2_fig/ ...        # generated at runtime
```

## Usage

Run the main OIC experiment (Facebook politician network by default):

```bash
python prop_propaganda.py
```

Run the random-strategy comparison:

```bash
python prop_cntrlComparison.py
```

Run the no-awareness baseline:

```bash
python prop_noAwareness.py
```

Model and control parameters (`T`, `h`, `N`, `beta1`, `beta2`, `eta`, `delta`, `gamma_a`, `gamma_u`, `omega`, `pulse_interval`, impulse bounds, etc.) can be edited at the top of each script. The output `.csv` files and figures are written to the `./data/exp_data/` and `./data/expX_fig/` directories; create them beforehand if you want the writes in the scripts to succeed.

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{10835850,
  author={Cheng, Xiaojuan and Yang, Lu-Xing and Zhu, Qingyi and Gan, Chenquan and Li, Gang},
  journal={IEEE Transactions on Computational Social Systems},
  title={Impulse Strategies for Suppressing Cyber Propaganda With Awareness},
  year={2025},
  volume={12},
  number={5},
  pages={2685-2695},
  doi={10.1109/TCSS.2024.3522889}
}
```

## License

This code is released under the [MIT License](LICENSE).
