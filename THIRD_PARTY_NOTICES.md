# Third-party Notices

This repository references three external research-code repositories. They remain the work of their original authors and retain their original licenses.

## OpinionMalware_TIFS_2025_Code

- Upstream repository: https://github.com/XiaojuanCheng/OpinionMalware_TIFS_2025_Code
- License in upstream repository: Apache License 2.0
- Main files discussed here:
  - `network.py`
  - `opinionMalware.py`
- Research topic: coupled malware-opinion dynamics with optimal impulse control.
- Citation from upstream README:

```text
Xiaojuan Cheng, Lu-Xing Yang, Gang Li, Zenan Ma, Tianqing Zhu, Lidan Wang, Shukai Duan.
"Modeling and Mitigating Social Engineering Malware: Integrating Malware-Opinion Dynamics With Optimal Impulse Control Approaches."
IEEE Transactions on Information Forensics and Security, vol. 20, pp. 12231-12244, 2025.
DOI: 10.1109/TIFS.2025.3630005
```

## PropagandaWar_TIFS_2024_Code

- Upstream repository: https://github.com/XiaojuanCheng/PropagandaWar_TIFS_2024_Code
- License in upstream repository: Apache License 2.0
- Main files discussed here:
  - `demo_network.py`
  - `propWar.py`
  - `comparison.py`
- Research topic: cost-effective hybrid control strategies for propaganda-war differential games.
- Citation from upstream README:

```text
Xiaojuan Cheng, Lu-Xing Yang, Qingyi Zhu, Chenquan Gan, Xiaofan Yang, Gang Li.
"Cost-Effective Hybrid Control Strategies for Dynamical Propaganda War Game."
IEEE Transactions on Information Forensics and Security, vol. 19, pp. 9789-9802, 2024.
DOI: 10.1109/TIFS.2024.3468903
```

## Propaganda_TCSS_2025_Code

- Upstream repository: https://github.com/XiaojuanCheng/Propaganda_TCSS_2025_Code
- License in upstream repository: MIT License
- Main files discussed here:
  - `prop_network.py`
  - `prop_propaganda.py`
  - `prop_cntrlComparison.py`
  - `prop_noAwareness.py`
- Research topic: impulse strategies for suppressing cyber propaganda with awareness.
- Citation from upstream README:

```text
Xiaojuan Cheng, Lu-Xing Yang, Qingyi Zhu, Chenquan Gan, Gang Li.
"Impulse Strategies for Suppressing Cyber Propaganda With Awareness."
IEEE Transactions on Computational Social Systems, vol. 12, no. 5, pp. 2685-2695, 2025.
DOI: 10.1109/TCSS.2024.3522889
```

## How this repository uses them

This repository does not need to vendor the upstream repositories by default. Instead, it provides:

```text
examples/reference/download_reference_repositories.sh
examples/reference/run_reference_smoke.py
examples/reference/patches/opinion_malware_numpy_networkx_compat.patch
```

The smoke runner imports the upstream code after the repositories are downloaded locally. It uses small networks and short horizons to verify that the model workflow can be executed in a teaching environment.

If you intentionally commit downloaded copies of these repositories, preserve their upstream `LICENSE` and `README` files and keep these notices.
