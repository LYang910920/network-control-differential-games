# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

from __future__ import annotations

import importlib
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]
PACKAGE_DIR = HERE
REF_DIR = PACKAGE_DIR / "reference_repositories"
OUT_DIR = HERE / "results" / "reference_repos_rerun"
PYDEPS = HERE / "pydeps"

if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

OUT_DIR.mkdir(parents=True, exist_ok=True)


def empirical_degree_distribution(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    degrees = np.array([d for _, d in graph.degree()], dtype=int)
    k, counts = np.unique(degrees, return_counts=True)
    pk = counts / counts.sum()
    return k.astype(float), pk.astype(float)


def apply_opinion_malware_compat(repo: Path) -> None:
    """Patch old matrix .A1 assumptions for current NumPy/NetworkX versions."""
    path = repo / "opinionMalware.py"
    text = path.read_text()
    if "def _flat(x):" in text:
        return

    text = text.replace(
        "from network import create_multiplex_network_with_connected_nodes_edges, normalized_matrix\n",
        "from network import create_multiplex_network_with_connected_nodes_edges, normalized_matrix\n\n\n"
        "def _flat(x):\n"
        "    return np.asarray(x).ravel()\n",
    )
    replacements = {
        "((A @ c[t]).A1)": "_flat(A @ c[t])",
        "((W - D) @ o[t]).A1": "_flat((W - D) @ o[t])",
        "mat_lamb1.A1": "_flat(mat_lamb1)",
        "mat_lamb2.A1": "_flat(mat_lamb2)",
        "mat_mu3.A1": "_flat(mat_mu3)",
        "((1 - o[t]) @ W).A1": "_flat((1 - o[t]) @ W)",
        "(W @ (1 - rho)).A1": "_flat(W @ (1 - rho))",
        "D = np.diag(D.A1)": "D = np.diag(_flat(D))",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)


def run_opinion_malware() -> dict[str, float]:
    repo = REF_DIR / "OpinionMalware_TIFS_2025_Code"
    apply_opinion_malware_compat(repo)
    sys.path.insert(0, str(repo))
    om = importlib.import_module("opinionMalware")

    file_path = PACKAGE_DIR / "sample_data" / "opinion_malware_edges.edges"
    _, social_network, scale_free_network = om.create_multiplex_network_with_connected_nodes_edges(
        str(file_path), max_nodes=60, seed=3, map=True, visualize=False
    )

    om.A = nx.adjacency_matrix(scale_free_network).todense()
    W = om.normalized_matrix(nx.adjacency_matrix(social_network))
    om.W = W
    om.D = np.diag(om._flat(np.sum(W, axis=1)))
    om.nodes = social_network.nodes

    om.T = 0.6
    om.h = 0.01
    om.t_scale = int(om.T / om.h) + 1
    n = len(om.nodes)

    om.c = np.full((om.t_scale, n), 0.5)
    om.o = np.full((om.t_scale, n), 0.3)
    om.lamb = np.zeros((om.t_scale, n))
    om.mu = np.zeros((om.t_scale, n))
    om.omega = 0.2
    om.zeta = 0.2
    om.rho = np.full(n, 0.7)
    om.u1 = np.zeros(om.t_scale)
    om.u2 = np.zeros(om.t_scale)
    om.optimal_u1 = np.zeros(om.t_scale)
    om.optimal_u2 = np.zeros(om.t_scale)
    om.up = 5
    om.low = 0.1
    om.step = 0.4
    om.u1_low = om.low + 0.4
    om.u1_upp = om.up
    om.u2_low = om.low + 0.2
    om.u2_upp = om.up - 2
    om.step_u = om.step

    imp_a = om.impsequence(start=8, interval=18, tscale=om.t_scale, max_count=3, mode=2)
    imp_b = om.impsequence(start=10, interval=20, tscale=om.t_scale, max_count=3, mode=2)

    maxiter = 5
    J = np.zeros(maxiter + 1)
    om.c = om.forward_C(imp_a)
    om.o = om.forward_O(imp_b)
    J[0] = om.payoff()
    for idx in range(maxiter):
        om.lamb = np.zeros((om.t_scale, n))
        om.mu = np.zeros((om.t_scale, n))
        om.backward_dlamb(imp_a)
        om.backward_dmu(imp_b)
        om.u1, om.u2 = om.optimalStrategy(imp_a, imp_b)
        om.c = om.forward_C(imp_a)
        om.o = om.forward_O(imp_b)
        J[idx + 1] = om.payoff()

    t = np.arange(om.t_scale) * om.h
    pd.DataFrame({"iteration": np.arange(maxiter + 1), "J": J}).to_csv(
        OUT_DIR / "opinion_malware_payoff.csv", index=False
    )
    pd.DataFrame(
        {
            "time": t,
            "mean_malware_c": om.c.mean(axis=1),
            "mean_opinion_o": om.o.mean(axis=1),
            "u1": om.u1,
            "u2": om.u2,
        }
    ).to_csv(OUT_DIR / "opinion_malware_timeseries.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(np.arange(maxiter + 1), J, marker="o")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("J")
    axes[0].set_title("OpinionMalware payoff")
    axes[1].plot(t, om.c.mean(axis=1), label="mean c")
    axes[1].plot(t, om.o.mean(axis=1), label="mean o")
    for tau in imp_a:
        axes[1].axvline(tau * om.h, color="tab:red", alpha=0.25, linewidth=1)
    for tau in imp_b:
        axes[1].axvline(tau * om.h, color="tab:green", alpha=0.25, linewidth=1)
    axes[1].set_xlabel("time")
    axes[1].set_title("state trajectories")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "opinion_malware.png", dpi=180)
    plt.close(fig)

    sys.path.remove(str(repo))
    return {"nodes": float(n), "J0": float(J[0]), "J_final": float(J[-1])}


def run_propaganda_war() -> dict[str, float]:
    repo = REF_DIR / "PropagandaWar_TIFS_2024_Code"
    sys.path.insert(0, str(repo))
    pw = importlib.import_module("propWar")

    red_graph = nx.barabasi_albert_graph(36, 2, seed=10)
    blue_graph = nx.watts_strogatz_graph(34, 4, 0.18, seed=11)
    pw.kr, pw.pkr = empirical_degree_distribution(red_graph)
    pw.kb, pw.pkb = empirical_degree_distribution(blue_graph)
    pw.avd_r = float(np.dot(pw.kr, pw.pkr))
    pw.avd_b = float(np.dot(pw.kb, pw.pkb))

    pw.T = 0.8
    pw.h = 0.02
    pw.t_interval = int(pw.T / pw.h) + 1
    pw.pulse_interval_r = 12
    pw.pulse_interval_b = 10
    pw.maxiter = 5
    pw.step = 0.25

    pw.beta_r = 0.02
    pw.gamma_r1 = 0.03
    pw.gamma_r2 = 0.01
    pw.delta_r = 0.02
    pw.beta_b = 0.03
    pw.gamma_b1 = 0.04
    pw.gamma_b2 = 0.02
    pw.delta_b = 0.01

    pw.c_r = 0.6
    pw.c_b = 0.05
    pw.brp = pw.brc = pw.lr = pw.bbp = pw.bbc = pw.lb = 1
    pw.ur_low = pw.vr_low = 1
    pw.ur_upp = pw.vr_upp = 3
    pw.ub_low = pw.vb_low = 1
    pw.ub_upp = pw.vb_upp = 3

    pw.pr = np.full((pw.t_interval, len(pw.kr)), 0.3)
    pw.cr = np.full((pw.t_interval, len(pw.kr)), 0.4)
    pw.pb = np.full((pw.t_interval, len(pw.kb)), 0.1)
    pw.cb = np.full((pw.t_interval, len(pw.kb)), 0.2)
    pw.lambda_r = np.zeros((pw.t_interval, len(pw.kr)))
    pw.mu_r = np.zeros((pw.t_interval, len(pw.kr)))
    pw.lambda_b = np.zeros((pw.t_interval, len(pw.kb)))
    pw.mu_b = np.zeros((pw.t_interval, len(pw.kb)))
    pw.phi_r = np.zeros((pw.t_interval, len(pw.kr)))
    pw.psi_r = np.zeros((pw.t_interval, len(pw.kr)))
    pw.phi_b = np.zeros((pw.t_interval, len(pw.kb)))
    pw.psi_b = np.zeros((pw.t_interval, len(pw.kb)))
    pw.ur = np.zeros(pw.t_interval)
    pw.ub = np.zeros(pw.t_interval)
    pw.vr = np.zeros(pw.t_interval)
    pw.vb = np.zeros(pw.t_interval)

    jr = np.zeros(pw.maxiter + 1)
    jb = np.zeros(pw.maxiter + 1)
    pw.pr, pw.cr = pw.forwardRed()
    pw.pb, pw.cb = pw.forwardBlue()
    jr[0] = pw.payoffRed()
    jb[0] = pw.payoffBlue()
    for idx in range(pw.maxiter):
        pw.lambda_r, pw.mu_r, pw.lambda_b, pw.mu_b = pw.backwardRed()
        pw.phi_r, pw.psi_r, pw.phi_b, pw.psi_b = pw.backwardBlue()
        pw.vr, pw.ur = pw.optimalStrategyRed()
        pw.vb, pw.ub = pw.optimalStrategyBlue()
        pw.pr, pw.cr = pw.forwardRed()
        pw.pb, pw.cb = pw.forwardBlue()
        jr[idx + 1] = pw.payoffRed()
        jb[idx + 1] = pw.payoffBlue()

    t = np.arange(pw.t_interval) * pw.h
    pd.DataFrame({"iteration": np.arange(pw.maxiter + 1), "J_red": jr, "J_blue": jb}).to_csv(
        OUT_DIR / "propaganda_war_payoff.csv", index=False
    )
    pd.DataFrame(
        {
            "time": t,
            "mean_pr": pw.pr @ pw.pkr,
            "mean_cr": pw.cr @ pw.pkr,
            "mean_pb": pw.pb @ pw.pkb,
            "mean_cb": pw.cb @ pw.pkb,
            "ur": pw.ur,
            "ub": pw.ub,
            "vr": pw.vr,
            "vb": pw.vb,
        }
    ).to_csv(OUT_DIR / "propaganda_war_timeseries.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(np.arange(pw.maxiter + 1), jr, marker="o", label="J red")
    axes[0].plot(np.arange(pw.maxiter + 1), jb, marker="o", label="J blue")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("PropagandaWar game payoff")
    axes[0].legend()
    axes[1].plot(t, pw.pr @ pw.pkr, label="red P")
    axes[1].plot(t, pw.cr @ pw.pkr, label="red C")
    axes[1].plot(t, pw.pb @ pw.pkb, label="blue P")
    axes[1].plot(t, pw.cb @ pw.pkb, label="blue C")
    axes[1].set_xlabel("time")
    axes[1].set_title("degree-level states")
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "propaganda_war.png", dpi=180)
    plt.close(fig)

    sys.path.remove(str(repo))
    return {"red_degree_classes": float(len(pw.kr)), "J_red_final": float(jr[-1]), "J_blue_final": float(jb[-1])}


def run_propaganda_tcss() -> dict[str, float]:
    repo = REF_DIR / "Propaganda_TCSS_2025_Code"
    sys.path.insert(0, str(repo))
    pp = importlib.import_module("prop_propaganda")

    A = pd.read_csv(PACKAGE_DIR / "sample_data" / "sample_adjacency.csv", header=None).to_numpy(float)
    A[A < 0] = 0
    np.fill_diagonal(A, 0)
    n = A.shape[0]
    T = 0.8
    h = 0.02
    t_interval = int(T / h) + 1
    maxiter = 5
    pulse_interval = 10
    beta1, beta2, eta, delta, gamma_a, gamma_u, omega = 0.001, 0.002, 0.001, 0.001, 0.002, 0.001, 0.01
    cg = cf = 1
    a_low, a_upp, u_low, u_upp = 0.1, 0.8, 0.3, 1.0

    sa = np.full((t_interval, n), 0.2)
    su = np.full((t_interval, n), 0.2)
    r = np.full((t_interval, n), 0.5)
    lambda_a = np.zeros((t_interval, n))
    lambda_u = np.zeros((t_interval, n))
    mu = np.zeros((t_interval, n))
    ca = np.zeros(t_interval)
    cu = np.zeros(t_interval)
    J = np.zeros(maxiter + 1)

    sa, su, r = pp.forward(
        sa, su, r, ca, cu, t_interval, pulse_interval, A,
        beta1, beta2, eta, delta, gamma_a, gamma_u, h,
        pp.g_func, pp.f_func, cg, cf,
    )
    J[0] = pp.profit_sim(r, ca, cu, t_interval, pulse_interval, h, omega)
    for idx in range(maxiter):
        lambda_a, lambda_u, mu = pp.backward(
            lambda_a, lambda_u, mu, ca, cu, sa, su, r, t_interval,
            pulse_interval, A, beta1, beta2, eta, delta, gamma_a,
            gamma_u, omega, h, cg, cf,
        )
        ca, cu = pp.policy_sim(
            lambda_a, lambda_u, mu, sa, su, ca, cu, t_interval,
            pulse_interval, a_low, a_upp, u_low, u_upp, cg, cf,
        )
        sa, su, r = pp.forward(
            sa, su, r, ca, cu, t_interval, pulse_interval, A,
            beta1, beta2, eta, delta, gamma_a, gamma_u, h,
            pp.g_func, pp.f_func, cg, cf,
        )
        J[idx + 1] = pp.profit_sim(r, ca, cu, t_interval, pulse_interval, h, omega)

    t = np.arange(t_interval) * h
    pd.DataFrame({"iteration": np.arange(maxiter + 1), "J": J}).to_csv(
        OUT_DIR / "propaganda_tcss_payoff.csv", index=False
    )
    pd.DataFrame(
        {
            "time": t,
            "mean_sa": sa.mean(axis=1),
            "mean_su": su.mean(axis=1),
            "mean_r": r.mean(axis=1),
            "ca": ca,
            "cu": cu,
        }
    ).to_csv(OUT_DIR / "propaganda_tcss_timeseries.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(np.arange(maxiter + 1), J, marker="o")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("J")
    axes[0].set_title("TCSS OIC profit")
    axes[1].plot(t, sa.mean(axis=1), label="Sa")
    axes[1].plot(t, su.mean(axis=1), label="Su")
    axes[1].plot(t, r.mean(axis=1), label="R")
    axes[1].set_xlabel("time")
    axes[1].set_title("node-level states")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "propaganda_tcss.png", dpi=180)
    plt.close(fig)

    sys.path.remove(str(repo))
    return {"nodes": float(n), "J0": float(J[0]), "J_final": float(J[-1])}


def make_contact_sheet() -> None:
    images = [
        ("OpinionMalware", OUT_DIR / "opinion_malware.png"),
        ("PropagandaWar", OUT_DIR / "propaganda_war.png"),
        ("Propaganda TCSS", OUT_DIR / "propaganda_tcss.png"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5))
    for ax, (title, path) in zip(axes, images):
        ax.imshow(plt.imread(path))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reference_repo_contact_sheet.png", dpi=180)
    plt.close(fig)


def main() -> None:
    global PACKAGE_DIR, REF_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description="Run lightweight smoke tests for the three reference repositories.")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=PACKAGE_DIR,
        help="Path to the examples/reference folder containing reference_repositories/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where reference-repo smoke-run outputs will be written.",
    )
    parser.add_argument(
        "--pydeps",
        type=Path,
        default=PYDEPS,
        help="Optional local dependency directory containing python-igraph.",
    )
    args = parser.parse_args()

    PACKAGE_DIR = args.package_dir.resolve()
    REF_DIR = PACKAGE_DIR / "reference_repositories"
    OUT_DIR = args.output_dir.resolve()
    if args.pydeps.exists():
        sys.path.insert(0, str(args.pydeps.resolve()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append({"repo": "OpinionMalware_TIFS_2025_Code", **run_opinion_malware()})
    rows.append({"repo": "PropagandaWar_TIFS_2024_Code", **run_propaganda_war()})
    rows.append({"repo": "Propaganda_TCSS_2025_Code", **run_propaganda_tcss()})
    pd.DataFrame(rows).to_csv(OUT_DIR / "smoke_run_summary.csv", index=False)
    make_contact_sheet()
    print(f"saved outputs to {OUT_DIR}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
