# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

from __future__ import annotations

import importlib
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str((HERE / ".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((HERE / ".cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT_DIR = HERE.parents[1]
PACKAGE_DIR = HERE
REF_DIR = PACKAGE_DIR / "reference_repositories"
OUT_DIR = HERE / "results" / "reference_repos_rerun"
PYDEPS = HERE / "pydeps"
EXAMPLES_DIR = HERE.parent
LINE_WIDTH = 2.0
FIGSIZE_REFERENCE = (14.4, 3.9)
BASELINE_ROWS: list[dict[str, object]] = []
CONVERGENCE_ROWS: list[dict[str, object]] = []

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

OUT_DIR.mkdir(parents=True, exist_ok=True)

from common_diagnostics import (  # noqa: E402
    RANDOM_BASELINE_COUNT,
    RANDOM_BASELINE_SEED,
    random_impulse_series,
    save_control_baseline_plot,
    save_game_baseline_plot,
    smooth_random_controls,
    write_baseline_table,
)


def apply_clean_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None,
                     title: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.8)


def plot_time_series(ax, t: np.ndarray, y: np.ndarray, label: str, **kwargs) -> None:
    kwargs.setdefault("linewidth", LINE_WIDTH)
    ax.plot(t, y, label=label, **kwargs)


def event_indices_from_values(
    values: np.ndarray,
    *,
    event_indices: np.ndarray | list[int] | None = None,
    tol: float = 1e-9,
    drop_endpoints: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if event_indices is None:
        indices = np.flatnonzero(np.abs(values) > tol)
    else:
        indices = np.asarray(event_indices, dtype=int)
    indices = indices[(indices >= 0) & (indices < len(values))]
    if drop_endpoints:
        indices = indices[(indices != 0) & (indices != len(values) - 1)]
    return np.unique(indices)


def pulse_indices(interval: int, length: int) -> np.ndarray:
    if interval <= 0 or length <= 0:
        return np.array([], dtype=int)
    return event_indices_from_values(
        np.ones(length, dtype=float),
        event_indices=np.arange(interval, length, interval),
        drop_endpoints=True,
    )


def mark_event_times(ax, t: np.ndarray, indices: np.ndarray | list[int], *,
                     color: str = "0.35", label: str | None = None) -> None:
    for count, idx in enumerate(np.asarray(indices, dtype=int)):
        ax.axvline(
            t[idx],
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.50,
            label=label if count == 0 else None,
        )


def plot_impulse_events(
    ax,
    t: np.ndarray,
    values: np.ndarray,
    label: str,
    *,
    color: str = "tab:red",
    event_indices: np.ndarray | list[int] | None = None,
    tol: float = 1e-9,
    linewidth: float = 2.2,
    linestyle: str = "-",
) -> np.ndarray:
    indices = event_indices_from_values(values, event_indices=event_indices, tol=tol)
    if len(indices) == 0:
        return indices
    event_t = np.asarray(t, dtype=float)[indices]
    heights = np.asarray(values, dtype=float)[indices]
    ax.vlines(event_t, 0.0, heights, color=color, linewidth=linewidth, linestyles=linestyle, label=label)
    ax.scatter(event_t, heights, color=color, s=24, zorder=3)
    return indices


def record_payoff_convergence(repo: str, values: np.ndarray, *, label: str = "payoff_change") -> None:
    """Record absolute iteration-to-iteration objective changes."""
    values = np.asarray(values, dtype=float)
    for iteration in range(1, len(values)):
        CONVERGENCE_ROWS.append(
            {
                "repo": repo,
                "iteration": iteration,
                "metric": label,
                "value": float(abs(values[iteration] - values[iteration - 1])),
            }
        )


def record_baseline_value(
    repo: str,
    scenario: str,
    sample_type: str,
    sample_id: str,
    metric: str,
    value: float,
    *,
    better: str = "higher",
    panel: str | None = None,
) -> None:
    """Record one computed, deterministic, or random baseline value."""
    BASELINE_ROWS.append(
        {
            "repo": repo,
            "panel": panel or "",
            "scenario": scenario,
            "sample_type": sample_type,
            "sample_id": sample_id,
            "metric": metric,
            "value": float(value),
            "better": better,
        }
    )


def require_outputs(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    empty = [path for path in paths if path.exists() and path.stat().st_size == 0]
    if missing or empty:
        details = []
        if missing:
            details.append("missing: " + ", ".join(str(path) for path in missing))
        if empty:
            details.append("empty: " + ", ".join(str(path) for path in empty))
        raise RuntimeError("Expected output check failed; " + "; ".join(details))


def require_reference_repos() -> None:
    expected = [
        REF_DIR / "OpinionMalware_TIFS_2025_Code" / "opinionMalware.py",
        REF_DIR / "PropagandaWar_TIFS_2024_Code" / "propWar.py",
        REF_DIR / "Propaganda_TCSS_2025_Code" / "prop_propaganda.py",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing reference source files: "
            + ", ".join(str(path) for path in missing)
            + ". Refresh snapshots with download_reference_repositories.sh or restore reference_repositories/."
        )


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

    opt_c, opt_o, opt_u1, opt_u2 = om.c.copy(), om.o.copy(), om.u1.copy(), om.u2.copy()

    def evaluate_impulse_policy(u1: np.ndarray, u2: np.ndarray) -> float:
        om.c = np.full((om.t_scale, n), 0.5)
        om.o = np.full((om.t_scale, n), 0.3)
        om.u1 = u1.copy()
        om.u2 = u2.copy()
        om.c = om.forward_C(imp_a)
        om.o = om.forward_O(imp_b)
        return float(om.payoff())

    record_baseline_value(
        "OpinionMalware_TIFS_2025_Code",
        "computed impulse control",
        "computed",
        "computed",
        "objective J",
        float(J[-1]),
        better="model-defined",
    )
    record_baseline_value(
        "OpinionMalware_TIFS_2025_Code",
        "no impulse",
        "deterministic",
        "no_impulse",
        "objective J",
        evaluate_impulse_policy(np.zeros(om.t_scale), np.zeros(om.t_scale)),
        better="model-defined",
    )
    rng = np.random.default_rng(RANDOM_BASELINE_SEED + 101)
    for idx in range(RANDOM_BASELINE_COUNT):
        random_u1 = random_impulse_series(om.t_scale, imp_a, lower=0.0, upper=om.u1_upp, rng=rng)
        random_u2 = random_impulse_series(om.t_scale, imp_b, lower=0.0, upper=om.u2_upp, rng=rng)
        record_baseline_value(
            "OpinionMalware_TIFS_2025_Code",
            "random impulse control",
            "random",
            f"random_{idx + 1:03d}",
            "objective J",
            evaluate_impulse_policy(random_u1, random_u2),
            better="model-defined",
        )
    om.c, om.o, om.u1, om.u2 = opt_c, opt_o, opt_u1, opt_u2
    record_payoff_convergence("OpinionMalware_TIFS_2025_Code", J)

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

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_REFERENCE)
    axes[0].plot(np.arange(maxiter + 1), J, marker="o", linewidth=LINE_WIDTH)
    apply_clean_axes(axes[0], xlabel="iteration", ylabel="J", title="OpinionMalware impulse-control payoff")

    plot_time_series(axes[1], t, om.c.mean(axis=1), "state: mean malware c (all nodes)", linestyle="-")
    plot_time_series(axes[1], t, om.o.mean(axis=1), "state: mean opinion o (all nodes)", linestyle="--")
    mark_event_times(axes[1], t, imp_a, color="tab:red", label="u1 impulse time")
    mark_event_times(axes[1], t, imp_b, color="tab:green", label="u2 impulse time")
    apply_clean_axes(axes[1], xlabel="time", title="node-level mean states (all nodes)")
    axes[1].legend(frameon=False, fontsize=8)

    plot_impulse_events(axes[2], t, om.u1, "impulse control u1 (discrete)", color="tab:red", event_indices=imp_a)
    plot_impulse_events(axes[2], t, om.u2, "impulse control u2 (discrete)", color="tab:green", event_indices=imp_b, linestyle="--")
    axes[2].set_ylim(0.0, max(0.65, float(max(np.max(om.u1), np.max(om.u2))) + 0.08))
    apply_clean_axes(axes[2], xlabel="time", ylabel="impulse magnitude", title="impulse controls: discrete event lines")
    axes[2].legend(frameon=False, fontsize=8)
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

    pw.T = 1.5
    pw.h = 0.02
    pw.t_interval = int(pw.T / pw.h) + 1
    pw.pulse_interval_r = 18
    pw.pulse_interval_b = 15
    pw.maxiter = 60
    pw.step = 0.05
    strategy_damping = 0.15

    pw.beta_r = 0.045
    pw.gamma_r1 = 0.0675
    pw.gamma_r2 = 0.0225
    pw.delta_r = 0.02
    pw.beta_b = 0.0675
    pw.gamma_b1 = 0.09
    pw.gamma_b2 = 0.045
    pw.delta_b = 0.01

    pw.c_r = 2.0
    pw.c_b = 0.6
    pw.brp = 3
    pw.brc = 3
    pw.lr = 0.1
    pw.bbp = 5
    pw.bbc = 5
    pw.lb = 0.05
    pw.ur_low = pw.vr_low = 0.2
    pw.ur_upp = pw.vr_upp = 4.0
    pw.ub_low = pw.vb_low = 0.2
    pw.ub_upp = pw.vb_upp = 4.0

    pw.pr = np.full((pw.t_interval, len(pw.kr)), 0.25)
    pw.cr = np.full((pw.t_interval, len(pw.kr)), 0.25)
    pw.pb = np.full((pw.t_interval, len(pw.kb)), 0.25)
    pw.cb = np.full((pw.t_interval, len(pw.kb)), 0.15)
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
    strategy_delta = np.zeros(pw.maxiter)
    pw.pr, pw.cr = pw.forwardRed()
    pw.pb, pw.cb = pw.forwardBlue()
    jr[0] = pw.payoffRed()
    jb[0] = pw.payoffBlue()
    for idx in range(pw.maxiter):
        old_vr, old_ur = pw.vr.copy(), pw.ur.copy()
        old_vb, old_ub = pw.vb.copy(), pw.ub.copy()
        pw.lambda_r, pw.mu_r, pw.lambda_b, pw.mu_b = pw.backwardRed()
        pw.phi_r, pw.psi_r, pw.phi_b, pw.psi_b = pw.backwardBlue()
        candidate_vr, candidate_ur = pw.optimalStrategyRed()
        candidate_vb, candidate_ub = pw.optimalStrategyBlue()
        pw.vr = (1.0 - strategy_damping) * old_vr + strategy_damping * candidate_vr
        pw.ur = (1.0 - strategy_damping) * old_ur + strategy_damping * candidate_ur
        pw.vb = (1.0 - strategy_damping) * old_vb + strategy_damping * candidate_vb
        pw.ub = (1.0 - strategy_damping) * old_ub + strategy_damping * candidate_ub
        strategy_delta[idx] = max(
            float(np.max(abs(pw.vr - old_vr))),
            float(np.max(abs(pw.ur - old_ur))),
            float(np.max(abs(pw.vb - old_vb))),
            float(np.max(abs(pw.ub - old_ub))),
        )
        pw.pr, pw.cr = pw.forwardRed()
        pw.pb, pw.cb = pw.forwardBlue()
        jr[idx + 1] = pw.payoffRed()
        jb[idx + 1] = pw.payoffBlue()

    opt_pr, opt_cr, opt_pb, opt_cb = pw.pr.copy(), pw.cr.copy(), pw.pb.copy(), pw.cb.copy()
    opt_ur, opt_ub, opt_vr, opt_vb = pw.ur.copy(), pw.ub.copy(), pw.vr.copy(), pw.vb.copy()
    red_pulses = pulse_indices(pw.pulse_interval_r, pw.t_interval)
    blue_pulses = pulse_indices(pw.pulse_interval_b, pw.t_interval)

    def evaluate_hybrid_strategies(
        ur: np.ndarray,
        vr: np.ndarray,
        ub: np.ndarray,
        vb: np.ndarray,
    ) -> tuple[float, float]:
        pw.pr = np.full((pw.t_interval, len(pw.kr)), 0.25)
        pw.cr = np.full((pw.t_interval, len(pw.kr)), 0.25)
        pw.pb = np.full((pw.t_interval, len(pw.kb)), 0.25)
        pw.cb = np.full((pw.t_interval, len(pw.kb)), 0.15)
        pw.ur, pw.vr = ur.copy(), vr.copy()
        pw.ub, pw.vb = ub.copy(), vb.copy()
        pw.pr, pw.cr = pw.forwardRed()
        pw.pb, pw.cb = pw.forwardBlue()
        return float(pw.payoffRed()), float(pw.payoffBlue())

    _, zero_blue_payoff = evaluate_hybrid_strategies(opt_ur, opt_vr, np.zeros(pw.t_interval), np.zeros(pw.t_interval))
    zero_red_payoff, _ = evaluate_hybrid_strategies(np.zeros(pw.t_interval), np.zeros(pw.t_interval), opt_ub, opt_vb)
    record_baseline_value(
        "PropagandaWar_TIFS_2024_Code",
        "computed blue strategy",
        "computed",
        "computed_blue",
        "blue payoff",
        float(jb[-1]),
        panel="fixed computed red strategy; vary blue",
    )
    record_baseline_value(
        "PropagandaWar_TIFS_2024_Code",
        "zero blue strategy",
        "deterministic",
        "zero_blue",
        "blue payoff",
        zero_blue_payoff,
        panel="fixed computed red strategy; vary blue",
    )
    record_baseline_value(
        "PropagandaWar_TIFS_2024_Code",
        "computed red strategy",
        "computed",
        "computed_red",
        "red payoff",
        float(jr[-1]),
        panel="fixed computed blue strategy; vary red",
    )
    record_baseline_value(
        "PropagandaWar_TIFS_2024_Code",
        "zero red strategy",
        "deterministic",
        "zero_red",
        "red payoff",
        zero_red_payoff,
        panel="fixed computed blue strategy; vary red",
    )

    rng = np.random.default_rng(RANDOM_BASELINE_SEED + 201)
    for idx in range(RANDOM_BASELINE_COUNT):
        random_ub = smooth_random_controls((pw.t_interval, 1), lower=pw.ub_low, upper=pw.ub_upp, rng=rng).ravel()
        random_vb = random_impulse_series(pw.t_interval, blue_pulses, lower=pw.vb_low, upper=pw.vb_upp, rng=rng)
        _, random_blue_payoff = evaluate_hybrid_strategies(opt_ur, opt_vr, random_ub, random_vb)
        record_baseline_value(
            "PropagandaWar_TIFS_2024_Code",
            "random blue strategy",
            "random",
            f"random_blue_{idx + 1:03d}",
            "blue payoff",
            random_blue_payoff,
            panel="fixed computed red strategy; vary blue",
        )

        random_ur = smooth_random_controls((pw.t_interval, 1), lower=pw.ur_low, upper=pw.ur_upp, rng=rng).ravel()
        random_vr = random_impulse_series(pw.t_interval, red_pulses, lower=pw.vr_low, upper=pw.vr_upp, rng=rng)
        random_red_payoff, _ = evaluate_hybrid_strategies(random_ur, random_vr, opt_ub, opt_vb)
        record_baseline_value(
            "PropagandaWar_TIFS_2024_Code",
            "random red strategy",
            "random",
            f"random_red_{idx + 1:03d}",
            "red payoff",
            random_red_payoff,
            panel="fixed computed blue strategy; vary red",
        )

    pw.pr, pw.cr, pw.pb, pw.cb = opt_pr, opt_cr, opt_pb, opt_cb
    pw.ur, pw.ub, pw.vr, pw.vb = opt_ur, opt_ub, opt_vr, opt_vb
    record_payoff_convergence("PropagandaWar_TIFS_2024_Code", jr, label="red_payoff_change")
    record_payoff_convergence("PropagandaWar_TIFS_2024_Code", jb, label="blue_payoff_change")
    for iteration, delta in enumerate(strategy_delta, start=1):
        CONVERGENCE_ROWS.append(
            {
                "repo": "PropagandaWar_TIFS_2024_Code",
                "iteration": iteration,
                "metric": "strategy_change",
                "value": float(delta),
            }
        )

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

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_REFERENCE)
    axes[0].plot(np.arange(pw.maxiter + 1), jr, marker="o", linewidth=LINE_WIDTH, label="J red")
    axes[0].plot(np.arange(pw.maxiter + 1), jb, marker="o", linewidth=LINE_WIDTH, label="J blue")
    apply_clean_axes(axes[0], xlabel="iteration", title="PropagandaWar hybrid differential-game payoff")
    axes[0].legend(frameon=False, fontsize=8)

    plot_time_series(axes[1], t, pw.pr @ pw.pkr, "state: red P (degree-weighted mean)", linestyle="-")
    plot_time_series(axes[1], t, pw.cr @ pw.pkr, "state: red C (degree-weighted mean)", linestyle="--")
    plot_time_series(axes[1], t, pw.pb @ pw.pkb, "state: blue P (degree-weighted mean)", linestyle="-.")
    plot_time_series(axes[1], t, pw.cb @ pw.pkb, "state: blue C (degree-weighted mean)", linestyle=":")
    mark_event_times(axes[1], t, red_pulses, color="tab:red", label="red pulse")
    mark_event_times(axes[1], t, blue_pulses, color="tab:blue", label="blue pulse")
    apply_clean_axes(axes[1], xlabel="time", title="degree-level states (degree-weighted means)")
    axes[1].legend(frameon=False, ncol=1, fontsize=7, loc="center left")

    plot_time_series(axes[2], t, pw.ur, "continuous strategy ur(t)", color="tab:red", linestyle="-")
    plot_time_series(axes[2], t, pw.ub, "continuous strategy ub(t)", color="tab:blue", linestyle="--")
    plot_impulse_events(axes[2], t, pw.vr, "impulse vr(tau)", color="tab:orange", event_indices=red_pulses)
    plot_impulse_events(axes[2], t, pw.vb, "impulse vb(tau)", color="tab:cyan", event_indices=blue_pulses, linestyle="--")
    axes[2].set_ylim(0.0, max(1.15, float(max(np.max(pw.ur), np.max(pw.ub), np.max(pw.vr), np.max(pw.vb))) + 0.12))
    apply_clean_axes(axes[2], xlabel="time", ylabel="strategy value", title="hybrid control: continuous + impulse strategies")
    axes[2].legend(frameon=False, ncol=2, fontsize=8)
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
    T = 1.0
    h = 0.02
    t_interval = int(T / h) + 1
    maxiter = 5
    pulse_interval = 10
    beta1, beta2, eta, delta, gamma_a, gamma_u, omega = 0.003, 0.004, 0.0015, 0.001, 0.004, 0.003, 40.0
    cg, cf = 1.5, 2.0
    a_low, a_upp, u_low, u_upp = 0.8, 2.0, 1.0, 2.4

    sa = np.full((t_interval, n), 0.45)
    su = np.full((t_interval, n), 0.30)
    r = np.full((t_interval, n), 0.08)
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

    opt_sa, opt_su, opt_r, opt_ca, opt_cu = sa.copy(), su.copy(), r.copy(), ca.copy(), cu.copy()
    pulse_events = pulse_indices(pulse_interval, t_interval)

    def evaluate_impulse_policy(ca_candidate: np.ndarray, cu_candidate: np.ndarray) -> float:
        candidate_sa = np.full((t_interval, n), 0.45)
        candidate_su = np.full((t_interval, n), 0.30)
        candidate_r = np.full((t_interval, n), 0.08)
        candidate_sa, candidate_su, candidate_r = pp.forward(
            candidate_sa,
            candidate_su,
            candidate_r,
            ca_candidate.copy(),
            cu_candidate.copy(),
            t_interval,
            pulse_interval,
            A,
            beta1,
            beta2,
            eta,
            delta,
            gamma_a,
            gamma_u,
            h,
            pp.g_func,
            pp.f_func,
            cg,
            cf,
        )
        return float(pp.profit_sim(candidate_r, ca_candidate, cu_candidate, t_interval, pulse_interval, h, omega))

    record_baseline_value(
        "Propaganda_TCSS_2025_Code",
        "computed impulse control",
        "computed",
        "computed",
        "profit J",
        float(J[-1]),
    )
    record_baseline_value(
        "Propaganda_TCSS_2025_Code",
        "no impulse",
        "deterministic",
        "no_impulse",
        "profit J",
        evaluate_impulse_policy(np.zeros(t_interval), np.zeros(t_interval)),
    )
    rng = np.random.default_rng(RANDOM_BASELINE_SEED + 301)
    for idx in range(RANDOM_BASELINE_COUNT):
        random_ca = random_impulse_series(t_interval, pulse_events, lower=a_low, upper=a_upp, rng=rng)
        random_cu = random_impulse_series(t_interval, pulse_events, lower=u_low, upper=u_upp, rng=rng)
        record_baseline_value(
            "Propaganda_TCSS_2025_Code",
            "random impulse control",
            "random",
            f"random_{idx + 1:03d}",
            "profit J",
            evaluate_impulse_policy(random_ca, random_cu),
        )
    sa, su, r, ca, cu = opt_sa, opt_su, opt_r, opt_ca, opt_cu
    record_payoff_convergence("Propaganda_TCSS_2025_Code", J, label="profit_change")

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

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_REFERENCE)
    axes[0].plot(np.arange(maxiter + 1), J, marker="o", linewidth=LINE_WIDTH)
    apply_clean_axes(axes[0], xlabel="iteration", ylabel="J", title="TCSS optimal impulse-control profit")

    plot_time_series(axes[1], t, sa.mean(axis=1), "state: mean Sa over all nodes", linestyle="-")
    plot_time_series(axes[1], t, su.mean(axis=1), "state: mean Su over all nodes", linestyle="--")
    plot_time_series(axes[1], t, r.mean(axis=1), "state: mean R over all nodes", linestyle="-.")
    mark_event_times(axes[1], t, pulse_events, color="0.35", label="pulse time")
    apply_clean_axes(axes[1], xlabel="time", title="node-level mean states (all nodes)")
    axes[1].legend(frameon=False, fontsize=8)

    plot_impulse_events(
        axes[2],
        t,
        ca,
        "impulse control ca(tau)",
        color="tab:red",
        event_indices=pulse_events,
        linewidth=4.0,
    )
    plot_impulse_events(
        axes[2],
        t,
        cu,
        "impulse control cu(tau)",
        color="tab:blue",
        event_indices=pulse_events,
        linewidth=2.0,
        linestyle="--",
    )
    axes[2].set_ylim(0.0, max(2.25, float(max(np.max(ca), np.max(cu))) + 0.12))
    apply_clean_axes(axes[2], xlabel="time", ylabel="impulse magnitude", title="impulse controls: discrete event lines")
    axes[2].legend(frameon=False, fontsize=8)
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
        ("Opinion baseline", OUT_DIR / "opinion_malware_baseline_comparison.png"),
        ("PropagandaWar baseline", OUT_DIR / "propaganda_war_baseline_comparison.png"),
        ("TCSS baseline", OUT_DIR / "propaganda_tcss_baseline_comparison.png"),
    ]
    require_outputs([path for _, path in images])
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 10.8))
    flat_axes = axes.ravel()
    for ax, (title, path) in zip(flat_axes, images):
        ax.imshow(plt.imread(path))
        ax.set_title(title)
        ax.axis("off")
    for ax in flat_axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reference_repo_contact_sheet.png", dpi=180)
    plt.close(fig)


def write_reference_diagnostics() -> None:
    """Save convergence and baseline diagnostics shared by all reference runs."""
    if CONVERGENCE_ROWS:
        convergence = pd.DataFrame(CONVERGENCE_ROWS)
        convergence.to_csv(OUT_DIR / "reference_convergence.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for (repo, metric), data in convergence.groupby(["repo", "metric"], sort=False):
            data = data.sort_values("iteration")
            ax.semilogy(data["iteration"], data["value"], marker="o", linewidth=LINE_WIDTH, label=f"{repo}: {metric}")
        apply_clean_axes(ax, xlabel="iteration", ylabel="absolute change", title="Reference smoke-run convergence diagnostics")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "reference_convergence.png", dpi=180)
        plt.close(fig)

    if BASELINE_ROWS:
        baseline = write_baseline_table(BASELINE_ROWS, OUT_DIR / "baseline_comparison.csv")
        old_mixed = OUT_DIR / "baseline_comparison.png"
        if old_mixed.exists():
            old_mixed.unlink()

        plot_map = {
            "OpinionMalware_TIFS_2025_Code": (
                "opinion_malware_baseline_comparison.png",
                "OpinionMalware impulse-control objective vs baselines",
                "objective J (reported by reference code)",
            ),
            "Propaganda_TCSS_2025_Code": (
                "propaganda_tcss_baseline_comparison.png",
                "TCSS impulse-control profit vs baselines",
                "profit J",
            ),
        }
        for repo, (file_name, title, ylabel) in plot_map.items():
            rows = baseline[baseline["repo"] == repo].to_dict("records")
            if rows:
                save_control_baseline_plot(rows, OUT_DIR / file_name, title=title, ylabel=ylabel)

        war_rows = baseline[baseline["repo"] == "PropagandaWar_TIFS_2024_Code"].to_dict("records")
        if war_rows:
            save_game_baseline_plot(
                war_rows,
                OUT_DIR / "propaganda_war_baseline_comparison.png",
                title="PropagandaWar hybrid differential game: unilateral baselines",
                ylabel="payoff (higher is better)",
            )


def _format_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def write_smoke_run_report(rows: list[dict[str, float]]) -> None:
    headers = ["repo", "nodes", "J0", "J_final", "red_degree_classes", "J_red_final", "J_blue_final"]
    summary_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        summary_lines.append("| " + " | ".join(_format_value(row.get(header)) for header in headers) + " |")

    text = """# Reference Smoke Run Report

This file is generated by `run_reference_smoke.py` for the current output directory.

## What this smoke run checks

- The three reference source snapshots can be imported.
- Small local sample graphs can drive each model without redistributing full paper datasets.
- Forward state equations, backward adjoint equations, strategy/control updates, payoff calculations, CSV export, and plotting all execute successfully.
- Iteration diagnostics remain finite; damped reference sweeps report payoff/strategy changes when available.
- Computed strategies are compared against no-impulse, zero-strategy, and random-strategy baselines.
- Expected output files exist and are non-empty.

## How to read the figures

Each reference figure has three panels:

| Panel type | X-axis | Purpose |
| --- | --- | --- |
| Payoff/profit panel | iteration | Shows the numerical optimization or game-strategy iteration. This is a smoke-level convergence diagnostic: the curve should be finite and should not break or explode. |
| State panel | time | Shows system state evolution under the computed control/game strategy. Labels state whether trajectories are node means over all nodes or degree-weighted means over degree classes. |
| Control/strategy panel | time | Continuous controls/strategies are time-indexed sampled curves. Impulse controls/strategies are vertical event lines. Hybrid examples show both. |

Continuous control is a time-indexed strategy sampled on the simulation grid; projected strategies can include flat segments at bounds, but they still enter the ODE between event times. Impulse control acts only at discrete event times and may make the state jump or change direction. Hybrid control combines the two. In the state panels, vertical markers show impulse or pulse times. The PropagandaWar smoke-run parameters are chosen so the continuous strategies `ur(t)` and `ub(t)` vary visibly; the TCSS smoke-run parameters are chosen to make impulse-induced state changes visible in a small local graph.

## Model classification

| Repository | Modeling level | Control/game type | Main idea |
| --- | --- | --- | --- |
| `OpinionMalware_TIFS_2025_Code` | Node-level coupled malware-opinion model | Optimal impulse control | Track malware and opinion states on coupled network layers. |
| `PropagandaWar_TIFS_2024_Code` | Degree-level red/blue population model | Hybrid/impulsive differential game | Aggregate graph structure into degree distributions, then compute interacting red/blue strategies. |
| `Propaganda_TCSS_2025_Code` | Node-level awareness-aware propagation model | Optimal impulse control | Track awareness-aware propagation states and compute impulse interventions. |

## Figure-specific notes

| File | Model class | Interpretation |
| --- | --- | --- |
| `opinion_malware.png` | Node-level optimal impulse control | Left: payoff over forward-backward iterations. Middle: node-mean malware state `c(t)` and opinion state `o(t)` over all nodes; vertical markers show `u1`/`u2` impulse times. Right: `u1` and `u2` are plotted only as discrete impulse magnitudes. |
| `propaganda_war.png` | Degree-level hybrid/impulsive differential game | Left: red and blue player payoffs over game iterations. Middle: degree-weighted red/blue propaganda state means with pulse markers. Right: `ur`/`ub` are continuous strategies; `vr`/`vb` are discrete impulse strategies. |
| `propaganda_tcss.png` | Node-level optimal impulse control with awareness | Left: profit over impulse-policy iterations. Middle: node-mean awareness/unawareness/removed states over all nodes with pulse markers; the tuned smoke-run parameters make the pulse effects visually clear. Right: `ca` and `cu` are plotted only as discrete impulse magnitudes. |
| `reference_repo_contact_sheet.png` | Mixed overview | Compact visual index for the three reference smoke runs and their model-specific baseline comparisons. |
| `reference_convergence.png` | Iteration diagnostics | Absolute payoff/profit or strategy changes across smoke-run iterations. |
| `opinion_malware_baseline_comparison.png` | Node-level impulse-control baseline comparison | Computed impulse policy compared with no-impulse and random impulse policies for the same model. |
| `propaganda_war_baseline_comparison.png` | Degree-level hybrid-game baseline comparison | Two unilateral panels: one fixes the computed red strategy and varies blue; the other fixes the computed blue strategy and varies red. |
| `propaganda_tcss_baseline_comparison.png` | Node-level impulse-control baseline comparison | Computed impulse policy compared with no-impulse and random impulse policies for the same model. |

## CSV outputs

| File pattern | Contents |
| --- | --- |
| `*_payoff.csv` | Iteration-indexed payoff/profit values for convergence inspection. |
| `*_timeseries.csv` | Time-indexed state and control/strategy trajectories. |
| `reference_convergence.csv` | Iteration-to-iteration diagnostic changes used in `reference_convergence.png`. |
| `baseline_comparison.csv` | Computed, deterministic, and random baseline values used in the model-specific baseline-comparison figures. |
| `smoke_run_summary.csv` | One-row-per-repository smoke-run summary. |

## Smoke-run summary

""" + "\n".join(summary_lines) + "\n"
    (OUT_DIR / "smoke_run_report.md").write_text(text)


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
    require_reference_repos()

    rows = []
    rows.append({"repo": "OpinionMalware_TIFS_2025_Code", **run_opinion_malware()})
    rows.append({"repo": "PropagandaWar_TIFS_2024_Code", **run_propaganda_war()})
    rows.append({"repo": "Propaganda_TCSS_2025_Code", **run_propaganda_tcss()})
    pd.DataFrame(rows).to_csv(OUT_DIR / "smoke_run_summary.csv", index=False)
    write_reference_diagnostics()
    make_contact_sheet()
    write_smoke_run_report(rows)
    require_outputs(
        [
            OUT_DIR / "opinion_malware_payoff.csv",
            OUT_DIR / "opinion_malware_timeseries.csv",
            OUT_DIR / "opinion_malware.png",
            OUT_DIR / "propaganda_war_payoff.csv",
            OUT_DIR / "propaganda_war_timeseries.csv",
            OUT_DIR / "propaganda_war.png",
            OUT_DIR / "propaganda_tcss_payoff.csv",
            OUT_DIR / "propaganda_tcss_timeseries.csv",
            OUT_DIR / "propaganda_tcss.png",
            OUT_DIR / "reference_convergence.csv",
            OUT_DIR / "reference_convergence.png",
            OUT_DIR / "baseline_comparison.csv",
            OUT_DIR / "opinion_malware_baseline_comparison.png",
            OUT_DIR / "propaganda_war_baseline_comparison.png",
            OUT_DIR / "propaganda_tcss_baseline_comparison.png",
            OUT_DIR / "smoke_run_summary.csv",
            OUT_DIR / "reference_repo_contact_sheet.png",
            OUT_DIR / "smoke_run_report.md",
        ]
    )
    print(f"saved outputs to {OUT_DIR}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
