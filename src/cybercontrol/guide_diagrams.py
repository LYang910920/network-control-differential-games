"""Reproducible guide diagrams shared by the three repository notes.

The functions in this module describe the mathematical or algorithmic content
of each diagram.  File output and cross-repository synchronization are handled
by ``scripts/generate_diagrams.py`` so the visual source remains unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


NAVY = "#17324D"
BLUE = "#276FBF"
TEAL = "#168C8C"
ORANGE = "#D97706"
RED = "#B54747"
INK = "#24313D"
MUTED = "#667685"
PALE_BLUE = "#EAF2FA"
PALE_TEAL = "#E7F5F3"
PALE_ORANGE = "#FFF1DD"
PALE_GRAY = "#F3F6F8"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class DiagramDefinition:
    """Metadata and renderer for one maintained guide diagram."""

    diagram_id: str
    renderer: Callable[[], object]
    alt_text: str
    used_in: str


class Canvas:
    """Small normalized-coordinate canvas for uncluttered technical diagrams."""

    def __init__(self, title: str, *, subtitle: str | None = None, height: float = 6.4):
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(11.2, height))
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.axis("off")
        self.fig.patch.set_facecolor(WHITE)
        self.ax.text(
            0.04,
            0.95,
            title,
            color=NAVY,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="top",
        )
        if subtitle:
            self.ax.text(0.04, 0.895, subtitle, color=MUTED, fontsize=10.5, ha="left", va="top")

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        fill: str = PALE_BLUE,
        edge: str = BLUE,
        fontsize: float = 10.0,
        weight: str = "normal",
        radius: float = 0.015,
        align: str = "center",
    ):
        from matplotlib.patches import FancyBboxPatch

        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
        self.ax.add_patch(patch)
        tx = x + w / 2 if align == "center" else x + 0.018
        self.ax.text(
            tx,
            y + h / 2,
            text,
            color=INK,
            fontsize=fontsize,
            fontweight=weight,
            ha=align,
            va="center",
            linespacing=1.25,
            zorder=3,
        )
        return patch

    def circle(
        self,
        x: float,
        y: float,
        r: float,
        text: str,
        *,
        fill: str,
        edge: str,
        fontsize: float = 13.0,
    ):
        from matplotlib.patches import Circle

        patch = Circle((x, y), r, facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=2)
        self.ax.add_patch(patch)
        self.ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=3)
        return patch

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        label: str | None = None,
        color: str = NAVY,
        bend: float = 0.0,
        label_offset: tuple[float, float] = (0.0, 0.018),
        style: str = "-|>",
        width: float = 1.5,
    ):
        from matplotlib.patches import FancyArrowPatch

        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=width,
            color=color,
            connectionstyle=f"arc3,rad={bend}",
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
        self.ax.add_patch(patch)
        if label:
            mx = (start[0] + end[0]) / 2 + label_offset[0]
            my = (start[1] + end[1]) / 2 + label_offset[1]
            self.ax.text(
                mx,
                my,
                label,
                color=color,
                fontsize=9.2,
                ha="center",
                va="center",
                bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.2},
                zorder=4,
            )
        return patch

    def label(
        self,
        x: float,
        y: float,
        text: str,
        *,
        fontsize: float = 10.0,
        color: str = INK,
        weight: str = "normal",
        ha: str = "left",
        va: str = "center",
    ) -> None:
        self.ax.text(x, y, text, fontsize=fontsize, color=color, fontweight=weight, ha=ha, va=va)

    def finish(self):
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)
        return self.fig


def repository_family():
    """Show package ownership and the complementary scope of the two notes."""

    c = Canvas(
        "Repository family and code ownership",
        subtitle="Shared scientific infrastructure lives in the foundation package; each note owns its method-specific layer.",
    )
    c.box(
        0.22,
        0.61,
        0.56,
        0.19,
        "Network Control and Differential Games\n"
        "cybercontrol: equations, graphs, FBSM and numerics\n"
        "shared metrics, experiment records and plotting",
        fill=PALE_BLUE,
        edge=BLUE,
        fontsize=12,
        weight="bold",
    )
    c.box(
        0.06,
        0.26,
        0.39,
        0.20,
        "Cyber Control and Game Learning\n"
        "cybergames: environments, DDQN/PPO, CTDE and MAPPO\n"
        "policy and attacker–defender evaluation",
        fill=PALE_TEAL,
        edge=TEAL,
        fontsize=11,
        weight="bold",
    )
    c.box(
        0.55,
        0.26,
        0.39,
        0.20,
        "Physics-Informed Cyber Control\n"
        "cyberpinn: inverse PINN, PIDL and neural control\n"
        "PMP-informed state, costate and stationarity residuals",
        fill=PALE_ORANGE,
        edge=ORANGE,
        fontsize=11,
        weight="bold",
    )
    c.arrow(
        (0.39, 0.60),
        (0.29, 0.47),
        label="imports cybercontrol",
        color=TEAL,
        label_offset=(-0.06, 0.02),
    )
    c.arrow(
        (0.61, 0.60),
        (0.71, 0.47),
        label="imports cybercontrol",
        color=ORANGE,
        label_offset=(0.06, 0.02),
    )
    c.box(
        0.18,
        0.08,
        0.64,
        0.10,
        "Common model vocabulary: SIR/SIPS states, heterogeneous parameters, sampled and impulse semantics",
        fill=PALE_GRAY,
        edge=MUTED,
        fontsize=10,
    )
    return c.finish()


def state_transitions():
    """Explain SIPS mass transfer and the node-indexed infection pressure."""

    c = Canvas(
        "Canonical SIPS state transitions",
        subtitle=r"State order is $[S,I,P]$ and $S+I+P=1$; every flow transfers mass between compartments.",
    )
    c.circle(0.17, 0.62, 0.075, "$S$\nsusceptible", fill=PALE_BLUE, edge=BLUE)
    c.circle(0.50, 0.62, 0.075, "$I$\ninfected", fill="#FBE9E9", edge=RED)
    c.circle(0.83, 0.62, 0.075, "$P$\nprotected", fill=PALE_TEAL, edge=TEAL)
    c.arrow((0.25, 0.62), (0.42, 0.62), label=r"infection: $S\lambda$", color=RED)
    c.arrow((0.58, 0.64), (0.75, 0.64), label=r"cleaning: $(\gamma+u_c)I$", color=TEAL)
    c.arrow(
        (0.18, 0.54),
        (0.78, 0.54),
        label=r"patching: $u_p S$",
        color=BLUE,
        bend=0.18,
        label_offset=(0.0, -0.025),
    )
    c.arrow(
        (0.78, 0.70),
        (0.22, 0.70),
        label=r"waning: $\omega P$",
        color=ORANGE,
        bend=0.20,
        label_offset=(0.0, 0.035),
    )
    c.box(
        0.07,
        0.22,
        0.40,
        0.18,
        r"Aggregate pressure"
        "\n"
        r"$\lambda(t)=\beta I(t)$"
        "\n"
        r"or degree mixing $\lambda_k=s_k k\sum_\ell M_{k\ell}\,\iota_\ell x_\ell$",
        fill=PALE_GRAY,
        edge=MUTED,
        fontsize=10.2,
    )
    c.box(
        0.53,
        0.22,
        0.40,
        0.18,
        r"Node-indexed pressure"
        "\n"
        r"$\lambda_i=\beta_i s_i\sum_j A_{ij}\,\iota_j I_j$"
        "\n"
        r"with node-specific $s_i,\iota_i,\gamma_i,\omega_i$ and control bounds",
        fill=PALE_GRAY,
        edge=MUTED,
        fontsize=10.2,
    )
    c.label(
        0.50,
        0.10,
        "The invariant is a modeling check, not an extra loss term in the ODE solver.",
        ha="center",
        color=MUTED,
    )
    return c.finish()


def degree_vs_node():
    """Contrast degree-class mixing with explicit node adjacency."""

    c = Canvas(
        "Degree-level and node-level network models",
        subtitle="Both can be heterogeneous; they differ in state resolution and the network operator applied to infection pressure.",
    )
    c.box(
        0.05,
        0.70,
        0.41,
        0.10,
        "Degree-level state $x_k(t)$",
        fill=PALE_BLUE,
        edge=BLUE,
        fontsize=12,
        weight="bold",
    )
    for x, degree in ((0.09, 1), (0.20, 3), (0.31, 8)):
        label = f"$k={degree}$"
        c.circle(x, 0.54, 0.045, label, fill=WHITE, edge=BLUE, fontsize=10)
        c.label(x, 0.45, f"$x_{degree}(t)$", ha="center", fontsize=9)
    c.box(
        0.10,
        0.27,
        0.31,
        0.11,
        r"mixing matrix $M_{k\ell}$" "\n" r"weights degree-class exposure",
        fill=PALE_GRAY,
        edge=MUTED,
    )
    c.arrow((0.25, 0.44), (0.25, 0.39), color=BLUE)

    c.box(
        0.54,
        0.70,
        0.41,
        0.10,
        "Node-level state $x_i(t)$",
        fill=PALE_TEAL,
        edge=TEAL,
        fontsize=12,
        weight="bold",
    )
    nodes = [(0.63, 0.56), (0.76, 0.61), (0.86, 0.52), (0.70, 0.43), (0.84, 0.38)]
    for start, end in ((0, 1), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4)):
        c.arrow(nodes[start], nodes[end], color="#9CB7B8", style="-", width=1.1)
    for idx, (x, y) in enumerate(nodes):
        c.circle(x, y, 0.033, f"$i_{idx + 1}$", fill=WHITE, edge=TEAL, fontsize=8.5)
    c.box(
        0.60,
        0.21,
        0.30,
        0.12,
        r"adjacency $A_{ij}$" "\n" r"preserves explicit contacts",
        fill=PALE_GRAY,
        edge=MUTED,
    )
    c.arrow((0.75, 0.37), (0.75, 0.34), color=TEAL)

    c.box(
        0.07,
        0.08,
        0.38,
        0.08,
        "Cost scales with number of degree classes and time points",
        fill=WHITE,
        edge=BLUE,
        fontsize=9.5,
    )
    c.box(
        0.55,
        0.08,
        0.38,
        0.08,
        "Cost scales with nodes, edges, controls and time points",
        fill=WHITE,
        edge=TEAL,
        fontsize=9.5,
    )
    return c.finish()


def control_timing():
    """Separate signal timing, sampled values, modes, impulses and hybrid flow."""

    import numpy as np

    c = Canvas(
        "Control timing and state effects",
        subtitle="Decision timing, action-value domain and state effect are separate classification axes.",
        height=7.0,
    )
    x0, x1 = 0.26, 0.94
    t = np.linspace(0.0, 1.0, 250)
    ys = [0.76, 0.61, 0.46, 0.31, 0.16]
    labels = [
        "Continuous-time signal $u(t)$",
        "Sampled action with ZOH",
        "Switched / parameterized action",
        "Impulse reset",
        "Continuous–impulsive control",
    ]
    for y, label in zip(ys, labels):
        c.label(0.04, y, label, fontsize=10.3, weight="bold")
        c.ax.plot([x0, x1], [y, y], color="#B8C3CC", linewidth=1.0)
    c.ax.plot(x0 + (x1 - x0) * t, ys[0] + 0.035 * np.sin(4 * np.pi * t), color=BLUE, linewidth=2.1)
    epochs = np.linspace(x0, x1, 6)
    values = np.array([-0.018, 0.030, 0.005, 0.040, -0.012, -0.012])
    c.ax.step(epochs, ys[1] + values, where="post", color=TEAL, linewidth=2.1)
    for xx in epochs[:-1]:
        c.ax.plot(
            [xx, xx], [ys[1] - 0.045, ys[1] + 0.05], color="#BCD9D7", linestyle=":", linewidth=1.0
        )
    modes = [0, 1, 1, 2, 0, 0]
    c.ax.step(epochs, ys[2] + 0.022 * np.asarray(modes), where="post", color=NAVY, linewidth=2.1)
    c.label(
        x1,
        ys[2] + 0.055,
        "$m_k$ selects $f_{m_k}$; no reset",
        fontsize=8.8,
        color=MUTED,
        ha="right",
    )
    impulses = [epochs[2], epochs[4]]
    for xx in impulses:
        c.arrow((xx, ys[3] - 0.045), (xx, ys[3] + 0.055), color=RED, width=2.0)
    c.label(
        x1, ys[3] + 0.055, r"$x(\tau_j^+)=G(x(\tau_j^-),v_j)$", fontsize=9.2, color=RED, ha="right"
    )
    c.ax.plot(x0 + (x1 - x0) * t, ys[4] + 0.025 * np.cos(3 * np.pi * t), color=BLUE, linewidth=2.0)
    for xx in impulses:
        c.arrow((xx, ys[4] - 0.045), (xx, ys[4] + 0.055), color=RED, width=1.8)
    c.label(x0, 0.095, "$t_0$", fontsize=9, ha="center")
    c.label(x1, 0.095, "$T$", fontsize=9, ha="center")
    c.label(
        0.50,
        0.025,
        "A continuous-valued sampled action may still be piecewise constant in time.",
        fontsize=9.5,
        color=MUTED,
        ha="center",
    )
    return c.finish()


def ode_environment():
    """Show one sampled transition from observation to next state."""

    c = Canvas(
        "One ODE-based environment transition",
        subtitle=r"At epoch $t_k$, the policy acts once; the simulator then advances the continuous state to $t_{k+1}$.",
    )
    items = [
        (0.04, "Observe\n" r"$o_k=h(x_k)$", PALE_BLUE, BLUE),
        (0.20, "Choose action\n" r"$a_k=(m_k,v_k)$", PALE_TEAL, TEAL),
        (0.37, "Optional reset\n" r"$x_k^+=G(x_k^-,v_k)$", "#FBE9E9", RED),
        (0.55, "Integrate flow\n" r"$\dot{x}=f_{m_k}(x,v_k)$", PALE_ORANGE, ORANGE),
        (0.73, "Accumulate reward\n" r"$r_k=-\int c\,dt-C_J$", PALE_GRAY, MUTED),
        (0.88, "Next state\n" r"$x_{k+1}$", PALE_BLUE, BLUE),
    ]
    widths = [0.13, 0.14, 0.15, 0.15, 0.13, 0.09]
    for idx, ((x, text, fill, edge), width) in enumerate(zip(items, widths)):
        c.box(x, 0.52, width, 0.18, text, fill=fill, edge=edge, fontsize=9.5, weight="bold")
        if idx < len(items) - 1:
            c.arrow((x + width + 0.005, 0.61), (items[idx + 1][0] - 0.005, 0.61), color=NAVY)
    c.box(
        0.19,
        0.22,
        0.62,
        0.15,
        "Ordering is part of the model: reset-before-flow and flow-before-reset generally produce different transitions.\n"
        "Running cost and impulse cost are recorded separately before they are combined in the reward.",
        fill=WHITE,
        edge=MUTED,
        fontsize=10.1,
    )
    c.arrow(
        (0.925, 0.50),
        (0.105, 0.41),
        label="repeat at $t_{k+1}$",
        color=BLUE,
        bend=-0.20,
        label_offset=(0.0, -0.015),
    )
    return c.finish()


def ddqn():
    """Show online and target networks, replay and the DDQN target."""

    c = Canvas(
        "Double DQN update for discrete cyber actions",
        subtitle="The online network selects the next action; the target network evaluates it.",
    )
    c.box(
        0.05,
        0.58,
        0.15,
        0.15,
        "Observation\n" r"$o_k\in\mathbb{R}^{d_o}$",
        fill=PALE_BLUE,
        edge=BLUE,
        weight="bold",
    )
    c.box(
        0.27,
        0.58,
        0.19,
        0.15,
        "Online Q-network\n" r"$d_o\to128\to128\to|\mathcal{A}|$",
        fill=PALE_TEAL,
        edge=TEAL,
        weight="bold",
    )
    c.box(
        0.54,
        0.58,
        0.15,
        0.15,
        r"$\varepsilon$-greedy" "\n" r"$a_k=\arg\max_a Q(o_k,a)$",
        fill=PALE_ORANGE,
        edge=ORANGE,
    )
    c.box(
        0.77,
        0.58,
        0.17,
        0.15,
        "ODE environment\n" r"$(o_k,a_k,r_k,o_{k+1})$",
        fill=PALE_BLUE,
        edge=BLUE,
    )
    for start, end in (
        ((0.20, 0.655), (0.27, 0.655)),
        ((0.46, 0.655), (0.54, 0.655)),
        ((0.69, 0.655), (0.77, 0.655)),
    ):
        c.arrow(start, end)
    c.box(
        0.67,
        0.27,
        0.22,
        0.14,
        "Replay buffer\nrandom mini-batch",
        fill=PALE_GRAY,
        edge=MUTED,
        weight="bold",
    )
    c.arrow((0.855, 0.57), (0.80, 0.42), label="store", color=MUTED)
    c.box(
        0.36,
        0.24,
        0.23,
        0.19,
        r"DDQN target"
        "\n"
        r"$y=r+\gamma(1-d)$"
        "\n"
        r"$Q_{\bar\theta}(o',\arg\max_a Q_\theta(o',a))$",
        fill="#FBE9E9",
        edge=RED,
        fontsize=9.5,
    )
    c.arrow((0.67, 0.34), (0.59, 0.34), label="sample", color=MUTED)
    c.arrow(
        (0.475, 0.44), (0.365, 0.57), label=r"$\nabla_\theta (Q_\theta-y)^2$", color=TEAL, bend=0.08
    )
    c.box(
        0.07,
        0.24,
        0.20,
        0.17,
        "Target Q-network\nperiodic or soft copy\n" r"$\bar\theta\leftarrow\theta$",
        fill=WHITE,
        edge=TEAL,
    )
    c.arrow((0.27, 0.325), (0.36, 0.325), label="evaluate", color=TEAL)
    return c.finish()


def mappo_ctde():
    """Show the repository's coordinated budgeted PPO implementation."""

    c = Canvas(
        "Budgeted cooperative PPO with centralized training",
        subtitle="Repository path: a state-conditioned coordinator chooses one joint community action; execution is not strictly decentralized.",
        height=6.8,
    )
    c.box(
        0.04,
        0.57,
        0.19,
        0.17,
        "Community summaries\n" r"$O_k\in\mathbb{R}^{M\times d_o}$" "\nstate + risk + time",
        fill=PALE_BLUE,
        edge=BLUE,
        weight="bold",
    )
    c.box(
        0.29,
        0.57,
        0.20,
        0.17,
        "Shared community encoder\nMLP or graph context\n+ joint-action logits",
        fill=PALE_TEAL,
        edge=TEAL,
        weight="bold",
    )
    c.arrow((0.23, 0.655), (0.29, 0.655), color=TEAL)
    c.box(
        0.55,
        0.57,
        0.16,
        0.17,
        "One categorical action\nno action or\n(community, mode)",
        fill=PALE_ORANGE,
        edge=ORANGE,
        weight="bold",
    )
    c.arrow((0.49, 0.655), (0.55, 0.655), color=TEAL)
    c.box(
        0.77,
        0.57,
        0.18,
        0.17,
        "Node-SIPS environment\nODE transition\nscalar team reward",
        fill=PALE_BLUE,
        edge=BLUE,
        weight="bold",
    )
    c.arrow((0.71, 0.655), (0.77, 0.655))
    c.box(
        0.73,
        0.20,
        0.22,
        0.17,
        "Rollout buffer\nobservations $[T,M,d_o]$\nactions/rewards $[T]$",
        fill=PALE_GRAY,
        edge=MUTED,
    )
    c.arrow((0.86, 0.57), (0.85, 0.38), label="collect", color=MUTED)
    c.box(
        0.42,
        0.18,
        0.23,
        0.19,
        "GAE + PPO update\nclipped ratios, entropy,\nmini-batches and epochs",
        fill="#FBE9E9",
        edge=RED,
        weight="bold",
    )
    c.arrow((0.73, 0.285), (0.65, 0.285), color=MUTED)
    c.box(
        0.08,
        0.18,
        0.25,
        0.19,
        r"Centralized value critic $V_\phi(O_k)$"
        "\npooled community summaries\ntraining and coordinated execution",
        fill=WHITE,
        edge=BLUE,
        weight="bold",
    )
    c.arrow((0.42, 0.275), (0.33, 0.275), label="value targets", color=BLUE)
    c.arrow((0.50, 0.38), (0.39, 0.57), label="update coordinator", color=TEAL, bend=0.12)
    return c.finish()


def hierarchical_game():
    """Show a state-conditioned hierarchical attacker-defender policy."""

    c = Canvas(
        "Hierarchical attacker–defender policy",
        subtitle="A high-level decision identifies where to act; a low-level allocator determines the bounded action intensity.",
    )
    c.box(
        0.05,
        0.55,
        0.17,
        0.18,
        "Global and local features\nstate, degree, community,\nasset risk and budget",
        fill=PALE_BLUE,
        edge=BLUE,
        weight="bold",
    )
    c.box(
        0.29,
        0.59,
        0.17,
        0.14,
        "High-level selector\ncommunity / target",
        fill=PALE_TEAL,
        edge=TEAL,
        weight="bold",
    )
    c.box(
        0.53,
        0.59,
        0.17,
        0.14,
        "Low-level allocator\nmode and intensity",
        fill=PALE_ORANGE,
        edge=ORANGE,
        weight="bold",
    )
    c.box(
        0.77,
        0.55,
        0.17,
        0.18,
        "Budget projection\nvalid joint action\nfor the simulator",
        fill=PALE_BLUE,
        edge=BLUE,
        weight="bold",
    )
    for start, end in (
        ((0.22, 0.64), (0.29, 0.66)),
        ((0.46, 0.66), (0.53, 0.66)),
        ((0.70, 0.66), (0.77, 0.64)),
    ):
        c.arrow(start, end)
    c.box(
        0.20,
        0.23,
        0.24,
        0.15,
        "Opponent pool\ncurrent, historical and\nrule-based policies",
        fill=PALE_GRAY,
        edge=MUTED,
    )
    c.box(
        0.56,
        0.23,
        0.24,
        0.15,
        "Evaluation evidence\nresponse matrix and\nunilateral deviations",
        fill="#FBE9E9",
        edge=RED,
    )
    c.arrow((0.32, 0.39), (0.37, 0.58), label="sample opponent", color=MUTED, bend=0.08)
    c.arrow((0.855, 0.54), (0.68, 0.39), label="deterministic evaluation", color=RED, bend=-0.08)
    c.label(
        0.50,
        0.10,
        "Training reward measures optimization progress; it does not by itself establish Nash equilibrium.",
        ha="center",
        color=MUTED,
    )
    return c.finish()


def pinn_loss():
    """Show state approximation, autodiff residuals and validation."""

    c = Canvas(
        "Physics-informed loss assembly",
        subtitle="Observation points and collocation points serve different roles and need not coincide.",
    )
    c.box(
        0.04,
        0.60,
        0.14,
        0.14,
        "Observed data\n$(t_d,y_d)$",
        fill=PALE_BLUE,
        edge=BLUE,
        weight="bold",
    )
    c.box(
        0.04,
        0.34,
        0.14,
        0.14,
        "Collocation points\n$t_f$",
        fill=PALE_GRAY,
        edge=MUTED,
        weight="bold",
    )
    c.box(
        0.25,
        0.49,
        0.18,
        0.19,
        "State network\n" r"$x_\theta(t)$" "\n" r"softmax $\rightarrow[S,I,P]$",
        fill=PALE_TEAL,
        edge=TEAL,
        weight="bold",
    )
    c.arrow((0.18, 0.67), (0.25, 0.61), color=BLUE)
    c.arrow((0.18, 0.41), (0.25, 0.54), color=MUTED)
    c.box(
        0.50,
        0.55,
        0.17,
        0.14,
        "Automatic differentiation\n" r"$\partial_t x_\theta$",
        fill=PALE_ORANGE,
        edge=ORANGE,
    )
    c.arrow((0.43, 0.59), (0.50, 0.62), color=ORANGE)
    c.box(
        0.50,
        0.31,
        0.17,
        0.14,
        "Model residual\n" r"$r_f=\partial_t x_\theta-f(x_\theta,p)$",
        fill="#FBE9E9",
        edge=RED,
    )
    c.arrow((0.585, 0.54), (0.585, 0.46), color=RED)
    c.box(
        0.74,
        0.43,
        0.21,
        0.25,
        r"Training objective"
        "\n"
        r"$\lambda_d\mathcal{L}_{data}$"
        "\n"
        r"$+\lambda_f\mathcal{L}_{ODE}$"
        "\n"
        r"$+\lambda_0\mathcal{L}_{IC}+\lambda_c\mathcal{L}_{con}$",
        fill=WHITE,
        edge=NAVY,
        fontsize=10.2,
        weight="bold",
    )
    c.arrow((0.67, 0.62), (0.74, 0.59), color=NAVY)
    c.arrow((0.67, 0.38), (0.74, 0.50), color=NAVY)
    c.box(
        0.31,
        0.12,
        0.38,
        0.10,
        "Independent validation: held-out times/nodes, ODE rollout, parameter error and residual maps",
        fill=PALE_GRAY,
        edge=MUTED,
        fontsize=9.8,
    )
    c.arrow((0.84, 0.42), (0.64, 0.23), label="evaluate after training", color=MUTED, bend=-0.10)
    return c.finish()


def node_time_encoder_decoder():
    """Show a factorized node-time architecture with explicit tensor shapes."""

    c = Canvas(
        "Node–time encoder–decoder for heterogeneous SIPS",
        subtitle="Node features encode reusable heterogeneity; the time branch represents temporal variation without one output head per node.",
    )
    blocks = [
        (
            0.04,
            0.57,
            0.15,
            "Node features\n" r"$F\in\mathbb{R}^{N\times d_f}$" "\ndegree, community, risk",
            PALE_BLUE,
            BLUE,
        ),
        (
            0.24,
            0.57,
            0.15,
            "Node/graph encoder\n" r"$d_f\to d\to d$" "\n" r"$H_x\in\mathbb{R}^{N\times d}$",
            PALE_TEAL,
            TEAL,
        ),
        (0.04, 0.28, 0.15, "Time input\n" r"$t\in\mathbb{R}^{B\times1}$", PALE_BLUE, BLUE),
        (
            0.24,
            0.28,
            0.15,
            "Optional Fourier + MLP\n" r"$1\to d\to d$" "\n" r"$H_t\in\mathbb{R}^{B\times d}$",
            PALE_ORANGE,
            ORANGE,
        ),
    ]
    for x, y, w, text_value, fill, edge in blocks:
        c.box(x, y, w, 0.18, text_value, fill=fill, edge=edge, fontsize=9.3, weight="bold")
    c.box(
        0.46,
        0.41,
        0.18,
        0.22,
        "Broadcast fusion\n"
        r"$[H_t,H_x]$"
        "\n"
        r"$\in\mathbb{R}^{B\times N\times2d}$",
        fill=PALE_GRAY,
        edge=MUTED,
        fontsize=9.8,
        weight="bold",
    )
    c.arrow((0.39, 0.66), (0.46, 0.56), color=TEAL)
    c.arrow((0.39, 0.37), (0.46, 0.47), color=ORANGE)
    c.box(
        0.71,
        0.46,
        0.13,
        0.16,
        "Shared decoder\n" r"$2d\to d\to3$",
        fill=PALE_TEAL,
        edge=TEAL,
        weight="bold",
    )
    c.arrow((0.64, 0.52), (0.71, 0.54), color=NAVY)
    c.box(
        0.88,
        0.44,
        0.09,
        0.20,
        "SIPS state\n" r"$X_\theta$" "\n" r"$[B,N,3]$",
        fill="#FBE9E9",
        edge=RED,
        fontsize=9.3,
        weight="bold",
    )
    c.arrow((0.84, 0.54), (0.88, 0.54), color=RED)
    c.box(
        0.29,
        0.10,
        0.48,
        0.10,
        "A separate positive rate head uses softplus; a gauge fixes the susceptibility–infectivity scale ambiguity.",
        fill=WHITE,
        edge=MUTED,
        fontsize=9.7,
    )
    return c.finish()


def pmp_informed_pinn():
    """Show state, costate and control heads tied by PMP residuals."""

    c = Canvas(
        "PMP-informed neural optimal control",
        subtitle="The network is trained against state dynamics, adjoint dynamics, stationarity and boundary conditions.",
    )
    c.box(0.04, 0.53, 0.12, 0.15, "Time\n$t$", fill=PALE_BLUE, edge=BLUE, weight="bold")
    heads = [
        (0.23, 0.69, "State head\n" r"$x_\theta(t)$", PALE_TEAL, TEAL),
        (0.23, 0.48, "Costate head\n" r"$\lambda_\psi(t)$", PALE_ORANGE, ORANGE),
        (0.23, 0.27, "Control head\n" r"$u_\phi(t)\in U$", PALE_BLUE, BLUE),
    ]
    for x, y, text_value, fill, edge in heads:
        c.box(x, y, 0.18, 0.13, text_value, fill=fill, edge=edge, weight="bold")
        c.arrow((0.16, 0.605), (x, y + 0.065), color=edge)
    c.box(
        0.49,
        0.43,
        0.18,
        0.25,
        r"Hamiltonian"
        "\n"
        r"$H=L(x,u)+\lambda^\top f(x,u)$"
        "\nmodel parameters\nand control bounds",
        fill=PALE_GRAY,
        edge=MUTED,
        fontsize=9.2,
        weight="bold",
    )
    for y, color in ((0.755, TEAL), (0.545, ORANGE), (0.335, BLUE)):
        c.arrow((0.41, y), (0.49, 0.56), color=color)
    residuals = [
        (0.75, 0.70, r"State: $\dot{x}-H_\lambda=0$"),
        (0.75, 0.53, r"Adjoint: $\dot{\lambda}+H_x=0$"),
        (0.75, 0.36, r"Interior stationarity: $H_u=0$"),
        (0.75, 0.19, r"Boundary: $x(0)=x_0$, $\lambda(T)=\Phi_x$"),
    ]
    for x, y, text_value in residuals:
        c.box(x, y, 0.22, 0.10, text_value, fill=WHITE, edge=RED, fontsize=9.1)
        c.arrow((0.67, 0.555), (x, y + 0.05), color=RED)
    c.label(
        0.48,
        0.11,
        "Bound-active controls require a projected/KKT residual; independent rollout remains the acceptance test.",
        ha="center",
        color=MUTED,
    )
    return c.finish()


def model_to_paper():
    """Show the evidence chain from a research question to a bounded claim."""

    c = Canvas(
        "From cyber mechanism to a defensible paper claim",
        subtitle="Each stage produces an artifact that can be inspected, rerun and challenged.",
        height=6.7,
    )
    stages = [
        ("1", "Question", "Threat, decision maker,\ninformation and horizon", PALE_BLUE, BLUE),
        ("2", "Model", "States, timing, dynamics,\nconstraints and objective", PALE_TEAL, TEAL),
        ("3", "Baseline", "No action, rule, random,\nclassical numerical method", PALE_GRAY, MUTED),
        ("4", "Method", "FBSM, RL/MARL, PINN/PIDL\nwith declared assumptions", PALE_ORANGE, ORANGE),
        ("5", "Ablation", "Remove components; vary\nheterogeneity and budgets", PALE_BLUE, BLUE),
        ("6", "Evidence", "Seeds, held-out cases,\nerrors, runtime and failures", PALE_TEAL, TEAL),
        ("7", "Claim", "Bound conclusion to the\nvalidated regime", "#FBE9E9", RED),
    ]
    positions = [
        (0.03, 0.57),
        (0.27, 0.57),
        (0.51, 0.57),
        (0.75, 0.57),
        (0.70, 0.24),
        (0.42, 0.24),
        (0.14, 0.24),
    ]
    for (number, title, body, fill, edge), (x, y) in zip(stages, positions):
        c.circle(x + 0.10, y + 0.23, 0.022, number, fill=WHITE, edge=edge, fontsize=8.5)
        c.box(
            x, y, 0.20, 0.19, f"{title}\n{body}", fill=fill, edge=edge, fontsize=9.0, weight="bold"
        )
    for idx in range(3):
        x, y = positions[idx]
        nx, ny = positions[idx + 1]
        c.arrow((x + 0.205, y + 0.095), (nx - 0.005, ny + 0.095), color=NAVY)
    c.arrow((0.85, 0.565), (0.80, 0.435), color=NAVY, bend=0.18)
    c.arrow((0.695, 0.335), (0.625, 0.335), color=NAVY)
    c.arrow((0.415, 0.335), (0.345, 0.335), color=NAVY)
    c.box(
        0.11,
        0.07,
        0.72,
        0.09,
        "Reproducibility record: resolved parameter arrays, random seeds, graph/data provenance, hardware, software versions,\n"
        "training diagnostics, figure source and a machine-readable run manifest.",
        fill=WHITE,
        edge=MUTED,
        fontsize=9.8,
    )
    return c.finish()


DIAGRAMS = (
    DiagramDefinition(
        "repository_family",
        repository_family,
        "Three-repository family showing the foundation package and method-specific Note 1 and Note 2 packages.",
        "all READMEs; all guides",
    ),
    DiagramDefinition(
        "state_transitions",
        state_transitions,
        "Canonical SIPS susceptible, infected and protected transitions with aggregate and node-indexed infection pressure.",
        "foundation model guide; Note 1 dynamics; Note 2 residuals",
    ),
    DiagramDefinition(
        "degree_vs_node",
        degree_vs_node,
        "Degree-class mixing model compared with explicit node adjacency and heterogeneous node states.",
        "foundation README and model guide",
    ),
    DiagramDefinition(
        "control_timing",
        control_timing,
        "Five timing lanes separating continuous-time, sampled ZOH, switched, impulse and continuous-impulsive control.",
        "foundation guide; Note 1 README and guide",
    ),
    DiagramDefinition(
        "ode_environment",
        ode_environment,
        "One sampled ODE environment transition from observation and action through reset, integration and reward.",
        "Note 1 README and model guide",
    ),
    DiagramDefinition(
        "ddqn",
        ddqn,
        "Double DQN data flow with online and target networks, replay buffer and the DDQN target equation.",
        "Note 1 method guide",
    ),
    DiagramDefinition(
        "mappo_ctde",
        mappo_ctde,
        "Budgeted cooperative PPO path with a joint community-action coordinator, centralized value critic, rollout buffer and GAE.",
        "Note 1 README and method guide",
    ),
    DiagramDefinition(
        "hierarchical_game",
        hierarchical_game,
        "Hierarchical attacker-defender policy with target selection, bounded allocation and opponent-pool evaluation.",
        "Note 1 method guide and paper workflow",
    ),
    DiagramDefinition(
        "pinn_loss",
        pinn_loss,
        "Physics-informed network connecting observations, collocation points, autodifferentiation, residuals and validation.",
        "Note 2 README and method guide",
    ),
    DiagramDefinition(
        "node_time_encoder_decoder",
        node_time_encoder_decoder,
        "Factorized node-time graph encoder-decoder with explicit tensor shapes for heterogeneous SIPS inference.",
        "Note 2 method guide",
    ),
    DiagramDefinition(
        "pmp_informed_pinn",
        pmp_informed_pinn,
        "PMP-informed neural control with state, costate and control heads feeding Hamiltonian residuals.",
        "foundation PMP guide; Note 2 method guide",
    ),
    DiagramDefinition(
        "model_to_paper",
        model_to_paper,
        "Seven-stage research workflow from question and model to baselines, evidence and bounded claims.",
        "all paper-workflow guides",
    ),
)


def get_diagram(diagram_id: str) -> DiagramDefinition:
    """Return one diagram definition or raise a descriptive key error."""

    for definition in DIAGRAMS:
        if definition.diagram_id == diagram_id:
            return definition
    known = ", ".join(definition.diagram_id for definition in DIAGRAMS)
    raise KeyError(f"unknown diagram {diagram_id!r}; expected one of: {known}")


def render_diagrams(
    output_dir: str | Path,
    *,
    diagram_ids: Iterable[str] | None = None,
    dpi: int = 240,
) -> dict[str, tuple[Path, Path, Path]]:
    """Render selected diagrams as PDF, SVG and PNG from the canonical source.

    Parameters
    ----------
    output_dir:
        Destination directory.  It is created when needed.
    diagram_ids:
        IDs to render.  ``None`` renders the complete maintained set.
    dpi:
        Raster resolution for PNG previews.  PDF and SVG remain vector files.
    """

    import matplotlib.pyplot as plt

    from .plotting import guide_style

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected = (
        tuple(diagram_ids)
        if diagram_ids is not None
        else tuple(definition.diagram_id for definition in DIAGRAMS)
    )
    written: dict[str, tuple[Path, Path, Path]] = {}
    for diagram_id in selected:
        definition = get_diagram(diagram_id)
        with guide_style():
            fig = definition.renderer()
        paths = tuple(destination / f"{diagram_id}.{suffix}" for suffix in ("pdf", "svg", "png"))
        for path in paths:
            kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
            if path.suffix == ".png":
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
            if path.suffix == ".svg":
                lines = path.read_text(encoding="utf-8").splitlines()
                path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        plt.close(fig)
        written[diagram_id] = paths
    return written
