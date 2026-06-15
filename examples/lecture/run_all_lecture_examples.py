# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def make_contact_sheet(items: list[tuple[str, Path]], out_path: Path, cols: int = 3) -> None:
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.5 * rows))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (title, path) in zip(flat_axes, items):
        ax.imshow(plt.imread(path))
        ax.set_title(title)
        ax.axis("off")

    for ax in flat_axes[len(items):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all lecture-package optimal-control/differential-game examples.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "results" / ("rerun_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Directory where rerun outputs will be written.",
    )
    parser.add_argument("--steps", type=int, default=45, help="Time-grid size for companion examples.")
    args = parser.parse_args()

    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str((HERE / ".mplconfig").resolve()))
    env.setdefault("XDG_CACHE_HOME", str((HERE / ".cache").resolve()))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    py = sys.executable
    simple = HERE / "code" / "simple_degree_k_control.py"
    examples = HERE / "code" / "network_control_examples.py"

    run([py, str(simple), "--output-dir", str(out / "simple_demo")], env)
    run([
        py, str(simple),
        "--edge-list", str(HERE / "sample_data" / "sample_edges.csv"),
        "--delimiter", ",",
        "--has-header",
        "--source-col", "source",
        "--target-col", "target",
        "--output-dir", str(out / "simple_edges"),
    ], env)
    run([
        py, str(simple),
        "--adjacency-csv", str(HERE / "sample_data" / "sample_adjacency.csv"),
        "--output-dir", str(out / "simple_adjacency"),
    ], env)

    run([py, str(examples), "--steps", str(args.steps), "--output-dir", str(out / "examples_demo")], env)
    run([
        py, str(examples),
        "--edge-list", str(HERE / "sample_data" / "sample_edges.csv"),
        "--delimiter", ",",
        "--has-header",
        "--source-col", "source",
        "--target-col", "target",
        "--steps", str(args.steps),
        "--output-dir", str(out / "examples_edges"),
    ], env)
    run([
        py, str(examples),
        "--adjacency-csv", str(HERE / "sample_data" / "sample_adjacency.csv"),
        "--steps", str(args.steps),
        "--output-dir", str(out / "examples_adjacency"),
    ], env)

    make_contact_sheet(
        [
            ("simple demo: degree", out / "simple_demo" / "degree_distribution.png"),
            ("simple demo: control", out / "simple_demo" / "degree_k_control.png"),
            ("simple edge: degree", out / "simple_edges" / "degree_distribution.png"),
            ("simple edge: control", out / "simple_edges" / "degree_k_control.png"),
            ("simple adjacency: degree", out / "simple_adjacency" / "degree_distribution.png"),
            ("simple adjacency: control", out / "simple_adjacency" / "degree_k_control.png"),
        ],
        out / "simple_contact_sheet.png",
    )
    make_contact_sheet(
        [
            ("demo: degree", out / "examples_demo" / "degree_distribution.png"),
            ("demo: control", out / "examples_demo" / "degree_control.png"),
            ("demo: game", out / "examples_demo" / "degree_game.png"),
            ("demo: node control", out / "examples_demo" / "node_control.png"),
            ("demo: node game", out / "examples_demo" / "node_game.png"),
            ("demo: hybrid", out / "examples_demo" / "hybrid_impulse.png"),
            ("edge: degree", out / "examples_edges" / "degree_distribution.png"),
            ("edge: control", out / "examples_edges" / "degree_control.png"),
            ("edge: game", out / "examples_edges" / "degree_game.png"),
            ("adjacency: degree", out / "examples_adjacency" / "degree_distribution.png"),
            ("adjacency: control", out / "examples_adjacency" / "degree_control.png"),
            ("adjacency: game", out / "examples_adjacency" / "degree_game.png"),
        ],
        out / "examples_contact_sheet.png",
    )

    print(f"\nDone. Outputs written to: {out}")


if __name__ == "__main__":
    main()
