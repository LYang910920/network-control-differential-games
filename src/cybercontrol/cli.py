"""Single public command-line interface for the foundation repository."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .experiments import git_sha, hardware_summary
from .fbs import (
    solve_degree_control,
    solve_degree_game,
    solve_node_control,
    solve_node_game,
)
from .graph_data import DegreeData
from .io import write_json
from .models import MalwareParams, controlled_sir_rhs, isolation_jump
from .numerics import project_simplex3, rk4_integrate

ROOT = Path(__file__).resolve().parents[2]


def _source_root() -> Path:
    """Return the repository root or explain that an operation needs source files."""

    required = (ROOT / "examples" / "foundations" / "code", ROOT / "docs" / "source")
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "This command needs a source checkout containing examples/ and docs/source/. "
            "The installed cybercontrol package can still be imported normally."
        )
    return ROOT


def _run(relative_path: str, *arguments: str) -> None:
    root = _source_root()
    subprocess.run([sys.executable, relative_path, *arguments], cwd=root, check=True)


def _run_foundation_examples(*, output_dir: Path, steps: int) -> None:
    _run(
        "examples/foundations/code/network_control_examples.py",
        "--steps",
        str(steps),
        "--output-dir",
        str(output_dir),
    )


def _package_smoke() -> dict[str, float | int | bool]:
    """Exercise public scientific kernels without relying on source-only files."""

    degree = np.array([1.0, 2.0, 4.0, 6.0])
    counts = np.array([12, 8, 5, 3])
    probability = counts / counts.sum()
    degree_data = DegreeData(
        k=degree,
        Nk=counts,
        pk=probability,
        kbar=float(degree @ probability),
        node_degree=np.repeat(degree.astype(int), counts),
    )
    nodes = 8
    adjacency = np.zeros((nodes, nodes), dtype=float)
    for node in range(nodes):
        adjacency[node, (node - 1) % nodes] = 0.5
        adjacency[node, (node + 1) % nodes] = 0.5

    solvers = {
        "degree_control": solve_degree_control(degree_data, steps=12, iterations=24, tol=1e-4),
        "degree_game": solve_degree_game(degree_data, steps=12, iterations=24, tol=1e-4),
        "node_control": solve_node_control(adjacency, steps=12, iterations=24, tol=1e-4),
        "node_game": solve_node_game(adjacency, steps=12, iterations=24, tol=1e-4),
    }
    for name, result in solvers.items():
        if not np.all(np.isfinite(result.x)) or np.min(result.x) < 0.0 or np.max(result.x) > 1.0:
            raise RuntimeError(f"{name} produced an invalid state trajectory")
        if not bool(result.value["converged"]):
            raise RuntimeError(f"{name} did not converge in the smoke profile")

    initial = np.array([0.78, 0.17, 0.05])
    flowed, _ = rk4_integrate(
        lambda state, _time: controlled_sir_rhs(
            state, u_patch=0.08, u_clean=0.12, p=MalwareParams()
        ),
        initial,
        t0=0.0,
        dt=0.5,
        substeps=8,
        project=project_simplex3,
    )
    jumped = isolation_jump(flowed, isolation_rate=0.5)
    if jumped[1] >= flowed[1]:
        raise RuntimeError("the isolation reset did not reduce infected mass")

    return {
        "degree_control_cost": float(solvers["degree_control"].value["cost"]),
        "degree_game_attacker_payoff": float(solvers["degree_game"].value["JA"]),
        "node_control_cost": float(solvers["node_control"].value["cost"]),
        "node_game_defender_payoff": float(solvers["node_game"].value["JD"]),
        "all_fbs_converged": True,
        "max_fbs_final_delta": max(
            float(result.value["final_delta"]) for result in solvers.values()
        ),
        "impulse_mass_error": float(abs(jumped.sum() - 1.0)),
        "impulse_infected_before": float(flowed[1]),
        "impulse_infected_after": float(jumped[1]),
    }


def _run_package_smoke(output_dir: Path) -> Path:
    """Run the package-only smoke profile and write its complete manifest."""

    started = time.perf_counter()
    metrics = _package_smoke()
    return write_json(
        output_dir / "run_manifest.json",
        {
            "command": "smoke",
            "commit_sha": git_sha(ROOT),
            "duration_seconds": time.perf_counter() - started,
            "hardware": hardware_summary(),
            "metrics": metrics,
        },
    )


def _run_scalability(
    *,
    output_dir: Path,
    sizes: str,
    steps: int,
    iterations: int,
) -> None:
    _run(
        "examples/foundations/code/scalability_analysis.py",
        "--model-level",
        "compare",
        "--sizes",
        sizes,
        "--steps",
        str(steps),
        "--iterations",
        str(iterations),
        "--repeats",
        "1",
        "--output-dir",
        str(output_dir),
    )


def _build_docs() -> None:
    root = _source_root()
    source = root / "docs" / "source"
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "network_control_foundations.tex"],
        cwd=source,
        check=True,
    )
    built = source / "network_control_foundations.pdf"
    if built.exists():
        (root / "docs" / built.name).write_bytes(built.read_bytes())


def build_parser() -> argparse.ArgumentParser:
    """Build the public five-command parser used across the repository family."""

    parser = argparse.ArgumentParser(
        prog="cybercontrol",
        description="Network control and differential-game examples.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Fast degree/node control and game execution check.")
    smoke.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    medium = sub.add_parser("medium", help="Bounded foundation and paired scalability experiment.")
    medium.add_argument("--output-dir", type=Path, default=Path("artifacts/medium"))
    medium.add_argument("--sizes", default="100,1000,10000")
    medium.add_argument("--steps", type=int, default=32)
    medium.add_argument("--iterations", type=int, default=35)
    sub.add_parser("figures", help="Regenerate the shared vector guide diagrams.")
    sub.add_parser("docs", help="Build the current foundation tutorial PDF.")
    sub.add_parser("all", help="Run smoke, figures, and docs.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute a public foundation command."""

    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        _run_package_smoke(args.output_dir)
    elif args.command == "medium":
        started = time.perf_counter()
        _run_foundation_examples(output_dir=args.output_dir / "models", steps=max(24, args.steps))
        _run_scalability(
            output_dir=args.output_dir / "scalability",
            sizes=args.sizes,
            steps=args.steps,
            iterations=args.iterations,
        )
        write_json(
            args.output_dir / "run_manifest.json",
            {
                "command": "medium",
                "commit_sha": git_sha(ROOT),
                "duration_seconds": time.perf_counter() - started,
                "hardware": hardware_summary(),
                "sizes": args.sizes,
                "steps": args.steps,
                "max_iterations": args.iterations,
            },
        )
    elif args.command == "figures":
        _run("scripts/generate_diagrams.py")
    elif args.command == "docs":
        _build_docs()
    elif args.command == "all":
        _run_package_smoke(Path("artifacts/smoke"))
        _run("scripts/generate_diagrams.py")
        _build_docs()


if __name__ == "__main__":
    main()
