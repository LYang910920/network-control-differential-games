# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see COPYRIGHT_AND_LICENSE.md.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def foundation_command(py: str, args: argparse.Namespace) -> list[str]:
    """Build the foundation-example command from root-level options.

    Keeping these flags at the root makes the common workflow discoverable:
    users can skip or shorten the scalability run without entering the nested
    `examples/foundations/code/` folder.
    """
    cmd = [
        py,
        "code/run_foundation_examples.py",
        "--steps",
        str(args.foundation_steps),
    ]
    if args.skip_scalability:
        cmd.append("--skip-scalability")
    if args.scalability_sizes:
        cmd.extend(["--scalability-sizes", args.scalability_sizes])
    if args.scalability_repeats is not None:
        cmd.extend(["--scalability-repeats", str(args.scalability_repeats)])
    if args.scalability_steps is not None:
        cmd.extend(["--scalability-steps", str(args.scalability_steps)])
    if args.scalability_iterations is not None:
        cmd.extend(["--scalability-iterations", str(args.scalability_iterations)])
    if args.include_node_scalability:
        cmd.append("--include-node-scalability")
    if args.node_scalability_sizes:
        cmd.extend(["--node-scalability-sizes", args.node_scalability_sizes])
    if args.node_scalability_repeats is not None:
        cmd.extend(["--node-scalability-repeats", str(args.node_scalability_repeats)])
    if args.node_scalability_steps is not None:
        cmd.extend(["--node-scalability-steps", str(args.node_scalability_steps)])
    if args.node_scalability_iterations is not None:
        cmd.extend(["--node-scalability-iterations", str(args.node_scalability_iterations)])
    return cmd


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"\n[{cwd.relative_to(ROOT)}] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the foundation examples and reference-repository smoke tests from the repository root."
    )
    parser.add_argument(
        "--skip-foundations",
        dest="skip_foundations",
        action="store_true",
        help="Do not run the foundation examples.",
    )
    parser.add_argument("--skip-tutorial", dest="skip_foundations", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-reference", action="store_true", help="Do not run the reference smoke tests.")
    parser.add_argument(
        "--foundation-steps",
        dest="foundation_steps",
        type=int,
        default=45,
        help="Time grid size for the main foundation examples.",
    )
    parser.add_argument("--tutorial-steps", dest="foundation_steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--skip-scalability", action="store_true", help="Skip the foundation scalability experiment.")
    parser.add_argument(
        "--scalability-sizes",
        default=None,
        help="Optional comma-separated node counts for the paired degree/node scalability experiment.",
    )
    parser.add_argument(
        "--scalability-repeats",
        type=int,
        default=None,
        help="Optional repeat count per network size for the paired degree/node scalability experiment.",
    )
    parser.add_argument(
        "--scalability-steps",
        type=int,
        default=None,
        help="Optional time-grid size for the paired degree/node scalability experiment.",
    )
    parser.add_argument(
        "--scalability-iterations",
        type=int,
        default=None,
        help="Optional max FBS iterations for the paired degree/node scalability experiment.",
    )
    parser.add_argument(
        "--include-node-scalability",
        action="store_true",
        help="Also run the heavier sparse node-level scalability experiment.",
    )
    parser.add_argument(
        "--node-scalability-sizes",
        default=None,
        help="Optional comma-separated node counts for sparse node-level scalability.",
    )
    parser.add_argument(
        "--node-scalability-repeats",
        type=int,
        default=None,
        help="Optional repeat count for sparse node-level scalability.",
    )
    parser.add_argument(
        "--node-scalability-steps",
        type=int,
        default=None,
        help="Optional time-grid size for sparse node-level scalability.",
    )
    parser.add_argument(
        "--node-scalability-iterations",
        type=int,
        default=None,
        help="Optional max FBS iterations for sparse node-level scalability.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    py = sys.executable

    if not args.skip_foundations:
        foundation_dir = ROOT / "examples" / "foundations"
        run(foundation_command(py, args), foundation_dir, env)

    if not args.skip_reference:
        reference_dir = ROOT / "examples" / "reference"
        cmd = [py, "run_reference_smoke.py"]
        run(cmd, reference_dir, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
