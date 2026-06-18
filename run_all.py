# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see COPYRIGHT_AND_LICENSE.md.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def tutorial_command(py: str, args: argparse.Namespace) -> list[str]:
    """Build the tutorial command from root-level options.

    Keeping these flags at the root makes the common workflow discoverable:
    users can skip or shorten the scalability run without entering the nested
    `examples/lecture/code/` folder.
    """
    cmd = [
        py,
        "code/run_all_lecture_examples.py",
        "--steps",
        str(args.tutorial_steps),
    ]
    if args.skip_scalability:
        cmd.append("--skip-scalability")
    if args.scalability_sizes:
        cmd.extend(["--scalability-sizes", args.scalability_sizes])
    if args.scalability_repeats is not None:
        cmd.extend(["--scalability-repeats", str(args.scalability_repeats)])
    return cmd


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"\n[{cwd.relative_to(ROOT)}] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the tutorial examples and reference-repository smoke tests from the repository root."
    )
    parser.add_argument(
        "--skip-tutorial",
        dest="skip_tutorial",
        action="store_true",
        help="Do not run the tutorial examples.",
    )
    parser.add_argument("--skip-lecture", dest="skip_tutorial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-reference", action="store_true", help="Do not run the reference smoke tests.")
    parser.add_argument(
        "--tutorial-steps",
        dest="tutorial_steps",
        type=int,
        default=45,
        help="Time grid size for tutorial companion examples.",
    )
    parser.add_argument("--lecture-steps", dest="tutorial_steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--skip-scalability", action="store_true", help="Skip the tutorial scalability experiment.")
    parser.add_argument(
        "--scalability-sizes",
        default=None,
        help="Optional comma-separated node counts for the tutorial scalability experiment.",
    )
    parser.add_argument(
        "--scalability-repeats",
        type=int,
        default=None,
        help="Optional repeat count per network size for the tutorial scalability experiment.",
    )
    parser.add_argument(
        "--reference-pydeps",
        type=Path,
        default=ROOT / "examples" / "reference" / "pydeps",
        help="Optional local dependency directory containing python-igraph.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    py = sys.executable

    if not args.skip_tutorial:
        tutorial_dir = ROOT / "examples" / "lecture"
        run(tutorial_command(py, args), tutorial_dir, env)

    if not args.skip_reference:
        reference_dir = ROOT / "examples" / "reference"
        cmd = [py, "run_reference_smoke.py"]
        if args.reference_pydeps.exists():
            cmd.extend(["--pydeps", str(args.reference_pydeps)])
        run(cmd, reference_dir, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
