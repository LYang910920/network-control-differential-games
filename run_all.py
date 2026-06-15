from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"\n[{cwd.relative_to(ROOT)}] $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lecture examples and reference-repository smoke tests from the repository root."
    )
    parser.add_argument("--skip-lecture", action="store_true", help="Do not run the lecture examples.")
    parser.add_argument("--skip-reference", action="store_true", help="Do not run the reference smoke tests.")
    parser.add_argument("--lecture-steps", type=int, default=45, help="Time grid size for lecture companion examples.")
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

    if not args.skip_lecture:
        lecture_dir = ROOT / "examples" / "lecture"
        run(
            [
                py,
                "run_all_lecture_examples.py",
                "--steps",
                str(args.lecture_steps),
            ],
            lecture_dir,
            env,
        )

    if not args.skip_reference:
        reference_dir = ROOT / "examples" / "reference"
        cmd = [py, "run_reference_smoke.py"]
        if args.reference_pydeps.exists():
            cmd.extend(["--pydeps", str(args.reference_pydeps)])
        run(cmd, reference_dir, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
