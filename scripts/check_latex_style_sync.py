"""Check that LaTeX style copies match the canonical file in this repo."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help="Style files that should match docs/source/cyberguide.sty.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    canonical = repo / "docs" / "source" / "cyberguide.sty"
    if not canonical.exists():
        print(f"missing canonical style: {canonical}")
        return 1
    if not args.targets:
        print("no target style files supplied; canonical style exists")
        return 0
    canonical_bytes = canonical.read_bytes()
    ok = True
    for target in args.targets:
        if not target.exists():
            print(f"missing synced style: {target}")
            ok = False
            continue
        if target.read_bytes() != canonical_bytes:
            print(f"style drift: {target}")
            ok = False
        else:
            print(f"ok: {target}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
