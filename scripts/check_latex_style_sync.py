"""Check that the shared LaTeX style file is byte-identical across repos."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    canonical = repo / "tex" / "cyberguide.sty"
    targets = [
        workspace / "note1-cyber-control-games" / "docs" / "latex" / "cyberguide.sty",
        workspace / "note2-pinn-pidl-cyber-control" / "docs" / "latex" / "cyberguide.sty",
    ]
    if not canonical.exists():
        print(f"missing canonical style: {canonical}")
        return 1
    canonical_bytes = canonical.read_bytes()
    ok = True
    for target in targets:
        if not target.exists():
            print(f"missing synced style: {target}")
            ok = False
            continue
        if target.read_bytes() != canonical_bytes:
            print(f"style drift: {target}")
            ok = False
        else:
            print(f"ok: {target.relative_to(workspace)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
