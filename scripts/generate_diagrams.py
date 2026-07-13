#!/usr/bin/env python3
"""Generate the shared guide-diagram set and synchronize curated copies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
from cybercontrol.guide_diagrams import DIAGRAMS, render_diagrams


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CANONICAL_DIR = ROOT / "docs" / "assets" / "diagrams"
MANIFEST = ROOT / "docs" / "assets" / "diagram_manifest.csv"

NOTE_SELECTIONS = {
    "note1-cyber-control-games": {
        "repository_family",
        "state_transitions",
        "ode_environment",
        "ddqn",
        "mappo_ctde",
        "hierarchical_game",
        "model_to_paper",
    },
    "note2-pinn-pidl-cyber-control": {
        "repository_family",
        "state_transitions",
        "pinn_loss",
        "node_time_encoder_decoder",
        "pmp_informed_pinn",
        "model_to_paper",
    },
}


def synchronize_notes(diagram_id: str, paths: tuple[Path, Path, Path]) -> None:
    """Copy only diagrams used by each self-contained note repository."""

    for repo_name, selected_ids in NOTE_SELECTIONS.items():
        if diagram_id not in selected_ids:
            continue
        repo = WORKSPACE / repo_name
        if not (repo / "pyproject.toml").is_file():
            continue
        destination = repo / "docs" / "assets" / "diagrams"
        destination.mkdir(parents=True, exist_ok=True)
        for source in paths:
            shutil.copy2(source, destination / source.name)


def write_manifest(rows: list[dict[str, str]]) -> None:
    """Write the machine-checkable canonical diagram manifest."""

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("id", "source", "svg_or_pdf", "png", "alt_text", "used_in"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=240, help="PNG resolution (default: 240 dpi)")
    parser.add_argument(
        "--sync-notes",
        action="store_true",
        help="Also copy selected assets to sibling Note 1 and Note 2 checkouts.",
    )
    parser.add_argument("--no-sync", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    rendered = render_diagrams(CANONICAL_DIR, dpi=args.dpi)
    rows = []
    for definition in DIAGRAMS:
        pdf_path, svg_path, png_path = rendered[definition.diagram_id]
        sync_notes = args.sync_notes and not args.no_sync
        if sync_notes:
            synchronize_notes(definition.diagram_id, (pdf_path, svg_path, png_path))
        rows.append(
            {
                "id": definition.diagram_id,
                "source": ("network-control-differential-games/src/cybercontrol/guide_diagrams.py"),
                "svg_or_pdf": (f"network-control-differential-games/{svg_path.relative_to(ROOT)}"),
                "png": (f"network-control-differential-games/{png_path.relative_to(ROOT)}"),
                "alt_text": definition.alt_text,
                "used_in": definition.used_in,
            }
        )
    write_manifest(rows)
    print(f"Generated {len(rows)} diagrams in {CANONICAL_DIR}")
    if args.sync_notes and not args.no_sync:
        print("Synchronized curated copies to Note 1 and Note 2")


if __name__ == "__main__":
    main()
