#!/usr/bin/env python3
"""Validate the canonical cross-repository diagram manifest."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "assets" / "diagram_manifest.csv"
REQUIRED_COLUMNS = {"id", "source", "svg_or_pdf", "png", "alt_text", "used_in"}
REQUIRED_CORE = {
    "repository_family",
    "state_transitions",
    "degree_vs_node",
    "ddqn",
    "mappo_ctde",
    "pinn_loss",
    "node_time_encoder_decoder",
}


def resolve_manifest_path(raw: str) -> Path:
    """Resolve a workspace path in both sibling and standalone checkouts."""

    path = Path(raw)
    workspace_candidate = ROOT.parent / path
    if workspace_candidate.exists():
        return workspace_candidate
    parts = path.parts
    if parts and parts[0] == "network-control-differential-games":
        return ROOT.joinpath(*parts[1:])
    return workspace_candidate


def main() -> None:
    errors: list[str] = []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if columns != REQUIRED_COLUMNS:
            errors.append(
                f"manifest columns must be {sorted(REQUIRED_COLUMNS)}; got {sorted(columns)}"
            )
        rows = list(reader)

    ids = {row.get("id", "") for row in rows}
    missing_core = REQUIRED_CORE - ids
    if missing_core:
        errors.append(f"missing core diagrams: {sorted(missing_core)}")
    if len(ids) != len(rows):
        errors.append("diagram ids must be unique")

    for row in rows:
        diagram_id = row.get("id", "<missing-id>")
        for field in ("source", "svg_or_pdf", "png"):
            value = row.get(field, "").strip()
            if not value:
                errors.append(f"{diagram_id}: missing {field}")
            elif not resolve_manifest_path(value).is_file():
                errors.append(f"{diagram_id}: missing {field} file {value}")
        for field in ("alt_text", "used_in"):
            if not row.get(field, "").strip():
                errors.append(f"{diagram_id}: missing {field}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"diagram manifest OK: {len(rows)} maintained diagrams")


if __name__ == "__main__":
    main()
