"""Input/output and reproducibility helpers."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_csv(path: Path | str, rows: list[dict]) -> None:
    """Write a list of dictionaries as a CSV file with stable line endings."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path | str, data: dict[str, Any]) -> None:
    """Write a small JSON artifact with deterministic formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_outputs(paths: list[Path | str]) -> None:
    """Fail fast when an expected artifact is missing or empty."""

    normalized = [Path(path) for path in paths]
    missing = [path for path in normalized if not path.exists()]
    empty = [path for path in normalized if path.exists() and path.stat().st_size == 0]
    if missing or empty:
        details = []
        if missing:
            details.append("missing: " + ", ".join(str(path) for path in missing))
        if empty:
            details.append("empty: " + ", ".join(str(path) for path in empty))
        raise RuntimeError("Expected output check failed; " + "; ".join(details))
