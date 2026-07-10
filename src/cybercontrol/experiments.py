"""Reproducible run configuration, hardware discovery, and manifests."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuntimeConfig:
    """Explicit runtime choices made by an experiment runner."""

    seed: int | None = 0
    device: str = "auto"
    dtype: str = "float32"
    threads: int | None = 1
    profile: str = "smoke"


def git_sha(path: str | Path = ".") -> str:
    """Return the current Git SHA, or ``unknown`` outside a worktree."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_metadata(path: str | Path = ".") -> dict[str, str | bool]:
    """Return commit, branch, and dirty state for experiment provenance."""

    root = str(path)
    try:
        branch = subprocess.check_output(
            ["git", "-C", root, "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        branch, status = "unknown", ""
    return {
        "commit_sha": git_sha(path),
        "git_branch": branch or "detached",
        "git_dirty": bool(status.strip()),
    }


def hardware_summary() -> dict[str, Any]:
    """Return portable CPU, RAM, accelerator, and software metadata."""

    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import torch

        summary.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "mps_available": bool(
                    getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
                ),
            }
        )
    except ImportError:
        summary["torch"] = "not-installed"
    return summary


def run_provenance(path: str | Path = ".") -> dict[str, Any]:
    """Return the common Git and hardware fields used by all public runners."""

    return {**git_metadata(path), "hardware": hardware_summary()}


def resolve_torch_runtime(config: RuntimeConfig):
    """Apply explicit seed/thread choices and resolve Torch device and dtype.

    This function is called by runners, never at import time. ``seed=None``
    preserves existing RNG states. The function does not alter Torch's
    process-wide default dtype; models and tensors should request the returned
    dtype explicitly when a non-default precision is needed.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Install the optional Torch dependency with `pip install -e .[torch]`."
        ) from exc

    if config.threads is not None and config.threads > 0:
        torch.set_num_threads(int(config.threads))
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    requested = config.device
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device == "mps" and not (
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ):
            raise RuntimeError("Apple MPS was requested but is unavailable.")
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
    try:
        dtype = getattr(torch, config.dtype)
    except AttributeError as exc:
        raise ValueError(f"unsupported Torch dtype: {config.dtype}") from exc
    return torch, device, dtype
