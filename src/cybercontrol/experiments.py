"""Reproducible run configuration, hardware discovery, and manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io import write_json


@dataclass(frozen=True)
class RuntimeConfig:
    """Explicit runtime choices made by an experiment runner."""

    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    threads: int | None = 1
    profile: str = "smoke"


@dataclass
class RunManifest:
    """Small serializable record for one bounded experiment."""

    experiment: str
    runtime: RuntimeConfig
    resolved_device: str
    started_at_unix: float
    duration_seconds: float = 0.0
    commit_sha: str = "unknown"
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def finish(self, *, metrics: Mapping[str, Any] | None = None) -> None:
        """Finalize elapsed time and optional scalar/list metrics."""

        self.duration_seconds = float(time.time() - self.started_at_unix)
        if metrics:
            self.metrics.update(dict(metrics))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        payload = asdict(self)
        payload["runtime"] = asdict(self.runtime)
        return payload


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


def resolve_torch_runtime(config: RuntimeConfig):
    """Apply explicit seed/thread choices and resolve Torch device and dtype.

    This function is called by runners, never at import time. It does not alter
    Torch's process-wide default dtype; models and tensors should request the
    returned dtype explicitly when a non-default precision is needed.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Install the optional Torch dependency with `pip install -e .[torch]`."
        ) from exc

    if config.threads is not None and config.threads > 0:
        torch.set_num_threads(int(config.threads))
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


def start_manifest(
    experiment: str,
    runtime: RuntimeConfig,
    *,
    repo: str | Path = ".",
    config: Mapping[str, Any] | None = None,
) -> RunManifest:
    """Create a manifest after resolving the requested runtime."""

    torch, device, _ = resolve_torch_runtime(runtime)
    return RunManifest(
        experiment=experiment,
        runtime=runtime,
        resolved_device=device,
        started_at_unix=time.time(),
        commit_sha=git_sha(repo),
        hardware=hardware_summary(),
        software={"torch": torch.__version__, "numpy": np.__version__},
        config=dict(config or {}),
    )


def write_run_manifest(path: str | Path, manifest: RunManifest) -> Path:
    """Write one completed manifest as formatted JSON."""

    out = Path(path)
    write_json(out, manifest.to_dict())
    return out
