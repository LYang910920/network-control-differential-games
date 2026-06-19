# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see ../../COPYRIGHT_AND_LICENSE.md.

"""Convenience wrapper for the foundation example runner.

The implementation lives in `code/run_foundation_examples.py` so the Python
code is grouped in one place. This wrapper lets users run
`python run_foundation_examples.py` from `examples/foundations/`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


RUNNER = Path(__file__).resolve().parent / "code" / "run_foundation_examples.py"


if __name__ == "__main__":
    runpy.run_path(str(RUNNER), run_name="__main__")
