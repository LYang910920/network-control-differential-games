# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see ../../COPYRIGHT_AND_LICENSE.md.

"""Compatibility wrapper for the tutorial example runner.

The implementation lives in `code/run_all_lecture_examples.py` so the tutorial
Python code is grouped in one place. This wrapper keeps the old command
`python run_all_lecture_examples.py` working from `examples/lecture/`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


RUNNER = Path(__file__).resolve().parent / "code" / "run_all_lecture_examples.py"


if __name__ == "__main__":
    runpy.run_path(str(RUNNER), run_name="__main__")
