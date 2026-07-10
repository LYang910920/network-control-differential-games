# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see COPYRIGHT_AND_LICENSE.md.

"""Deprecated root runner; use ``cybercontrol`` or ``python -m cybercontrol``."""

from __future__ import annotations

import sys
import warnings

from cybercontrol.cli import main as cli_main


def main() -> None:
    """Forward legacy invocations to the public package CLI."""

    warnings.warn(
        "run_all.py is deprecated; use 'python -m cybercontrol <command>'",
        DeprecationWarning,
        stacklevel=2,
    )
    cli_main(sys.argv[1:] or ["all"])


if __name__ == "__main__":
    main()
