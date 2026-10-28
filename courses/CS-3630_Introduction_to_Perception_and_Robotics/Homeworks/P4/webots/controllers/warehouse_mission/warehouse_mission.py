"""CS 3630 Project 4: the controller of the warehouse robot (``DEF AMR``).

Students: you do not need to read or change this file. It loads your
``project4.py`` (the one next to the ``webots/`` folder) and runs the whole
warehouse mission with it. See "Play it in Webots" in ``README.md``.

Staff: the show lives in ``wm/viz`` (presentation) and ``wm/core`` (the
mission engine). Controller arguments (``controllerArgs``, else the
``WAREHOUSE_ARGS`` environment variable) are documented in ``wm/viz/app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _environment_ok() -> bool:
    """Explain the two classic setup problems instead of crashing with a traceback."""
    # Webots can launch an interpreter older than this repository supports.
    if sys.version_info < (3, 9):  # noqa: UP036
        print(
            f"warehouse_mission: Python {sys.version.split()[0]} is too old (3.9 or newer "
            f"is needed).\n"
            "Set Webots > Preferences > General > Python command to the cs3630_p4 interpreter "
            '(README.md, "Setup").',
            flush=True,
        )
        return False
    try:
        import numpy  # noqa: F401
    except ImportError:
        print(
            f"warehouse_mission: NumPy is not installed for {sys.executable}.\n"
            "Webots is not using the course Python. Set Webots > Preferences > General > "
            "Python command "
            'to the cs3630_p4 interpreter and restart Webots (README.md, "Setup").',
            flush=True,
        )
        return False
    return True


if __name__ == "__main__" and _environment_ok():
    from wm.viz.app import main

    main(HERE)
