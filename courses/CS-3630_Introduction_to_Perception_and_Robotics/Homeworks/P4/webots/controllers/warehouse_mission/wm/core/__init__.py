"""Warehouse mission engine core: pure Python + NumPy, no Webots.

The Webots controller (``wm.viz``) and the offline runner
(``student/tests/run_mission.py``) both drive the same :class:`Mission`.
The API is documented in ``ENGINE_API.md`` (staff docs, not shipped to students).
"""

from __future__ import annotations

from . import config
from .mission import (
    BOOT,
    BRIEFING,
    COMPLETE,
    COUNTDOWN,
    DOCK_DROPOFF,
    DOCK_PICKUP,
    DOCKING_PHASES,
    FAILED,
    FAULT,
    LOCALIZING,
    PHASES,
    TERMINAL_PHASES,
    TO_DOCK,
    TO_PICKUP,
    TRAVEL_PHASES,
    ChaosScript,
    Mission,
    MissionOptions,
    Order,
    random_code,
    run_to_completion,
)
from .records import BeliefSnapshot, Event, Sensors, StepResult
from .scenarios import BY_KEY as SCENARIOS_BY_KEY
from .scenarios import PENALTY_TESTS, SCENARIOS, SENSOR_TESTS, START_TESTS, Scenario, format_metric
from .scoring import MissionStats, score_breakdown
from .student_bridge import (
    CORE_TODOS,
    ERROR,
    LOCALIZATION_TODOS,
    MISSING,
    OK,
    PLANNING_TODOS,
    TODOS,
    CheckResult,
    StudentBridge,
    StudentFault,
    StudentImportError,
    import_student_code,
)
from .world import BEACONS, DOCKS, SHELF_RECTS, SLOTS, SLOTS_BY_NAME, Dock, Slot, World

__all__ = [
    "config",
    "Mission",
    "MissionOptions",
    "Order",
    "ChaosScript",
    "random_code",
    "run_to_completion",
    "StepResult",
    "Event",
    "Sensors",
    "BeliefSnapshot",
    "MissionStats",
    "score_breakdown",
    "PENALTY_TESTS",
    "START_TESTS",
    "SCENARIOS",
    "SCENARIOS_BY_KEY",
    "SENSOR_TESTS",
    "Scenario",
    "format_metric",
    "StudentBridge",
    "StudentFault",
    "StudentImportError",
    "import_student_code",
    "CheckResult",
    "TODOS",
    "CORE_TODOS",
    "LOCALIZATION_TODOS",
    "PLANNING_TODOS",
    "OK",
    "MISSING",
    "ERROR",
    "World",
    "Slot",
    "Dock",
    "SLOTS",
    "SLOTS_BY_NAME",
    "DOCKS",
    "BEACONS",
    "SHELF_RECTS",
    "PHASES",
    "TRAVEL_PHASES",
    "DOCKING_PHASES",
    "TERMINAL_PHASES",
    "BOOT",
    "BRIEFING",
    "COUNTDOWN",
    "LOCALIZING",
    "TO_PICKUP",
    "DOCK_PICKUP",
    "TO_DOCK",
    "DOCK_DROPOFF",
    "COMPLETE",
    "FAULT",
    "FAILED",
]
