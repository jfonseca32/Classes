"""Plain record types shared by the mission engine and its renderers.

Everything here is data that the Webots layer and the offline runner read.
Positions are world-frame meters ``(x, y)`` (x east, y north, origin at the
center of cell (0, 0)); cells are ``(i, j) = (row, col) = (y, x)``;
the dock-marker fix is in centimeters; times are simulated seconds.
See ``README.md`` in this folder for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Event:
    """Something worth announcing.

    ``name`` is an UPPER_SNAKE identifier (the README lists all of them),
    ``text`` a short ASCII line for the HUD, and ``data`` plain values
    (numbers, strings, lists, tuples, dicts) for renderers and logs.
    """

    name: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sensors:
    """One round of sensor readings taken at the robot's true position.

    * ``z_on``: proximity sensor (book 4.3.1), True near shelves, walls, obstacles.
    * ``z_rfid``: ``(beacon_id, range_m)`` of the nearest beacon within 8 m, or
      ``(None, inf)``.
    * ``z_zone``: the zone reader's zone id (an index into ``config.ZONE_NAMES``),
      or None when switched off.
    * ``z_fiducial``: the precise dock-marker fix in cm (only at MARKER_ACQUIRED).

    A sensor that an experiment scenario switched off reads None (``z_on``,
    ``z_rfid``, or ``z_zone``): no reading at all, which is not the same as the
    RFID reading ``(None, inf)`` ("no beacon in range").
    """

    z_on: bool | None
    z_rfid: tuple[int | None, float] | None
    z_zone: int | None = None
    z_fiducial: np.ndarray | None = None


@dataclass
class BeliefSnapshot:
    """The robot's belief after a result (copies: safe to keep and to modify).

    ``particles`` (N, 2) m and ``weights`` (N,) normalized to sum to 1 (N is
    2000 while localizing, 600 once LOCKED). ``mean`` (2,) m and ``cov``
    (2, 2) m^2 come from the student's ``estimate_position``; ``confidence``
    is the weight fraction within 1 m of the mean. ``grid`` is the Markov
    (grid) filter belief and ``likelihood`` the last measurement's likelihood
    grid, both (50, 100) and summing to 1 / unnormalized, or None when
    disabled. ``relocalizing`` is the LOST flag (re-localizing after a loss).
    """

    particles: np.ndarray
    weights: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    confidence: float
    locked: bool
    relocalizing: bool
    grid: np.ndarray | None = None
    likelihood: np.ndarray | None = None
    dead_reckoning: bool = False  # the NO SENSORS test: motion alone

    @property
    def n_particles(self) -> int:
        return int(len(self.particles))

    @property
    def sigma(self) -> float:
        """Largest standard deviation of the belief (m), for "estimate +/- sigma"."""
        try:
            return float(np.sqrt(max(float(np.max(np.linalg.eigvalsh(self.cov))), 0.0)))
        except (np.linalg.LinAlgError, ValueError):
            return float("nan")


@dataclass
class StepResult:
    """One call of :meth:`Mission.step`: a control step or a scripted animation.

    The Webots layer plays a result over ``duration`` simulated seconds: it
    moves the robot along the truth segment ``start -> end`` (a straight line;
    ``kind == "teleport"`` jumps instead), spins the wheels at
    ``wheel_omegas``, and renders the belief, plan, actors, and events. The
    mission clock runs from ``clock_start`` to ``clock`` during the result
    (both stay 0 before deployment, e.g. during the countdown).

    ``kind`` is one of ``"move" | "dock" | "undock" | "pickup" | "dropoff" |
    "wait" | "teleport" | "event_only"`` (``event_only`` results have
    duration 0). See ``README.md`` for every field.
    """

    index: int  # 0, 1, 2, ... per mission
    kind: str
    phase: str  # the phase at the END of the result (see mission.PHASES)
    banner: str  # HUD phase banner, e.g. "TO PICKUP B07-E"
    duration: float  # s
    clock_start: float  # mission time (s) when the result starts
    clock: float  # mission time (s) when the result ends
    start: np.ndarray  # (2,) m, true position at the start
    end: np.ndarray  # (2,) m, true position at the end
    wheel_omegas: np.ndarray  # (3,) rad/s that reproduce the true motion (real wheels)
    wheel_omegas_cmd: np.ndarray  # (3,) rad/s commanded with the student's Jacobian
    action: int | None = None  # 0 WEST, 1 EAST, 2 SOUTH, 3 NORTH (move/wait)
    u: np.ndarray | None = None  # (2,) m, commanded displacement
    u_eff: np.ndarray | None = None  # (2,) m, what the brain was told it executed
    sensors: Sensors | None = None
    belief: BeliefSnapshot | None = None
    value: np.ndarray | None = None  # (50, 100) read-only: the student's V for the current goal
    value_change: np.ndarray | None = (
        None  # (50, 100): V after minus V before the last obstacle drop
    )
    goal_mask: np.ndarray | None = None  # (50, 100) bool, read-only
    goal_cell: tuple[int, int] | None = None
    goal_name: str | None = None  # e.g. "B07-E" or "DOCK 2"
    path: list[tuple[int, int]] = field(
        default_factory=list
    )  # greedy preview cells from the estimate
    events: list[Event] = field(default_factory=list)
    stats: dict[str, Any] = field(
        default_factory=dict
    )  # MissionStats.snapshot(): counters + live score
    actors: list[dict[str, Any]] = field(default_factory=list)  # forklift poses
    obstacles: list[dict[str, Any]] = field(default_factory=list)  # dropped obstacles
    parcel: dict[str, Any] | None = None  # the current order's parcel and where it is
    dock: dict[str, Any] | None = None  # docking animation targets (dock/pickup/dropoff/undock)
    order: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # persistent HUD warnings
    done: bool = False
    success: bool = False

    @property
    def truth(self) -> np.ndarray:
        """True robot position at the end of the result."""
        return self.end

    def event_names(self) -> list[str]:
        return [event.name for event in self.events]
