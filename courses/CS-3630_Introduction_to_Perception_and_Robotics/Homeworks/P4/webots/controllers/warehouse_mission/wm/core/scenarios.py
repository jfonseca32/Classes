"""Experiment scenarios: press E at the briefing (Webots) or pass ``--scenario`` (offline runner).

Each scenario is preconfigured, so an experiment is one choice and then runs on its own:

* ``mission``: the normal delivery mission.
* two START TESTS (E picks START TEST, keys 1-2): the sensor tests' loop with all sensors and 2000
  particles, from a known start (a good belief around the truth, ``start_known``) or an unknown
  one (particles spread over the whole floor: global localization, ``start_unknown``).
* five SENSOR TESTS, one per sensor setting (in Webots: E picks SENSOR TEST, keys 1-5 the setting):
  all sensors, each sensor alone, and none: ``sensors_all``, ``sensors_zone_only``,
  ``sensors_rfid_only``, ``sensors_proximity_only``, and ``sensors_none``. The robot drives the
  same loop (24 m x 9 m, around the south-west corner of shelf block A) once. It knows its start (a
  good belief around the truth), follows the loop by
  its TRUE position (a fixed path, like a robot on rails) and only its belief depends on the
  sensors. The noise is fixed too (``noise_seed``): every sensor test has exactly the same true
  path, wheel slips, and sensor noise, in every run, so the tests differ only in which sensors
  are on, and rerunning a test gives the same result. The result is the mean belief spread
  (the size of the particle cloud) and the mean estimate error.
* ``obstacle_test``: the normal mission, plus an obstacle that drops on the robot's route
  ``obstacle_after`` control steps after the first LOCKED; ``StepResult.value_change`` holds
  the change of the value function (new minus old).
* two PENALTY TESTS (E picks PENALTY TEST, keys 1-2): one fixed delivery (``mission_code``'s
  start and order), with fixed noise, planned with the proximity penalty (``penalty_50``) or
  without it (``penalty_0``: the planner gets an empty proximity map). They test PLANNING only:
  the robot is told its exact position (``perfect_localization``: no particle filter, no student
  localization code), and each run is capped at ``max_steps`` control steps (about 2 minutes).

The sensor and particle tests need only the localization TODOs (1-6), the penalty tests only the
planning TODOs (7-8), and the mission and the obstacle test all of them.
Scenarios with the same ``group`` share one entry of the briefing's E menu; keys 1-N pick the
``option``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SENSOR_NAMES = ("proximity", "rfid", "zone")
ALL_ON = (True, True, True)

Point = tuple[float, float]


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    needs_planning: bool
    description: tuple[str, ...]  # briefing lines
    route: tuple[
        Point, ...
    ] = ()  # a tour (sensor and particle tests): the fixed route (first point = start)
    group: str = ""  # tests of one experiment ("start", "sensors", "penalty") share one E entry
    option: str = (
        ""  # its setting within the group, e.g. "RFID ONLY" (the menu entry and table row)
    )
    sensors: tuple[bool, bool, bool] = ALL_ON  # proximity, RFID, zone
    noise_seed: int | None = None  # the noise after deployment (a test: the same every run)
    known_start: bool = False  # deploy with a good belief around the truth (no global localization)
    particles: int | None = None  # a fixed particle count (else 2000 while lost, 600 once LOCKED)
    mission_code: int | None = None  # use this code's start and orders (a fixed delivery)
    penalty: bool = True  # False: plan without the proximity penalty
    perfect_localization: bool = (
        False  # planning only: the belief IS the true position (no student filter)
    )
    max_steps: int | None = None  # a cap on the control steps of the whole run (else per order)
    obstacle_after: int | None = None  # control steps after the first LOCKED

    @property
    def needs_localization(self) -> bool:
        """Whether the scenario runs the student's particle filter (TODOs 1-6)."""
        return not self.perfect_localization

    @property
    def is_tour(self) -> bool:
        """A tour (sensor and particle tests): a fixed route, driven by the true position."""
        return bool(self.route)


MISSION = Scenario(
    "mission", "MISSION", True, ("Pick up the parcel and deliver it to the dock.",)
)  # (briefing lines: <= 48 characters)

# The sensor tests' loop (64 m, counterclockwise), from the middle of the RECEIVING 1W bay: east
# along
# the open floor (y = 3) into the south cross aisle, up and west along the middle of shelf A's south
# end (y = 7.6: proximity ON, beacon 0 right overhead), keeping clear of the shelf's corner, north
# into RECEIVING 2E, west into 2W, and back south. It crosses six zone boundaries.
_LOOP = (
    (7.0, 3.0),
    (31.0, 3.0),
    (31.0, 5.5),
    (27.0, 7.6),
    (17.0, 7.6),
    (17.0, 12.0),
    (7.0, 12.0),
    (7.0, 3.0),
)
SENSOR_TEST_LAPS = 1  # one lap: a second one would draw over the first
SENSOR_TEST_SEED = (
    3634  # the fixed noise of every sensor test (a typical one: see INSTRUCTOR_NOTES)
)
# All sensors, each sensor alone ("keep one"), and none: (menu entry, (proximity, RFID, zone)).
SENSOR_SETTINGS = (
    ("ALL SENSORS", (True, True, True)),
    ("ZONE ONLY", (False, False, True)),
    ("RFID ONLY", (False, True, False)),
    ("PROXIMITY ONLY", (True, False, False)),
    ("NO SENSORS", (False, False, False)),
)
_SENSOR_KEYS = (
    "sensors_all",
    "sensors_zone_only",
    "sensors_rfid_only",
    "sensors_proximity_only",
    "sensors_none",
)


def _sensor_test(key: str, option: str, sensors: tuple[bool, bool, bool]) -> Scenario:
    return Scenario(
        key,
        "SENSOR TEST",
        False,
        (
            "The robot drives a fixed loop once. Every run",
            "has the same path and noise; only the sensors",
            "change. Pick them with keys 1-5.",
        ),
        route=_LOOP + _LOOP[1:] * (SENSOR_TEST_LAPS - 1),
        group="sensors",
        option=option,
        sensors=sensors,
        noise_seed=SENSOR_TEST_SEED,
        known_start=True,
    )


SENSOR_TESTS: tuple[Scenario, ...] = tuple(
    _sensor_test(key, option, sensors)
    for key, (option, sensors) in zip(_SENSOR_KEYS, SENSOR_SETTINGS, strict=False)
)

# The start tests: the sensor tests' loop, all sensors, 2000 particles (a fixed count, no switch to
# 600), from a known or an unknown start.
START_TEST_SEED = (
    3610  # an unknown start costs accuracy in 20/20 filter random streams (see INSTRUCTOR_NOTES)
)
START_TEST_PARTICLES = 2000
START_SETTINGS = (("KNOWN START", True), ("UNKNOWN START", False))


def _start_test(option: str, known: bool) -> Scenario:
    return Scenario(
        f"start_{'known' if known else 'unknown'}",
        "START TEST",
        False,
        (
            "The robot drives a fixed loop with all sensors",
            "and 2000 particles, from a known start (key 1)",
            "or an unknown one (key 2).",
        ),
        route=_LOOP,
        group="start",
        option=option,
        noise_seed=START_TEST_SEED,
        known_start=known,
        particles=START_TEST_PARTICLES,
    )


START_TESTS: tuple[Scenario, ...] = tuple(_start_test(*setting) for setting in START_SETTINGS)

OBSTACLE_TEST = Scenario(
    "obstacle_test",
    "OBSTACLE TEST",
    True,
    (
        "The normal mission, but an obstacle drops on",
        "the route. The floor then shows how the value",
        "function changed (red: lower value).",
    ),
    obstacle_after=8,
)

# The penalty tests: one fixed delivery (mission PENALTY_TEST_CODE: its start and order), with
# fixed noise, planned with or without the proximity penalty, driven with an exact position
# (planning
# only) and capped at PENALTY_TEST_STEPS control steps (a robot stuck at a shelf corner then FAILS).
PENALTY_TEST_CODE = 3606  # its route passes the shelves (see INSTRUCTOR_NOTES)
PENALTY_TEST_SEED = 3642  # penalty 0 bumps but finishes (other seeds can get stuck at a corner)
PENALTY_TEST_STEPS = 240  # control steps (0.5 s each): about 2 minutes of driving


def _penalty_test(option: str, penalty: bool) -> Scenario:
    return Scenario(
        f"penalty_{50 if penalty else 0}",
        "PENALTY TEST",
        True,
        (
            "Planning only: the robot knows its exact",
            "position. One delivery, planned with the",
            "proximity penalty (key 1) or without (key 2).",
        ),
        group="penalty",
        option=option,
        noise_seed=PENALTY_TEST_SEED,
        known_start=True,
        mission_code=PENALTY_TEST_CODE,
        penalty=penalty,
        perfect_localization=True,
        max_steps=PENALTY_TEST_STEPS,
    )


PENALTY_TESTS: tuple[Scenario, ...] = (
    _penalty_test("PENALTY 50", True),
    _penalty_test("PENALTY 0", False),
)

SCENARIOS: tuple[Scenario, ...] = (
    (MISSION,) + START_TESTS + SENSOR_TESTS + PENALTY_TESTS + (OBSTACLE_TEST,)
)
BY_KEY: dict[str, Scenario] = {scenario.key: scenario for scenario in SCENARIOS}


def format_metric(value, unit: str) -> str:
    """A result table's cell: ``0.45 m``, ``76.0 s``, ``41 %``, ``27`` (a count), a text such as
    ``COMPLETE``, or ``-`` (none)."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    v = float(value)
    if unit == "m":
        return f"{v:.0f} m" if v >= 10 and v.is_integer() else f"{v:.2f} m"
    if unit == "s":
        return f"{v:.1f} s"
    if unit == "%":
        return f"{v:.0f} %"
    return f"{v:.0f}"


class Route:
    """A polyline route that the robot follows by chasing a "carrot" one step ahead.

    ``s`` is the progress along the route (meters). The robot steers toward the carrot
    ``point_at(s + STEP)``; once it is within ``REACH`` of it, the carrot moves one step on.
    So the robot makes one step per meter, also where a lap turns back on itself.
    """

    STEP = 1.0  # m: the carrot is one control step ahead
    REACH = 0.75  # m: close enough to the carrot to move it on

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        seg = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.lengths)])
        self.length = float(self.cum[-1])

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        k = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.lengths) - 1))
        t = (s - self.cum[k]) / max(self.lengths[k], 1e-9)
        return self.points[k] + t * (self.points[k + 1] - self.points[k])

    def carrot(self, s: float) -> np.ndarray:
        return self.point_at(s + self.STEP)

    def control(self, position, s: float) -> np.ndarray:
        """The displacement (at most one step) toward the carrot."""
        u = self.carrot(s) - np.asarray(position, dtype=float)
        n = float(np.linalg.norm(u))
        return u * (self.STEP / n) if n > self.STEP else u

    def advance(self, position, s: float) -> float:
        """The new progress after a move: one step on if the robot reached the carrot."""
        if float(np.linalg.norm(self.carrot(s) - np.asarray(position, dtype=float))) < self.REACH:
            return min(s + self.STEP, self.length)
        return s

    def cells_ahead(self, s: float, count: int) -> list:
        """Grid cells ``(i, j)`` along the route ahead of ``s`` (the path preview)."""
        cells = []
        for k in range(count):
            x, y = self.point_at(s + k)
            cell = (int(round(y)), int(round(x)))
            if not cells or cells[-1] != cell:
                cells.append(cell)
            if s + k >= self.length:
                break
        return cells
