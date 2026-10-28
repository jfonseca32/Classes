"""The robot's brain: engine glue around the student's Chapter 4 functions.

What the brain does every control step (DESIGN 5.2):

* **MCL** (book 4.4.4): ``particles = mcl_predict(particles, weights, u_eff,
  sigma_hat, rng, n)``, ``weights = mcl_update(particles, z_on, z_rfid, z_zone)``,
  ``mean, cov = estimate_position(particles, weights)``. ``sigma_hat`` comes
  from the student's own system identification (book 4.6).
* **Confidence** = weight fraction within 1 m of the mean. LOCKED after two
  confident steps (then 600 particles); LOST when confidence collapses or the
  weights degenerate (kidnapped) -> 2000 uniform particles again.
* **Markov grid filter** (book 4.4.3) in parallel, for visualization only.
* **Planning** (book 4.5): value iteration per goal, ``vote_action`` on the
  whole belief, and a greedy path preview from the estimate.

Every student call goes through :class:`StudentBridge` with fresh copies of
the engine's arrays, and every returned value is validated; anything unusable
raises :class:`StudentFault`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from . import config as C
from .records import BeliefSnapshot, Event, Sensors
from .student_bridge import StudentBridge, StudentFault
from .world import MAP_POINTS, MAP_SHAPE, ZONE_MAP, World, in_bounds

Cell = tuple[int, int]


def _as_array(value, function: str, shape: tuple[int, ...], what: str = "returned") -> np.ndarray:
    """Copy a student return value into a float array of ``shape`` or fault.

    A ``-1`` in ``shape`` accepts any positive length on that axis. A 1-D
    target also accepts a column, e.g. weights of shape (N, 1).
    """
    try:
        array = np.array(value, dtype=float)
    except Exception as exc:  # malformed or nonnumeric student output
        raise StudentFault(
            function, f"{what} {type(value).__name__}, expected an array of shape {shape}"
        ) from exc
    if len(shape) == 1 and array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    fits = array.ndim == len(shape) and all(
        want in (-1, got) for want, got in zip(shape, array.shape, strict=False)
    )
    if not fits or array.size == 0:
        expected = str(tuple(shape)).replace("-1", "N")
        raise StudentFault(function, f"{what} shape {array.shape}, expected {expected}")
    return array


def _as_action(value, function: str) -> int:
    try:
        action = int(value)
        valid = float(value) == action and 0 <= action <= 3
    except Exception:
        valid = False
    if not valid:
        raise StudentFault(
            function, f"returned {value!r}; expected an action 0 (WEST) .. 3 (NORTH)"
        )
    return action


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _cells(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows, columns (clipped to the map), and the on-map mask of points (N, 2)."""
    cols = np.round(points[:, 0]).astype(int)
    rows = np.round(points[:, 1]).astype(int)
    inside = (rows >= 0) & (rows < MAP_SHAPE[0]) & (cols >= 0) & (cols < MAP_SHAPE[1])
    return np.clip(rows, 0, MAP_SHAPE[0] - 1), np.clip(cols, 0, MAP_SHAPE[1] - 1), inside


def _marker_likelihood(points: np.ndarray, z_fiducial) -> np.ndarray:
    """Gaussian likelihood of the dock-marker fix z = 100 x + w, w ~ N(0, FIDUCIAL_R), at each
    point.
    """
    e = np.asarray(z_fiducial, dtype=float)[None, :] - C.FIDUCIAL_C[0, 0] * np.asarray(
        points, dtype=float
    )
    return np.exp(
        -0.5 * np.sum(e**2, axis=1) / C.FIDUCIAL_SIGMA_CM**2
    )  # (C and R are multiples of I)


def _spread(cov: np.ndarray) -> float:
    """Largest standard deviation (m) of a 2x2 covariance."""
    try:
        return float(np.sqrt(max(float(np.max(np.linalg.eigvalsh(cov))), 0.0)))
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")


###############################################################################
#                                  Planning                                  #
###############################################################################


@dataclass
class Plan:
    """A value function for one goal on one version of the map (all arrays read-only)."""

    goal_name: str
    goal_cell: Cell
    goal_mask: np.ndarray  # (50, 100) bool
    V: np.ndarray  # (50, 100): the student's value function, as returned
    V_act: np.ndarray  # V with obstacle cells masked (used to pick actions)
    obstacle_map: np.ndarray  # the maps value iteration used
    prox_on: np.ndarray
    map_version: int
    seconds: float  # how long value_iteration took


class Planner:
    """Runs the student's ``value_iteration`` once per (goal, map version)."""

    def __init__(self, bridge: StudentBridge, world: World):
        self.bridge = bridge
        self.world = world
        self.penalty = (
            True  # False (PENALTY TEST): plan with an empty proximity map, i.e. no penalty
        )
        self._cache: dict[tuple, Plan] = {}

    def plan(
        self, goal_name: str, goal_cell: Cell, more_cells: tuple[Cell, ...] = ()
    ) -> tuple[Plan, bool]:
        """The plan for a goal (``more_cells``: more goal cells, any of them will do); ``computed``
        is
        True if value iteration just ran.
        """
        key = (tuple(goal_cell), tuple(more_cells), self.world.map_version, self.penalty)
        if key in self._cache:
            return self._cache[key], False
        mask = np.zeros(MAP_SHAPE, dtype=bool)
        for cell in (goal_cell,) + tuple(more_cells):
            mask[cell] = True
        prox = self.world.prox_on if self.penalty else np.zeros_like(self.world.prox_on)
        start = time.perf_counter()
        raw = self.bridge.call(
            "value_iteration", mask.copy(), self.world.obstacle_map.copy(), prox.copy(), C.GAMMA
        )
        seconds = time.perf_counter() - start
        V = _as_array(raw, "value_iteration", MAP_SHAPE)
        if not np.all(np.isfinite(V)):
            raise StudentFault("value_iteration", "returned a value function with NaN or inf")
        V_act = V.copy()
        V_act[self.world.obstacle_map != 0] = C.OBSTACLE_ACTION_VALUE
        plan = Plan(
            goal_name,
            (int(goal_cell[0]), int(goal_cell[1])),
            _read_only(mask),
            _read_only(V),
            _read_only(V_act),
            self.world.obstacle_map,
            prox,
            self.world.map_version,
            seconds,
        )
        self._cache[key] = plan
        return plan, True

    def invalidate(self) -> None:
        self._cache.clear()


###############################################################################
#                                 The brain                                  #
###############################################################################


class Brain:
    """Belief (particles + grid) and decisions, driven by the student's code."""

    def __init__(
        self,
        bridge: StudentBridge,
        sigma: float,
        rng: np.random.Generator,
        grid_filter: bool = True,
        likelihood_layer: bool = True,
    ):
        self.bridge = bridge
        self.sigma = float(sigma)  # the student's sigma_hat (system ID), used by MCL
        self.rng = rng
        self.particles = np.zeros((0, 2))
        self.weights = np.zeros(0)
        self.mean = np.zeros(2)
        self.cov = np.eye(2)
        self.confidence = 0.0
        self.locked = False
        self.relocalizing = False  # the LOST flag: re-localizing after a loss
        self.dead_reckoning = False  # the NO SENSORS test: motion alone (shown on the HUD)
        self.tracking_only = (
            False  # sensor tests: never re-seed (LOST); the test drives by the truth
        )
        self.n_fixed: int | None = None  # START TEST: always this many particles
        # PENALTY TEST (planning only): the belief is the true position, no student filter
        self.exact = False
        self.initialized = False
        self._lock_streak = 0
        self._degenerate_streak = 0
        self._zone_miss_streak = (
            0  # steps in a row with (almost) no particle in the zone the reader names
        )
        self._invalid_streak = 0
        self._updates = 0  # measurement updates since the particles were (re)seeded
        self.heard_beacon = False  # an RFID beacon was in range since the particles were (re)seeded
        self.grid_enabled = bool(grid_filter)
        self.grid: np.ndarray | None = None
        self.likelihood_enabled = bool(likelihood_layer)
        self.likelihood: np.ndarray | None = None
        self.last_grid_seconds = 0.0
        self._held_action: int | None = None
        self._hold_left = 0
        self._path_key = None
        self._path: list[Cell] = []
        self._path_index: dict[Cell, int] = {}
        self._events: list[Event] = []

    # --- events ----------------------------------------------------------------
    def drain_events(self) -> list[Event]:
        events, self._events = self._events, []
        return events

    def _emit(self, name: str, text: str, **data) -> None:
        self._events.append(Event(name, text, data))

    # --- snapshots -------------------------------------------------------------
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            particles=self.particles.copy(),
            weights=self.weights.copy(),
            mean=self.mean.copy(),
            cov=self.cov.copy(),
            confidence=float(self.confidence),
            locked=self.locked,
            relocalizing=self.relocalizing,
            grid=None if self.grid is None or not self.grid_enabled else self.grid.copy(),
            likelihood=None if self.likelihood is None else self.likelihood.copy(),
            dead_reckoning=self.dead_reckoning,
        )

    @property
    def n_next(self) -> int:
        """Particle count for the next prediction (fewer once LOCKED; a START TEST: fixed)."""
        if self.n_fixed:
            return self.n_fixed
        return (
            C.N_PARTICLES_TRACKING if self.locked or self.dead_reckoning else C.N_PARTICLES_GLOBAL
        )

    @property
    def n_global(self) -> int:
        """Particle count of a uniform (re)start."""
        return self.n_fixed or C.N_PARTICLES_GLOBAL

    @property
    def spread(self) -> float:
        """Largest standard deviation of the belief (m)."""
        return _spread(self.cov)

    @property
    def updates(self) -> int:
        """Measurement updates since the particles were last (re)seeded."""
        return self._updates

    @property
    def healthy(self) -> bool:
        """False right after a measurement that no particle explained (kidnapped?)."""
        return self._degenerate_streak == 0

    # --- particle filter (book 4.4.4) ------------------------------------------------
    def _uniform_particles(self, n: int) -> np.ndarray:
        return self.rng.uniform(
            [C.FLOOR_X[0], C.FLOOR_Y[0]], [C.FLOOR_X[1], C.FLOOR_Y[1]], size=(n, 2)
        )

    def _raw_weights(self, particles: np.ndarray, sensors: Sensors) -> np.ndarray:
        zone = None if sensors.z_zone is None else int(sensors.z_zone)
        if sensors.z_on is not None and sensors.z_rfid is not None:
            raw = self.bridge.call(
                "mcl_update", particles.copy(), bool(sensors.z_on), tuple(sensors.z_rfid), zone
            )
            return _as_array(raw, "mcl_update", (len(particles),))
        # An experiment switched proximity or RFID off: multiply the student's likelihoods of the
        # sensors that are on (exactly the product mcl_update computes); no sensor -> all ones.
        n = len(particles)
        raw = np.ones(n)
        if sensors.z_on is not None:
            raw = raw * _as_array(
                self.bridge.call("proximity_likelihood", particles.copy(), bool(sensors.z_on)),
                "proximity_likelihood",
                (n,),
            )
        if sensors.z_rfid is not None:
            raw = raw * _as_array(
                self.bridge.call("rfid_likelihood", particles.copy(), tuple(sensors.z_rfid)),
                "rfid_likelihood",
                (n,),
            )
        if zone is not None:
            raw = raw * _as_array(
                self.bridge.call("zone_likelihood", particles.copy(), zone), "zone_likelihood", (n,)
            )
        return raw

    def _estimate(self) -> None:
        result = self.bridge.call("estimate_position", self.particles.copy(), self.weights.copy())
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise StudentFault("estimate_position", "must return a (mean, cov) pair")
        mean = _as_array(result[0], "estimate_position", (2,), "returned a mean of")
        cov = _as_array(result[1], "estimate_position", (2, 2), "returned a covariance of")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
            raise StudentFault("estimate_position", "returned NaN or inf for valid weights")
        self.mean, self.cov = mean, cov
        close = np.linalg.norm(self.particles - self.mean[None, :], axis=1) <= C.CONFIDENCE_RADIUS
        self.confidence = float(np.sum(self.weights[close]))

    def set_exact(self, position: np.ndarray) -> None:
        """Planning only: a tight belief at the TRUE ``position`` (EXACT_PARTICLES particles within
        a few
        cm), LOCKED, computed without any student code. While ``exact``, every belief update below
        (predict, measure, marker fix, re-localization, the likelihood layer) does nothing; the
        mission calls this again after every step.
        """
        position = np.asarray(position, dtype=float)
        n = C.EXACT_PARTICLES
        self.initialized = self.exact = self.tracking_only = True
        self.dead_reckoning = self.relocalizing = False
        self.particles = position[None, :] + self.rng.normal(0.0, C.EXACT_SIGMA, size=(n, 2))
        self.weights = np.full(n, 1.0 / n)
        self.mean, self.cov = position.copy(), C.EXACT_SIGMA**2 * np.eye(2)
        self.confidence, self.locked = 1.0, True
        self._lock_streak, self._degenerate_streak = C.LOCK_STREAK, 0
        self._updates = max(self._updates, C.LOCK_MIN_UPDATES)
        self._held_action, self._hold_left = None, 0

    def reset_to(
        self, position: np.ndarray, sigma: float, n: int | None = None, tracking_only: bool = True
    ) -> None:
        """A known start: a good belief around ``position`` (``n`` or 600 particles, N(position,
        sigma^2 I)), LOCKED.

        Every sensor test starts from this belief, so the tests differ only in their sensors.
        With ``tracking_only`` the brain then only tracks: it never re-seeds (LOST) during the test.
        """
        self.initialized = True
        self.tracking_only = tracking_only
        self.dead_reckoning = False
        n = int(n or C.N_PARTICLES_TRACKING)
        self.particles = np.asarray(position, dtype=float)[None, :] + self.rng.normal(
            0.0, sigma, size=(n, 2)
        )
        self.weights = np.full(n, 1.0 / n)
        self.locked, self.relocalizing = True, False
        self._lock_streak, self._degenerate_streak = C.LOCK_STREAK, 0
        self._updates = max(self._updates, C.LOCK_MIN_UPDATES)
        self._held_action, self._hold_left = None, 0
        self._estimate()
        if self.grid_enabled:
            self.grid = np.zeros(MAP_SHAPE)
            i, j = (
                int(np.clip(round(position[1]), 0, MAP_SHAPE[0] - 1)),
                int(np.clip(round(position[0]), 0, MAP_SHAPE[1] - 1)),
            )
            self.grid[i, j] = 1.0

    def global_init(self, sensors: Sensors) -> None:
        """Wake up lost: uniform particles weighted by the first measurement (eq. 4.46)."""
        if self.exact:  # (planning only: the mission keeps the belief at the truth)
            return
        self.initialized = True
        self.dead_reckoning = False
        self.tracking_only = False
        self._reset_particles(sensors)
        self.relocalizing = False
        if self.grid_enabled:
            self.grid = np.full(MAP_SHAPE, 1.0 / (MAP_SHAPE[0] * MAP_SHAPE[1]))
        self.grid_step(None, sensors)

    def _reset_particles(self, sensors: Sensors) -> None:
        self.particles = self._uniform_particles(self.n_global)
        raw = self._raw_weights(self.particles, sensors)
        total = float(np.sum(raw)) if np.all(np.isfinite(raw)) and np.all(raw >= 0) else 0.0
        if total > 0 and math.isfinite(total):
            self.weights = raw / total
        else:  # the sensors contradict the whole map (or bad weights): stay uniform
            self.weights = np.full(len(self.particles), 1.0 / len(self.particles))
        self.locked = False
        self._lock_streak = 0
        self._degenerate_streak = 0
        self._zone_miss_streak = 0
        self._updates = 1
        self.heard_beacon = self._beacon_in(sensors)
        self._held_action, self._hold_left = None, 0
        self._estimate()

    def relocalize(self, sensors: Sensors, reason: str, lost_event: bool = True) -> None:
        """Start over with 2000 uniform particles (RE-LOCALIZING; a START TEST: its count).

        Losing a LOCKED belief is a LOST event (it costs points); re-seeding a
        belief that never locked is a RESEED (free: the robot was still searching).
        """
        if self.exact:  # (planning only: the mission keeps the belief at the truth)
            return
        if lost_event and self.locked:
            self._emit("LOST", "LOST! RE-LOCALIZING", reason=reason)
        elif lost_event:
            self._emit("RESEED", "RE-SEEDING PARTICLES", reason=reason)
        self._reset_particles(sensors)
        self.relocalizing = True
        if self.grid_enabled:
            self.grid = np.full(MAP_SHAPE, 1.0 / (MAP_SHAPE[0] * MAP_SHAPE[1]))
            self.grid_step(None, sensors)

    def _predict(self, u: np.ndarray, sigma: float, n: int) -> np.ndarray:
        weights = self.weights / np.sum(self.weights)
        new = self.bridge.call(
            "mcl_predict",
            self.particles.copy(),
            weights.copy(),
            np.array(u, dtype=float),
            float(sigma),
            self.rng,
            int(n),
        )
        # Any (M, 2) is usable: a student who ignores num_samples just keeps more particles.
        new = _as_array(new, "mcl_predict", (-1, 2))
        if not np.all(np.isfinite(new)):
            raise StudentFault("mcl_predict", "returned particles with NaN or inf")
        return new

    def predict_and_update(self, u_eff: np.ndarray, sensors: Sensors) -> None:
        """One MCL cycle with the executed control ``u_eff`` and new readings."""
        if self.exact:  # (planning only: the mission keeps the belief at the truth)
            return
        new = self._predict(u_eff, self.sigma, self.n_next)
        raw = self._raw_weights(new, sensors)
        if self.locked and np.all(np.isfinite(raw)) and not np.sum(raw) > 0:
            # Every particle died at a sensor boundary: retry once, wider.
            wide = self._predict(u_eff, C.RECOVERY_SIGMA, self.n_global)
            wide_raw = self._raw_weights(wide, sensors)
            if (
                np.all(np.isfinite(wide_raw))
                and np.all(wide_raw >= 0)
                and np.mean(wide_raw) >= C.DEGENERATE_MEAN_WEIGHT
            ):
                new, raw = wide, wide_raw
                self._emit("RECOVERED", "BELIEF WIDENED", sigma=C.RECOVERY_SIGMA)
        self.particles = new
        if not self._absorb(raw, sensors):
            self.grid_step(u_eff, sensors)

    def measure(self, sensors: Sensors) -> None:
        """Measurement update only (the robot did not move, or moved precisely)."""
        if self.exact:  # (planning only: the mission keeps the belief at the truth)
            return
        if not self._absorb(self._raw_weights(self.particles, sensors), sensors):
            self.grid_step(None, sensors)

    @staticmethod
    def _beacon_in(sensors: Sensors | None) -> bool:
        return sensors is not None and sensors.z_rfid is not None and sensors.z_rfid[0] is not None

    def _absorb(self, raw: np.ndarray, sensors: Sensors) -> bool:
        """Validate new raw weights, estimate, and run the LOCKED / LOST logic.

        Returns True if the belief was re-initialized.
        """
        self.heard_beacon = self.heard_beacon or self._beacon_in(sensors)
        if not (np.all(np.isfinite(raw)) and np.all(raw >= 0)):
            self._invalid_streak += 1
            if self._invalid_streak >= C.INVALID_WEIGHTS_FAULT_STREAK:
                raise StudentFault(
                    "mcl_update",
                    f"returned NaN, inf, or negative weights {self._invalid_streak} steps in a row",
                )
            if self.tracking_only or self.dead_reckoning:
                return False  # an experiment tour never re-seeds: keep the belief
            self.relocalize(sensors, "invalid weights (NaN or negative)")
            return True
        self._invalid_streak = 0
        total = float(np.sum(raw))
        if (
            self.dead_reckoning or self.tracking_only
        ):  # an experiment tour: just track (never re-seed)
            if total > 0 and math.isfinite(total):
                self.weights = raw / total
            self._updates += 1
            self._estimate()
            return False
        if not (total > 0 and math.isfinite(total)):
            self.relocalize(sensors, "no particle agrees with the sensors")
            return True
        if total / len(raw) < C.DEGENERATE_MEAN_WEIGHT:
            self._degenerate_streak += 1
        else:
            self._degenerate_streak = 0
        if self._degenerate_streak >= C.DEGENERATE_STREAK:
            self.relocalize(sensors, "the sensors disagree with the belief (kidnapped?)")
            return True
        if sensors.z_zone is not None:
            rows, cols, inside = _cells(self.particles)
            support = float(np.mean(inside & (ZONE_MAP[rows, cols] == int(sensors.z_zone))))
            self._zone_miss_streak = (
                self._zone_miss_streak + 1 if support < C.ZONE_MISS_SUPPORT else 0
            )
            if self._zone_miss_streak >= C.ZONE_MISS_STEPS:
                self.relocalize(sensors, "the zone reader disagrees with the belief (kidnapped?)")
                return True
        self.weights = raw / total
        self._updates += 1
        self._estimate()
        if self.locked:
            if self.confidence < C.LOST_CONFIDENCE:
                self.relocalize(sensors, f"confidence dropped to {100 * self.confidence:.0f}%")
                return True
            return False
        self._lock_streak = self._lock_streak + 1 if self.confidence >= C.LOCK_CONFIDENCE else 0
        if self._lock_streak >= C.LOCK_STREAK and self._updates >= C.LOCK_MIN_UPDATES:
            self.locked = True
            self.relocalizing = False
            self._held_action, self._hold_left = None, 0
            self._emit(
                "LOCKED",
                f"LOCKED ON  ({self.mean[0]:.1f}, {self.mean[1]:.1f})",
                mean=self.mean.tolist(),
                confidence=self.confidence,
            )
        return False

    def marker_fix(self, z_fiducial: np.ndarray, station: np.ndarray) -> None:
        """MARKER ACQUIRED: a precise position fix z = 100 x + w (R = 10^2 I cm^2) at the station.

        The marker is only visible inside the capture disk, so the robot scatters
        fresh particles over that disk and weights them with the marker's
        Gaussian likelihood (engine code: the camera is not part of the assignment).
        """
        if self.exact:  # (planning only: the mission keeps the belief at the truth)
            return
        n = C.MARKER_PARTICLES
        radius = C.CAPTURE_RADIUS * np.sqrt(self.rng.uniform(0.0, 1.0, n))
        angle = self.rng.uniform(0.0, 2.0 * np.pi, n)
        particles = np.asarray(station, dtype=float)[None, :] + radius[:, None] * np.column_stack(
            (np.cos(angle), np.sin(angle))
        )
        raw = _marker_likelihood(particles, z_fiducial)
        total = float(np.sum(raw))
        if not total > 0:  # (cannot happen: the true position is inside the capture disk)
            raw, total = np.ones(n), float(n)
        self.particles = particles
        self.weights = raw / total
        self.dead_reckoning = False
        self._estimate()
        if not self.locked:
            self._emit(
                "LOCKED",
                f"LOCKED ON  ({self.mean[0]:.1f}, {self.mean[1]:.1f})",
                mean=self.mean.tolist(),
                confidence=self.confidence,
                marker=True,
            )
        self.locked, self.relocalizing = True, False
        self._lock_streak = C.LOCK_STREAK
        self._degenerate_streak = 0
        self._updates = max(self._updates, C.LOCK_MIN_UPDATES)
        if self.grid_enabled and self.grid is not None:
            self.grid_step(
                None, None, likelihood=_marker_likelihood(MAP_POINTS, z_fiducial).reshape(MAP_SHAPE)
            )

    def known_motion(self, displacement: np.ndarray) -> None:
        """Precise, marker-guided motion (docking): predict with a tiny sigma."""
        if self.exact:  # (planning only: the mission keeps the belief at the truth)
            return
        self.particles = self._predict(displacement, C.DOCK_MOTION_SIGMA, C.N_PARTICLES_TRACKING)
        n = len(self.particles)
        self.weights = np.full(n, 1.0 / n)
        self._estimate()
        if self.grid_enabled and self.grid is not None:
            self.grid_step(np.array(displacement, dtype=float), None)

    def dock_ready(self, station) -> bool:
        """Confident enough, although not LOCKED, that the dock marker is in view.

        True when at least ``DOCK_READY_CONFIDENCE`` of the belief weight lies
        within ``DOCK_READY_RADIUS`` of the station (and the belief has had as
        many updates as a lock would need).
        """
        if C.DOCK_READY_RADIUS <= 0 or self._updates < C.LOCK_MIN_UPDATES or not self.healthy:
            return False
        offsets = self.particles - np.asarray(station, dtype=float)[None, :]
        near = np.linalg.norm(offsets, axis=1) <= C.DOCK_READY_RADIUS
        return float(np.sum(self.weights[near])) >= C.DOCK_READY_CONFIDENCE

    # --- Markov grid filter (book 4.4.3) ----------------------------------------------
    def _call_grid(self, function: str, *args) -> np.ndarray | None:
        try:
            values = _as_array(self.bridge.call(function, *args), function, (MAP_POINTS.shape[0],))
        except StudentFault as fault:
            self.disable_grid(f"{fault.function}: {fault.message}")
            return None
        return values

    def _likelihood_grid(self, sensors: Sensors) -> np.ndarray | None:
        """Product of the student's likelihoods at every cell center (of the sensors that are
        on).
        """
        likelihood = np.ones(MAP_POINTS.shape[0])
        if sensors.z_on is not None:
            prox = self._call_grid("proximity_likelihood", MAP_POINTS.copy(), bool(sensors.z_on))
            if prox is None:
                return None
            likelihood = likelihood * prox
        if sensors.z_rfid is not None:
            rfid = self._call_grid("rfid_likelihood", MAP_POINTS.copy(), tuple(sensors.z_rfid))
            if rfid is None:
                return None
            likelihood = likelihood * rfid
        if sensors.z_zone is not None:
            zone = self._call_grid("zone_likelihood", MAP_POINTS.copy(), int(sensors.z_zone))
            if zone is None:
                return None
            likelihood = likelihood * zone
        if not np.all(np.isfinite(likelihood)):
            self.disable_grid("the likelihood grid contains NaN or inf")
            return None
        return likelihood.reshape(MAP_SHAPE)

    def disable_grid(self, reason: str) -> None:
        """Turn the grid filter and the likelihood layer off (GRID_FILTER_OFF)."""
        if self.grid_enabled or self.likelihood_enabled:
            self._emit("GRID_FILTER_OFF", "GRID FILTER OFF", reason=reason)
        self.grid_enabled = False
        self.likelihood_enabled = False
        self.grid = None
        self.likelihood = None

    def grid_step(
        self, u: np.ndarray | None, sensors: Sensors | None, likelihood: np.ndarray | None = None
    ) -> None:
        """Markov localization: predict with ``u`` (if any), update with the sensors."""
        if self.exact:  # (planning only: the mission keeps the belief at the truth)
            return
        if not (self.grid_enabled or self.likelihood_enabled):
            return
        start = time.perf_counter()
        if likelihood is None and sensors is not None:
            likelihood = self._likelihood_grid(sensors)
            if likelihood is None:
                return
        if likelihood is not None and self.likelihood_enabled:
            self.likelihood = likelihood
        if self.grid_enabled and self.grid is not None:
            try:
                grid = self.grid
                if u is not None:
                    grid = _as_array(
                        self.bridge.call(
                            "markov_predict", grid.copy(), np.array(u, dtype=float), C.GRID_SIGMA
                        ),
                        "markov_predict",
                        MAP_SHAPE,
                    )
                if likelihood is not None:
                    grid = _as_array(
                        self.bridge.call("markov_update", grid.copy(), likelihood.copy()),
                        "markov_update",
                        MAP_SHAPE,
                    )
                total = float(np.sum(grid))
                if not (np.all(np.isfinite(grid)) and np.all(grid >= 0) and total > 0):
                    raise StudentFault(
                        "markov_update" if likelihood is not None else "markov_predict",
                        "returned a grid with NaN, negative values, or zero mass",
                    )
                self.grid = grid / total
            except StudentFault as fault:
                self.disable_grid(f"{fault.function}: {fault.message}")
                return
        self.last_grid_seconds = time.perf_counter() - start
        if self.grid_enabled and self.last_grid_seconds > C.GRID_MAX_SECONDS:
            self.disable_grid(f"too slow ({self.last_grid_seconds:.2f} s per update)")

    # --- acting (book 4.5.3) ------------------------------------------------------------
    def choose_action(self, plan: Plan, last_blocked: bool) -> tuple[int, bool]:
        """Vote on the whole belief; returns ``(action, voted_now)``.

        Anti-dither rule (``config.ANTI_DITHER_HOLD``): while not LOCKED and
        the belief is still spread out, a voted action is held for a few steps
        unless the safety layer stopped the robot.
        """
        if (
            not self.locked
            and self._held_action is not None
            and self._hold_left > 0
            and not last_blocked
        ):
            self._hold_left -= 1
            return self._held_action, False
        action = _as_action(
            self.bridge.call(
                "vote_action",
                plan.V_act.copy(),
                self.particles.copy(),
                self.weights.copy(),
                plan.goal_mask.copy(),
                plan.obstacle_map.copy(),
                plan.prox_on.copy(),
                C.GAMMA,
            ),
            "vote_action",
        )
        if self.locked or self.spread < C.HOLD_MIN_SPREAD:
            self._held_action, self._hold_left = None, 0
        else:
            self._held_action, self._hold_left = action, C.ANTI_DITHER_HOLD - 1
        return action, True

    def seek_action(self, plan: Plan) -> int:
        """The beacon search's action (engine code): the largest belief-weighted value one step
        ahead,
        sum_i w_i V(next cell of particle i) (QMDP). Unlike a vote, an action that is good for every
        mode of the belief (e.g. NORTH for two look-alike places) wins over each mode's favorite.
        """
        rows, cols, inside = _cells(self.particles)
        valid = inside & ~plan.goal_mask[rows, cols] & (plan.obstacle_map[rows, cols] == 0)
        if not np.any(valid):
            return self._held_action if self._held_action is not None else 0
        rows, cols, weights = rows[valid], cols[valid], self.weights[valid]
        scores = []
        for di, dj in C.ACTION_DELTAS:
            r = np.clip(rows + di, 0, MAP_SHAPE[0] - 1)
            c = np.clip(cols + dj, 0, MAP_SHAPE[1] - 1)
            scores.append(float(np.sum(weights * plan.V_act[r, c])))
        action = int(np.argmax(scores))
        self._held_action, self._hold_left = action, 0
        return action

    def greedy_path(self, plan: Plan, start: Cell) -> list[Cell]:
        """Roll out the student's ``greedy_action`` from ``start`` (the path preview)."""
        start = (int(start[0]), int(start[1]))
        key = (id(plan), plan.map_version, plan.goal_cell)
        if key == self._path_key and start in self._path_index:
            return self._path[self._path_index[start] :]
        if not in_bounds(start) or plan.obstacle_map[start]:
            return []
        V, goal, obstacles, proximity = (
            plan.V_act.copy(),
            plan.goal_mask.copy(),
            plan.obstacle_map.copy(),
            plan.prox_on.copy(),
        )
        path, seen, cell = [start], {start}, start
        with self.bridge.quiet():  # one NumPy error-state switch for the whole rollout
            for _ in range(C.PATH_PREVIEW_MAX):
                if plan.goal_mask[cell]:
                    break
                action = _as_action(
                    self.bridge.call("greedy_action", V, cell, goal, obstacles, proximity, C.GAMMA),
                    "greedy_action",
                )
                di, dj = C.ACTION_DELTAS[action]
                nxt = (cell[0] + di, cell[1] + dj)
                if not in_bounds(nxt) or nxt in seen:
                    break
                path.append(nxt)
                seen.add(nxt)
                cell = nxt
        self._path_key, self._path = key, path
        self._path_index = {c: k for k, c in enumerate(path)}
        return list(path)
