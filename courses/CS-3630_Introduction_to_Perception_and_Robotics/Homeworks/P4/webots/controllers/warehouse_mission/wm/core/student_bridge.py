"""The only door between the mission engine and the student's ``project4.py``.

* :func:`import_student_code` imports the student's file (course convention,
  as in Project 2).
* :meth:`StudentBridge.system_check` runs a quick sanity probe of every TODO
  and reports ``(todo_number, name, status, message)`` rows with status
  ``OK``, ``MISSING`` (``NotImplementedError`` or not defined), or ``ERROR``.
* :meth:`StudentBridge.call` calls a student function, times it (calls
  slower than ``config.SLOW_CALL_SECONDS`` are reported once as SLOW CODE),
  silences NumPy warnings, and turns any exception into a
  :class:`StudentFault` that keeps the innermost traceback lines.
* Fallbacks (DESIGN 5.1/5.2): a missing or invalid Jacobian -> the true one,
  a missing or invalid system ID -> sigma 0.2, each with a warning.

The probes compare against hard-coded numbers, never against a reference
implementation, so no solution code ships with the engine.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from . import config as C

# (TODO number, function name), in the order of project4.py: the eight functions students write.
TODOS: tuple[tuple[int, str], ...] = (
    (1, "sample_motion_model"),
    (2, "rfid_likelihood"),
    (3, "zone_likelihood"),
    (4, "mcl_predict"),
    (5, "mcl_update"),
    (6, "estimate_position"),
    (7, "q_value"),
    (8, "value_iteration"),
)
TODO_NUMBER: dict[str, int] = {name: number for number, name in TODOS}
TODO_NAME: dict[int, str] = {number: name for number, name in TODOS}

# The robot cannot deploy without these (particle filter + planning). The experiment
# tours (scenarios.py) drive a fixed route, so they need only the localization ones.
LOCALIZATION_TODOS = frozenset({1, 2, 3, 4, 5, 6})
PLANNING_TODOS = frozenset({7, 8})
CORE_TODOS = LOCALIZATION_TODOS | PLANNING_TODOS
# project4.py also PROVIDES functions the engine calls (proximity_likelihood, greedy_action,
# vote_action, omni_jacobian, estimate_motion_model). They are not probed at boot; omni_jacobian
# and estimate_motion_model keep their fallbacks (the true Jacobian, sigma 0.2).

OK, MISSING, ERROR = "OK", "MISSING", "ERROR"
_PRISTINE_ATTR = "_wm_pristine_maps"


class StudentImportError(RuntimeError):
    """The student's file could not be found or imported."""


class StudentFault(RuntimeError):
    """A student function raised, or returned something unusable."""

    def __init__(self, function: str, message: str, missing: bool = False, details: str = ""):
        super().__init__(f"{function}: {message}")
        self.function = function
        self.todo = TODO_NUMBER.get(function)
        self.message = message
        self.missing = missing
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "todo": self.todo,
            "message": self.message,
            "missing": self.missing,
            "details": self.details,
        }


class CheckResult(NamedTuple):
    """One SYSTEM CHECK row."""

    todo: int
    name: str
    status: str
    message: str


@dataclass
class SysIdResult:
    """Result of the boot-time system identification (book Section 4.6)."""

    A: np.ndarray | None
    B: np.ndarray | None
    sigma_raw: float | None
    sigma: float
    fallback: bool
    message: str


def import_student_code(path) -> Any:
    """Import the student's ``project4.py`` (or any file) as a fresh module."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise StudentImportError(f"Student module does not exist: {path}")
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)  # keep sibling imports working
    spec = importlib.util.spec_from_file_location("cs3630_project4_student", path)
    if spec is None or spec.loader is None:
        raise StudentImportError(f"Could not create an import specification for {path}.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit) as exc:  # syntax errors, top-level crashes, exit() ...
        raise StudentImportError(f"Importing {path.name} failed: {_describe(exc)}") from exc
    module.__student_path__ = str(path)
    return module


def _describe(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _short(value, digits: int = 3) -> str:
    return np.array2string(
        np.asarray(value, dtype=float),
        precision=digits,
        suppress_small=True,
        separator=", ",
        max_line_width=200,
    )


class _ProbeFailure(Exception):
    """A probe found a wrong value (status ERROR)."""


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise _ProbeFailure(message)


def _array(value, shape: tuple[int, ...] | None, what: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise _ProbeFailure(f"{what} is not numeric ({type(value).__name__})") from exc
    if shape is not None:
        _expect(array.shape == shape, f"{what} has shape {array.shape}, expected {shape}")
    return array


def _close(value, expected, what: str, atol: float = 1e-6, rtol: float = 1e-4) -> None:
    expected = np.asarray(expected, dtype=float)
    value = _array(value, expected.shape, what)
    _expect(bool(np.all(np.isfinite(value))), f"{what} contains NaN or inf")
    _expect(
        np.allclose(value, expected, atol=atol, rtol=rtol),
        f"{what} = {_short(value)}, expected {_short(expected)}",
    )


###############################################################################
#                                  Probes                                    #
###############################################################################
# Each probe calls the student function through ``call`` and raises
# _ProbeFailure (-> ERROR) with a short message when a value is wrong. The
# expected numbers come from the book's formulas.


def _book_goal() -> np.ndarray:
    goal = np.zeros((C.MAP_HEIGHT, C.MAP_WIDTH), dtype=bool)
    goal[35, 50] = True
    return goal


def _probe_omni_jacobian(call) -> str:
    J = _array(call(), (3, 2), "omni_jacobian()")
    _close(J @ np.array([0.3, 0.1]), [0.5, -1.549, 1.049], "J @ (0.3, 0.1)", atol=2e-3, rtol=0)
    J4 = _array(call(4, 0.1), (4, 2), "omni_jacobian(4, 0.1)")
    _expect(bool(np.all(np.isfinite(J4))), "omni_jacobian(4, 0.1) contains NaN")
    return "J @ (0.3, 0.1) = [0.5, -1.549, 1.049]"


def _probe_sample_motion_model(call) -> str:
    states = np.tile([5.0, 7.0], (3000, 1))
    out = _array(
        call(states.copy(), np.array([1.0, -2.0]), 0.5, np.random.default_rng(7)),
        (3000, 2),
        "output",
    )
    _expect(bool(np.all(np.isfinite(out))), "output contains NaN or inf")
    _expect(
        np.allclose(out.mean(axis=0), [6.0, 5.0], atol=0.05),
        f"sample mean {_short(out.mean(axis=0))}, expected ~[6, 5]",
    )
    _expect(
        np.allclose(out.std(axis=0), [0.5, 0.5], atol=0.05),
        f"sample std {_short(out.std(axis=0))}, expected ~0.5",
    )
    again = _array(
        call(states.copy(), np.array([1.0, -2.0]), 0.5, np.random.default_rng(7)),
        (3000, 2),
        "output",
    )
    _expect(np.array_equal(out, again), "results differ for the same seed: use the rng argument")
    exact = _array(
        call(np.array([[1.0, 2.0]]), np.array([0.5, 0.5]), 0.0, np.random.default_rng(1)),
        (1, 2),
        "output",
    )
    _close(exact, [[1.5, 2.5]], "sigma = 0 sample")
    return "mean, spread and rng use look right"


_PROX_POINTS = np.array([[5.0, 25.0], [18.0, 20.0], [24.0, 20.0], [0.2, 0.2], [-3.0, 10.0]])


def _probe_proximity_likelihood(call) -> str:
    _close(call(_PROX_POINTS.copy(), True), [0, 1, 0, 1, 0], "z_on=True likelihood")
    _close(call(_PROX_POINTS.copy(), False), [1, 0, 0, 0, 0], "z_on=False likelihood")
    return "ON/OFF maps and off-map points"


def _probe_rfid_likelihood(call) -> str:
    points = np.array([[24.0, 5.0], [24.0, 0.0], [5.0, 25.0]])
    _close(call(points.copy(), (0, 4.2)), [0.9559974818, 0.0, 0.0], "beacon 0 at 4.2 m")
    _close(call(points.copy(), (None, float("inf"))), [0.0, 1.0, 1.0], "(None, inf) reading")
    return "range and out-of-range cases"


# AISLE 1 (zone 10), RECEIVING 3W (4), inside shelf A, off the map, CROSS AISLE N (14).
_ZONE_POINTS = np.array([[32.5, 25.0], [5.0, 25.0], [25.0, 25.0], [-3.0, 5.0], [50.0, 45.0]])


def _probe_zone_likelihood(call) -> str:
    _close(
        call(_ZONE_POINTS.copy(), 10),
        [0.9, 0.1 / 24, 0.0, 0.0, 0.1 / 24],
        "zone likelihood for AISLE 1",
    )
    _close(
        call(_ZONE_POINTS.copy(), 14, p_correct=0.7),
        [0.3 / 24, 0.3 / 24, 0.0, 0.0, 0.7],
        "zone likelihood for zone 14 with p_correct = 0.7",
    )
    return "zone lookup, misreads, shelves, off-map points"


def _probe_mcl_predict(call) -> str:
    particles = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    out = _array(
        call(
            particles.copy(),
            np.array([0.0, 5.0, 0.0]),
            np.array([1.0, 2.0]),
            0.1,
            np.random.default_rng(3),
        ),
        (3, 2),
        "output",
    )
    _expect(
        bool(np.all(np.linalg.norm(out - [11.0, 12.0], axis=1) < 1.0)),
        f"samples {_short(out, 2)} should all be near (11, 12)",
    )
    more = _array(
        call(
            particles.copy(),
            np.array([1.0, 1.0, 1.0]),
            np.zeros(2),
            0.1,
            np.random.default_rng(3),
            7,
        ),
        (7, 2),
        "num_samples=7 output",
    )
    _expect(bool(np.all(np.isfinite(more))), "output contains NaN")
    return "resampling + motion model"


def _probe_mcl_update(call) -> str:
    particles = np.array([[24.0, 5.0], [5.0, 25.0], [18.0, 20.0]])
    _close(
        call(particles.copy(), False, (0, 4.2), 13),
        [0.8603977336, 0.0, 0.0],
        "weights with a zone reading",
    )
    _close(
        call(particles.copy(), False, (0, 4.2)),
        [0.9559974818, 0.0, 0.0],
        "weights without a zone reading",
    )
    return "product of the three likelihoods"


def _probe_estimate_position(call) -> str:
    result = call(np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]]), np.array([1.0, 1.0, 2.0]))
    _expect(isinstance(result, (tuple, list)) and len(result) == 2, "expected a (mean, cov) pair")
    _close(result[0], [0.5, 2.0], "mean")
    _close(result[1], [[0.75, -1.0], [-1.0, 4.0]], "cov")
    return "weighted mean and covariance"


def _book_maps() -> tuple[np.ndarray, np.ndarray]:
    """Writable copies of the book's obstacle and proximity maps."""
    from .world import BASE_MAP, PROXIMITY_MAP_ON

    return BASE_MAP.copy(), PROXIMITY_MAP_ON.copy()


def _probe_q_value(call) -> str:
    goal = _book_goal()
    V = np.zeros((C.MAP_HEIGHT, C.MAP_WIDTH))
    obstacles, proximity = _book_maps()

    def q(values, cell, action):
        return float(
            call(
                values.copy(), cell, action, goal.copy(), obstacles.copy(), proximity.copy(), 0.975
            )
        )

    _close(q(V, (35, 49), 1), 100.0, "Q entering the goal")
    _close(q(V, (35, 50), 0), 0.0, "Q at the goal")
    _close(q(V, (20, 25), 0), 0.0, "Q inside a shelf")
    _close(q(V, (8, 25), 3), -100.0, "Q inside the proximity band")
    _close(q(V, (0, 5), 2), -100.0, "Q moving off the map")
    V2 = V.copy()
    V2[35, 48] = 10.0
    _close(q(V2, (35, 47), 1), -40.25, "Q with V = 10 next door")
    return "rewards, penalties, discount"


def _probe_value_iteration(call) -> str:
    obstacles, proximity = _book_maps()
    start = time.perf_counter()
    V = _array(call(_book_goal(), obstacles, proximity), (C.MAP_HEIGHT, C.MAP_WIDTH), "V")
    elapsed = time.perf_counter() - start
    cells = [(35, 49), (35, 48), (35, 47), (35, 50), (20, 25), (5, 5), (45, 90), (25, 10)]
    expected = [100.0, 97.5, 45.0625, 0.0, 0.0, 15.3582439521, 28.9218566504, 20.2904084244]
    _close([V[c] for c in cells], expected, "V at test cells", atol=1e-3, rtol=1e-5)
    return f"book map converges ({elapsed:.2f} s)"


def _probe_greedy_action(call) -> str:
    obstacles, proximity = _book_maps()
    goal = _book_goal()
    V = np.zeros((C.MAP_HEIGHT, C.MAP_WIDTH))
    V[25, 51] = 10.0
    action = call(V, (25, 50), goal, obstacles, proximity, 0.975)
    _expect(
        isinstance(action, (int, np.integer)), f"returned {type(action).__name__}, expected int"
    )
    _expect(int(action) == 1, f"returned {action} for a better cell to the EAST (1)")
    tie = call(np.zeros_like(V), (25, 50), goal, obstacles, proximity, 0.975)
    _expect(int(tie) == 0, f"ties returned {tie}, expected the lowest index 0 (WEST)")
    return "argmax with low-index ties"


def _probe_vote_action(call) -> str:
    obstacles, proximity = _book_maps()
    goal = _book_goal()
    V = np.zeros((C.MAP_HEIGHT, C.MAP_WIDTH))
    V[25, 51] = 10.0  # cell (25, 50) prefers EAST
    V[31, 50] = 10.0  # cell (30, 50) prefers NORTH
    particles = np.array(
        [
            [50.0, 25.0],
            [50.2, 24.9],
            [49.8, 25.1],  # EAST, weight 3
            [50.0, 30.0],  # NORTH, weight 5
            [25.0, 20.0],
            [-5.0, -5.0],
            [50.0, 35.0],
        ]
    )  # shelf, off map, goal
    weights = np.array([1.0, 1.0, 1.0, 5.0, 100.0, 100.0, 100.0])
    action = call(V, particles, weights, goal, obstacles, proximity, 0.975)
    cause = {
        0: " (do particles in a shelf, at the goal, or off the map vote?)",  # all three vote WEST
        1: " (votes count by weight, not by head count)",
    }.get(int(action), "")  # 3 EAST vs 1 NORTH
    _expect(
        int(action) == 3, f"returned {action}, expected NORTH (3) from the weighted vote{cause}"
    )
    none = call(V, particles[4:], weights[4:], goal, obstacles, proximity, 0.975)
    _expect(int(none) == 0, f"no valid particles returned {none}, expected 0")
    return "weighted vote of valid particles"


def _probe_estimate_motion_model(call) -> str:
    rng = np.random.default_rng(16)
    A = np.array([[1.0, 0.05], [0.0, 0.95]])
    B = np.array([[0.9, 0.0], [0.1, 1.1]])
    X = rng.uniform([0.0, 0.0], [99.0, 49.0], size=(400, 2))
    U = rng.normal(0.0, 1.5, size=(400, 2))
    Z = X @ A.T + U @ B.T + rng.normal(0.0, 0.2, size=(400, 2))
    result = call(X, U, Z)
    _expect(
        isinstance(result, (tuple, list)) and len(result) == 3, "expected (A_hat, B_hat, sigma_hat)"
    )
    _close(result[0], A, "A_hat", atol=0.05, rtol=0)
    _close(result[1], B, "B_hat", atol=0.05, rtol=0)
    sigma = float(result[2])
    _expect(abs(sigma - 0.2) < 0.03, f"sigma_hat = {sigma:.3f}, expected ~0.2")
    return f"sigma_hat = {sigma:.3f} on test data"


PROBES: dict[str, Callable] = {
    "omni_jacobian": _probe_omni_jacobian,
    "sample_motion_model": _probe_sample_motion_model,
    "proximity_likelihood": _probe_proximity_likelihood,
    "rfid_likelihood": _probe_rfid_likelihood,
    "zone_likelihood": _probe_zone_likelihood,
    "mcl_predict": _probe_mcl_predict,
    "mcl_update": _probe_mcl_update,
    "estimate_position": _probe_estimate_position,
    "q_value": _probe_q_value,
    "value_iteration": _probe_value_iteration,
    "greedy_action": _probe_greedy_action,
    "vote_action": _probe_vote_action,
    "estimate_motion_model": _probe_estimate_motion_model,
}


###############################################################################
#                                 The bridge                                 #
###############################################################################


class StudentBridge:
    """Safe, timed access to the student's module."""

    def __init__(self, module: Any):
        self.module = module
        self.path = getattr(module, "__student_path__", getattr(module, "__file__", None))
        self.timings: dict[str, list[float]] = {}  # name -> [calls, total_s, max_s]
        self.probe_seconds: dict[str, float] = {}  # name -> duration of its SYSTEM CHECK probe
        self._checks: list[CheckResult] | None = None
        self._saved_maps: dict[str, np.ndarray] = {}
        self._quiet_depth = 0
        self.slow_calls: dict[
            str, float
        ] = {}  # functions that took > SLOW_CALL_SECONDS: worst time (s)
        self._probing = False

    # --- calling ------------------------------------------------------------
    @contextmanager
    def quiet(self):
        """Silence NumPy floating point warnings for a batch of calls."""
        if self._quiet_depth:
            self._quiet_depth += 1
            try:
                yield
            finally:
                self._quiet_depth -= 1
            return
        self._quiet_depth = 1
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            self._quiet_depth = 0

    def has(self, name: str) -> bool:
        return callable(getattr(self.module, name, None))

    def call(self, name: str, *args, **kwargs):
        """Call ``module.name(*args, **kwargs)``; raise :class:`StudentFault` on any error."""
        function = getattr(self.module, name, None)
        if not callable(function):
            raise StudentFault(name, "is not defined in the student module", missing=True)
        start = time.perf_counter()
        try:
            with self.quiet():
                return function(*args, **kwargs)
        except NotImplementedError as exc:
            raise StudentFault(name, _describe(exc), missing=True) from exc
        except (Exception, SystemExit) as exc:  # never let student code stop the engine
            raise StudentFault(
                name, _describe(exc), details=traceback.format_exc(limit=-8)
            ) from exc
        finally:
            elapsed = time.perf_counter() - start
            record = self.timings.setdefault(name, [0, 0.0, 0.0])
            record[0] += 1
            record[1] += elapsed
            record[2] = max(record[2], elapsed)
            if elapsed > C.SLOW_CALL_SECONDS and not self._probing:
                self.slow_calls[name] = max(self.slow_calls.get(name, 0.0), elapsed)

    def timing_summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "calls": int(n),
                "total_s": total,
                "mean_ms": 1000.0 * total / max(n, 1),
                "max_ms": 1000.0 * worst,
            }
            for name, (n, total, worst) in sorted(self.timings.items())
        }

    # --- system check ---------------------------------------------------------
    def system_check(self, refresh: bool = False) -> list[CheckResult]:
        """Probe every TODO once (cached). Rows are ``(todo, name, status, message)``."""
        if self._checks is not None and not refresh:
            return list(self._checks)
        rows = []
        for number, name in TODOS:
            rows.append(self._check_one(number, name))
        self._checks = rows
        return list(rows)

    def _check_one(self, number: int, name: str) -> CheckResult:
        if not self.has(name):
            return CheckResult(number, name, MISSING, "function is not defined")
        probe = PROBES[name]

        def call(*args, **kwargs):
            # Fresh copies, so that a function that modifies its inputs cannot spoil later probes.
            args = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
            return self.call(name, *args, **kwargs)

        start = time.perf_counter()
        self._probing = True
        try:
            message = probe(call)
            return CheckResult(number, name, OK, message)
        except _ProbeFailure as failure:
            return CheckResult(number, name, ERROR, str(failure))
        except StudentFault as fault:
            if fault.missing:
                needed = [int(n) for n in re.findall(r"TODO\s*(\d+)", fault.message)]
                others = [n for n in needed if n != number]
                if others:
                    return CheckResult(
                        number,
                        name,
                        MISSING,
                        f"needs TODO {others[0]} ({TODO_NAME.get(others[0], '?')})",
                    )
                return CheckResult(number, name, MISSING, "raises NotImplementedError")
            if "No module named 'gtsam'" in fault.message:
                return CheckResult(number, name, ERROR, "GTSAM is not installed in this Python")
            return CheckResult(number, name, ERROR, fault.message)
        except Exception as exc:  # a probe choking on a very odd return value
            return CheckResult(number, name, ERROR, _describe(exc))
        finally:
            self._probing = False
            self.probe_seconds[name] = time.perf_counter() - start

    def missing_core(self, todos=CORE_TODOS) -> list[tuple[int, str, str]]:
        """The ``todos`` (default: every core TODO) that are MISSING: ``(todo, name, message)``.

        Empty -> can deploy.
        """
        return [
            (row.todo, row.name, row.message)
            for row in self.system_check()
            if row.todo in todos and row.status == MISSING
        ]

    @property
    def can_start(self) -> bool:
        return not self.missing_core()

    def status(self, name: str) -> str:
        for row in self.system_check():
            if row.name == name:
                return row.status
        return MISSING

    # --- fallbacks --------------------------------------------------------------
    def jacobian(self) -> tuple[np.ndarray, str, str]:
        """The wheel Jacobian the robot will actually use.

        Returns ``(J, status, message)`` with status ``"ok"``, ``"mismatch"``
        (valid shape but wrong numbers: the robot will really drive wrong), or
        ``"fallback"`` (missing/invalid: the true Jacobian is used).
        """
        try:
            J = np.asarray(self.call("omni_jacobian"), dtype=float)
        except (StudentFault, TypeError, ValueError) as exc:
            reason = exc.message if isinstance(exc, StudentFault) else _describe(exc)
            return (
                C.J_TRUE.copy(),
                "fallback",
                f"omni_jacobian unavailable ({reason}); using the true Jacobian",
            )
        if J.shape != (3, 2) or not np.all(np.isfinite(J)):
            return (
                C.J_TRUE.copy(),
                "fallback",
                (f"omni_jacobian returned shape {J.shape} or NaN; using the true Jacobian"),
            )
        if not np.allclose(J, C.J_TRUE, atol=C.JACOBIAN_MATCH_TOL):
            return J, "mismatch", "KINEMATICS MISMATCH: omni_jacobian differs from the real wheels"
        return J, "ok", "wheel Jacobian OK"

    def system_id(
        self, states: np.ndarray, controls: np.ndarray, next_states: np.ndarray
    ) -> SysIdResult:
        """Run the student's ``estimate_motion_model`` on a calibration log."""
        try:
            result = self.call(
                "estimate_motion_model", states.copy(), controls.copy(), next_states.copy()
            )
            A = np.asarray(result[0], dtype=float)
            B = np.asarray(result[1], dtype=float)
            sigma_raw = float(np.asarray(result[2], dtype=float).reshape(()))
        except StudentFault as fault:
            return SysIdResult(
                None,
                None,
                None,
                C.SIGMA_FALLBACK,
                True,
                f"estimate_motion_model unavailable ({fault.message}); sigma = 0.2",
            )
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            return SysIdResult(
                None,
                None,
                None,
                C.SIGMA_FALLBACK,
                True,
                f"estimate_motion_model returned {_describe(exc)}; sigma = 0.2",
            )
        if (
            A.shape != (2, 2)
            or B.shape != (2, 2)
            or not math.isfinite(sigma_raw)
            or sigma_raw <= 0
            or not np.all(np.isfinite(A))
            or not np.all(np.isfinite(B))
        ):
            return SysIdResult(
                A if A.shape == (2, 2) else None,
                B if B.shape == (2, 2) else None,
                sigma_raw if math.isfinite(sigma_raw) else None,
                C.SIGMA_FALLBACK,
                True,
                "estimate_motion_model returned invalid values; sigma = 0.2",
            )
        sigma = float(np.clip(sigma_raw, *C.SIGMA_CLAMP))
        note = "" if sigma == sigma_raw else f" (clamped from {sigma_raw:.3f})"
        return SysIdResult(A, B, sigma_raw, sigma, False, f"sigma_hat = {sigma:.3f} m{note}")

    # --- personalization ----------------------------------------------------------
    def robot_profile(self) -> dict[str, str]:
        """``robot_profile()`` sanitized: ASCII name (<= 16 chars) and a known color."""
        profile = dict(C.DEFAULT_PROFILE)
        try:
            raw = self.call("robot_profile")
        except StudentFault:
            return profile
        if isinstance(raw, dict):
            name = str(raw.get("name", "")).encode("ascii", "replace").decode("ascii").strip()
            name = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in name)[: C.PROFILE_NAME_MAX]
            if name:
                profile["name"] = name
            color = str(raw.get("color", "")).strip().lower()
            if color in C.ROBOT_COLORS:
                profile["color"] = color
        return profile

    def student_info(self) -> dict[str, str] | None:
        try:
            info = self.call("student_info")
        except StudentFault:
            return None
        return {str(k): str(v) for k, v in info.items()} if isinstance(info, dict) else None

    # --- the robot's map in the student module ----------------------------------------
    def patch_proximity_maps(self, delta_mask: np.ndarray, on: np.ndarray, off: np.ndarray) -> bool:
        """Tell the student module's proximity maps about dropped obstacles.

        ``proximity_likelihood`` reads the module's ``PROXIMITY_MAP_ON/OFF``;
        the engine writes the obstacle cells and their bands (``delta_mask``)
        into those arrays *in place* so that the robot's sensor model agrees
        with the real proximity sensor. :meth:`restore_maps` undoes it.
        Returns False if the module has no usable map arrays.
        """
        saved = self._pristine_maps()
        patched = False
        for attr, new in (("PROXIMITY_MAP_ON", on), ("PROXIMITY_MAP_OFF", off)):
            array = getattr(self.module, attr, None)
            if (
                not isinstance(array, np.ndarray)
                or array.shape != new.shape
                or not array.flags.writeable
            ):
                continue
            if attr not in saved:
                saved[attr] = array.copy()
            updated = saved[attr].copy()
            updated[delta_mask] = new[delta_mask]
            np.copyto(array, updated, casting="unsafe")
            patched = True
        return patched

    def _pristine_maps(self) -> dict[str, np.ndarray]:
        """The module's original map contents, kept on the module itself so that
        every bridge (and every new mission) can restore them."""
        saved = getattr(self.module, _PRISTINE_ATTR, None)
        if not isinstance(saved, dict):
            saved = {}
            try:
                setattr(self.module, _PRISTINE_ATTR, saved)
            except (AttributeError, TypeError):  # an exotic module object: keep it local
                saved = self._saved_maps
        return saved

    def restore_maps(self) -> None:
        """Undo :meth:`patch_proximity_maps` (called when a new mission starts)."""
        saved = self._pristine_maps()
        for attr, original in saved.items():
            array = getattr(self.module, attr, None)
            if (
                isinstance(array, np.ndarray)
                and array.shape == original.shape
                and array.flags.writeable
            ):
                np.copyto(array, original, casting="unsafe")
        saved.clear()
