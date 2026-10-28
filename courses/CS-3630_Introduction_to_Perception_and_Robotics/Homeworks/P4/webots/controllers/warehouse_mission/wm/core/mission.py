"""The mission state machine (DESIGN 5.4): one ``Mission`` per mission code.

::

    BOOT -> BRIEFING -> COUNTDOWN -> LOCALIZING -> TO_PICKUP -> DOCK_PICKUP
         -> TO_DOCK -> DOCK_DROPOFF -> (next order: TO_PICKUP | COMPLETE)

``FAULT`` (student code raised or returned garbage, or core TODOs are missing)
and ``FAILED`` (out of time) are terminal. LOST is a flag of the belief, not a
phase: the current leg continues after the robot locks on again.

Usage (the offline runner and the Webots controller do exactly this)::

    mission = Mission("project4.py", code=4821, orders=1)
    while not mission.done:
        result = mission.step()      # a StepResult; play it for result.duration s

Chaos hooks can be called between steps: :meth:`Mission.kidnap`,
:meth:`Mission.drop_obstacle`, :meth:`Mission.set_forklifts`. Experiment
scenarios (``scenarios.py``: the SENSOR TESTS, OBSTACLE TEST) are chosen
with the ``scenario`` argument or :meth:`Mission.set_scenario` before deployment.
See ``ENGINE_API.md`` for the complete API.
"""

from __future__ import annotations

import os
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import config as C
from . import scenarios as SC
from .brain import Brain, Plan, Planner
from .records import Event, Sensors, StepResult
from .scoring import MissionStats, score_breakdown
from .sim import TruthRobot
from .student_bridge import (
    LOCALIZATION_TODOS,
    MISSING,
    PLANNING_TODOS,
    StudentBridge,
    StudentFault,
    import_student_code,
)
from .world import (
    BEACON_SEEK_CELLS,
    BEACONS,
    DOCKS,
    PROXIMITY_MAP_OFF,
    PROXIMITY_MAP_ON,
    SLOTS,
    Dock,
    Slot,
    World,
    box_distance,
    cell_center,
    point_to_cell,
)

BOOT, BRIEFING, COUNTDOWN = "BOOT", "BRIEFING", "COUNTDOWN"
LOCALIZING, TO_PICKUP, DOCK_PICKUP = "LOCALIZING", "TO_PICKUP", "DOCK_PICKUP"
TO_DOCK, DOCK_DROPOFF = "TO_DOCK", "DOCK_DROPOFF"
COMPLETE, FAULT, FAILED = "COMPLETE", "FAULT", "FAILED"
PHASES = (
    BOOT,
    BRIEFING,
    COUNTDOWN,
    LOCALIZING,
    TO_PICKUP,
    DOCK_PICKUP,
    TO_DOCK,
    DOCK_DROPOFF,
    COMPLETE,
    FAULT,
    FAILED,
)
TRAVEL_PHASES = (LOCALIZING, TO_PICKUP, TO_DOCK)
DOCKING_PHASES = (DOCK_PICKUP, DOCK_DROPOFF)
TERMINAL_PHASES = (COMPLETE, FAULT, FAILED)

Place = Slot | Dock


@dataclass
class MissionOptions:
    """Switches for the engine (the defaults are what Webots uses)."""

    grid_filter: bool = False  # the Markov grid filter (removed from project4.py: always off)
    likelihood_layer: bool = True  # compute the likelihood grid every step (visualization only)
    forklifts: bool = False  # forklift traffic at the start
    zone: bool = True  # the zone reader; False = the book's two sensors only ("hard mode")
    max_steps_per_order: int = C.MAX_STEPS_PER_ORDER


@dataclass(frozen=True)
class Order:
    """One delivery: pick up the parcel at ``slot``, deliver it to ``dock``."""

    number: int
    slot: Slot
    dock: Dock

    def as_dict(self) -> dict[str, Any]:
        return {"number": self.number, "pickup": self.slot.name, "dock": self.dock.name}


def random_code() -> int:
    """A fresh 4-digit mission code."""
    return int(np.random.default_rng().integers(0, 10000))


def _route_distance(point: np.ndarray) -> float:
    """Distance from a point to the nearest forklift route segment."""
    best = np.inf
    for route in C.FORKLIFT_ROUTES:
        pts = np.asarray(route, dtype=float)
        for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=False):
            ab = b - a
            t = float(np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            best = min(best, float(np.linalg.norm(point - (a + t * ab))))
    return best


class Mission:
    """One mission: the world, the robot, its brain, the orders, and the score.

    Args:
        student: the student's module object, a path to ``project4.py``, or a
            :class:`StudentBridge`.
        code: mission code (seed) 0-9999; random if None. The same code and the
            same student module always give the identical run.
        orders: number of deliveries (1 by default, 3 for a "double shift");
            can be changed with :meth:`set_orders` until the robot deploys.
        options: :class:`MissionOptions`.
        scenario: ``"mission"`` (default), an experiment (a sensor test such as
            ``"sensors_rfid_only"``, or ``"obstacle_test"``; see ``scenarios.py``), or
            ``"auto"``: the mission if the planning TODOs work, else the first
            particle test (Webots). Can be changed with :meth:`set_scenario` until
            the robot deploys.

    Raises :class:`StudentImportError` if a path cannot be imported, and
    ValueError for an unknown scenario. Nothing else raises: student errors end
    the mission in the FAULT phase.
    """

    def __init__(
        self,
        student: Any,
        code: int | None = None,
        orders: int = 1,
        options: MissionOptions | None = None,
        scenario: str = "mission",
    ):
        self.options = options or MissionOptions()
        if isinstance(student, StudentBridge):
            self.bridge = student
        elif isinstance(student, (str, os.PathLike)):
            self.bridge = StudentBridge(import_student_code(student))
        else:
            self.bridge = StudentBridge(student)
        self.bridge.restore_maps()  # undo obstacles of a previous mission on this module
        self.phase = BOOT
        self._deployed = False
        self.code = int(code) % 10000 if code is not None else random_code()
        self.world = World()
        # Two RNG streams (DESIGN 5.4): the world (truth noise, start, orders)
        # is identical for everyone with the same code; the brain's stream is
        # what the student's functions receive.
        self.world_rng = np.random.default_rng(self.code)
        self.brain_rng = np.random.default_rng(self.code + 1)
        self.start = self.world.sample_free_position(self.world_rng)
        # Always draw MAX_ORDERS orders (distinct pickup slots), so that the
        # first orders of a code do not depend on how many are delivered.
        slots = self.world_rng.choice(len(SLOTS), size=C.MAX_ORDERS, replace=False)
        docks = self.world_rng.integers(len(DOCKS), size=C.MAX_ORDERS)
        self._all_orders = [
            Order(n + 1, SLOTS[int(s)], DOCKS[int(d)])
            for n, (s, d) in enumerate(zip(slots, docks, strict=False))
        ]
        self._code_orders = list(self._all_orders)  # (a PENALTY TEST uses its own fixed delivery)
        self.orders: list[Order] = []
        self.stats = MissionStats()
        self.set_orders(orders)
        self.world.set_traffic(self.options.forklifts)

        self.robot = TruthRobot(self.start, C.J_TRUE)
        self.brain: Brain | None = None
        self.planner = Planner(self.bridge, self.world)
        self.plan: Plan | None = None
        self.path: list[tuple] = []
        self.profile = self.bridge.robot_profile()
        self.student_info = self.bridge.student_info()
        self.checks = None  # SYSTEM CHECK rows, after BOOT
        self.sysid = None  # SysIdResult, after BOOT
        self.jacobian = C.J_TRUE.copy()
        self.jacobian_status = "unknown"  # "ok" | "mismatch" | "fallback" after BOOT
        self.notes: list[str] = []

        self.clock = 0.0
        self.index = 0
        self.order_index = 0
        self.leg = "pickup"  # or "dropoff"
        self.parcel_state = "rack"  # rack -> robot -> conveyor -> delivered
        self.done = False
        self.success = False
        self.fault: dict[str, Any] | None = None
        self._countdown_left = C.COUNTDOWN_SECONDS
        self._pending: list[Event] = []
        self._kidnap_pending = False
        self._deferred_fault: StudentFault | None = None
        self._script: deque[Callable[[], StepResult]] = deque()
        self._last_blocked = False
        self._amr_wait_streak = 0
        self._standoff: dict[int, int] = {}
        self._rfid_last = None
        self._last_sensors: Sensors | None = None
        self._slow_reported: set = set()

        # Experiment scenarios (scenarios.py).
        self._scenario_auto = scenario == "auto"
        key = SC.MISSION.key if self._scenario_auto else str(scenario)
        if key not in SC.BY_KEY:
            raise ValueError(
                f"unknown scenario {scenario!r}; choose one of {', '.join(SC.BY_KEY)} or 'auto'"
            )
        self.scenario: SC.Scenario = SC.BY_KEY[key]
        self.planning_ready = True  # False after BOOT if TODOs 7-8 are missing (the tours only)
        self.localization_ready = (
            True  # False after BOOT if TODOs 1-6 are missing (the penalty tests only)
        )
        self.route: SC.Route | None = None  # a sensor test's route
        self.experiment: dict[str, Any] | None = None  # a sensor test's result, at COMPLETE
        self.value_change: np.ndarray | None = None  # V after - V before the last obstacle drop
        self._mission_start = self.start.copy()  # tours start on their route instead
        self._route_s = 0.0
        self._tour_blocked = 0
        self._tour_stats: dict[str, Any] = {}
        self._deploy_clock = 0.0
        self._planned_routes: list[
            list[tuple]
        ] = []  # PENALTY TEST: each leg's planned route (cells)
        self._travel_moves = (
            0  # travel moves, and those that ended in the proximity band (PENALTY TEST)
        )
        self._band_moves = 0
        self._trace: list[tuple[int, float, float]] = []  # (step, belief spread, estimate error)
        self._lock_step: int | None = None
        self._seek_announced = False  # SEEK_BEACON was announced for the current search
        self._obstacle_done = False
        self._apply_scenario()

    # =====================================================================
    # Public API
    # =====================================================================
    @property
    def current_order(self) -> Order:
        return self.orders[self.order_index]

    @property
    def goal(self) -> Place:
        """The current leg's destination (a Slot or a Dock)."""
        order = self.current_order
        return order.slot if self.leg == "pickup" else order.dock

    @property
    def deployed(self) -> bool:
        return self._deployed

    @property
    def step_limit(self) -> int:
        """Control steps (moves + waits) allowed for the whole mission (a scenario may cap it)."""
        if self.scenario.max_steps:
            return int(self.scenario.max_steps)
        return self.options.max_steps_per_order * len(self.orders)

    def set_orders(self, orders: int) -> bool:
        """Change the number of deliveries (1..MAX_ORDERS) before deployment.

        Returns False (and changes nothing) once the robot has deployed.
        """
        if self._deployed_or_later():
            return False
        n = int(np.clip(int(orders), 1, C.MAX_ORDERS))
        self.orders = self._all_orders[:n]
        self.stats.orders_total = n
        return True

    def available_scenarios(self) -> list[str]:
        """Scenario keys that can run with the TODOs that work: all of them, only the tours (TODOs
        7-8
        missing), or only the penalty tests (TODOs 1-6 missing).
        """
        return [
            s.key
            for s in SC.SCENARIOS
            if (self.planning_ready or not s.needs_planning)
            and (self.localization_ready or not s.needs_localization)
        ]

    def set_scenario(self, key: str) -> bool:
        """Choose the scenario (``scenarios.py``) before deployment (E at the briefing).

        Returns False (and changes nothing) once the robot has deployed, for an
        unknown key, or for a scenario that needs the missing planning TODOs.
        """
        if (
            self._deployed_or_later()
            or key not in SC.BY_KEY
            or key not in self.available_scenarios()
        ):
            return False
        self.scenario = SC.BY_KEY[key]
        self._scenario_auto = False
        self._apply_scenario()
        return True

    def layout(self) -> dict[str, Any]:
        """Static layout (shelves, beacons, slots, docks, forklift routes, obstacles) + orders and
        start.
        """
        layout = self.world.layout()
        layout["orders"] = [order.as_dict() for order in self.orders]
        layout["start"] = tuple(float(v) for v in self.start)
        layout["route"] = [tuple(p) for p in self.scenario.route]
        return layout

    def step(self) -> StepResult:
        """Advance the mission by one control step or one scripted animation."""
        if self.done:
            return self._result("event_only", 0.0, [])
        try:
            if self._deferred_fault is not None:
                fault, self._deferred_fault = self._deferred_fault, None
                raise fault
            if self.index >= C.MAX_RESULTS_PER_ORDER * len(self.orders):
                return self._fail("result limit", "OUT OF TIME")
            return self._dispatch()
        except StudentFault as fault:
            return self._fault_result(fault)
        except Exception as exc:  # an engine bug must never take the Webots controller down
            fault = StudentFault(
                "engine",
                f"internal error {type(exc).__name__}: {exc}",
                details=traceback.format_exc(),
            )
            try:
                return self._fault_result(fault, reason="engine_error")
            except Exception:  # even the FAULT result could not be built: return a bare one
                return self._bare_fault_result(fault)

    # --- chaos hooks -------------------------------------------------------
    def kidnap(self) -> bool:
        """Teleport the robot to a random far-away free cell at the next step.

        Returns False (nothing happens) unless the robot is driving a leg.
        """
        if (
            self.done
            or self.phase not in TRAVEL_PHASES
            or not self._deployed
            or self._kidnap_pending
            or self.route is not None
        ):  # a tour's route is fixed: no kidnapping
            return False
        self._kidnap_pending = True
        return True

    def drop_obstacle(self) -> dict[str, Any] | None:
        """Drop an obstacle on the robot's path ahead and re-plan (value iteration).

        Returns the obstacle (``id, name, x, y, cell, half_size``) or None if
        there is no good spot right now (or the robot is not driving a leg).
        """
        if self.done or self.phase not in TRAVEL_PHASES or not self._deployed or self.plan is None:
            return None
        if len(self.world.obstacles) >= C.MAX_OBSTACLES:
            return None
        try:
            cell = self._choose_obstacle_cell()
        except StudentFault as fault:
            self._deferred_fault = fault
            return None
        if cell is None:
            return None
        old_plan = self.plan
        obstacle = self.world.add_obstacle(cell)
        delta = (self.world.prox_on != PROXIMITY_MAP_ON) | (
            self.world.prox_off != PROXIMITY_MAP_OFF
        )
        if not self.bridge.patch_proximity_maps(delta, self.world.prox_on, self.world.prox_off):
            self._note("ROBOT MAP NOT UPDATED (no PROXIMITY_MAP arrays)")
        self.planner.invalidate()
        self.stats.obstacles += 1
        self.stats.replans += 1
        info = obstacle.as_dict()
        self._pending.append(Event("OBSTACLE_DROPPED", "OBSTACLE AHEAD!", info))
        try:
            self.plan, _ = self.planner.plan(self.goal.name, self.goal.goal_cell)
            change = None
            if old_plan is not None and old_plan.goal_cell == self.plan.goal_cell:
                change = np.asarray(self.plan.V, dtype=float) - np.asarray(old_plan.V, dtype=float)
                self.value_change = change
            self._pending.append(
                Event(
                    "REPLANNED",
                    "RE-PLANNING ROUTE",
                    {
                        "reason": "obstacle",
                        "goal": self.goal.name,
                        "seconds": self.plan.seconds,
                        "value_drop": None
                        if change is None
                        else float(max(0.0, -float(np.min(change)))),
                        "changed_cells": None
                        if change is None
                        else int(np.count_nonzero(np.abs(change) > 1e-6)),
                    },
                )
            )
            self._update_path()
        except StudentFault as fault:
            self._deferred_fault = fault
        return info

    def set_forklifts(self, on: bool) -> None:
        """Forklift traffic on/off (off: forklifts finish their lap and park)."""
        self.world.set_traffic(on)
        self._pending.append(
            Event("FORKLIFTS", "FORKLIFT TRAFFIC " + ("ON" if on else "OFF"), {"on": bool(on)})
        )

    # --- summaries ---------------------------------------------------------
    def score(self) -> dict[str, Any]:
        """Score, stars, title, and every score term (see scoring.py)."""
        return score_breakdown(self.stats, success=self.success, final=self.done)

    def summary(self) -> dict[str, Any]:
        """One row of the offline runner's final table."""
        score = self.score()
        return {
            "code": self.code,
            "success": self.success,
            "phase": self.phase,
            "time": self.stats.mission_time,
            "score": score["score"],
            "stars": score["stars"],
            "title": score["title"],
            "safety_stops": self.stats.safety_stops,
            "failed_dockings": self.stats.failed_dockings,
            "relocalizations": self.stats.relocalizations,
            "mean_error": self.stats.mean_tracking_error,
            "steps": self.stats.control_steps,
            "time_to_lock": self.stats.time_to_lock,
            "orders": len(self.orders),
            "orders_completed": self.stats.orders_completed,
            "fault": self.fault,
            "scenario": self.scenario.key,
            "experiment": self.experiment,
        }

    # =====================================================================
    # Internals: dispatch and bookkeeping
    # =====================================================================
    def _deployed_or_later(self) -> bool:
        return self._deployed or self.phase in TERMINAL_PHASES

    def _dispatch(self) -> StepResult:
        if self.phase == BOOT:
            return self._boot()
        if self.phase == BRIEFING:
            return self._briefing()
        if self.phase == COUNTDOWN:
            return self._countdown()
        if self.phase in TRAVEL_PHASES:
            if not self._deployed:
                return self._deploy()
            if self._kidnap_pending:
                return self._teleport()
            return self._travel()
        if self.phase in DOCKING_PHASES and self._script:
            return self._script.popleft()()
        return self._result("event_only", 0.0, [])

    def _tick(self, duration: float) -> None:
        if self._deployed:
            self.clock += duration
            self.stats.mission_time = self.clock

    def _note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def _banner(self, phase: str) -> str:
        sc = self.scenario
        if self.route is not None and phase in TRAVEL_PHASES:
            return f"{sc.title}: {sc.option}"
        if self.route is not None and phase == COMPLETE:
            return f"{sc.title} COMPLETE"
        order = self.current_order
        return {
            BOOT: "SYSTEM BOOT",
            BRIEFING: "MISSION BRIEFING",
            COUNTDOWN: "DEPLOYING",
            LOCALIZING: "LOCALIZING",
            TO_PICKUP: f"TO PICKUP {order.slot.name}",
            DOCK_PICKUP: f"DOCKING AT {order.slot.name}",
            TO_DOCK: f"TO {order.dock.name}",
            DOCK_DROPOFF: f"DOCKING AT {order.dock.name}",
            COMPLETE: "MISSION COMPLETE",
            FAULT: "FAULT",
            FAILED: "MISSION FAILED",
        }[phase]

    def _set_phase(self, phase: str, events: list[Event]) -> None:
        if phase == self.phase:
            return
        events.append(Event("PHASE", self._banner(phase), {"from": self.phase, "to": phase}))
        self.phase = phase

    def _handle_brain_events(self, events: list[Event]) -> None:
        for event in self.brain.drain_events():
            if event.name == "LOST":
                self.stats.relocalizations += 1
                if self.route is not None:
                    self._tour_stats["lost"] += 1
            elif event.name == "LOCKED":
                if self.stats.time_to_lock is None:
                    self.stats.time_to_lock = self.clock
                    self._lock_step = self.stats.control_steps
                event.data["time"] = self.clock
            elif event.name == "GRID_FILTER_OFF":
                self._note("GRID FILTER OFF")
            events.append(event)
            if event.name == "LOCKED" and self.phase == LOCALIZING:
                self._set_phase(TO_PICKUP, events)

    def _slow_code_events(self, events: list[Event]) -> None:
        """Report each slow student function once per mission (also if it was slow in an earlier
        one).
        """
        for name, seconds in list(self.bridge.slow_calls.items()):
            if name not in self._slow_reported:
                self._slow_reported.add(name)
                self._note(f"SLOW CODE: {name}")
                events.append(
                    Event(
                        "SLOW_CODE",
                        f"SLOW CODE: {name} took {seconds:.1f} s",
                        {"function": name, "seconds": seconds},
                    )
                )

    def _update_path(self) -> None:
        if self.route is not None:  # a tour: the fixed route ahead
            self.path = (
                [] if self.done else self.route.cells_ahead(self._route_s, C.TOUR_PATH_PREVIEW)
            )
            return
        if self.brain is None or not self.brain.initialized or self.plan is None:
            self.path = []
            return
        self.path = self.brain.greedy_path(self._active_plan(), point_to_cell(self.brain.mean))

    def _seeking_beacon(self) -> bool:
        """Not LOCKED after BEACON_SEEK_AFTER updates and no beacon heard yet (RFID on): look for
        one first.
        """
        brain = self.brain
        return (
            brain is not None
            and brain.initialized
            and not brain.locked
            and not brain.exact
            and not brain.heard_beacon
            and brain.updates >= C.BEACON_SEEK_AFTER
            and self._sensor_mask()[1]
        )

    def _active_plan(self) -> Plan:
        """The plan the robot votes on: its goal's, or (searching) the nearest beacon's."""
        if not self._seeking_beacon():
            return self.plan
        plan, _ = self.planner.plan("RFID BEACON", BEACON_SEEK_CELLS[0], BEACON_SEEK_CELLS[1:])
        return plan

    def _check_timeout(self, events: list[Event]) -> None:
        if not self.done and self.stats.control_steps >= self.step_limit:
            self._finish_failed("timeout", "OUT OF TIME", events)

    def _finish_failed(self, reason: str, text: str, events: list[Event]) -> None:
        self.done, self.success = True, False
        self._script.clear()
        self._set_phase(FAILED, events)
        events.append(
            Event("MISSION_FAILED", text, {"reason": reason, "step_limit": self.step_limit})
        )
        if self.scenario.group == "penalty":  # a failed run is a result too
            self.experiment = self._mission_experiment()
            events.append(
                Event("EXPERIMENT_COMPLETE", f"{self.scenario.title} FAILED", dict(self.experiment))
            )

    def _fail(self, reason: str, text: str) -> StepResult:
        events: list[Event] = []
        self._finish_failed(reason, text, events)
        return self._result("event_only", 0.0, events)

    def _rfid_event(self, sensors: Sensors, events: list[Event]) -> None:
        if sensors.z_rfid is None:  # switched off (an experiment)
            self._rfid_last = None
            return
        beacon, distance = sensors.z_rfid
        if beacon is None:
            self._rfid_last = None
            return
        step = self.stats.control_steps
        if (
            self._rfid_last is None
            or self._rfid_last[0] != beacon
            or step - self._rfid_last[1] >= C.RFID_PING_REPEAT_STEPS
        ):
            self._rfid_last = (beacon, step)
            events.append(
                Event(
                    "RFID_PING",
                    f"RFID {beacon}: {distance:.1f} m",
                    {
                        "beacon": int(beacon),
                        "range": float(distance),
                        "position": tuple(float(v) for v in BEACONS[int(beacon)]),
                    },
                )
            )

    def _resolve_standoffs(self, waiting_on: int | None, events: list[Event]) -> None:
        """Forklifts yield to the AMR: break AMR-waits-for-forklift-waits-for-AMR."""
        for forklift in self.world.forklifts:
            if waiting_on == forklift.index and not forklift.moving:
                count = self._standoff.get(forklift.index, 0) + 1
                if count >= C.FORKLIFT_DEADLOCK_STEPS:
                    if forklift.active:
                        forklift.reverse()
                        action, text = "reverse", f"{forklift.name} BACKS OFF"
                    else:
                        forklift.make_way(self.robot.position)
                        action, text = "make_way", f"{forklift.name} MAKES WAY"
                    events.append(
                        Event(
                            "FORKLIFT_YIELD", text, {"forklift": forklift.index, "action": action}
                        )
                    )
                    count = 0
                self._standoff[forklift.index] = count
            else:
                self._standoff[forklift.index] = 0

    def _sensor_mask(self) -> tuple[bool, bool, bool]:
        """(proximity, RFID, zone) on/off: the scenario's sensors (zone: also the option)."""
        prox, rfid, zone = self.scenario.sensors
        return prox, rfid, zone and self.options.zone

    def _sense(self, position: np.ndarray) -> Sensors:
        prox, rfid, zone = self._sensor_mask()
        sensors = self.world.sense(position, self.world_rng, zone=zone, proximity=prox, rfid=rfid)
        self._last_sensors = sensors
        return sensors

    # --- results -----------------------------------------------------------
    def _parcel_info(self) -> dict[str, Any]:
        order = self.current_order
        return {
            "id": order.number,
            "name": f"ORDER_PARCEL_{order.number}",
            "state": self.parcel_state,
            "slot": order.slot.name,
            "dock": order.dock.name,
            "rack_xy": order.slot.parcel_xy,
            "deck_height": C.PARCEL_DECK_HEIGHT,
            "robot_xy": tuple(float(v) for v in self.robot.position),
            "conveyor_from": order.dock.conveyor_from,
            "conveyor_to": order.dock.conveyor_to,
        }

    @staticmethod
    def _dock_info(place: Place, stage: str) -> dict[str, Any]:
        if isinstance(place, Slot):
            return {
                "stage": stage,
                "target": "rack",
                "name": place.name,
                "station": place.station,
                "pose": place.dock_pose,
                "facing": place.facing,
                "parcel_from": place.parcel_xy,
                "parcel_to": place.dock_pose,
                "dock_index": None,
            }
        return {
            "stage": stage,
            "target": "dock",
            "name": place.name,
            "station": place.station,
            "pose": place.dock_pose,
            "facing": place.facing,
            "parcel_from": place.dock_pose,
            "parcel_to": place.conveyor_to,
            "conveyor_from": place.conveyor_from,
            "dock_index": place.index,
        }

    def _result(
        self,
        kind: str,
        duration: float,
        events: list[Event],
        start=None,
        end=None,
        omega=None,
        omega_cmd=None,
        action: int | None = None,
        u=None,
        u_eff=None,
        sensors: Sensors | None = None,
        dock: dict[str, Any] | None = None,
        clock_start: float | None = None,
    ) -> StepResult:
        position = self.robot.position.copy()
        if self.brain is not None and self._deployed and self.scenario.perfect_localization:
            self.brain.set_exact(position)  # planning only: the belief follows the truth exactly
        start = position if start is None else np.asarray(start, dtype=float).copy()
        end = position if end is None else np.asarray(end, dtype=float).copy()
        zeros = np.zeros(3)
        all_events = self._pending + events
        self._pending = []
        if self.brain is not None:
            self._handle_brain_events(all_events)
        self._slow_code_events(all_events)
        plan = self.plan
        result = StepResult(
            index=self.index,
            kind=kind,
            phase=self.phase,
            banner=self._banner(self.phase),
            duration=float(duration),
            clock_start=self.clock if clock_start is None else float(clock_start),
            clock=self.clock,
            start=start,
            end=end,
            wheel_omegas=zeros.copy() if omega is None else np.asarray(omega, dtype=float).copy(),
            wheel_omegas_cmd=zeros.copy()
            if omega_cmd is None
            else np.asarray(omega_cmd, dtype=float).copy(),
            action=action,
            u=None if u is None else np.asarray(u, dtype=float).copy(),
            u_eff=None if u_eff is None else np.asarray(u_eff, dtype=float).copy(),
            sensors=sensors,
            belief=self.brain.snapshot()
            if self.brain is not None and self.brain.initialized
            else None,
            value=None if plan is None else plan.V,
            value_change=None if plan is None else self.value_change,
            goal_mask=None if plan is None else plan.goal_mask,
            goal_cell=None if plan is None else plan.goal_cell,
            goal_name=None if plan is None else plan.goal_name,
            path=list(self.path),
            events=all_events,
            stats=self.stats.snapshot(success=self.success),
            actors=[f.as_dict() for f in self.world.forklifts],
            obstacles=[ob.as_dict() for ob in self.world.obstacles],
            parcel=self._parcel_info(),
            dock=dock,
            order={**self.current_order.as_dict(), "total": len(self.orders), "leg": self.leg},
            notes=list(self.notes),
            done=self.done,
            success=self.success,
        )
        self.index += 1
        return result

    def _fault_result(self, fault: StudentFault, reason: str = "exception") -> StepResult:
        self.stats.faults += 1
        self.fault = fault.as_dict()
        self.fault["reason"] = reason
        if reason == "engine_error":
            text = f"ENGINE ERROR: {fault.message}"
        else:
            text = f"FAULT in {fault.function}: {fault.message}"
        events = [Event("FAULT", text[:160], dict(self.fault))]
        self._script.clear()
        self._kidnap_pending = False
        self._set_phase(FAULT, events)
        self.done, self.success = True, False
        return self._result("event_only", 0.0, events)

    def _bare_fault_result(self, fault: StudentFault) -> StepResult:
        """Last resort after an internal error: a FAULT result without belief or plan."""
        self.done, self.success, self.phase = True, False, FAULT
        self.fault = dict(fault.as_dict(), reason="engine_error")
        position = np.array(self.robot.position, dtype=float)
        result = StepResult(
            index=self.index,
            kind="event_only",
            phase=FAULT,
            banner="FAULT",
            duration=0.0,
            clock_start=self.clock,
            clock=self.clock,
            start=position.copy(),
            end=position.copy(),
            wheel_omegas=np.zeros(3),
            wheel_omegas_cmd=np.zeros(3),
            events=[Event("FAULT", f"ENGINE ERROR: {fault.message}"[:160], dict(self.fault))],
            done=True,
        )
        self.index += 1
        return result

    # =====================================================================
    # Phases
    # =====================================================================
    def _boot(self) -> StepResult:
        """System check, kinematics, system identification, first plans."""
        events: list[Event] = []
        checks = self.bridge.system_check()
        self.checks = checks
        missing_planning = self.bridge.missing_core(PLANNING_TODOS)
        missing_localization = self.bridge.missing_core(LOCALIZATION_TODOS)
        self.planning_ready = not missing_planning
        self.localization_ready = not missing_localization
        available = self.available_scenarios()
        if self._scenario_auto and self.scenario.key not in available and available:
            self.scenario = SC.BY_KEY[
                available[0]
            ]  # Webots: the first experiment the working TODOs allow
            self._apply_scenario()
        missing = []
        if self.scenario.needs_localization:
            missing += missing_localization
        if self.scenario.needs_planning:
            missing += missing_planning
        missing = sorted(missing)
        events.append(
            Event(
                "SYSTEM_CHECK",
                "SYSTEM CHECK",
                {
                    "rows": [tuple(row) for row in checks],
                    "can_start": not missing,
                    "missing": missing,
                    "profile": dict(self.profile),
                    "planning_ready": self.planning_ready,
                    "scenario": self.scenario.key,
                },
            )
        )
        if missing:
            listed = ", ".join(f"{n} {name}" for n, name, _ in missing)
            self.fault = {
                "reason": "missing_todos",
                "function": None,
                "todo": None,
                "missing": missing,
                "message": f"implement TODO {listed} to deploy the robot",
                "details": "",
            }
            self._set_phase(FAULT, events)
            self.done = True
            events.append(
                Event(
                    "FAULT",
                    "MISSING TODOS: " + ", ".join(str(n) for n, _, _ in missing),
                    dict(self.fault),
                )
            )
            return self._result("event_only", 0.0, events)

        J, status, message = self.bridge.jacobian()
        self.jacobian, self.jacobian_status = J, status
        self.robot = TruthRobot(self.start, J)
        if status == "mismatch":
            self._note("KINEMATICS MISMATCH")
            events.append(
                Event(
                    "WARNING",
                    "KINEMATICS MISMATCH",
                    {"code": "KINEMATICS_MISMATCH", "message": message, "J": J.tolist()},
                )
            )
        elif status == "fallback":
            self._note("JACOBIAN FALLBACK: using the true wheels")
            events.append(
                Event(
                    "WARNING",
                    "JACOBIAN FALLBACK",
                    {"code": "JACOBIAN_FALLBACK", "message": message},
                )
            )

        X, U, Z = self._calibration_log()
        self.sysid = self.bridge.system_id(X, U, Z)
        sysid = self.sysid
        events.append(
            Event(
                "SYSID",
                f"SYSTEM ID: sigma = {sysid.sigma:.3f} m",
                {
                    "A": None if sysid.A is None else sysid.A.tolist(),
                    "B": None if sysid.B is None else sysid.B.tolist(),
                    "sigma": sysid.sigma,
                    "sigma_raw": sysid.sigma_raw,
                    "fallback": sysid.fallback,
                    "message": sysid.message,
                    "samples": len(X),
                },
            )
        )
        if sysid.fallback:
            self._note("SIGMA FALLBACK: 0.2 m")
            events.append(
                Event(
                    "WARNING",
                    "SIGMA FALLBACK 0.2",
                    {"code": "SIGMA_FALLBACK", "message": sysid.message},
                )
            )

        grid, reason = self.options.grid_filter, ""
        if grid and any(
            self.bridge.status(name) == MISSING for name in ("markov_update", "markov_predict")
        ):
            grid, reason = False, "TODO 7/8 missing"
        elif grid and self.bridge.probe_seconds.get("markov_predict", 0.0) > C.GRID_MAX_SECONDS:
            grid, reason = (
                False,
                f"markov_predict too slow ({self.bridge.probe_seconds['markov_predict']:.2f} s)",
            )
        self.brain = Brain(
            self.bridge,
            sysid.sigma,
            self.brain_rng,
            grid_filter=grid,
            likelihood_layer=self.options.likelihood_layer,
        )
        if reason:
            self._note("GRID FILTER OFF")
            events.append(Event("GRID_FILTER_OFF", "GRID FILTER OFF", {"reason": reason}))

        if self.planning_ready:
            start = time.perf_counter()
            order = self.orders[0]
            self.planner.plan(order.dock.name, order.dock.goal_cell)
            self.plan, _ = self.planner.plan(order.slot.name, order.slot.goal_cell)
            events.append(
                Event(
                    "PLAN_READY",
                    "ROUTES PLANNED",
                    {
                        "goals": [order.slot.name, order.dock.name],
                        "seconds": time.perf_counter() - start,
                    },
                )
            )
            if self.route is not None:
                self.plan = None  # a sensor test drives its fixed route
        else:
            events.append(
                Event(
                    "PLANNING_OFF",
                    "PLANNING TODOS MISSING: EXPERIMENTS ONLY",
                    {"missing": missing_planning, "available": self.available_scenarios()},
                )
            )
        if not self.localization_ready:
            events.append(
                Event(
                    "LOCALIZATION_OFF",
                    "LOCALIZATION TODOS MISSING: PENALTY TEST ONLY",
                    {"missing": missing_localization, "available": self.available_scenarios()},
                )
            )
        self._set_phase(BRIEFING, events)
        return self._result("event_only", 0.0, events)

    def _calibration_log(self):
        """Synthetic calibration drive: 400 states, N(0, 1.5^2 I) controls (DESIGN 5.1).

        The next states come from the robot's real kinematics, so a wrong
        Jacobian shows up in the student's B_hat.
        """
        rng = self.world_rng
        cells = self.world.free_cells(non_proximity=False)
        picks = cells[rng.integers(len(cells), size=C.SYSID_SAMPLES)]
        states = picks[:, ::-1].astype(float) + rng.uniform(-0.5, 0.5, size=(C.SYSID_SAMPLES, 2))
        controls = rng.normal(0.0, C.SYSID_CONTROL_SIGMA, size=(C.SYSID_SAMPLES, 2))
        next_states = (
            states
            + controls @ self.robot.B_real.T
            + rng.normal(0.0, C.SYSID_TRUE_SIGMA, size=(C.SYSID_SAMPLES, 2))
        )
        return states, controls, next_states

    def _briefing(self) -> StepResult:
        events: list[Event] = []
        orders = [order.as_dict() for order in self.orders]
        first = self.orders[0]
        sc = self.scenario
        text = f"PICK UP {first.slot.name} -> {first.dock.name}" if not sc.is_tour else sc.title
        events.append(
            Event(
                "BRIEFING",
                text,
                {
                    "code": self.code,
                    "orders": orders,
                    "profile": dict(self.profile),
                    "student": self.student_info,
                    "scenario": sc.key,
                    "title": sc.title,
                    "description": list(sc.description),
                    "available": self.available_scenarios(),
                },
            )
        )
        self._set_phase(COUNTDOWN, events)
        return self._result("event_only", 0.0, events)

    def _countdown(self) -> StepResult:
        events = [Event("COUNTDOWN", str(self._countdown_left), {"count": self._countdown_left})]
        self._countdown_left -= 1
        position = self.robot.position
        self.world.step_forklifts(1.0, position, position)
        if self._countdown_left <= 0:
            self._set_phase(LOCALIZING, events)
        return self._result("wait", 1.0, events)

    def _deploy(self) -> StepResult:
        """GO: the robot wakes up anywhere and takes its first measurement."""
        self._deployed = True
        clock_start = self.clock
        self._deploy_clock = self.clock
        self._tick(C.SCAN_DURATION)
        sc = self.scenario
        known = sc.known_start
        n = sc.particles or (C.N_PARTICLES_TRACKING if known else C.N_PARTICLES_GLOBAL)
        text = (
            "GO! EXACT POSITION (PLANNING ONLY)"
            if sc.perfect_localization
            else (f"GO! {n} PARTICLES" + (" AT THE START" if known else ""))
        )
        events = [
            Event(
                "DEPLOYED",
                text,
                {
                    "particles": n,
                    "known_start": known,
                    "perfect_localization": sc.perfect_localization,
                },
            )
        ]
        seed = sc.noise_seed
        if seed is not None:  # a test: the same noise (truth and belief) in every run
            self.world_rng = np.random.default_rng(seed)
            self.brain.rng = np.random.default_rng(seed + 1)
        self.brain.n_fixed = sc.particles
        sensors = self._sense(self.robot.position)
        if (
            sc.perfect_localization
        ):  # planning only: the belief is the truth, no student filter code
            self.brain.set_exact(self.robot.position)
        elif known:  # (a tour only tracks; a known-start mission can still get LOST)
            self.brain.reset_to(
                self.robot.position, C.TOUR_RESET_SIGMA, n, tracking_only=sc.is_tour
            )
        else:
            self.brain.global_init(sensors)
        self._handle_brain_events(events)
        if self.route is not None:
            self._start_tour(events)
        else:
            if known:
                self._set_phase(TO_PICKUP, events)  # (it knows where it starts: no localizing)
            if sc.group:  # a mission test (PENALTY TEST): its setting, announced
                events.append(
                    Event(
                        "TEST_START",
                        f"{sc.title}: {sc.option}",
                        {"scenario": sc.key, "label": sc.option},
                    )
                )
            self._start_leg(events)
        self._rfid_event(sensors, events)
        position = self.robot.position
        self.world.step_forklifts(C.SCAN_DURATION, position, position)
        return self._result(
            "wait", C.SCAN_DURATION, events, sensors=sensors, clock_start=clock_start
        )

    def _start_leg(self, events: list[Event]) -> None:
        goal = self.goal
        self.value_change = None  # (a new goal: the last obstacle's value change no longer applies)
        self.plan, computed = self.planner.plan(goal.name, goal.goal_cell)
        if self.scenario.group == "penalty":  # the route the value function plans from here (Q5)
            self._planned_routes.append(
                list(self.brain.greedy_path(self.plan, point_to_cell(self.robot.position)))
            )
        events.append(
            Event(
                "NEW_GOAL",
                f"NEXT: {goal.name}",
                {
                    "goal": goal.name,
                    "cell": goal.goal_cell,
                    "leg": self.leg,
                    "computed": computed,
                    "seconds": self.plan.seconds,
                },
            )
        )
        self._update_path()

    def _travel(self) -> StepResult:
        """One control step: vote, drive (student Jacobian), sense, update the belief."""
        if self.route is not None:
            return self._tour_step()
        brain, robot = self.brain, self.robot
        if (
            point_to_cell(brain.mean) == self.plan.goal_cell
            and brain.healthy
            and (brain.locked or brain.dock_ready(self.goal.station))
        ):
            return self._begin_docking()
        seeking = self._seeking_beacon()
        if seeking:
            action = brain.seek_action(self._active_plan())
        else:
            action, _ = brain.choose_action(self.plan, self._last_blocked)
        u = C.ACTION_VECTORS[action].copy()
        omega_cmd, displacement = robot.plan(u, self.world_rng)
        start = robot.position.copy()
        blocking = self.world.forklift_blocking(start, start + displacement)
        if blocking is not None and self._amr_wait_streak < C.AMR_MAX_WAIT_STEPS:
            return self._wait_for_forklift(blocking, action, u)
        self._amr_wait_streak = 0

        clock_start = self.clock
        self._tick(C.MOVE_DURATION)
        events: list[Event] = []
        if seeking and not self._seek_announced:
            events.append(
                Event(
                    "SEEK_BEACON",
                    "NOT SURE YET: LOOKING FOR AN RFID BEACON",
                    {"updates": brain.updates},
                )
            )
        self._seek_announced = seeking
        outcome = robot.execute(self.world, u, omega_cmd, displacement)
        self._last_blocked = outcome.safety_stop
        if outcome.safety_stop:
            self.stats.safety_stops += 1
            events.append(
                Event(
                    "SAFETY_STOP",
                    f"SAFETY STOP ({outcome.blocked_by})",
                    {"cause": outcome.blocked_by, "fraction": outcome.fraction},
                )
            )
        self.world.step_forklifts(C.MOVE_DURATION, outcome.start, outcome.end)
        self._resolve_standoffs(None, events)
        sensors = self._sense(robot.position)
        brain.predict_and_update(outcome.u_eff, sensors)
        self.stats.moves += 1
        self.stats.control_steps += 1
        self._travel_moves += 1
        self._band_moves += int(self.world.prox_on[point_to_cell(robot.position)] > 0)
        self._handle_brain_events(events)
        if brain.locked:
            self.stats.record_tracking_error(float(np.linalg.norm(brain.mean - robot.position)))
        self._rfid_event(sensors, events)
        self._update_path()
        self._obstacle_test_hook()
        self._check_timeout(events)
        return self._result(
            "move",
            C.MOVE_DURATION,
            events,
            start=outcome.start,
            end=outcome.end,
            omega=outcome.omega,
            omega_cmd=outcome.omega_cmd,
            action=action,
            u=u,
            u_eff=outcome.u_eff,
            sensors=sensors,
            clock_start=clock_start,
        )

    def _obstacle_test_hook(self) -> None:
        """OBSTACLE TEST: drop the obstacle ``obstacle_after`` steps after the first LOCKED."""
        after = self.scenario.obstacle_after
        if after is None or self._obstacle_done or self._lock_step is None or self.done:
            return
        steps = self.stats.control_steps - self._lock_step
        # Give up if no safe drop location has been found after 40 steps.
        if steps >= after and (self.drop_obstacle() is not None or steps > after + 40):
            self._obstacle_done = True

    def _wait_for_forklift(self, index: int, action: int, u: np.ndarray) -> StepResult:
        clock_start = self.clock
        self._tick(C.WAIT_DURATION)
        events: list[Event] = []
        self._amr_wait_streak += 1
        if self._amr_wait_streak == 1:
            events.append(Event("WAITING", f"WAITING FOR FORKLIFT_{index}", {"forklift": index}))
        position = self.robot.position
        self.world.step_forklifts(C.WAIT_DURATION, position, position)
        self._resolve_standoffs(index, events)
        self._last_blocked = False
        self.stats.waits += 1
        self.stats.control_steps += 1
        self._check_timeout(events)
        return self._result(
            "wait",
            C.WAIT_DURATION,
            events,
            action=action,
            u=u,
            u_eff=np.zeros(2),
            sensors=self._last_sensors,
            clock_start=clock_start,
        )

    def _teleport(self) -> StepResult:
        """KIDNAPPED: an unseen hand carries the robot far away (book 4.4 exercise).

        The brain is not told; it only notices through its next measurements,
        the first of which it takes right after being put down.
        """
        self._kidnap_pending = False
        clock_start = self.clock
        self._tick(C.TELEPORT_DURATION)
        start = self.robot.position.copy()
        target = self.world.sample_free_position(
            self.world_rng,
            far_from=start,
            min_distance=C.KIDNAP_MIN_DISTANCE,
            avoid_cells=[self.plan.goal_cell],
            avoid_radius=3,
        )
        self.robot.teleport(target)
        self.stats.kidnaps += 1
        self._last_blocked = False
        self._amr_wait_streak = 0
        events = [Event("KIDNAPPED", "KIDNAPPED!", {"from": start.tolist(), "to": target.tolist()})]
        self.world.step_forklifts(C.TELEPORT_DURATION, target, target)
        sensors = self._sense(target)
        self.brain.measure(sensors)
        self._handle_brain_events(events)
        self._rfid_event(sensors, events)
        self._update_path()
        return self._result(
            "teleport",
            C.TELEPORT_DURATION,
            events,
            start=start,
            end=target,
            sensors=sensors,
            clock_start=clock_start,
        )

    # --- docking ------------------------------------------------------------
    def _begin_docking(self) -> StepResult:
        """Arrived (the estimate is in the goal cell): look for the dock marker."""
        place = self.goal
        station = np.array(place.station)
        position = self.robot.position
        distance = float(np.linalg.norm(position - station))
        self.stats.arrival_errors.append(distance)
        events: list[Event] = []
        clock_start = self.clock
        if distance > C.CAPTURE_RADIUS:
            self._tick(C.DOCK_FAIL_DURATION)
            self.stats.failed_dockings += 1
            events.append(
                Event(
                    "DOCKING_FAILED",
                    "DOCKING FAILED: MARKER NOT FOUND",
                    {"place": place.name, "distance": distance, "capture_radius": C.CAPTURE_RADIUS},
                )
            )
            sensors = self._sense(position)
            self.brain.relocalize(sensors, "marker not found", lost_event=False)
            events.append(
                Event("RELOCALIZING", "FORCED RE-LOCALIZATION", {"reason": "marker not found"})
            )
            self._handle_brain_events(events)
            self._update_path()
            self.world.step_forklifts(C.DOCK_FAIL_DURATION, position, position)
            return self._result(
                "wait", C.DOCK_FAIL_DURATION, events, sensors=sensors, clock_start=clock_start
            )

        sensors = self._sense(position)
        sensors.z_fiducial = self.world.sense_fiducial(position, self.world_rng)
        error_before = float(np.linalg.norm(self.brain.mean - position))
        self.brain.marker_fix(sensors.z_fiducial, station)
        error_after = float(np.linalg.norm(self.brain.mean - position))
        self._set_phase(DOCK_PICKUP if self.leg == "pickup" else DOCK_DROPOFF, events)
        events.append(
            Event(
                "MARKER_ACQUIRED",
                f"MARKER ACQUIRED: {place.name}",
                {
                    "place": place.name,
                    "station": place.station,
                    "distance": distance,
                    "error_before": error_before,
                    "error_after": error_after,
                    "fix_cm": sensors.z_fiducial.tolist(),
                },
            )
        )
        self._script = deque([self._dock_transfer, self._undock])
        return self._dock_approach(events, sensors, clock_start)

    def _dock_approach(
        self, events: list[Event], sensors: Sensors, clock_start: float
    ) -> StepResult:
        place = self.goal
        self._tick(C.DOCK_DURATION)
        outcome = self.robot.drive_to(self.world, np.array(place.dock_pose), C.DOCK_DURATION)
        if outcome.safety_stop:
            self.stats.safety_stops += 1
            events.append(
                Event(
                    "SAFETY_STOP",
                    f"SAFETY STOP ({outcome.blocked_by})",
                    {"cause": outcome.blocked_by, "fraction": outcome.fraction},
                )
            )
        self.brain.known_motion(outcome.end - outcome.start)
        self.world.step_forklifts(C.DOCK_DURATION, outcome.start, outcome.end)
        self._update_path()
        return self._result(
            "dock",
            C.DOCK_DURATION,
            events,
            start=outcome.start,
            end=outcome.end,
            omega=outcome.omega,
            omega_cmd=outcome.omega_cmd,
            sensors=sensors,
            dock=self._dock_info(place, "approach"),
            clock_start=clock_start,
        )

    def _dock_transfer(self) -> StepResult:
        place = self.goal
        events: list[Event] = []
        clock_start = self.clock
        if self.leg == "pickup":
            kind, duration = "pickup", C.PICKUP_DURATION
            self._tick(duration)
            self.parcel_state = "robot"
            events.append(
                Event(
                    "PACKAGE_LOADED",
                    f"PACKAGE LOADED: {place.name}",
                    {"slot": place.name, "order": self.current_order.number},
                )
            )
        else:
            kind, duration = "dropoff", C.DROPOFF_DURATION
            self._tick(duration)
            self.parcel_state = "conveyor"
            self.stats.orders_completed += 1
            events.append(
                Event(
                    "DELIVERED",
                    f"DELIVERED TO {place.name}!",
                    {
                        "dock": place.name,
                        "order": self.current_order.number,
                        "orders_completed": self.stats.orders_completed,
                    },
                )
            )
        position = self.robot.position
        self.world.step_forklifts(duration, position, position)
        return self._result(
            kind, duration, events, dock=self._dock_info(place, kind), clock_start=clock_start
        )

    def _undock(self) -> StepResult:
        place = self.goal
        events: list[Event] = []
        clock_start = self.clock
        self._tick(C.UNDOCK_DURATION)
        target = np.array(place.station) + self.world_rng.normal(0.0, C.DOCK_MOTION_SIGMA, 2)
        outcome = self.robot.drive_to(self.world, target, C.UNDOCK_DURATION)
        self.brain.known_motion(outcome.end - outcome.start)
        sensors = self._sense(self.robot.position)
        self.brain.measure(sensors)
        self._handle_brain_events(events)
        self.world.step_forklifts(C.UNDOCK_DURATION, outcome.start, outcome.end)
        dock = self._dock_info(place, "undock")
        if self.leg == "pickup":
            self.leg = "dropoff"
            self._set_phase(TO_DOCK, events)
            self._start_leg(events)
        elif self.order_index + 1 < len(self.orders):
            self.parcel_state = "delivered"
            delivered = self._parcel_info()
            self.order_index += 1
            self.leg = "pickup"
            self.parcel_state = "rack"
            order = self.current_order
            events.append(
                Event(
                    "NEW_ORDER",
                    f"NEW ORDER: {order.slot.name} -> {order.dock.name}",
                    {**order.as_dict(), "delivered": delivered},
                )
            )
            self._set_phase(TO_PICKUP, events)
            self._start_leg(events)
        else:
            self.parcel_state = "delivered"
            self.done, self.success = True, True
            self.path = []
            self._set_phase(COMPLETE, events)
            final = score_breakdown(self.stats, success=True, final=True)
            if self.scenario.group:  # a PENALTY TEST: its results table instead of the score
                self.experiment = self._mission_experiment()
                events.append(
                    Event(
                        "EXPERIMENT_COMPLETE",
                        f"{self.scenario.title} COMPLETE",
                        dict(self.experiment),
                    )
                )
            events.append(
                Event(
                    "MISSION_COMPLETE",
                    f"MISSION COMPLETE: {final['score']} PTS",
                    {
                        **final,
                        "time": self.clock,
                        "code": self.code,
                        "profile": dict(self.profile),
                        "stats": self.stats.snapshot(success=True),
                    },
                )
            )
        return self._result(
            "undock",
            C.UNDOCK_DURATION,
            events,
            start=outcome.start,
            end=outcome.end,
            omega=outcome.omega,
            omega_cmd=outcome.omega_cmd,
            sensors=sensors,
            dock=dock,
            clock_start=clock_start,
        )

    # --- experiment tours (scenarios.py) ---------------------------------------
    def _apply_scenario(self) -> None:
        """Set up the chosen scenario: a sensor test's route and start, or the mission's random
        start.
        """
        sc = self.scenario
        count = max(1, len(self.orders))
        self._all_orders = list(self._code_orders)
        if sc.is_tour:
            self.route = SC.Route(sc.route)
            self.start = np.array(sc.route[0], dtype=float)
        elif sc.mission_code is not None:  # a fixed delivery: that code's start and orders
            self.route = None
            self.start, self._all_orders = self._layout_of(sc.mission_code)
        else:
            self.route = None
            self.start = self._mission_start.copy()
        self.orders = self._all_orders[:count]
        self.planner.penalty = sc.penalty
        self.robot = TruthRobot(self.start, self.jacobian)
        self._route_s, self._trace = 0.0, []
        self._tour_stats = {"steps": 0, "error_sum": 0.0, "spread_sum": 0.0, "lost": 0}
        if self.phase == BRIEFING and self.planning_ready:  # (after BOOT: the cached plans)
            self.plan = (
                None
                if sc.is_tour
                else self.planner.plan(self.orders[0].slot.name, self.orders[0].slot.goal_cell)[0]
            )

    def _layout_of(self, code: int) -> tuple[np.ndarray, list[Order]]:
        """The start and orders that mission ``code`` draws (as in the constructor)."""
        rng = np.random.default_rng(code)
        start = self.world.sample_free_position(rng)
        slots = rng.choice(len(SLOTS), size=C.MAX_ORDERS, replace=False)
        docks = rng.integers(len(DOCKS), size=C.MAX_ORDERS)
        return start, [
            Order(n + 1, SLOTS[int(s)], DOCKS[int(d)])
            for n, (s, d) in enumerate(zip(slots, docks, strict=False))
        ]

    def _start_tour(self, events: list[Event]) -> None:
        """A sensor test starts: its sensors are announced; it never localizes (it knows its
        start).
        """
        sc = self.scenario
        self.brain.dead_reckoning = not any(sc.sensors)  # (NO SENSORS: motion alone)
        events.append(
            Event(
                "SENSORS",
                f"{sc.title}: {sc.option}",
                {
                    "scenario": sc.key,
                    "label": sc.option,
                    "sensors": dict(zip(SC.SENSOR_NAMES, sc.sensors, strict=False)),
                    "known_start": sc.known_start,
                    "particles": sc.particles,
                },
            )
        )
        self._set_phase(TO_PICKUP, events)
        self._update_path()

    def _record_tour_stats(self) -> None:
        brain = self.brain
        error = float(np.linalg.norm(brain.mean - self.robot.position))
        spread = float(brain.spread)
        stats = self._tour_stats
        stats["steps"] += 1
        stats["error_sum"] += error
        stats["spread_sum"] += spread
        self._trace.append((self.stats.control_steps, spread, error))

    def _tour_step(self) -> StepResult:
        """One control step of a sensor test: the robot follows its fixed route by its TRUE
        position (so the path never depends on the belief), senses with the test's sensors, and
        updates the student's belief."""
        brain, robot, route = self.brain, self.robot, self.route
        if self._route_s >= route.length:
            return self._finish_tour()
        u = route.control(robot.position, self._route_s)
        action = int(np.argmax(C.ACTION_VECTORS @ u))
        omega_cmd, displacement = robot.plan(u, self.world_rng)
        start = robot.position.copy()
        blocking = self.world.forklift_blocking(start, start + displacement)
        if blocking is not None and self._amr_wait_streak < C.AMR_MAX_WAIT_STEPS:
            return self._wait_for_forklift(blocking, action, u)
        self._amr_wait_streak = 0

        clock_start = self.clock
        self._tick(C.MOVE_DURATION)
        events: list[Event] = []
        outcome = robot.execute(self.world, u, omega_cmd, displacement)
        self._last_blocked = outcome.safety_stop
        if outcome.safety_stop:
            self.stats.safety_stops += 1
            events.append(
                Event(
                    "SAFETY_STOP",
                    f"SAFETY STOP ({outcome.blocked_by})",
                    {"cause": outcome.blocked_by, "fraction": outcome.fraction},
                )
            )
        self.world.step_forklifts(C.MOVE_DURATION, outcome.start, outcome.end)
        self._resolve_standoffs(None, events)
        sensors = self._sense(robot.position)
        brain.predict_and_update(outcome.u_eff, sensors)
        self._route_s = route.advance(robot.position, self._route_s)
        self._tour_blocked = self._tour_blocked + 1 if outcome.safety_stop else 0
        if (
            self._tour_blocked >= C.TOUR_BLOCKED_SKIP
        ):  # pushed against something: move the carrot on
            self._route_s = min(self._route_s + route.STEP, route.length)
            self._tour_blocked = 0
        self.stats.moves += 1
        self.stats.control_steps += 1
        self._handle_brain_events(events)
        self._record_tour_stats()
        if brain.locked:
            self.stats.record_tracking_error(float(np.linalg.norm(brain.mean - robot.position)))
        self._rfid_event(sensors, events)
        self._update_path()
        self._check_timeout(events)
        return self._result(
            "move",
            C.MOVE_DURATION,
            events,
            start=outcome.start,
            end=outcome.end,
            omega=outcome.omega,
            omega_cmd=outcome.omega_cmd,
            action=action,
            u=u,
            u_eff=outcome.u_eff,
            sensors=sensors,
            clock_start=clock_start,
        )

    def _finish_tour(self) -> StepResult:
        """End of the route: the experiment's result."""
        sc = self.scenario
        events: list[Event] = []
        self.done, self.success = True, True
        self.path = []
        self._set_phase(COMPLETE, events)
        self.experiment = self._experiment_summary()
        events.append(Event("EXPERIMENT_COMPLETE", f"{sc.title} COMPLETE", dict(self.experiment)))
        return self._result("event_only", 0.0, events)

    def _experiment_summary(self) -> dict[str, Any]:
        """A tour's result (sensor and particle tests): its setting, the steps, the mean estimate
        error and belief spread (m), the error at the end, the mean error of the first EARLY_STEPS
        steps, when it first LOCKED (s after deployment; None for a known start), LOST events, the
        trace (step, spread, error), and ``metrics``: the
        columns of its results table, (name, value, unit) with the first one shown as a bar.
        """
        sc = self.scenario
        stats = self._tour_stats
        n = stats["steps"]
        error = stats["error_sum"] / n if n else None
        spread = stats["spread_sum"] / n if n else None
        locked = (
            None
            if self.stats.time_to_lock is None
            else self.stats.time_to_lock - self._deploy_clock
        )
        first = [row[2] for row in self._trace[: C.EARLY_STEPS]]
        early = (
            float(np.mean(first)) if first else None
        )  # while an unknown start is still being found
        if sc.group == "start":
            metrics = [
                ("ERROR", error, "m"),
                (f"FIRST {C.EARLY_STEPS * C.STEP_DT:.0f} S", early, "m"),
                ("SPREAD", spread, "m"),
            ]
        else:
            metrics = [("SPREAD", spread, "m"), ("ERROR", error, "m")]
        return {
            "scenario": sc.key,
            "title": sc.title,
            "group": sc.group,
            "label": sc.option,
            "sensors": dict(zip(SC.SENSOR_NAMES, sc.sensors, strict=False)),
            "particles": sc.particles,
            "steps": n,
            "mean_error": error,
            "mean_spread": spread,
            "final_error": float(np.linalg.norm(self.brain.mean - self.robot.position)),
            "early_error": early,
            "time_to_lock": locked,
            "lost": stats["lost"],
            "time": self.clock,
            "trace": [list(row) for row in self._trace],
            "metrics": metrics,
        }

    def _mission_experiment(self) -> dict[str, Any]:
        """A PENALTY TEST's result (Q5): the planned route (the value function's route from the
        start
        to the rack and from the rack to the dock: its length in m and its cells in the proximity
        band), and what happened when the robot drove it: safety stops, time (s), COMPLETE or
        FAILED.
        Also the share of travel moves that ended next to a shelf or a wall (%).
        """
        sc = self.scenario
        cells = [
            cell
            for k, path in enumerate(self._planned_routes)
            for cell in (path if k == 0 else path[1:])
        ]
        length = float(sum(max(0, len(path) - 1) for path in self._planned_routes))
        prox = float(sum(int(PROXIMITY_MAP_ON[cell] > 0) for cell in cells))
        near = 100.0 * self._band_moves / self._travel_moves if self._travel_moves else None
        result = "COMPLETE" if self.success else "FAILED"
        return {
            "scenario": sc.key,
            "title": sc.title,
            "group": sc.group,
            "label": sc.option,
            "steps": self.stats.control_steps,
            "route_length": length,
            "route_prox_cells": prox,
            "route": [list(cell) for cell in cells],
            "safety_stops": self.stats.safety_stops,
            "time": self.stats.mission_time,
            "near_shelves": near,
            "success": self.success,
            "result": result,
            "lost": self.stats.relocalizations,
            "metrics": [
                ("ROUTE", length, "m"),
                ("PROX ON", prox, ""),
                ("STOPS", float(self.stats.safety_stops), ""),
                ("TIME", self.stats.mission_time, "s"),
                ("RESULT", result, ""),
            ],
        }

    # --- chaos helpers ----------------------------------------------------------
    def _choose_obstacle_cell(self) -> tuple | None:
        """A free, open cell 4-16 cells ahead on the planned path, away from goals and lanes."""
        if self.brain.locked and len(self.path) > C.OBSTACLE_MIN_AHEAD:
            path = self.path
        else:
            path = self.brain.greedy_path(self.plan, point_to_cell(self.robot.position))
        goals = [self.plan.goal_cell]
        for order in self.orders[self.order_index :]:
            goals += [order.slot.goal_cell, order.dock.goal_cell]
        boxes = self.world.forklift_boxes()
        for k in range(C.OBSTACLE_MIN_AHEAD, min(len(path), C.OBSTACLE_MAX_AHEAD + 1)):
            cell = path[k]
            if self.world.obstacle_map[cell] or self.world.prox_on[cell]:
                continue
            center = cell_center(cell)
            if np.linalg.norm(center - self.robot.position) < C.OBSTACLE_ROBOT_CLEARANCE:
                continue
            if any(
                max(abs(cell[0] - g[0]), abs(cell[1] - g[1])) <= C.OBSTACLE_GOAL_CLEARANCE
                for g in goals
            ):
                continue
            if _route_distance(center) < C.OBSTACLE_LANE_CLEARANCE:
                continue
            if float(box_distance(center[None, :], boxes)[0]) < C.OBSTACLE_LANE_CLEARANCE:
                continue
            return cell
        return None


###############################################################################
#                    Running a whole mission without Webots                  #
###############################################################################


class ChaosScript:
    """The scripted chaos of ``--chaos`` (offline runner, tests, Webots autotest).

    ``drop_after`` control steps after the first LOCKED an obstacle is dropped
    on the robot's path, and ``kidnap_after`` steps after it the robot is
    kidnapped. Each hook is retried after every result until it is accepted
    (for example once a docking animation is over); the obstacle is given up
    after ``give_up_after`` more steps without a suitable spot.

    Usage: ``script = ChaosScript()``, then ``script.after_step(mission)``
    after every ``mission.step()``.
    """

    def __init__(self, drop_after: int = 10, kidnap_after: int = 40, give_up_after: int = 40):
        self.drop_after, self.kidnap_after, self.give_up_after = (
            drop_after,
            kidnap_after,
            give_up_after,
        )
        self._drop_at: int | None = None
        self._kidnap_at: int | None = None
        self.fired: list[str] = []  # hooks fired so far, in order

    def after_step(self, mission: Mission) -> list[str]:
        """Fire the hooks that are due; returns their names (``"drop_obstacle"``, ``"kidnap"``)."""
        fired: list[str] = []
        if mission.done:
            return fired
        steps = mission.stats.control_steps
        if self._drop_at is None and mission.stats.time_to_lock is not None:
            self._drop_at = steps + self.drop_after
            self._kidnap_at = steps + self.kidnap_after
        if self._drop_at is not None and self._drop_at >= 0 and steps >= self._drop_at:
            if mission.drop_obstacle() is not None:
                fired.append("drop_obstacle")
                self._drop_at = -1
            elif steps > self._drop_at + self.give_up_after:
                self._drop_at = -1
        if (
            self._kidnap_at is not None
            and self._kidnap_at >= 0
            and steps >= self._kidnap_at
            and mission.kidnap()
        ):
            fired.append("kidnap")
            self._kidnap_at = -1
        self.fired += fired
        return fired


def run_to_completion(
    mission: Mission,
    chaos: bool = False,
    forklifts: bool = False,
    on_result: Callable[[StepResult], None] | None = None,
    chaos_script: ChaosScript | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Step a mission until it is done (offline runner, tests, autograder).

    ``forklifts=True`` turns forklift traffic on first. ``chaos=True`` runs
    a :class:`ChaosScript` (or the given ``chaos_script``): an obstacle drop
    10 control steps after the first LOCKED and a kidnapping 40 steps after
    it. ``should_stop`` (optional) is polled after every result; returning
    True abandons the mission (the summary then shows it unfinished).
    """
    if forklifts:
        mission.set_forklifts(True)
    script = chaos_script or (ChaosScript() if chaos else None)
    while not mission.done:
        result = mission.step()
        if on_result is not None:
            on_result(result)
        if should_stop is not None and should_stop():
            break
        if script is not None:
            script.after_step(mission)
    return mission.summary()
