"""The warehouse show: presentation state machine and controller main loop.

``warehouse_mission.py`` calls :func:`main`. The mission itself is
:class:`wm.core.Mission`; this module only decides *when* to call
``mission.step()`` and plays every returned ``StepResult`` in the 3D world
(VIZ_SPEC section 3.3):

``BOOT`` (lights power up, establishing flyover, SYSTEM CHECK; SPACE skips
it) -> ``BRIEFING`` (camera preview, wait for SPACE; E = experiment) ->
``MISSION`` (the countdown, then one result after the other) -> ``END``
(celebration and score, or the failure screen) | ``FAULT`` (student code
error) | ``IMPORT_ERROR``. R / N restart with the same / a new mission code,
reusing the imported ``project4.py``; after a FAULT or an import error they
re-import it (so "fix, save, R" works without resetting Webots).

Keys (click the 3D view first; edge-triggered and debounced): SPACE, D,
1-5 and 0 (cameras), V, B, M (Robot Vision layers), K, O, F (chaos), L
(night), T (turbo), P (photo), S (sound), H (help), R, N.

Controller arguments (``controllerArgs`` of ``DEF AMR``, else the environment
variable ``WAREHOUSE_ARGS``): ``--student-code PATH``, ``--seed N``,
``--orders N``, ``--autotest``, ``--chaos``, ``--screenshots DIR``,
``--no-sound``, ``--key-script "t:KEY,..."`` (simulated key presses at
simulation times, e.g. ``"20:V,21.5:B,30:SPACE"``), ``--no-grid``; and, for
staff: ``--camera N`` (initial camera mode), ``--forklifts`` (traffic from the
start), ``--no-zone`` (the book's two sensors only), ``--sound`` (play sounds
in ``--autotest``, which is silent by default), ``--volume V`` (0..1),
``--realtime`` (run ``--autotest`` in REAL_TIME mode to measure the real-time
factor; it runs in FAST mode otherwise), ``--shot-set docs`` (with
``--screenshots``: stage the student guide's pictures instead of the QA set),
``--trace CSV`` (one line per frame: camera, robot, eye and frame time, for
checking the smoothness of the show offline).

The HUD (``hud.py``) and the Robot Vision renderers (``hologram.py``) are
regular parts of the show. Every call into them goes through :class:`Guarded`:
an exception is reported once with its traceback (and fails ``--autotest``)
instead of stopping the controller. Only if one of them cannot even be
imported or built (a broken installation) does the mission run with inert
stand-ins, with a loud console warning (``--autotest`` then FAILs too).
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
import time
import traceback
import types
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from controller import Keyboard, Supervisor

from .. import core
from ..core import config as C
from . import util
from .audio import Audio
from .camera import (
    MODE_AUTO,
    MODE_FREE,
    MODE_NAMES,
    CameraDirector,
    CameraInput,
    briefing_holds,
    overhead_shot,
)
from .scene import (
    CONVEYOR_SPEED,
    CONVEYOR_Z,
    DECK_LIFT,
    DECK_TOP,
    PARCEL_GLOW,
    RACK_Z,
    Scene,
    drop_height,
)

# The HUD and the renderers. A failed import is a broken installation: the
# mission still runs (with NullHud / NullHologram) so that the student sees the
# console message, and --autotest FAILs (see App.__init__).
_IMPORT_PROBLEMS: list[str] = []
try:
    from .hud import Hud
except Exception:  # noqa: BLE001 - last-resort safety net
    Hud = None
    _IMPORT_PROBLEMS.append("wm/viz/hud.py:\n" + traceback.format_exc().rstrip())
try:
    from .hologram import HoloFrame, HologramRenderer, HoloLayers
except Exception:  # noqa: BLE001 - last-resort safety net
    HoloFrame = HoloLayers = HologramRenderer = None
    _IMPORT_PROBLEMS.append("wm/viz/hologram.py:\n" + traceback.format_exc().rstrip())

BANNER = "CS 3630 PROJECT 4 - AUTONOMOUS WAREHOUSE CHALLENGE"
DOCKS_BY_NAME = {dock.name: dock for dock in core.DOCKS}
MOVING_KINDS = ("move", "dock", "undock")
# Events whose result carries a fresh uniform particle set (weighed by one measurement).
FRESH_PARTICLES = ("DEPLOYED", "LOST", "RESEED", "RELOCALIZING")
BELIEF_LAYERS = ("particles", "off")  # B: the particles on or off
MAP_LAYERS = (
    "none",
    "value",
    "change",
    "likelihood",
    "zones",
)  # "change": only after an obstacle drop
CHANGE_LAYER_SECONDS = 6.0  # OBSTACLE TEST: how long the value-change layer shows by itself
# The deployment's global scan (a 1 s "wait" result) plays in slow motion: the 2000 equal guesses
# appear all over the floor, then the first measurement weighs them (the HUD clock slows down too).
DEPLOY_SLOW_MOTION = 0.5
LOOK_AHEAD = 3.0  # m: a LOCKED robot looks at the point of its planned path this far ahead
EYE_TURN_RATE = math.radians(360.0)  # rad/s: the fastest turn of the robot's eye
# Events worth a console line (as in tests/run_mission.py).
LOGGED_EVENTS = {
    "LOCKED",
    "LOST",
    "RESEED",
    "RECOVERED",
    "SAFETY_STOP",
    "WAITING",
    "FORKLIFT_YIELD",
    "MARKER_ACQUIRED",
    "DOCKING_FAILED",
    "PACKAGE_LOADED",
    "DELIVERED",
    "NEW_ORDER",
    "OBSTACLE_DROPPED",
    "REPLANNED",
    "KIDNAPPED",
    "FORKLIFTS",
    "GRID_FILTER_OFF",
    "SLOW_CODE",
    "MISSION_COMPLETE",
    "MISSION_FAILED",
    "FAULT",
    "DEPLOYED",
}
BRIEFING_HINTS = [  # each <= 48 characters (the briefing card's width)
    "V VISION   B PARTICLES   M MAP LAYER",
    "1-5 CAMERAS (5 = AUTO)   0 FREE CAMERA",
    "K KIDNAP   O OBSTACLE   F FORKLIFTS",
    "L NIGHT SHIFT   T TURBO   H HELP",
]
GROUP_HINTS = {  # the briefing hints of each experiment group: each <= 48 characters
    "sensors": [
        "WATCH THE PARTICLE CLOUD: HOW BIG DOES IT GET?",
        "RUN ALL 5: THE END TABLE COLLECTS THEM",
    ],
    "start": ["WATCH THE PARTICLES FIND THE ROBOT", "RUN BOTH: THE END TABLE COLLECTS THEM"],
    "penalty": [
        "PRESS M: THE VALUE FUNCTION ON THE FLOOR",
        "RUN BOTH: THE END TABLE COLLECTS THEM",
    ],
}
GROUP_MENU = {"sensors": "SENSORS", "start": "START", "penalty": "PENALTY"}  # the 1-N menu's name
TOUR_VIEW_MARGIN = 3.0  # m: the OVERHEAD camera of a sensor test frames its route plus this margin


def tour_area(route) -> tuple[float, float, float, float]:
    """The part of the floor (x0, y0, x1, y1) that the OVERHEAD camera frames in a sensor test."""
    xs, ys = [float(p[0]) for p in route], [float(p[1]) for p in route]
    m = TOUR_VIEW_MARGIN
    return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)


def tour_end_area(route) -> tuple[float, float, float, float]:
    """The end screen of a test: the floor around ``route`` (its points) with room on the left and
    on
    top, so the route (the trails, the final gap) sits in the lower right, beside the results table.
    """
    x0, y0, x1, y1 = tour_area(route)
    return (x0 - 20.0, y0, x1, y1 + 12.0)


def group_tests(group: str) -> list:
    """The scenarios of an experiment group, in menu order (keys 1-N)."""
    return [sc for sc in core.SCENARIOS if sc.group == group]


FAULT_HINTS = {  # R re-imports project4.py after a fault (see App.restart)
    "missing_todos": "Implement the missing TODOs in project4.py, save, then press R to retry.",
    "exception": "Fix project4.py (the console shows the traceback), save, then press R to retry.",
    "import_error": "Fix project4.py, save, then press R to retry.",
    "engine_error": (
        "Internal engine error: please post the mission code and the console output on the forum."
    ),
}


def log(message: str) -> None:
    print(message, flush=True)


###############################################################################
#          Guards for the HUD / hologram, and last-resort stand-ins           #
###############################################################################


class NullHud:
    """Inert stand-in for ``hud.Hud`` (only if it cannot be imported or built): every call does
    nothing.
    """

    aspect = 16.0 / 9.0

    def __init__(self, supervisor=None):
        pass

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class NullHologram:
    """Inert stand-in for ``hologram.HologramRenderer`` (only if it cannot be imported or built)."""

    def render_floor(self, frame):
        return None

    def render_minimap(self, frame):
        return None


class Guarded:
    """Wraps another module's object: exceptions are reported once per method and counted.

    A method that keeps failing (``max_failures``) is switched off so that a
    bug cannot flood the console every frame. ``errors`` (shared) feeds the
    ``--autotest`` verdict.
    """

    def __init__(self, obj: Any, label: str, errors: dict[str, int], max_failures: int = 25):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_errors", errors)
        object.__setattr__(self, "_max", max_failures)

    def __getattr__(self, name: str):
        attr = getattr(self._obj, name)
        if not callable(attr):
            return attr
        key = f"{self._label}.{name}"
        errors, limit = self._errors, self._max

        def call(*args, **kwargs):
            if errors.get(key, 0) >= limit:
                return None
            try:
                return attr(*args, **kwargs)
            except Exception:  # noqa: BLE001 - report, count, carry on
                errors[key] = errors.get(key, 0) + 1
                if errors[key] == 1:
                    log(
                        f"warehouse_mission: exception in "
                        f"{key}():\n{traceback.format_exc().rstrip()}"
                    )
                elif errors[key] == limit:
                    log(f"warehouse_mission: {key}() failed {limit} times; it is switched off.")
                return None

        object.__setattr__(self, name, call)  # cache: later lookups skip __getattr__
        return call


###############################################################################
#                                  Keyboard                                   #
###############################################################################


def parse_key_script(text: str) -> list[tuple[float, int]]:
    """``"12.5:V,14:B,20:SPACE"`` -> [(12.5, ord('V')), ...] sorted by time."""
    out = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        t_text, _, key = item.partition(":")
        try:
            t = float(t_text)
        except ValueError:
            log(f"warehouse_mission: bad --key-script item {item!r} (expected TIME:KEY)")
            continue
        name = key.strip().upper()
        if name in ("SPACE", "SP", "") and key:
            code = 32
        elif len(name) == 1:
            code = ord(name)
        else:
            log(f"warehouse_mission: unknown key {key!r} in --key-script")
            continue
        out.append((t, code))
    return sorted(out)


class KeyInput:
    """Edge-triggered, debounced keys (Webots reports held keys every step), plus a key script."""

    DEBOUNCE = (
        0.15  # s of wall time between two presses of the same key (longer than OS auto-repeat)
    )

    def __init__(self, supervisor, time_step: int, script: list[tuple[float, int]]):
        self.keyboard = supervisor.getKeyboard()
        self.keyboard.enable(time_step)
        self.held: set = set()
        self.last_press: dict[int, float] = {}
        self.script = list(script)

    def poll(self, t: float) -> list[int]:
        pressed = set()
        for _ in range(7):
            key = self.keyboard.getKey()
            if key < 0:
                break
            if key & (Keyboard.CONTROL | Keyboard.ALT):
                continue  # leave Ctrl/Cmd/Alt shortcuts to Webots
            key &= Keyboard.KEY
            if 97 <= key <= 122:
                key -= 32
            pressed.add(key)
        new = pressed - self.held
        self.held = pressed
        now = time.monotonic()
        out = []
        for key in sorted(new):
            if now - self.last_press.get(key, -1e9) >= self.DEBOUNCE:
                self.last_press[key] = now
                out.append(key)
        while self.script and self.script[0][0] <= t:
            out.append(self.script.pop(0)[1])
        return out


###############################################################################
#                              Performance log                                #
###############################################################################

SECTIONS = ("logic", "scene", "camera", "holo", "hud", "other")


class Perf:
    """Per-frame Python time (ms): plain frames, frames that ran ``mission.step()`` (result
    boundaries), and frames that rendered a hologram image; plus the costs of the parts."""

    def __init__(self):
        self.frames: dict[str, list[float]] = {"plain": [], "step": [], "holo": []}
        self.step_ms: list[float] = []
        self.floor_ms: list[float] = []
        self.mini_ms: list[float] = []
        self.paste_ms: list[float] = []
        # Where a frame's time goes: logic (keys, the state machine, mission.step, the robot's
        # motion, telemetry), scene (robot looks and effects), camera, holo (renders and pastes),
        # hud (labels), other (sound, photo, screenshots, autotest).
        self.sections: dict[str, list[float]] = {name: [] for name in SECTIONS}
        self.wall0 = time.perf_counter()
        self.sim0: float | None = None

    @staticmethod
    def _stats(values: list[float]) -> str:
        if not values:
            return "-"
        v = sorted(values)
        p95 = v[min(len(v) - 1, int(0.95 * len(v)))]
        return f"mean {sum(v) / len(v):.2f} ms, p95 {p95:.2f} ms, max {v[-1]:.1f} ms (n={len(v)})"

    def report(self) -> str:
        every = [ms for values in self.frames.values() for ms in values]
        return (
            "VIZ PERF  every frame: "
            + self._stats(every)
            + "\n          plain frames: "
            + self._stats(self.frames["plain"])
            + "\n          result boundaries (frames running mission.step): "
            + self._stats(self.frames["step"])
            + "\n          hologram frames (one render + paste): "
            + self._stats(self.frames["holo"])
            + "\n          mission.step(): "
            + self._stats(self.step_ms)
            + "\n          render_floor: "
            + self._stats(self.floor_ms)
            + "\n          render_minimap: "
            + self._stats(self.mini_ms)
            + "\n          floor image paste: "
            + self._stats(self.paste_ms)
            + "\n          per frame, by part: "
            + ", ".join(
                f"{name} {sum(v) / max(len(v), 1):.2f}" for name, v in self.sections.items()
            )
            + " ms (means)"
        )


###############################################################################
#                                 Arguments                                   #
###############################################################################


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="warehouse_mission", add_help=False)
    parser.add_argument("--student-code", default=None, metavar="PATH")
    parser.add_argument("--seed", type=int, default=None, metavar="CODE")
    parser.add_argument("--orders", type=int, default=1, help="staff QA: deliveries per mission")
    parser.add_argument(
        "--scenario",
        default="auto",
        choices=("auto",) + tuple(core.SCENARIOS_BY_KEY),
        help="preselect an experiment (E at the briefing switches it)",
    )
    parser.add_argument("--autotest", action="store_true")
    parser.add_argument("--chaos", action="store_true")
    parser.add_argument("--screenshots", default=None, metavar="DIR")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--sound", action="store_true", help="play sounds even in --autotest")
    parser.add_argument(
        "--volume", type=float, default=None, help="global volume 0..1 (default 1; 0 in --autotest)"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="--autotest in REAL_TIME mode (measures the RTF)"
    )
    parser.add_argument("--key-script", default="")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--no-zone", action="store_true")
    parser.add_argument("--forklifts", action="store_true")
    parser.add_argument("--camera", type=int, default=MODE_AUTO, choices=sorted(MODE_NAMES))
    parser.add_argument(
        "--shot-set",
        default="qa",
        choices=("qa", "docs"),
        help="--screenshots: the QA set (default) or the student guide's pictures",
    )
    parser.add_argument(
        "--trace",
        default=None,
        metavar="CSV",
        help="staff QA: write one line per frame (camera, robot, eye, timing) to CSV",
    )
    if argv is None:
        argv = sys.argv[1:]
        if not argv:
            text = os.environ.get("WAREHOUSE_ARGS", "")
            try:
                argv = shlex.split(text)
            except ValueError:  # unbalanced quotes: best effort
                log(
                    f"warehouse_mission: could not parse WAREHOUSE_ARGS={text!r}; splitting "
                    f"on spaces."
                )
                argv = text.split()
    try:
        args, unknown = parser.parse_known_args(argv)
    except SystemExit:  # argparse reports the problem on stderr; never kill the controller
        log(f"warehouse_mission: could not parse the arguments {argv}; using the defaults.")
        args, unknown = parser.parse_known_args([])
    if unknown:
        log(f"warehouse_mission: ignoring unknown arguments {unknown}")
    return args


###############################################################################
#                                  The app                                    #
###############################################################################


class App:
    """The presentation state machine (VIZ_SPEC section 3.3)."""

    def __init__(self, args: argparse.Namespace, controller_dir: Path):
        self.args = args
        self.sup = Supervisor()
        self.ts = int(self.sup.getBasicTimeStep())
        self.dt = self.ts / 1000.0
        self.controller_dir = controller_dir
        self.webots_dir = controller_dir.parents[1]
        self.student_path = (
            Path(args.student_code).expanduser()
            if args.student_code
            else controller_dir.parents[2] / "project4.py"
        )
        self.errors: dict[str, int] = {}  # exceptions in this layer (autotest verdict)
        self.perf = Perf()
        self.t = 0.0
        self.frame_kind = (
            "plain"  # this frame: plain | step (ran mission.step) | holo (rendered an image)
        )

        volume = args.volume
        if volume is None:  # autotests exercise the audio API silently
            volume = 0.0 if (args.autotest and not args.sound) else 1.0
        self.audio = Audio(
            self.sup,
            controller_dir / "sounds",
            enabled=not args.no_sound,
            volume_scale=util.clamp(volume, 0.0, 1.0),
        )
        # The HUD first: it creates its labels at construction, and Webots draws labels in creation
        # order.
        self.hud = Guarded(self._build("hud", Hud, self.sup) or NullHud(), "hud", self.errors)
        self.holo = Guarded(
            self._build("hologram", HologramRenderer) or NullHologram(), "hologram", self.errors
        )
        self.layers = (
            HoloLayers()
            if HoloLayers is not None
            else types.SimpleNamespace(vision=True, belief="particles", map_layer="none")
        )
        self.scene = Scene(self.sup, self.audio)
        self.camera = CameraDirector(self.sup, self.scene)
        self.floor_display = self.sup.getDevice("floor_holo")
        self.mini_display = self.sup.getDevice("minimap")
        self.keys = KeyInput(self.sup, self.ts, parse_key_script(args.key_script))
        self.shots: ShotPlanner | None = None
        if args.screenshots:
            planner = DocsShotPlanner if args.shot_set == "docs" else ShotPlanner
            self.shots = planner(self, Path(args.screenshots).expanduser().resolve())

        # Mission and presentation state.
        self.mission: core.Mission | None = None
        self.bridge = None
        self.orders_n = max(1, int(args.orders))
        self.scenario_key = str(args.scenario)  # "auto" until E picks one (R and N keep it)
        picked = core.SCENARIOS_BY_KEY.get(self.scenario_key)
        self.choices: dict[str, int] = {}  # per experiment group: the test picked with keys 1-N
        if picked is not None and picked.group:
            self.choices[picked.group] = group_tests(picked.group).index(picked)
        self.results: dict[
            str, dict[str, dict]
        ] = {}  # per group: this session's tests (the end table)
        self.change_until = -1e9  # OBSTACLE TEST: the value-change layer shows until then
        self.forklift_traffic = bool(args.forklifts)
        self.chaos = core.ChaosScript() if args.chaos else None
        self.state = "INIT"
        self.state_t0 = 0.0
        self.deployed = False
        self.result = None
        self.play_t = 0.0  # seconds of the current result played (result time)
        self.play_rate = (
            1.0  # result seconds per simulated second (DEPLOY_SLOW_MOTION for the global scan)
        )
        self.boot_result = None
        self.boot_lines: list[tuple[str, str]] = []
        self.boot_full = True
        self.brief_wait = math.inf
        self.pending: list[tuple[float, Callable[[], None]]] = []
        self.help_visible = False
        self.photo: dict | None = None
        self.verdict_at: float | None = None
        self.quitting = False
        self.fault_info: dict | None = None
        # Robot and effects.
        self.robot_xy = (0.0, 0.0)
        self.robot_z = 0.0
        self.psi = 0.0  # eye pan (rad)
        self.look_vec = (0.0, 0.0)  # smoothed travel direction (the eye's fallback target)
        self.travel: tuple[float, float] | None = None
        self.actor_from: dict[int, tuple[float, float, float]] = {}
        self.actor_to: dict[int, tuple[float, float, float]] = {}
        self.active_parcel = 1
        self.parcel_on_robot = False
        self.flash: tuple[tuple[float, float, float], float] | None = None
        self.zone_t = -1e9
        self.trail: dict[str, list] | None = None
        self.trail_next = None
        self.last_ping: tuple[int, float, float] | None = None
        self.scan_psi0: float | None = None
        self.prev_score = float(C.SCORE_BASE)
        self.odometer = 0.0  # meters driven (score screen)
        self.tele_base: dict[str, Any] = {}
        self.said: set = set()
        self.fail_text = ""  # the MISSION_FAILED event's text
        self.smoothed: dict[str, Any] | None = (
            None  # TODO 17 at MISSION COMPLETE (core smoother.py)
        )
        self.smoothed_t0 = 0.0
        # Hologram scheduling.
        self.floor_due = False
        self.mini_due = False
        self.holo_last = -1e9
        self.floor_t = -1e9  # time of the last floor render
        self.weigh_result = (
            None  # the result whose fresh particle set is being "weighed" (HoloFrame.weigh)
        )
        self.floor_ms_last = 0.0
        self.mini_ms_last = 0.0
        self.near = 0.0  # HoloFrame.near: 0 = the camera sees the whole warehouse, 1 = a close shot
        self.reticle_angle = 0.0
        self.lock_t = -1e9  # time of the last LOCKED event (the reticle snaps shut)
        self.robot_xy = self.scene.robot_xyz[
            :2
        ]  # where the world put the robot (until a mission places it)
        # Real-time factor (HUD "speed_factor").
        self.rtf = 1.0
        self._rtf_mark = (time.perf_counter(), 0.0)
        self._trace_files = ExitStack()
        self.trace = None  # --trace: per-frame QA log
        if args.trace:
            try:
                self.trace = self._trace_files.enter_context(
                    Path(args.trace).expanduser().open("w")
                )
                self.trace.write(
                    "t,state,shot,eye_x,eye_y,eye_z,target_x,target_y,target_z,fov,"
                    "robot_x,robot_y,robot_z,psi,kind,play_t,frame_ms,frame_kind,"
                    "shot_x,shot_y,shot_z,retrack,blend,wall\n"
                )
            except OSError as exc:
                log(f"warehouse_mission: cannot write --trace {args.trace}: {exc}")

    def _build(self, name: str, cls, *args):
        """Build the HUD or the renderer; None (with a loud report) if it cannot be imported or
        built.
        """
        problem = next((p for p in _IMPORT_PROBLEMS if f"/{name}.py" in p), None)
        if cls is not None:
            try:
                return cls(*args)
            except Exception:  # noqa: BLE001 - last-resort safety net
                problem = (
                    f"wm/viz/{name}.py ({cls.__name__}() failed):\n"
                    + traceback.format_exc().rstrip()
                )
        self.errors[f"{name}.unavailable"] = 1  # --autotest FAILs
        log("=" * 78)
        log(f"warehouse_mission: ERROR, the {name} is unavailable; the mission runs without it.")
        log(problem or f"wm/viz/{name}.py: not loaded")
        log("Re-download the project files (wm/viz must not be modified).")
        log("=" * 78)
        return None

    # ======================================================================
    # Main loop
    # ======================================================================
    def run(self) -> None:
        if self.args.realtime:
            self.sup.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
        elif self.args.autotest or self.shots is not None:
            self.sup.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        self.camera.set_mode(self.args.camera if self.args.camera in MODE_NAMES else MODE_AUTO)
        self.clear_displays()
        if self.create_mission(self.args.seed):
            self.reset_scene()
            if self.args.autotest:  # (QA: compare with the counts logged at R / N restarts)
                log(f"World nodes: {self.scene.root_children.getCount()}")
            self.enter_boot(full=True)
        else:
            self.enter_import_error_boot()
        while self.sup.step(self.ts) != -1:
            if self.quitting:
                continue
            start = time.perf_counter()
            self.frame_kind = "plain"
            try:
                self.frame()
            except Exception:  # noqa: BLE001 - a bug here must not freeze the show
                self.report_exception("frame")
            ms = 1000.0 * (time.perf_counter() - start)
            self.perf.frames[self.frame_kind].append(ms)
            if self.trace is not None:
                self.write_trace(ms)

    def report_exception(self, where: str) -> None:
        tb = traceback.format_exc()
        key = where + ":" + tb.strip().splitlines()[-1][:120]
        self.errors[key] = self.errors.get(key, 0) + 1
        if self.errors[key] == 1:
            log(f"warehouse_mission: exception in {where}:\n{tb.rstrip()}")

    def write_trace(self, frame_ms: float) -> None:
        """--trace: one CSV line per frame (see the header written in __init__)."""
        eye, target, fov = self.camera.out
        r = self.result
        fields = [f"{self.t:.3f}", self.state, self.camera.shot_name.replace(" ", "_")]
        fields += [
            f"{v:.3f}"
            for v in (
                *eye,
                *target,
                fov,
                self.robot_xy[0],
                self.robot_xy[1],
                self.robot_z,
                self.psi,
            )
        ]
        fields += [
            r.kind if r is not None else "-",
            f"{self.play_t:.3f}",
            f"{frame_ms:.2f}",
            self.frame_kind,
        ]
        shot_eye, retracked, blending = self.camera.debug
        fields += [f"{v:.3f}" for v in shot_eye] + [str(int(retracked)), str(int(blending))]
        fields.append(
            f"{time.perf_counter() - self.perf.wall0:.4f}"
        )  # wall clock (s), for real-time factors
        self.trace.write(",".join(fields) + "\n")

    def frame(self) -> None:
        clock = time.perf_counter
        sections = self.perf.sections
        c0 = clock()
        t = self.sup.getTime()
        dt = self.dt
        self.t = t
        if self.perf.sim0 is None:
            self.perf.sim0 = t
        for key in self.keys.poll(t):
            self.on_key(key)
        self.run_pending(t)
        state = self.state
        if state == "BOOT":
            self.frame_boot(t)
        elif state == "BRIEFING":
            self.frame_briefing(t)
        elif state == "MISSION":
            self.frame_mission(t, dt)
        elif state == "IMPORT_BOOT":
            self.frame_import_boot(t)
        c1 = clock()
        self.update_robot_look(t, dt)
        self.update_doors()
        self.update_parcel_glow(t)
        self.scene.update(dt, t)
        c2 = clock()
        self.camera.aspect = self.hud.aspect or self.camera.aspect
        self.camera.update(
            dt,
            t,
            CameraInput(
                (self.robot_xy[0], self.robot_xy[1], self.robot_z), self.travel, self.current_dock()
            ),
        )
        self.near = self.camera_near(t)
        self.update_reticle(t, dt)
        c3 = clock()
        self.update_hologram(t)
        c4 = clock()
        self.hud.tick(dt)
        c5 = clock()
        self.audio.update()
        self.update_photo()
        self.update_rtf(t)
        if self.shots is not None:
            self.shots.update(t)
        self.check_autotest(t)
        c6 = clock()
        for name, a, b in (
            ("logic", c0, c1),
            ("scene", c1, c2),
            ("camera", c2, c3),
            ("holo", c3, c4),
            ("hud", c4, c5),
            ("other", c5, c6),
        ):
            sections[name].append(1000.0 * (b - a))

    def later(self, delay: float, action: Callable[[], None]) -> None:
        self.pending.append((self.t + delay, action))

    def run_pending(self, t: float) -> None:
        if not self.pending:
            return
        due = [p for p in self.pending if p[0] <= t]
        if due:
            self.pending = [p for p in self.pending if p[0] > t]
            for _, action in due:
                action()

    def update_rtf(self, t: float) -> None:
        wall = time.perf_counter()
        w0, s0 = self._rtf_mark
        if wall - w0 >= 0.5:
            self.rtf = (t - s0) / max(wall - w0, 1e-6)
            self._rtf_mark = (wall, t)

    # ======================================================================
    # Mission creation and scene reset
    # ======================================================================
    def options(self) -> core.MissionOptions:
        # (no Markov grid filter; --no-grid also skips the sensor-likelihood layer of key M)
        return core.MissionOptions(
            grid_filter=False,
            likelihood_layer=not self.args.no_grid,
            forklifts=self.forklift_traffic,
            zone=not self.args.no_zone,
        )

    def create_mission(self, code: int | None) -> bool:
        """Build a Mission (importing the student's file the first time). False on
        StudentImportError.
        """
        try:
            student = self.bridge if self.bridge is not None else str(self.student_path)
            self.mission = core.Mission(
                student,
                code=code,
                orders=self.orders_n,
                options=self.options(),
                scenario=self.scenario_key,
            )
        except core.StudentImportError as exc:
            self.mission = None
            detail = traceback.format_exc()
            cause = exc.__cause__
            if cause is not None:
                detail = "".join(
                    traceback.format_exception(type(cause), cause, cause.__traceback__)
                )
            log("=" * 78)
            log(f"STUDENT CODE ERROR: {exc}")
            log(detail.rstrip())
            log("=" * 78)
            self.fault_info = {
                "reason": "import_error",
                "function": None,
                "todo": None,
                "message": str(exc),
                "missing": [],
                "details": detail,
                "path": str(self.student_path),
                "hint": FAULT_HINTS["import_error"],
            }
            return False
        self.bridge = self.mission.bridge
        m = self.mission
        log("=" * 78)
        log(f" {BANNER}")
        log("=" * 78)
        info = m.student_info or {}
        log(f"Student code : {self.bridge.path}")
        log(f"Student      : {info.get('name', '?')} ({info.get('gt_username', '?')})")
        log(f"Robot        : {m.profile['name']} ({m.profile['color']})")
        orders = "; ".join(f"pick up {o.slot.name}, deliver to {o.dock.name}" for o in m.orders)
        log(f"Mission code : {m.code:04d}   {len(m.orders)} order(s): {orders}")
        log(f"               (replay offline: python tests/run_mission.py --seeds {m.code})")
        return True

    def reset_scene(self) -> None:
        m = self.mission
        layout = m.layout()
        self.scene.reset(m.start, m.profile.get("color", "gold"), layout["forklift_routes"])
        self.scene.import_parcels(m.orders)
        self.robot_xy = (float(m.start[0]), float(m.start[1]))
        self.robot_z = 0.0
        self.psi = 0.0
        self.look_vec = (0.0, 0.0)
        self.travel = None
        self.result = None
        self.play_t = 0.0
        self.play_rate = 1.0
        self.deployed = False
        self.active_parcel = m.orders[0].number
        self.parcel_on_robot = False
        self.flash = None
        self.zone_t = -1e9
        self.trail: dict[str, list] | None = (
            None  # a sensor test: the true and believed paths so far
        )
        self.trail_next = None  # (true end, believed end) of the result playing now
        self.last_ping = None
        self.scan_psi0 = None
        self.prev_score = float(C.SCORE_BASE)
        self.odometer = 0.0
        self.tele_base = {}
        self.fail_text = ""
        self.smoothed = None
        self.pending = []
        self.weigh_result = None
        self.lock_t = -1e9
        self.actor_to = {}
        for k, route in enumerate(layout["forklift_routes"][:2]):
            (x0, y0), (x1, y1) = route[0], route[1]
            self.actor_to[k] = (float(x0), float(y0), math.atan2(y1 - y0, x1 - x0))
        self.actor_from = dict(self.actor_to)
        self.floor_due = self.mini_due = True

    def clear_displays(self) -> None:
        """The floor_holo plane renders opaque white until something is drawn: clear it (alpha
        0).
        """
        for display in (self.floor_display, self.mini_display):
            if display is None:
                continue
            display.setAlpha(0.0)
            display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
            display.setAlpha(1.0)

    def restart(self, same_code: bool) -> None:
        """R (same code) / N (new code) (VIZ_SPEC 3.3 step 8).

        A restart reuses the imported student module (fast: the SYSTEM CHECK is
        cached), except after a FAULT or an import error: then R *retries*, i.e.
        re-imports ``project4.py``, so a student can fix the file, save, and press
        R instead of resetting the whole simulation.
        """
        retry = self.state in ("FAULT", "IMPORT_ERROR", "IMPORT_BOOT")
        # After an import error, keep the requested code.
        code = self.args.seed
        if self.mission is not None:
            code = self.mission.code if same_code else None
        if retry:
            self.bridge = None
        if self.mission is not None:
            self.orders_n = len(self.mission.orders)
        self.pending = []
        self.photo_abort()
        self.hud.clear()
        self.help_visible = False
        self.scene.set_panel_dim(False)
        self.fault_info = None
        self.verdict_at = None
        self.said.clear()
        self.scene.stop_celebration()
        if self.chaos is not None:
            self.chaos = core.ChaosScript()
        if not self.create_mission(code):
            self.enter_import_error_boot()
            return
        self.reset_scene()
        nodes = (
            self.scene.root_children.getCount()
        )  # (QA: imported parcels/obstacles must not pile up)
        if retry:
            log(
                f"RETRY: project4.py re-imported, mission code {self.mission.code:04d} "
                f"({nodes} world nodes)"
            )
        else:
            log(
                f"RESTART: mission code {self.mission.code:04d} (the loaded project4.py is reused; "
                f"press Webots' Reset Simulation to load an edited file) ({nodes} world nodes)"
            )
        self.enter_boot(full=False)

    # ======================================================================
    # BOOT
    # ======================================================================
    def enter_boot(self, full: bool) -> None:
        t = self.t
        self.state, self.state_t0, self.boot_full = "BOOT", t, full
        if full:
            # The self test types itself in the dark (the text reads best on the dark building);
            # once it is complete the lamp rows power up (with clicks), while the establishing
            # flyover rises over the wall into the hall.
            reveal_start, reveal_span, _ = self.boot_timing()
            self.scene.lights_off()
            self.scene.power_up(t + reveal_start + reveal_span + 0.1, row_gap=0.2)
            if self.camera.mode != MODE_FREE:
                self.camera.start_script("boot", t, snap=True, duration=5.4)
        elif self.camera.mode != MODE_FREE:
            self.camera.start_script("idle", t)
        # Run BOOT right away, before any animation starts, so its ~0.15 s never stutters one.
        start = time.perf_counter()
        self.boot_result = self.mission.step()
        self.perf.step_ms.append(1000.0 * (time.perf_counter() - start))
        self.frame_kind = "step"
        self.boot_lines = self.build_boot_lines(self.boot_result)
        self.print_system_check(self.boot_result)
        if full:
            self.later(0.25, lambda: self.audio.play("boot_chime"))
        else:
            self.audio.play("beep_go", pitch=0.8)
        self.floor_due = self.mini_due = True
        if self.shots is not None:
            self.shots.on_state("BOOT", t)

    def boot_timing(self) -> tuple[float, float, float]:
        """(reveal start, reveal duration, total) of the boot screen."""
        return (0.5, 3.2, 5.6) if self.boot_full else (0.1, 1.0, 1.8)

    def frame_boot(self, t: float) -> None:
        u = t - self.state_t0
        start, span, total = self.boot_timing()
        reveal = util.clamp((u - start) / span, 0.0, 1.0)
        m = self.mission
        title = f"SYSTEM CHECK  //  {m.profile['name']}  //  CODE {m.code:04d}"
        self.hud.boot(title, self.boot_lines, reveal)
        faulted = m.phase == core.FAULT
        if u >= total + (1.5 if faulted else 0.0):
            if faulted:
                self.enter_fault(m.fault)
            else:
                self.enter_briefing()

    @staticmethod
    def build_boot_lines(result) -> list[tuple[str, str]]:
        """SYSTEM CHECK rows for ``hud.boot`` (each text <= 34 characters; the console has the
        details).
        """

        def num(v: float) -> str:
            text = f"{float(v):.2f}"
            return "0.00" if text == "-0.00" else text

        lines: list[tuple[str, str]] = []
        for event in result.events:
            data = event.data
            if event.name == "SYSTEM_CHECK":
                for todo, name, status, _ in data.get("rows", []):
                    lines.append((f"TODO {int(todo):>2}  {name}", str(status)))
            elif event.name == "SYSID":
                lines.append(
                    (
                        f"SYSTEM ID: sigma_hat = {float(data.get('sigma', 0.0)):.3f} m",
                        "WARN" if data.get("fallback") else "INFO",
                    )
                )
                for label in ("A", "B"):
                    m = data.get(label)
                    if m is not None and len(m) == 2 and len(m[0]) == 2:
                        lines.append(
                            (
                                f"  {label}hat = [{num(m[0][0])} {num(m[0][1])}; "
                                f"{num(m[1][0])} {num(m[1][1])}]",
                                "INFO",
                            )
                        )
            elif event.name in ("WARNING", "GRID_FILTER_OFF"):
                lines.append((event.text[:34], "WARN"))
            elif event.name == "SLOW_CODE":
                lines.append(
                    (
                        f"SLOW CODE: {data.get('function', '?')} "
                        f"{float(data.get('seconds', 0.0)):.1f} "
                        f"s"[:34],
                        "WARN",
                    )
                )
            elif event.name == "PLAN_READY":
                lines.append((f"VALUE ITERATION  {float(data.get('seconds', 0.0)):.2f} s", "OK"))
            elif event.name == "PLANNING_OFF":
                lines.append(("PLANNING MISSING: EXPERIMENTS ONLY", "WARN"))
            elif event.name == "LOCALIZATION_OFF":
                lines.append(("LOCALIZATION MISSING: PENALTY ONLY", "WARN"))
            elif event.name == "FAULT":
                lines.append(("CANNOT DEPLOY: TODOS MISSING", "ERROR"))
        return lines

    def print_system_check(self, result) -> None:
        for event in result.events:
            if event.name == "SYSTEM_CHECK":
                log(
                    "SYSTEM CHECK (a quick probe of every TODO; MISSING = still raises "
                    "NotImplementedError)"
                )
                for todo, name, status, message in event.data.get("rows", []):
                    flag = ""
                    if status == core.MISSING and todo in core.LOCALIZATION_TODOS:
                        flag = "  <- needed to deploy"
                    elif status == core.MISSING and todo in core.PLANNING_TODOS:
                        flag = "  <- needed for MISSION / OBSTACLE TEST"
                    log(f"  {todo:>2}  {name:<22} {status:<8} {message}{flag}")
            elif event.name == "SYSID":
                log(
                    f"System ID: {event.data.get('message', '')} -> sigma = "
                    f"{event.data.get('sigma', 0.0):.3f} m"
                )
            elif event.name in ("WARNING", "GRID_FILTER_OFF"):
                message = str(event.data.get("message", event.data.get("reason", "")))
                log(
                    f"WARNING: {message}"
                    if message.startswith(event.text)
                    else f"WARNING: {event.text}: {message}"
                )
            elif event.name == "SLOW_CODE":
                log(
                    f"WARNING: {event.text} (more than {C.SLOW_CALL_SECONDS:.0f} s; the "
                    f"simulation waits for "
                    "your code: check it for Python loops over cells or particles)"
                )

    # ======================================================================
    # BRIEFING
    # ======================================================================
    def enter_briefing(self) -> None:
        t = self.t
        self.state, self.state_t0 = "BRIEFING", t
        m = self.mission
        order = m.orders[0]
        self.show_scenario_markers()
        self.show_briefing()
        if self.camera.mode != MODE_FREE:
            self.camera.start_script("briefing", t, holds=briefing_holds(order.slot, order.dock))
        if self.shots is not None:
            self.brief_wait = 11.0  # the whole preview: rack, dock, robot
        elif self.args.autotest:
            self.brief_wait = 1.5
        else:
            self.brief_wait = math.inf
        self.floor_due = self.mini_due = True
        if self.shots is not None:
            self.shots.on_state("BRIEFING", t)

    def show_scenario_markers(self) -> None:
        """Briefing: the order's rack and dock (a tour has none)."""
        m = self.mission
        sc = m.scenario
        if not sc.is_tour:
            order = m.orders[0]
            self.scene.show_marker("pickup", order.slot.station)
            self.scene.show_marker("dropoff", order.dock.station)
            return
        self.scene.hide_marker("pickup", animate=False)
        self.scene.hide_marker("dropoff", animate=False)

    def show_briefing(self) -> None:
        m = self.mission
        sc = m.scenario
        info = m.student_info or {}
        name = info.get("name", "")
        if not name or name.upper().startswith("TODO"):
            name = ""
        self.hud.briefing(
            {
                "code": m.code,
                "robot_name": m.profile["name"],
                "student_name": name,
                "orders": [] if sc.is_tour else [o.as_dict() for o in m.orders],
                "scenario": sc.key,
                "scenario_title": sc.title,
                "description": list(sc.description),
                "scenarios": self.scenario_groups(),
                "options": [test.option for test in group_tests(sc.group)] if sc.group else [],
                "option": sc.option,
                "option_title": GROUP_MENU.get(sc.group, "PICK"),
                "prompt": "PRESS SPACE TO START" if sc.group else "PRESS SPACE TO DEPLOY",
                "hints": list(GROUP_HINTS.get(sc.group, BRIEFING_HINTS)),
            }
        )

    def scenario_groups(self) -> list[tuple[str, str]]:
        """The briefing's [E] choices, (key, title): the mission, ONE entry per experiment group
        (the
        test picked with keys 1-N), and the obstacle test; only the tours while TODOs 12-15 are
        missing.
        """
        m = self.mission
        groups: list[tuple[str, str]] = []
        seen = set()
        for key in m.available_scenarios():
            sc = core.SCENARIOS_BY_KEY[key]
            if sc.group:
                if sc.group in seen:
                    continue
                seen.add(sc.group)
                tests = group_tests(sc.group)
                key = (
                    m.scenario.key
                    if m.scenario.group == sc.group
                    else tests[self.choices.get(sc.group, 0)].key
                )
            groups.append((key, sc.title))
        return groups

    def cycle_scenario(self) -> None:
        """E at the briefing: the next experiment (MISSION, START TEST, SENSOR TEST, ...)."""
        keys = [key for key, _ in self.scenario_groups()]
        current = self.mission.scenario.key
        if len(keys) > 1 or current not in keys:
            self.select_scenario(
                keys[(keys.index(current) + 1) % len(keys)] if current in keys else keys[0]
            )

    def choose_option(self, index: int) -> None:
        """Keys 1-N at the briefing of an experiment group: which of its tests (e.g. which
        sensors).
        """
        group = self.mission.scenario.group
        tests = group_tests(group)
        if not group or not 0 <= index < len(tests):
            return
        self.choices[group] = index
        if tests[index].key != self.mission.scenario.key:
            self.select_scenario(tests[index].key)

    def select_scenario(self, key: str) -> None:
        m = self.mission
        if not m.set_scenario(key):
            return
        sc = m.scenario
        self.scenario_key = key
        self.robot_xy = (float(m.start[0]), float(m.start[1]))
        self.scene.place_robot(self.robot_xy[0], self.robot_xy[1], 0.0)
        self.show_scenario_markers()
        self.show_briefing()
        self.hud.toast(
            f"{GROUP_MENU.get(sc.group, 'PICK')}: {sc.option}"
            if sc.group
            else f"EXPERIMENT: {sc.title}"
        )
        self.audio.play("ui_click")
        self.floor_due = self.mini_due = True
        n = len(group_tests(sc.group))
        log(
            f"Scenario     : {sc.title}"
            + (f": {sc.option}" if sc.group else "")
            + " (E at the briefing switches it"
            + (f", keys 1-{n} the test)" if sc.group else ")")
        )

    def frame_briefing(self, t: float) -> None:
        if t - self.state_t0 >= self.brief_wait:
            self.deploy()

    def deploy(self) -> None:
        """SPACE: the BRIEFING step; then the countdown and the mission play as results."""
        self.audio.play("beep_go")
        start = time.perf_counter()
        result = self.mission.step()
        self.perf.step_ms.append(1000.0 * (time.perf_counter() - start))
        self.frame_kind = "step"
        self.state, self.state_t0 = "MISSION", self.t
        self.deployed = False
        if self.camera.mode != MODE_FREE:
            self.camera.start_script("countdown", self.t)
        self.result, self.play_t = None, 0.0
        self.start_result(result)

    # ======================================================================
    # MISSION: playing StepResults
    # ======================================================================
    def frame_mission(self, t: float, dt: float) -> None:
        self.play_t += dt * self.play_rate
        for _ in range(16):  # zero-length results are consumed in the same frame
            r = self.result
            if r is not None and self.play_t < r.duration:
                break
            if r is not None:
                self.play_t -= r.duration
                self.finish_result(r)
            if self.mission.done:
                self.enter_end()
                return
            start = time.perf_counter()
            new = self.mission.step()
            self.perf.step_ms.append(1000.0 * (time.perf_counter() - start))
            self.frame_kind = "step"
            if self.chaos is not None:
                self.run_chaos()
            self.start_result(new)
        self.animate(t, dt)
        if self.deployed:
            self.hud.telemetry(self.telemetry(t))

    def run_chaos(self) -> None:
        fired = self.chaos.after_step(self.mission)
        for name in fired:
            log(f"  [{self.mission.clock:6.1f} s] CHAOS: {name}")
            if name == "drop_obstacle" and self.mission.world.obstacles:
                self.drop_obstacle_visual(self.mission.world.obstacles[-1].as_dict())

    def drop_obstacle_visual(self, info: dict) -> None:
        """Start the SpillBarrier drop right away (the engine reports OBSTACLE_DROPPED one result
        later).
        """
        if not self.scene.drop_obstacle(info, self.t):
            return
        if (
            self.camera.mode == MODE_AUTO and self.camera.script is None
        ):  # show the drop, not the report
            self.camera.auto_event(
                "OBSTACLE_DROPPED", self.t, focus=(float(info["x"]), float(info["y"]))
            )
        if self.shots is not None:
            self.shots.on_event("OBSTACLE_FALLING", info, self.t)

    def start_result(self, r) -> None:
        t = self.t
        self.result = r
        if self.mission is not None and self.mission.scenario.group:
            self.extend_trail(r)
        self.actor_from = dict(self.actor_to)
        for actor in r.actors:
            self.actor_to[int(actor["index"])] = (
                float(actor["x"]),
                float(actor["y"]),
                float(actor["heading"]),
            )
        self.handle_events(r)
        if (
            r.sensors is not None
            and r.sensors.z_zone is not None
            and r.kind in ("move", "wait", "undock")
        ):
            self.zone_t = t
        for info in r.obstacles:  # anything not dropped visually yet (e.g. after a missed event)
            if int(info["id"]) not in self.scene.obstacles:
                self.scene.drop_obstacle(info, t, animate=False)
        self.update_markers(r)
        if r.kind == "dropoff":
            self.parcel_on_robot = False
        self.tele_base = self.telemetry_base(r)
        if r.duration > 0 or r.belief is not None:
            self.floor_due = self.mini_due = True
        if r.belief is not None and any(e.name in FRESH_PARTICLES for e in r.events):
            self.weigh_result = r
        self.play_rate = 1.0
        if r.kind == "wait" and any(e.name == "DEPLOYED" for e in r.events):
            self.scan_psi0 = (
                self.psi
            )  # the eye looks all around (360 degrees) during the global scan
            self.play_rate = DEPLOY_SLOW_MOTION
        else:
            self.scan_psi0 = None

    def finish_result(self, r) -> None:
        end = (float(r.end[0]), float(r.end[1]))
        if r.kind in MOVING_KINDS:
            self.odometer += math.hypot(end[0] - float(r.start[0]), end[1] - float(r.start[1]))
        if r.kind != "teleport":
            self.scene.roll(end[0] - self.robot_xy[0], end[1] - self.robot_xy[1])
        self.robot_xy, self.robot_z = end, 0.0
        self.scene.place_robot(end[0], end[1], 0.0)
        if r.kind == "pickup":
            self.parcel_on_robot = True
            self.scene.set_deck(DECK_LIFT)
        elif r.kind == "dropoff" and r.dock is not None:
            vis = self.scene.parcel(self.active_parcel)
            if vis is not None:
                vis.move((max(vis.pos[0], r.dock["conveyor_from"][0]), vis.pos[1], CONVEYOR_Z))
                self.scene.ride_conveyor(
                    self.active_parcel, r.dock["parcel_to"][0], r.dock["dock_index"] or 0
                )
        self.prev_score = float(r.stats.get("score", self.prev_score))

    def animate(self, t: float, dt: float) -> None:
        """Play the current result at ``self.play_t`` seconds (every frame)."""
        r = self.result
        if r is None:
            return
        u = util.clamp(self.play_t, 0.0, r.duration)
        s = u / r.duration if r.duration > 0 else 1.0
        sx, sy = float(r.start[0]), float(r.start[1])
        ex, ey = float(r.end[0]), float(r.end[1])
        kind = r.kind
        z = 0.0
        if kind == "teleport":
            x, y, z = self.teleport_pose(u, (sx, sy), (ex, ey))
        else:
            e = (
                util.smoothstep(s) if kind in ("dock", "undock") else s
            )  # (peaks at ~2 m/s, like driving)
            x, y = sx + (ex - sx) * e, sy + (ey - sy) * e
            self.scene.roll(x - self.robot_xy[0], y - self.robot_xy[1])
        self.robot_xy, self.robot_z = (x, y), z
        self.scene.place_robot(x, y, z)
        moving = kind in MOVING_KINDS and abs(ex - sx) + abs(ey - sy) > 1e-3
        self.travel = util.unit2(ex - sx, ey - sy) if moving else None
        if self.travel is not None and r.dock is None:
            k = util.smooth_factor(dt, 0.5)
            lx, ly = self.look_vec
            self.look_vec = (lx + (self.travel[0] - lx) * k, ly + (self.travel[1] - ly) * k)
        # Eye (and its headlights): a 360 degree scan at deployment, else see look_direction().
        if self.scan_psi0 is not None:
            self.psi = self.scan_psi0 + util.TAU * util.smoothstep(s)
        else:
            want = self.look_direction(r, x, y)
            if want is not None:  # a smooth turn, at most EYE_TURN_RATE
                step = util.wrap_angle(want - self.psi) * util.smooth_factor(dt, 0.2)
                limit = EYE_TURN_RATE * dt
                self.psi += util.clamp(step, -limit, limit)
        # Forklifts: from their poses at the end of the previous result to the end of this one.
        for k, pose in self.actor_to.items():
            x0, y0, h0 = self.actor_from.get(k, pose)
            x1, y1, h1 = pose
            self.scene.set_forklift(
                k, x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, util.angle_lerp(h0, h1, s)
            )
        self.animate_parcels(r, u, t)

    def look_direction(self, r, x: float, y: float) -> float | None:
        """Where the robot's eye (and headlights) should look (rad), or None to keep its direction.

        Docking: at the rack slot or the dock door. Driving while LOCKED: the
        point of the planned path LOOK_AHEAD meters ahead, so the eye turns
        into a corner before the robot does. Otherwise the smoothed travel
        direction: a west-north-west-north staircase reads as north-west, not
        as a twitch every half second.
        """
        if r.dock is not None:
            return math.atan2(r.dock["facing"][1], r.dock["facing"][0])
        b = r.belief
        path = r.path if r.path is not None else ()
        if b is not None and b.locked and not b.relocalizing and len(path) >= 2:
            reach = LOOK_AHEAD * LOOK_AHEAD
            for i, j in path[1:]:  # cell (i, j) is centered at x = j, y = i
                if (j - x) ** 2 + (i - y) ** 2 >= reach:
                    return math.atan2(i - y, j - x)
            i, j = path[-1]
            if (j - x) ** 2 + (i - y) ** 2 >= 1.0:
                return math.atan2(i - y, j - x)
        lx, ly = self.look_vec
        if lx * lx + ly * ly > 0.09:
            return math.atan2(ly, lx)
        return None

    @staticmethod
    def teleport_pose(u: float, start, end) -> tuple[float, float, float]:
        """KIDNAPPED (1 s): lifted away by an invisible hand, dropped far away (lands at ~0.9 s)."""
        if u < 0.35:
            return start[0], start[1], 1.5 * util.ease_in(u / 0.35)
        z, _ = drop_height(u - 0.35, h0=1.5, restitution=0.0, bounces=0)
        return end[0], end[1], z

    def animate_parcels(self, r, u: float, t: float) -> None:
        scene = self.scene
        vis = scene.parcel(self.active_parcel)
        if vis is not None and r.dock is not None and r.kind == "pickup":
            # 2.5 s: glow, slide from the rack level onto the deck with a small arc, lift the deck.
            (fx, fy), (tx, ty) = r.dock["parcel_from"], r.dock["parcel_to"]
            a = util.ease_in_out((u - 0.25) / 1.6)
            lift = DECK_LIFT * util.smoothstep((u - 1.95) / 0.35)
            scene.set_deck(lift)
            zz = util.lerp(RACK_Z, DECK_TOP, util.smoothstep(a * 1.4)) + 0.08 * math.sin(
                math.pi * a
            )
            if a >= 1.0:
                zz = DECK_TOP + lift
            vis.move((fx + (tx - fx) * a, fy + (ty - fy) * a, zz))
        elif vis is not None and r.dock is not None and r.kind == "dropoff":
            # 3 s: lower the deck, push the parcel onto the belt, then it rides at the belt speed.
            px, py = self.robot_xy
            cx = r.dock["conveyor_from"][0]
            scene.set_deck(DECK_LIFT * (1.0 - util.smoothstep(u / 0.35)))
            if u < 0.35:
                vis.move((px, py, DECK_TOP + scene.deck_z))
            elif u < 1.1:
                a = util.ease_in_out((u - 0.35) / 0.75)
                vis.move((px + (cx - px) * a, py, DECK_TOP))
            else:
                vis.move((cx + CONVEYOR_SPEED * (u - 1.1), py, CONVEYOR_Z))
        elif vis is not None and self.parcel_on_robot:
            x, y = self.robot_xy
            vis.move((x, y, DECK_TOP + scene.deck_z + self.robot_z))

    def update_parcel_glow(self, t: float) -> None:
        """The active parcel pulses orange on its rack and glows softly while carried (every
        state).
        """
        loading = (
            self.result is not None and self.result.kind == "pickup" and self.state == "MISSION"
        )
        for n, p in self.scene.parcels.items():
            if p.pos[0] > 100.2:
                glow = 0.0  # through the door: delivered
            elif p.riding is not None:
                glow = 0.45  # on the conveyor, on its way out
            elif n == self.active_parcel:
                glow = 0.45 if self.parcel_on_robot or loading else util.pulse(t, 0.8, 0.3, 0.85)
            else:
                glow = 0.1  # a later order (multi-order missions: staff --orders N)
            p.glow.set(util.scale(PARCEL_GLOW, glow))

    def update_markers(self, r) -> None:
        """Pickup marker (yellow) while going to the rack; drop-off marker (green) while going to
        the dock.
        """
        if not self.deployed or self.state != "MISSION":
            return
        if self.mission is not None and self.mission.scenario.is_tour:
            # a tour has no orders (show_scenario_markers set its dock marker at the briefing)
            return
        scene, phase = self.scene, r.phase
        slot = core.SLOTS_BY_NAME.get(r.order.get("pickup"))
        dock = DOCKS_BY_NAME.get(r.order.get("dock"))
        if phase in (core.LOCALIZING, core.TO_PICKUP) and slot is not None:
            if scene.marker_xy("pickup") != tuple(slot.station):
                scene.show_marker("pickup", slot.station)
            if scene.marker_xy("dropoff") is not None:
                scene.hide_marker("dropoff", animate=False)
        elif phase == core.TO_DOCK and dock is not None:
            if scene.marker_xy("dropoff") != tuple(dock.station):
                scene.show_marker("dropoff", dock.station)
            if scene.marker_xy("pickup") is not None:
                scene.hide_marker("pickup", animate=False)

    def current_dock(self):
        r = self.result
        if self.state == "MISSION" and r is not None and r.dock is not None:
            return r.dock
        return None

    def update_doors(self) -> None:
        want = set(self.scene.parcels_in_doorway())
        r = self.result
        if r is not None and self.deployed and r.order.get("leg") == "dropoff":
            dock = DOCKS_BY_NAME.get(r.order.get("dock"))
            if dock is not None and (
                r.phase == core.DOCK_DROPOFF or util.dist2(self.robot_xy, dock.station) < 16.0
            ):
                want.add(dock.index)
        if self.state == "END" and self.mission is not None and self.mission.success:
            dock = DOCKS_BY_NAME.get(self.mission.orders[-1].dock.name)
            if dock is not None:
                want.add(dock.index)
        for n in (1, 2, 3):
            self.scene.set_dock_open(n, n in want)

    # ======================================================================
    # Events
    # ======================================================================
    def handle_events(self, r) -> None:
        t = self.t
        for event in r.events:
            name, data = event.name, event.data
            if name in LOGGED_EVENTS:
                self.log_event(r, event)
            if self.camera.mode == MODE_AUTO and self.camera.script is None:
                self.camera.auto_event(name, t)
            if self.shots is not None:
                self.shots.on_event(name, data, t)
            handler = getattr(
                self, "event_" + name.lower(), None
            )  # event_locked(), event_lost(), ...
            if handler is not None:
                handler(event, r)

    def log_event(self, r, event) -> None:
        data = event.data
        text = event.text
        if event.name == "LOCKED":
            text += f"  confidence {100 * data.get('confidence', 0):.0f}%"
        elif event.name == "MARKER_ACQUIRED":
            text += (
                f"  (estimate error {data['error_before']:.2f} m -> {data['error_after']:.2f} m)"
            )
        elif event.name == "REPLANNED":
            text += f"  (value_iteration took {data.get('seconds', 0):.2f} s)"
        log(f"  [{r.clock:6.1f} s] {text}")  # a FAULT's traceback is printed by enter_fault()

    def say_once(self, key: str, text: str) -> None:
        if key not in self.said:
            self.said.add(key)
            self.audio.say(text)

    def event_countdown(self, event, r) -> None:
        self.hud.countdown(int(event.data.get("count", 0)))
        self.audio.play("beep", force=True)

    def event_deployed(self, event, r) -> None:
        self.deployed = True
        # No big centered word now: the camera cuts to the whole floor, covered with the 2000
        # particles of the global localization, and "GO! 2000 PARTICLES" types in above it.
        self.hud.countdown(None)
        self.audio.play("beep_go", force=True)
        self.hud.announce(event.text, "big", 2.6)
        self.camera.end_script()
        sc = self.mission.scenario if self.mission is not None else None
        # Tours: the loop from above (an unknown start: the whole floor until the robot is found).
        self.camera.set_area(
            tour_area(sc.route) if sc is not None and sc.is_tour and sc.known_start else None
        )
        if self.camera.mode == MODE_AUTO:
            self.camera.auto_event("DEPLOYED", self.t)
        self.scene.hide_marker("dropoff", animate=False)
        self.update_markers(r)
        self.say_once("deploy", "Localizing.")

    def event_locked(self, event, r) -> None:
        self.lock_t = self.t
        m = self.mission
        if (
            m is not None and m.scenario.is_tour and self.camera.area is None
        ):  # found: zoom to the loop
            route = m.scenario.route
            self.later(
                2.5, lambda: self.camera.set_area(tour_area(route)) if self.mission is m else None
            )
        self.audio.play("lock_on")
        self.hud.announce(event.text, "good", 2.5)
        self.flash = (util.GREEN, self.t + 0.5)
        # A green shock ring out of the robot (it also reads well from the OVERHEAD camera).
        self.scene.ping(
            self.robot_xy, 5.0, self.t, util.GREEN, duration=1.3, opacity=0.8, priority=1
        )
        self.say_once("lock", "Position locked.")

    def event_lost(self, event, r) -> None:
        self.audio.play("safety_buzz", pitch=0.7)
        self.hud.announce(event.text, "bad", 2.8)

    def event_reseed(self, event, r) -> None:
        self.hud.announce(event.text, "info", 1.6)

    def event_recovered(self, event, r) -> None:
        self.hud.announce(event.text, "info", 1.6)

    def event_rfid_ping(self, event, r) -> None:
        data = event.data
        beacon, rng = int(data["beacon"]), float(data["range"])
        self.scene.flash_beacon(beacon, self.t)
        center = data.get("position", core.BEACONS[beacon])
        eye = self.camera.out[0]
        if (
            eye[2] > 6.0 or util.dist2(eye, center) > rng + 2.5
        ):  # not through a low camera near the beacon
            self.scene.ping(center, rng, self.t, opacity=0.45)
        self.audio.play("rfid_ping")
        self.last_ping = (beacon, rng, self.t)

    def event_safety_stop(self, event, r) -> None:
        self.audio.play("safety_buzz")
        self.hud.announce(event.text, "warn", 2.0)
        self.flash = (util.RED, self.t + 0.8)

    def event_waiting(self, event, r) -> None:
        self.hud.announce(event.text, "info", 1.8)

    def event_forklift_yield(self, event, r) -> None:
        self.hud.announce(event.text, "info", 1.8)

    def event_forklifts(self, event, r) -> None:
        self.forklift_traffic = bool(event.data.get("on"))
        self.hud.announce(event.text, "info", 2.0)

    def event_obstacle_dropped(self, event, r) -> None:
        self.drop_obstacle_visual(event.data)
        self.hud.announce(event.text, "warn", 2.4)

    def event_replanned(self, event, r) -> None:
        self.hud.announce(f"{event.text} ({event.data.get('seconds', 0.0):.2f} s)", "info", 2.2)
        if (
            self.mission is not None
            and self.mission.scenario.key == "obstacle_test"
            and getattr(r, "value_change", None) is not None
        ):
            # OBSTACLE TEST: once the barrier has landed, show how the value function changed (from
            # above, if the director runs the camera), then go back to the previous layer.
            previous = self.layers.map_layer if self.layers.map_layer != "change" else "none"

            def show():
                self.layers.map_layer = "change"
                self.change_until = self.t + CHANGE_LAYER_SECONDS
                self.floor_due = self.mini_due = True
                self.hud.announce("VALUE FUNCTION CHANGED: RED = LOWER VALUE", "warn", 4.0)
                if self.camera.mode == MODE_AUTO and self.camera.script is None:
                    self.camera.auto_event("VALUE_CHANGE", self.t)
                self.later(CHANGE_LAYER_SECONDS, restore)

            def restore():
                if self.layers.map_layer == "change" and self.t >= self.change_until - 1e-6:
                    self.layers.map_layer = previous
                    self.floor_due = self.mini_due = True

            self.later(2.5, show)

    def event_test_start(self, event, r) -> None:
        self.event_sensors(event, r)

    def event_sensors(self, event, r) -> None:
        """A test starts (a tour's SENSORS event, a mission test's TEST_START): its setting,
        announced.
        """
        self.audio.play("beep_go", pitch=1.2)
        self.hud.announce(event.text, "big", 3.0)
        on = [name for name, active in event.data.get("sensors", {}).items() if active]
        log(f"  [{r.clock:6.1f} s] {event.text}  (sensors on: {', '.join(on) or 'none'})")

    def event_experiment_complete(self, event, r) -> None:
        self.say_once("experiment", "Experiment complete.")

    def event_kidnapped(self, event, r) -> None:
        self.audio.play("kidnap_whoosh")
        self.hud.announce(event.text, "bad", 2.5)
        self.flash = ((1.0, 1.0, 1.0), self.t + 1.0)
        start = event.data.get("from", r.start)
        self.scene.ping(
            (float(start[0]), float(start[1])),
            3.0,
            self.t,
            (1.0, 1.0, 1.0),
            duration=1.0,
            opacity=0.75,
            priority=2,
        )

    def event_marker_acquired(self, event, r) -> None:
        self.audio.play("lock_on", pitch=1.25)
        self.hud.announce(event.text, "good", 2.2)
        kind = "pickup" if r.phase == core.DOCK_PICKUP else "dropoff"
        self.scene.hide_marker(kind, animate=True, t=self.t)

    def event_docking_failed(self, event, r) -> None:
        self.audio.play("safety_buzz", pitch=0.8)
        self.hud.announce(event.text, "bad", 2.8)
        self.flash = (util.RED, self.t + 1.0)

    def event_relocalizing(self, event, r) -> None:
        self.hud.announce(event.text, "bad", 2.2)

    def event_package_loaded(self, event, r) -> None:
        def landed():
            self.audio.play("pickup")
            self.hud.announce(event.text, "good", 2.4)

        self.later(1.9, landed)

    def event_delivered(self, event, r) -> None:
        dock = DOCKS_BY_NAME.get(event.data.get("dock"))

        def through_the_door():
            self.audio.play("delivered_fanfare")
            self.hud.announce(event.text, "good", 2.6)
            if dock is not None:
                self.scene.dock_celebrate(dock.index, self.t)

        self.later(2.2, through_the_door)

    def event_new_order(self, event, r) -> None:
        self.active_parcel = int(event.data.get("number", self.active_parcel))
        self.parcel_on_robot = False

        def announce():
            self.hud.announce(event.text, "info", 2.6)
            self.audio.play("beep_go")

        self.later(1.0, announce)

    def event_slow_code(self, event, r) -> None:
        self.hud.announce(event.text, "warn", 3.5)

    def event_grid_filter_off(self, event, r) -> None:
        if self.deployed:
            self.hud.announce(event.text, "warn", 2.5)

    def event_mission_complete(self, event, r) -> None:
        self.say_once("complete", "Mission complete!")

    def event_mission_failed(self, event, r) -> None:
        self.fail_text = event.text  # e.g. "OUT OF TIME": the subtitle of the MISSION FAILED screen
        self.audio.play("safety_buzz", pitch=0.6)
        self.hud.announce(event.text, "bad", 3.0)

    def event_fault(self, event, r) -> None:
        self.audio.play("safety_buzz", pitch=0.6, force=True)

    # ======================================================================
    # HUD telemetry
    # ======================================================================
    @staticmethod
    def test_subline(sc) -> str | None:
        """The banner's subline during a test: what it changes."""
        if sc.group == "sensors":
            on = [
                name
                for name, active in zip(("PROXIMITY", "RFID", "ZONE"), sc.sensors, strict=False)
                if active
            ]
            return ("SENSORS ON: " + " + ".join(on)) if on else "NO SENSORS: DEAD RECKONING"
        if sc.group == "start":
            return f"{'KNOWN' if sc.known_start else 'UNKNOWN'} START, {sc.particles} PARTICLES"
        if sc.group == "penalty":
            return (
                "PENALTY -50" if sc.penalty else "NO PENALTY"
            ) + " - EXACT POSITION (PLANNING ONLY)"
        return None

    def telemetry_base(self, r) -> dict[str, Any]:
        """Everything that is constant during a result (computed once)."""
        m = self.mission
        b = r.belief
        sensors = r.sensors
        base: dict[str, Any] = {
            "robot_name": m.profile["name"],
            "code": m.code,
            "order": None
            if m.scenario.is_tour
            else f"{r.order.get('number', 1)}/{r.order.get('total', 1)}",
            "banner": r.banner,
            "phase": r.phase,
            "stars": int(r.stats.get("stars", 0)),
            "wheels": tuple(float(w) for w in r.wheel_omegas_cmd),
            "notes": list(r.notes),
            "parcel_state": (r.parcel or {}).get("state"),
            "waiting": r.kind == "wait" and r.action is not None,
            "goal": r.goal_name,
            "leg": r.order.get("leg"),
            "route": f"{r.order.get('pickup', '?')} -> {r.order.get('dock', '?')}",
        }
        base["experiment"] = self.test_subline(m.scenario)
        if b is not None:
            mean = (float(b.mean[0]), float(b.mean[1]))
            base.update(
                confidence=float(b.confidence),
                locked=bool(b.locked),
                relocalizing=bool(b.relocalizing),
                estimate=mean,
                sigma=float(b.sigma),
                particles=int(b.n_particles),
                error=math.hypot(mean[0] - float(r.end[0]), mean[1] - float(r.end[1])),
                dead_reckoning=bool(getattr(b, "dead_reckoning", False)),
            )
        else:
            base.update(
                confidence=0.0,
                locked=False,
                relocalizing=False,
                estimate=None,
                sigma=None,
                particles=0,
                error=None,
            )
        if sensors is not None:
            # A sensor that an experiment switched off reads None (the HUD shows it DISABLED).
            base["disabled"] = [
                name
                for name, value in (
                    ("proximity", sensors.z_on),
                    ("rfid", sensors.z_rfid),
                    ("zone", sensors.z_zone),
                )
                if value is None
            ]
            base["z_on"] = None if sensors.z_on is None else bool(sensors.z_on)
            beacon, rng = sensors.z_rfid if sensors.z_rfid is not None else (None, float("inf"))
            base["rfid"] = (int(beacon), float(rng)) if beacon is not None else None
            base["zone"] = C.ZONE_NAMES[int(sensors.z_zone)] if sensors.z_zone is not None else None
        else:  # pickup / drop-off animations take no readings: keep showing the last ones
            for key in ("z_on", "rfid", "zone", "disabled"):
                base[key] = self.tele_base.get(key)
        base["_score"] = float(r.stats.get("score", self.prev_score))
        return base

    def telemetry(self, t: float) -> dict[str, Any]:
        r = self.result
        state = dict(self.tele_base)
        s = util.clamp(self.play_t / r.duration, 0.0, 1.0) if r.duration > 0 else 1.0
        state["clock"] = r.clock_start + (r.clock - r.clock_start) * s
        # The live score slides toward the result's score in 4 steps per second (every changed
        # label is a round trip to Webots; a new number every frame would cost ~1.5 ms per frame).
        steps = max(1.0, 4.0 * r.duration)
        state["score"] = int(
            round(util.lerp(self.prev_score, state.pop("_score"), math.floor(s * steps) / steps))
        )
        if state.get("experiment"):  # a tour is an experiment, not a scored mission
            state["score"] = None
        turbo, rtf = self.turbo_on(), self.rtf
        if self.args.screenshots and self.args.shot_set == "docs":
            turbo, rtf = False, 1.0  # the guide's pictures show what a student sees in real time
        state.update(
            camera=self.camera.mode_name,
            belief=self.layers.belief,
            map_layer=self.layers.map_layer,
            vision=bool(self.layers.vision),
            turbo=turbo,
            muted=not self.audio.active,
            night=self.scene.night,
            forklifts=self.forklift_traffic,
            speed_factor=rtf,
            view="overhead" if self.camera.overhead_view else "",
        )
        return state

    def turbo_on(self) -> bool:
        try:
            return self.sup.simulationGetMode() == Supervisor.SIMULATION_MODE_FAST
        except Exception:  # noqa: BLE001
            return False

    # ======================================================================
    # Robot looks (LED ring, iris, zone reader light, idle eye)
    # ======================================================================
    def update_robot_look(self, t: float, dt: float) -> None:
        state = self.state
        led = util.CYAN
        if state == "BOOT":
            u = t - self.state_t0
            wake = util.smoothstep((u - (2.2 if self.boot_full else 0.0)) / 0.8)
            led = util.scale(util.CYAN, wake)
            self.psi = wake * 0.8 * math.sin(0.7 * u)  # the eye wakes up and looks around
        elif state in ("FAULT", "IMPORT_ERROR", "IMPORT_BOOT"):
            led = util.RED if util.blink(t, 1.2) else util.scale(util.RED, 0.15)
        elif state == "END":
            if self.mission is not None and self.mission.success:
                palette = (util.GREEN, util.CYAN, util.YELLOW, util.MAGENTA)
                k = int(t * 2.0)
                led = util.mix(palette[k % 4], palette[(k + 1) % 4], (t * 2.0) % 1.0)
            else:
                led = util.scale(util.RED, 0.6)
        elif state == "BRIEFING" or not self.deployed:
            led = util.scale(util.CYAN, util.pulse(t, 0.5, 0.45, 1.0))
            if self.result is None or self.result.kind == "wait":
                self.psi = util.angle_lerp(
                    self.psi, 0.9 * math.sin(0.5 * t), util.smooth_factor(dt, 0.4)
                )
        else:
            b = self.result.belief if self.result is not None else None
            if b is None:
                led = util.CYAN
            elif b.relocalizing:
                led = util.MAGENTA if util.blink(t, 3.0) else util.scale(util.MAGENTA, 0.2)
            elif not b.locked:
                led = util.scale(util.CYAN, util.pulse(t, 1.2, 0.55, 1.0))
            elif self.parcel_on_robot:
                led = util.ORANGE
            else:
                led = util.GREEN
        if self.flash is not None:
            color, until = self.flash
            if t < until:
                led = color if util.blink(t, 6.0) else util.scale(color, 0.25)
            else:
                self.flash = None
        self.scene.set_led(led)
        self.scene.set_eye(self.psi)
        on = (t - self.zone_t) < 0.15  # the zone reader's light blinks with every zone reading
        self.scene.set_tag((0.67, 1.0, 0.31) if on else (0.08, 0.14, 0.04))

    # ======================================================================
    # Hologram and minimap
    # ======================================================================
    def extend_trail(self, r) -> None:
        """A sensor test: the previous result's end joins the trails; this one's end waits until it
        has played (meanwhile the trails end at the moving robot and its sliding estimate)."""
        if self.trail is None:
            self.trail = {"truth": [], "belief": []}
        if self.trail_next is not None:
            truth, belief = self.trail_next
            self.trail["truth"].append(truth)
            self.trail["belief"].append(belief)
            self.trail_next = None
        if (
            r.belief is not None
            and r.end is not None
            and r.kind in ("move", "wait", "dock", "undock")
        ):
            self.trail_next = (
                (float(r.end[0]), float(r.end[1])),
                (float(r.belief.mean[0]), float(r.belief.mean[1])),
            )

    def trails(self) -> dict | None:
        """HoloFrame.trails: the committed trails plus their live heads (the robot, its
        estimate).
        """
        r = self.result
        tour = (
            self.mission is not None and self.mission.scenario.is_tour
        )  # (a mission test: its true path only)
        if self.trail is None or r is None or r.belief is None:
            trail = self.trail
            return trail and {
                "truth": trail["truth"],
                "belief": trail["belief"] if tour else [],
                "gap": tour and self.state == "END",
                "other": self.other_route(),
            }
        dx, dy = self.belief_shift()
        head = (float(r.belief.mean[0]) + dx, float(r.belief.mean[1]) + dy)
        return {
            "truth": self.trail["truth"] + [tuple(self.robot_xy)],
            "belief": self.trail["belief"] + [head] if tour else [],
            "gap": tour and self.state == "END",
            "other": self.other_route(),
        }

    def other_route(self) -> list:
        """A PENALTY TEST: the true path of the other test of this session (the latest one), to
        compare.
        """
        m = self.mission
        if m is None or m.scenario.group != "penalty":
            return []
        done = self.results.get("penalty", {})
        others = [ex for key, ex in done.items() if key != m.scenario.key and ex.get("route")]
        return others[-1]["route"] if others else []

    def holo_frame(self, t: float):
        r = self.result if self.result is not None else self.boot_result
        ping = None
        if self.last_ping is not None and t - self.last_ping[2] < 3.0:
            ping = (self.last_ping[0], self.last_ping[1], t - self.last_ping[2])
        goal = None
        if r is not None and r.goal_cell is not None:
            goal = (float(r.goal_cell[1]), float(r.goal_cell[0]))
        kwargs = dict(
            result=r,
            robot_xy=self.robot_xy,
            t=t,
            layers=self.layers,
            ping=ping,
            zone_age=max(0.0, t - self.zone_t),
            goal_xy=goal,
            phase=r.phase if r is not None else "",
            show_truth=True,
            weigh=self.weigh(t),
            near=self.near,
            belief_shift=self.belief_shift(),
            estimate_ring=not self.scene.reticle_shown,
            smoothed=self.smoothed if self.state == "END" else None,
            smoothed_reveal=util.clamp((t - self.smoothed_t0 - 1.0) / 2.5, 0.0, 1.0),
            trails=self.trails(),
        )
        return HoloFrame(**kwargs) if HoloFrame is not None else types.SimpleNamespace(**kwargs)

    def belief_shift(self) -> tuple[float, float]:
        """HoloFrame.belief_shift: slide the result's belief (computed at its END) along with the
        robot.

        While a ``move`` plays, the belief is drawn moved back by the part of the
        motion the robot has not driven yet (the displacement the brain was
        told, ``u_eff``), so the swarm and the estimate travel with the robot
        instead of jumping one meter ahead at the start of every result.
        Docking moves are known exactly (``end - start``) and eased like the robot.
        """
        r = self.result
        if r is None or r.belief is None or r.duration <= 0 or self.state != "MISSION":
            return (0.0, 0.0)
        s = util.clamp(self.play_t / r.duration, 0.0, 1.0)
        if r.kind == "move":
            motion = r.u_eff if r.u_eff is not None else (r.end - r.start)
        elif r.kind in ("dock", "undock"):
            motion, s = r.end - r.start, util.smoothstep(s)
        else:
            return (0.0, 0.0)
        rest = 1.0 - s
        return (-rest * float(motion[0]), -rest * float(motion[1]))

    def camera_near(self, t: float) -> float:
        """HoloFrame.near: 1 when the camera looks at the floor from within 15 m, 0 beyond 45 m.

        The distance is the one to what the camera looks at (the robot in the
        chase and orbit shots, the rack or the dock in the briefing preview, the
        floor below in OVERHEAD); a FREE camera is measured to the robot.
        """
        if self.camera.mode == MODE_FREE:  # (the camera reads the user's viewpoint at ~4 Hz)
            eye = self.camera.free_eye or self.camera.out[0]
            target = (self.robot_xy[0], self.robot_xy[1], 0.5)
        else:
            eye, target, _ = self.camera.out
        d = math.sqrt(
            (eye[0] - target[0]) ** 2 + (eye[1] - target[1]) ** 2 + (eye[2] - target[2]) ** 2
        )
        return util.smoothstep((45.0 - d) / 30.0)

    def update_reticle(self, t: float, dt: float) -> None:
        """The crisp 3D estimate reticle of close camera views (the floor draws the ring otherwise).

        It sits at the student's estimate (slid with the robot like the rest of
        the belief), turns slowly, and takes the belief's state colors:
        magenta locked, dimmer and faster while localizing, pulsing red when LOST.
        """
        r = self.result
        b = r.belief if (r is not None and self.state == "MISSION" and self.deployed) else None
        alpha = util.smoothstep((self.near - 0.4) / 0.2)
        if b is None or not self.layers.vision or self.floor_clear or alpha <= 0.0:
            self.scene.set_reticle(None)
            return
        mean = (float(b.mean[0]), float(b.mean[1]))
        if not (math.isfinite(mean[0]) and math.isfinite(mean[1])):
            self.scene.set_reticle(None)
            return
        dx, dy = self.belief_shift()
        scale = 1.0
        if b.relocalizing:
            color, rate = (util.RED if util.blink(t, 2.6) else util.MAGENTA), -1.4
            scale = 1.0 + 0.06 * math.sin(util.TAU * 1.3 * t)
        elif b.locked:
            color, rate = util.MAGENTA, 0.6
            snap = util.clamp((t - self.lock_t) / 0.45, 0.0, 1.0)  # lock-on: the reticle snaps shut
            scale = 1.0 + 0.6 * (1.0 - snap) ** 2
            if snap < 1.0:
                color = util.mix(util.WHITE, util.MAGENTA, snap)
        else:
            color, rate = util.scale(util.MAGENTA, 0.8), 1.8
            alpha *= 0.85
        self.reticle_angle += rate * dt
        self.scene.set_reticle(
            (mean[0] + dx, mean[1] + dy), color, alpha, self.reticle_angle, scale
        )

    def weigh(self, t: float) -> float:
        """HoloFrame.weigh: a fresh uniform particle set first shows its 2000 equal guesses, then
        the
        weights of its first measurement (over the middle half of its result, in result time).
        """
        r = self.result
        if r is None or r is not self.weigh_result:
            return 1.0
        span = max(float(r.duration), 0.4)
        return util.smoothstep((self.play_t - 0.3 * span) / (0.5 * span))

    def update_hologram(self, t: float) -> None:
        """Render at most one image per frame: when a result starts, and at ~4 Hz while it plays.

        A frame that ran ``mission.step()`` renders nothing: the floor image
        follows one frame (32 ms) later and the minimap one more frame later,
        so no frame pays for the student's code and a render together. The
        extra ~4 Hz renders (pulses, fading pings) are skipped while a render
        takes more than 20 ms (VIZ_SPEC section 3.1). A close camera following
        the moving robot gets the floor at 10 Hz (the belief slides with the
        robot, see :meth:`belief_shift`); a freshly weighed particle set at ~15 Hz.
        """
        if self.photo is not None or self.state == "INIT":
            return
        if self.frame_kind == "step":  # mission.step() ran in this frame: render in the next one
            return
        # MISSION COMPLETE and the fault screens show a clean floor (the minimap keeps the last
        # belief).
        vision = bool(self.layers.vision) and not self.floor_clear
        slow = max(self.floor_ms_last if vision else 0.0, self.mini_ms_last) > 20.0
        if vision and t - self.floor_t >= 0.066 and 0.0 < self.weigh(t) < 1.0:
            self.floor_due = True  # the particles are being weighed: ~15 floor images per second
        elif (
            vision
            and self.near >= 0.5
            and self.travel is not None
            and not slow
            and self.state == "MISSION"
            and t - self.floor_t >= 0.1
        ):
            self.floor_due = (
                True  # a close camera follows the moving robot: the swarm slides at 10 Hz
            )
        if (
            self.state == "END"
            and self.smoothed is not None
            and t - self.smoothed_t0 < 4.0
            and t - self.holo_last >= 0.1
        ):
            self.mini_due = True  # the smoothed path draws itself on the minimap (TODO 17)
        if not (self.floor_due or self.mini_due):
            if self.state not in ("MISSION", "BRIEFING") or t - self.holo_last < 0.25 or slow:
                return  # (the briefing's goal pulses too)
            # ~4 Hz: the minimap, and the floor unless the close-camera refresh just drew it.
            self.floor_due, self.mini_due = vision and t - self.floor_t >= 0.2, True
        if self.floor_due and not vision:
            self.floor_due = False
        if not (self.floor_due or self.mini_due):
            return
        frame = self.holo_frame(t)
        start = time.perf_counter()
        if self.floor_due:
            self.floor_due = False
            self.floor_t = t
            image = self.holo.render_floor(frame)
            self.floor_ms_last = 1000.0 * (time.perf_counter() - start)
            self.perf.floor_ms.append(self.floor_ms_last)
            if image is not None and self.floor_display is not None:
                pasted = time.perf_counter()
                self.paste(self.floor_display, image)
                self.perf.paste_ms.append(1000.0 * (time.perf_counter() - pasted))
                if not self.scene.holo_visible:
                    self.scene.set_holo_visible(True)  # only once the plane has an image
        else:
            self.mini_due = False
            image = self.holo.render_minimap(frame)
            self.mini_ms_last = 1000.0 * (time.perf_counter() - start)
            self.perf.mini_ms.append(self.mini_ms_last)
            if image is not None and self.mini_display is not None:
                self.paste(self.mini_display, image)
            self.holo_last = t
        self.frame_kind = "holo"

    @property
    def floor_clear(self) -> bool:
        """No floor hologram: the MISSION COMPLETE celebration and the fault screens."""
        if self.state in ("FAULT", "IMPORT_ERROR", "IMPORT_BOOT"):
            return True
        # (a sensor test keeps its floor: its final belief)
        return (
            self.state == "END"
            and self.mission is not None
            and self.mission.success
            and self.mission.experiment is None
        )

    def paste(self, display, image) -> None:
        """Push an (h, w, 4) uint8 RGBA image to a Display (VIZ_SPEC section 2)."""
        shape = (display.getHeight(), display.getWidth(), 4)
        if getattr(image, "shape", None) != shape or str(getattr(image, "dtype", "")) != "uint8":
            key = f"hologram.image_shape.{display.getName()}"
            self.errors[key] = self.errors.get(key, 0) + 1
            if self.errors[key] == 1:
                log(
                    f"warehouse_mission: {display.getName()} image has shape "
                    f"{getattr(image, 'shape', None)} "
                    f"dtype {getattr(image, 'dtype', None)}; expected {shape} uint8 (not drawn)"
                )
            return
        ref = display.imageNew(image.tobytes(), display.RGBA, shape[1], shape[0])
        display.imagePaste(ref, 0, 0, False)
        display.imageDelete(ref)

    # ======================================================================
    # END / FAULT
    # ======================================================================
    def enter_end(self) -> None:
        m = self.mission
        if m.phase == core.FAULT or m.fault is not None:
            self.enter_fault(m.fault)
            return
        t = self.t
        self.state, self.state_t0 = "END", t
        self.travel = None
        summary = m.summary()
        score = m.score()
        if m.experiment is not None:  # a test (also a failed penalty test): its table
            self.enter_experiment_end(summary)
            return
        if m.success:
            log(
                f"  MISSION COMPLETE in {summary['time']:.1f} s: {score['score']} points, "
                f"{'*' * score['stars']} {score['title']}"
            )
            log("  score = " + "  ".join(f"{k} {v:+.0f}" for k, v in score["score_terms"].items()))
            self.scene.celebrate(t, self.robot_xy)
            self.scene.set_holo_visible(False)  # see update_hologram()
            self.audio.play("delivered_fanfare", force=True)
            if self.camera.mode != MODE_FREE:
                dock = self.result.dock if self.result is not None else None
                facing = tuple(dock["facing"]) if dock and dock.get("target") == "dock" else None
                self.camera.start_script("celebrate", t, facing=facing)
        else:
            log(f"  MISSION FAILED: {summary.get('phase')} after {summary['steps']} control steps.")
            if self.camera.mode != MODE_FREE:
                self.camera.start_script("idle", t)
        board = dict(summary)
        info = m.student_info or {}
        board.update(
            robot_name=m.profile["name"],
            color=m.profile["color"],
            student_name=info.get("name", ""),
            score_terms=score["score_terms"],
            headline="MISSION COMPLETE" if m.success else "MISSION FAILED",
            kidnaps=m.stats.kidnaps,
            obstacles=m.stats.obstacles,
            replans=m.stats.replans,
            distance=self.odometer,
            prompt=f"R: REPLAY CODE {m.code:04d}    N: NEW CODE",
        )
        if (
            not m.success
        ):  # the reason under the headline (the rank title would repeat "MISSION FAILED")
            board["title"] = (
                self.fail_text if self.fail_text and self.fail_text != board["headline"] else ""
            )
        self.later(0.8 if m.success else 0.3, lambda: self.hud.scoreboard(board))
        if self.shots is not None:
            self.shots.on_state("END", t)
        if self.args.autotest:
            self.verdict_at = t + (self.shots.end_hold if self.shots is not None else 2.5)

    def enter_experiment_end(self, summary: dict) -> None:
        """A test is over: the table of this session's tests of its group (on screen and in the
        console), no score."""
        m = self.mission
        t = self.t
        experiment = m.experiment
        group = m.scenario.group
        tests = group_tests(group)
        done = self.results.setdefault(group, {})
        done[experiment["scenario"]] = dict(
            experiment, route=list((self.trail or {}).get("truth") or [])
        )

        def cells(ex: dict) -> str:
            return "   ".join(
                f"{name.lower()} {core.format_metric(value, unit)}"
                for name, value, unit in ex["metrics"]
            )

        log(
            f"  {experiment['title']} {experiment['label']} COMPLETE after {experiment['steps']} "
            f"control steps: "
            + cells(experiment)
            + (f", LOST {experiment['lost']}" if experiment.get("lost") else "")
        )
        log(f"  The {experiment['title'].lower()}s so far:")
        for n, test in enumerate(tests, start=1):
            log(
                f"    {n}. {test.option:<14} "
                + (cells(done[test.key]) if test.key in done else "(not run yet)")
            )
        session = [
            dict(
                done.get(test.key) or {"label": test.option},
                current=test.key == experiment["scenario"],
            )
            for test in tests
        ]
        self.audio.play("lock_on", force=True)
        if (
            self.camera.mode != MODE_FREE
        ):  # from above: the path (and a tour's final gap) beside the table
            points = m.scenario.route or (self.trail or {}).get("truth") or [tuple(self.robot_xy)]
            eye, target, fov = overhead_shot(self.camera.aspect, tour_end_area(points))
            self.camera.start_script("still", t, eye=eye, target=target, fov=fov)
        info = m.student_info or {}
        board = dict(summary)
        board.update(
            robot_name=m.profile["name"],
            color=m.profile["color"],
            student_name=info.get("name", ""),
            headline=f"{experiment['title']} {'COMPLETE' if m.success else 'FAILED'}",
            experiment=dict(experiment, session=session),
            prompt=f"R: NEXT TEST   (THEN 1-{len(tests)} AT THE BRIEFING)",
        )
        self.later(0.8, lambda: self.hud.scoreboard(board))
        if self.shots is not None:
            self.shots.on_state("END", t)
        if self.args.autotest:
            self.verdict_at = t + (self.shots.end_hold if self.shots is not None else 2.5)

    def enter_fault(self, fault: dict | None) -> None:
        t = self.t
        self.state, self.state_t0 = "FAULT", t
        self.travel = None
        info = dict(fault or {})
        info.setdefault("reason", "exception")
        info.setdefault("message", "unknown error")
        info["hint"] = FAULT_HINTS.get(info["reason"], FAULT_HINTS["exception"])
        self.fault_info = info
        log("=" * 78)
        if info["reason"] == "missing_todos":
            log("The robot cannot deploy yet: its particle filter and planner need these TODOs:")
            for todo, name, message in info.get("missing", []):
                log(f"  TODO {todo:>2}  {name:<22} {message}")
        else:
            where = f" in {info.get('function')}()" if info.get("function") else ""
            log(f"SYSTEM FAULT{where}: {info['message']}")
            if info.get("details"):
                log(str(info["details"]).rstrip())
        log(info["hint"])
        log("=" * 78)
        self.hud.fault(info)
        self.scene.set_fault_lights(True)
        self.scene.set_holo_visible(False)
        self.audio.play("safety_buzz", pitch=0.6, force=True)
        if self.camera.mode != MODE_FREE:
            self.camera.start_script("idle", t)
        if self.shots is not None:
            self.shots.on_state("FAULT", t)
        if self.args.autotest:
            self.verdict_at = t + (2.8 if self.shots is not None else 0.5)

    def enter_import_error_boot(self) -> None:
        """No student module: power the lights up anyway, then show the error."""
        t = self.t
        self.state, self.state_t0 = "IMPORT_BOOT", t
        self.scene.lights_on()
        if self.camera.mode != MODE_FREE:
            self.camera.start_script("idle", t)
        self.floor_due, self.mini_due = False, True

    def frame_import_boot(self, t: float) -> None:
        if self.state == "IMPORT_BOOT" and t - self.state_t0 >= 0.6:
            self.state, self.state_t0 = "IMPORT_ERROR", t
            self.hud.fault(
                self.fault_info or {"reason": "import_error", "message": "project4.py not loaded"}
            )
            self.scene.set_fault_lights(True)
            self.scene.set_holo_visible(False)
            self.audio.play("safety_buzz", pitch=0.6, force=True)
            if self.shots is not None:
                self.shots.on_state("IMPORT_ERROR", t)
            if self.args.autotest:
                self.verdict_at = t + (2.8 if self.shots is not None else 0.5)

    # ======================================================================
    # Keys
    # ======================================================================
    def on_key(self, key: int) -> None:
        state = self.state
        if (
            state == "BRIEFING"
            and self.mission is not None
            and self.mission.scenario.group
            and ord("1") <= key < ord("1") + len(group_tests(self.mission.scenario.group))
        ):
            self.choose_option(key - ord("1"))  # an experiment's briefing: 1-N pick its test
        elif key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            mode = key - ord("0")
            self.camera.set_mode(mode, self.t)
            self.hud.toast(f"CAMERA {mode}: {MODE_NAMES[mode]}")
            self.audio.play("ui_click")
        elif key == ord(" "):
            if state == "BRIEFING":
                self.deploy()
            elif state == "BOOT" and self.mission is not None and self.mission.phase != core.FAULT:
                self.scene.lights_on()  # skip the boot sequence
                self.enter_briefing()
        elif key == ord("E"):
            if state == "BRIEFING":
                self.cycle_scenario()
        elif key == ord("V"):
            self.layers.vision = not self.layers.vision
            if self.layers.vision:
                self.floor_due = True
            else:
                self.scene.set_holo_visible(False)
            self.mini_due = True
            self.hud.toast("ROBOT VISION " + ("ON" if self.layers.vision else "OFF"))
        elif key == ord("B"):
            i = (
                BELIEF_LAYERS.index(self.layers.belief)
                if self.layers.belief in BELIEF_LAYERS
                else 0
            )
            self.layers.belief = BELIEF_LAYERS[(i + 1) % len(BELIEF_LAYERS)]
            self.floor_due = self.mini_due = True
            self.hud.toast("PARTICLES " + ("ON" if self.layers.belief == "particles" else "OFF"))
        elif key == ord("M"):
            i = (
                MAP_LAYERS.index(self.layers.map_layer)
                if self.layers.map_layer in MAP_LAYERS
                else 0
            )
            has_change = (
                self.result is not None and getattr(self.result, "value_change", None) is not None
            )
            layers = [layer for layer in MAP_LAYERS if layer != "change" or has_change]
            current = self.layers.map_layer if self.layers.map_layer in layers else "none"
            self.layers.map_layer = layers[(layers.index(current) + 1) % len(layers)]
            self.change_until = -1e9  # (a key press ends the OBSTACLE TEST's automatic change view)
            self.floor_due = self.mini_due = True
            note = "  (RED: LOWER VALUE)" if self.layers.map_layer == "change" else ""
            self.hud.toast(f"MAP LAYER: {self.layers.map_layer.upper()}{note}")
        elif key == ord("K"):
            self.chaos_key("kidnap")
        elif key == ord("O"):
            self.chaos_key("obstacle")
        elif key == ord("F"):
            if self.mission is not None and not self.mission.done:
                self.mission.set_forklifts(not self.forklift_traffic)
                self.forklift_traffic = not self.forklift_traffic
                self.hud.toast("FORKLIFT TRAFFIC " + ("ON" if self.forklift_traffic else "OFF"))
        elif key == ord("L"):
            self.scene.set_night(not self.scene.night)
            self.hud.toast("NIGHT SHIFT " + ("ON" if self.scene.night else "OFF"))
        elif key == ord("T"):
            turbo = not self.turbo_on()
            self.sup.simulationSetMode(
                Supervisor.SIMULATION_MODE_FAST if turbo else Supervisor.SIMULATION_MODE_REAL_TIME
            )
            self.hud.toast("TURBO " + ("ON" if turbo else "OFF"))
        elif key == ord("P"):
            self.photo_start()
        elif key == ord("S"):
            on = self.audio.toggle_mute()
            self.hud.toast("SOUND " + ("ON" if on else "OFF"))
        elif key == ord("H"):
            self.help_visible = not self.help_visible
            self.hud.help(self.help_visible)
            self.scene.set_panel_dim(self.help_visible)
        elif key in (ord("R"), ord("N")):
            self.restart(same_code=key == ord("R"))

    def chaos_key(self, which: str) -> None:
        m = self.mission
        if m is None or self.state != "MISSION" or not self.deployed or m.done:
            self.hud.toast("CHAOS: ONLY WHILE THE ROBOT DRIVES")
            return
        if which == "kidnap":
            ok = m.kidnap()
            self.hud.toast("KIDNAP!" if ok else "KIDNAP: ONLY WHILE DRIVING A LEG")
        else:
            info = m.drop_obstacle()
            if info is None:
                self.hud.toast("NO SAFE SPOT FOR AN OBSTACLE RIGHT NOW")
            else:
                self.drop_obstacle_visual(info)
                self.hud.toast("OBSTACLE INCOMING")

    # ======================================================================
    # Photo mode (P)
    # ======================================================================
    def photo_start(self) -> None:
        if self.photo is not None:
            return
        folder = self.webots_dir / "screenshots"
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.hud.toast("PHOTO FAILED: CANNOT CREATE screenshots/")
            log(f"warehouse_mission: cannot create {folder}: {exc}")
            return
        code = self.mission.code if self.mission is not None else 0
        n = 1
        while (folder / f"p4_{code:04d}_{n}.png").exists():
            n += 1
        self.photo = {
            "path": folder / f"p4_{code:04d}_{n}.png",
            "frames": 0,
            "wall": time.monotonic(),
        }
        if self.help_visible:  # a photo of the scene, not of a dimmed help screen
            self.help_visible = False
            self.hud.help(False)
            self.scene.set_panel_dim(False, instant=True)
        self.hud.hide_all()
        if self.mini_display is not None:  # hide the minimap image too
            self.mini_display.setAlpha(0.0)
            self.mini_display.fillRectangle(
                0, 0, self.mini_display.getWidth(), self.mini_display.getHeight()
            )
            self.mini_display.setAlpha(1.0)

    def update_photo(self) -> None:
        p = self.photo
        if p is None:
            return
        p["frames"] += 1
        if p["frames"] == 3:  # the labels are gone from the rendered frame by now
            self.sup.exportImage(str(p["path"]), 95)
            log(f"PHOTO SAVED: {p['path']}")
        if p["frames"] >= 4 and time.monotonic() - p["wall"] >= 2.0:
            self.photo = None
            self.hud.restore()
            self.mini_due = True
            self.hud.toast(f"PHOTO SAVED: {p['path'].name}")
            self.audio.play("ui_click")

    def photo_abort(self) -> None:
        if self.photo is not None:
            self.photo = None
            self.hud.restore()
            self.mini_due = True

    # ======================================================================
    # Autotest
    # ======================================================================
    def check_autotest(self, t: float) -> None:
        if not self.args.autotest or self.quitting:
            return
        if self.verdict_at is None and (
            t > 1200.0 or time.perf_counter() - self.perf.wall0 > 600.0
        ):
            self.finish_autotest(False, f"timeout (state {self.state}, sim time {t:.0f} s)")
            return
        if self.verdict_at is None or t < self.verdict_at:
            return
        m = self.mission
        errors = sum(self.errors.values())
        if self.state == "IMPORT_ERROR":
            self.finish_autotest(
                False, f"student code could not be imported: {self.fault_info['message']}"
            )
        elif self.state == "FAULT":
            info = self.fault_info or {}
            self.finish_autotest(False, f"FAULT ({info.get('reason')}): {info.get('message')}")
        elif m is None or not m.success:
            self.finish_autotest(False, f"mission not complete (phase {m.phase if m else '?'})")
        elif any(key.endswith(".unavailable") for key in self.errors):
            missing = ", ".join(
                key.split(".")[0] for key in self.errors if key.endswith(".unavailable")
            )
            self.finish_autotest(False, f"wm/viz could not load: {missing} (see the console)")
        elif errors:
            self.finish_autotest(
                False, f"{errors} Python exception(s) in the Webots layer (see the console)"
            )
        else:
            self.finish_autotest(True, "")

    def finish_autotest(self, passed: bool, reason: str) -> None:
        wall = time.perf_counter() - self.perf.wall0
        sim = self.t - (self.perf.sim0 or 0.0)
        rtf = sim / max(wall, 1e-6)
        log(self.perf.report())
        stats = getattr(self.hud, "stats", None)  # (the NullHud stand-in has none)
        if isinstance(stats, dict) and stats.get("ticks"):
            n = stats["ticks"]
            log(
                f"VIZ HUD   {stats['labels'] / n:.2f} setLabel calls per tick, compose "
                f"{1000 * stats['compose_s'] / n:.2f} ms, "
                f"send {1000 * stats['send_s'] / n:.2f} ms per tick (setLabel "
                f"{1000 * stats['setlabel_s'] / n:.2f}), "
                f"{stats['measures']} view measurement(s)"
            )
        log(
            f"VIZ TIME  simulated {sim:.1f} s in {wall:.1f} s of wall time (real-time factor "
            f"{rtf:.2f}); "
            f"sounds played {self.audio.played}; screenshots "
            f"{self.shots.count if self.shots else 0}"
        )
        if passed:
            m = self.mission
            score = m.score()
            log(
                f"PROJECT 4 WEBOTS SELF-TEST PASS code={m.code:04d} "
                f"time={m.stats.mission_time:.1f} "
                f"score={score['score']} stars={score['stars']} fps={rtf:.2f}"
            )
        else:
            log(f"PROJECT 4 WEBOTS SELF-TEST FAIL {reason}")
        if self.trace is not None:
            self.trace.close()
            self.trace = None
        self.quitting = True
        self.sup.simulationQuit(0 if passed else 1)


###############################################################################
#                       Screenshots for QA and the docs                       #
###############################################################################


class ShotPlanner:
    """``--screenshots DIR``: export PNGs at key moments, in every camera mode, and at night.

    Milestones schedule actions relative to their time: ``("key", code)``
    presses a key exactly like the keyboard (so the key handlers are exercised
    too) and ``("shot", name)`` exports the 3D view. A plain run tours every
    camera mode and layer after the first lock; a ``--chaos`` run leaves the
    camera to the AUTO-DIRECTOR and shoots the obstacle drop, the kidnapping
    and the recovery instead; ``--forklifts`` adds the robot giving way.
    """

    def __init__(self, app: App, out_dir: Path):
        self.app = app
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.queue: list[tuple[float, str, Any, str]] = []  # (time, kind, argument, tag)
        self.done: set = set()
        self.count = 0
        self.tour_active = False
        self.end_hold = (
            4.4  # --autotest: seconds of the END screen before the verdict (and the quit)
        )
        log(f"warehouse_mission: screenshots -> {self.out}")

    def plan(self, t0: float, steps, tag: str = "") -> None:
        for dt, kind, arg in steps:
            self.queue.append((t0 + dt, kind, arg, tag))
        self.queue.sort(key=lambda item: item[0])

    def once(self, key: str) -> bool:
        if key in self.done:
            return False
        self.done.add(key)
        return True

    def on_state(self, state: str, t: float) -> None:
        if state == "BOOT" and self.once("boot"):
            self.plan(
                t,
                [
                    (1.8, "shot", "01a_boot_dark"),
                    (3.2, "shot", "01_boot_system_check"),
                    (5.0, "shot", "01b_boot_lights_on"),
                ],
            )
        elif state == "BRIEFING" and self.once("briefing"):
            self.plan(
                t,
                [
                    (3.4, "shot", "02_briefing_pickup_rack"),
                    (7.3, "shot", "03_briefing_dock"),
                    (10.7, "shot", "04_briefing_robot"),
                ],
            )
        elif state == "END" and self.once("end"):
            self.queue = [item for item in self.queue if item[3] != "tour"]
            # (the score has counted up, the rank title and the R / N prompt are shown by then)
            self.plan(t, [(3.8, "shot", "20_complete_scoreboard")])
        elif state == "FAULT" and self.once("fault"):
            self.queue = [item for item in self.queue if item[3] != "tour"]  # nothing left to tour
            self.plan(t, [(2.4, "shot", "fault_screen")])
        elif state == "IMPORT_ERROR" and self.once("import"):
            self.plan(t, [(2.4, "shot", "import_error_screen")])

    def on_event(self, name: str, data: dict, t: float) -> None:
        k = ord
        if name == "COUNTDOWN" and data.get("count") == 2 and self.once("countdown"):
            self.plan(t, [(0.5, "shot", "05_countdown")])
        elif name == "DEPLOYED" and self.once("deployed"):
            self.plan(
                t, [(0.45, "shot", "06a_deploy_particles"), (2.6, "shot", "06_localizing_vision")]
            )
        elif (
            name == "LOCKED"
            and self.app.mission is not None
            and self.app.mission.scenario.key == "obstacle_test"
            and "tour" not in self.done
        ):  # OBSTACLE TEST: the director keeps the camera (it shows the change)
            self.done.update(("locked", "tour"))
            self.plan(t, [(0.35, "shot", "07a_lock_ring")])
        elif name == "LOCKED" and self.once("locked") and self.app.chaos is not None:
            # --chaos: the AUTO-DIRECTOR keeps the camera for the obstacle and the kidnapping that
            # follow (the camera tour is the plain run's job).
            self.plan(t, [(0.35, "shot", "07a_lock_ring"), (2.0, "shot", "07_locked_overhead")])
        elif (
            name == "LOCKED" and self.app.mission is not None and self.app.mission.scenario.is_tour
        ):
            if self.once("locked"):  # a sensor test: no camera tour (the test is shot instead)
                self.plan(t, [(0.35, "shot", "07a_lock_ring")])
        elif name in ("SENSORS", "TEST_START") and self.once("sensors"):
            # A test: early (an unknown start: the particles all over the floor) and later on.
            key = str(data.get("scenario", "test"))
            early = 1.5 if name == "SENSORS" and not data.get("known_start", True) else 6.0
            steps = [(early, "shot", f"30_{key}_early"), (15.0, "shot", f"30_{key}_late")]
            self.plan(t, ([(0.1, "key", k("2"))] if name == "SENSORS" else []) + steps)
        elif name == "LOCKED" and "tour" not in self.done and self.app.chaos is None:
            self.done.update(("locked", "tour"))
            self.tour_active = True
            self.plan(
                t,
                [  # every camera mode once, the belief/map layers, and the night shift
                    (0.35, "shot", "07a_lock_ring"),
                    (2.0, "shot", "07_locked_overhead"),
                    (4.2, "key", k("1")),
                    (6.9, "shot", "08_cam1_chase"),
                    (7.1, "key", k("3")),
                    (8.6, "shot", "09_cam3_robot_pov"),
                    (8.8, "key", k("4")),
                    (10.6, "shot", "10_cam4_orbit"),
                    (10.8, "key", k("2")),
                    (10.9, "key", k("B")),
                    (11.0, "key", k("B")),
                    (11.1, "key", k("M")),
                    (13.6, "shot", "11_cam2_overhead_both_value"),
                    (13.7, "key", k("M")),
                    (13.8, "key", k("M")),
                    (13.9, "key", k("B")),
                    (14.0, "key", k("0")),
                    (14.6, "shot", "12_cam0_free"),
                    (14.62, "key", k("H")),
                    (15.4, "shot", "12b_help_panel"),
                    (15.45, "key", k("H")),
                    (15.5, "key", k("1")),
                    (15.6, "key", k("L")),
                    (18.3, "shot", "13_night_shift_chase"),
                    (18.5, "key", k("L")),
                    (18.7, "key", k("5")),
                    (18.8, "tour_end", None),
                ],
                tag="tour",
            )
        elif name == "MARKER_ACQUIRED" and self.tour_active:
            self.abort_tour()
        elif name == "PACKAGE_LOADED" and self.once("pickup"):
            self.plan(t, [(1.3, "shot", "14_pickup_closeup")])
        elif name == "PHASE" and data.get("to") == core.TO_DOCK and self.once("enroute"):
            self.plan(
                t,
                [
                    (3.0, "key", k("1")),
                    (5.2, "shot", "15_enroute_chase_path"),
                    (5.4, "key", k("5")),
                ],
            )
        elif name == "DELIVERED" and self.once("dropoff"):
            self.plan(t, [(1.9, "shot", "16_dropoff_conveyor")])
        elif name == "OBSTACLE_FALLING" and self.once(
            "obstacle"
        ):  # the drop animation just started
            self.plan(t, [(1.8, "shot", "17_chaos_obstacle"), (3.4, "shot", "17b_chaos_replanned")])
            if self.app.mission is not None and self.app.mission.scenario.key == "obstacle_test":
                self.plan(t, [(6.5, "shot", "17c_value_change")])  # (the overhead change view)
        elif name == "KIDNAPPED" and self.once("kidnap"):
            self.plan(t, [(0.3, "shot", "18_chaos_kidnap")])
        elif name == "WAITING" and self.once("forklift"):  # --forklifts: the robot gives way
            self.plan(t, [(0.3, "shot", "21_forklift_waiting")])
        elif name == "NEW_ORDER" and self.once(
            "new_order"
        ):  # --orders 3: the next parcel lights up
            self.plan(t, [(3.5, "shot", "22_new_order")])  # (the camera has left the dock by then)
        elif name == "LOST" and self.once("lost"):
            self.plan(t, [(1.0, "shot", "19_lost_relocalizing")])

    def abort_tour(self) -> None:
        """Docking started during the camera tour: restore the view and the lights now."""
        self.tour_active = False
        self.queue = [item for item in self.queue if item[3] != "tour"]
        log("screenshots: docking started during the camera tour; the rest of the tour is skipped")
        app = self.app
        if app.scene.night:
            app.on_key(ord("L"))
        while app.layers.map_layer != "none":
            app.on_key(ord("M"))
        while app.layers.belief != "particles":
            app.on_key(ord("B"))
        if app.camera.mode != MODE_AUTO:
            app.on_key(ord("5"))

    def update(self, t: float) -> None:
        while self.queue and self.queue[0][0] <= t:
            _, kind, arg, _ = self.queue.pop(0)
            self.handle(kind, arg, t)

    def handle(self, kind: str, arg: Any, t: float) -> None:
        """Run one planned step: ``key`` (press it), ``shot`` (export the view), ``tour_end``."""
        if kind == "key":
            self.app.on_key(arg)
        elif kind == "tour_end":
            self.tour_active = False
        elif kind == "shot":
            path = self.out / f"{arg}.png"
            self.app.sup.exportImage(str(path), 95)
            self.count += 1
            cam = self.app.camera
            eye = ", ".join(f"{v:.1f}" for v in cam.out[0])
            robot = ", ".join(f"{v:.1f}" for v in self.app.robot_xy)
            log(
                f"screenshot: {path.name}  [{cam.shot_name}]  eye ({eye})  robot ({robot})  "
                f"robot eye {math.degrees(self.app.psi) % 360:.0f} deg"
            )


class DocsShotPlanner(ShotPlanner):
    """``--screenshots DIR --shot-set docs`` (staff): the handful of polished pictures of the
    student guide.

    Each shot is staged once, as the mission allows: the particle cloud of
    the global localization, a chase shot in an aisle, an RFID ping seen
    from above its beacon, the value function in perspective, the pickup,
    the delivery, the night shift, and the MISSION COMPLETE screen. Staged
    shots are skipped (never forced) once the robot starts docking.
    """

    def __init__(self, app: App, out_dir: Path):
        super().__init__(app, out_dir)
        self.end_hold = 5.2
        self.locked_at: float | None = None
        self.phase_t0 = 0.0  # when the current mission phase began
        self.busy_until = -1e9  # a staged shot owns the camera until then
        self.aligned: tuple[str, float] | None = (
            None  # (name, deadline) of a shot waiting for a straight drive
        )

    def on_state(self, state: str, t: float) -> None:
        if state == "END" and self.once("end"):
            self.plan(t, [(4.6, "shot", "webots_mission_complete")])

    def on_event(self, name: str, data: dict, t: float) -> None:
        app = self.app
        if name == "PHASE":
            self.phase_t0 = t
        if name == "DEPLOYED" and self.once("deployed"):
            # Global localization: the 2000 guesses cover the whole floor while the first
            # measurement weighs them (the unlikely ones fade, the ones near the robot grow).
            self.plan(t, [(0.9, "shot", "webots_global_localization")])
        elif name == "LOCKED" and self.locked_at is None:
            self.locked_at = t
        elif (
            name == "RFID_PING"
            and self.ready(t)
            and float(data.get("range", 0.0)) >= 4.0
            and self.once("rfid")
        ):
            bx, by = (float(v) for v in data.get("position", core.BEACONS[int(data["beacon"])]))
            side = (
                -1.0 if by < 25.0 else 1.0
            )  # from the end of the block, high enough for the whole ring
            self.stage(
                t,
                1.6,
                [
                    (
                        0.0,
                        "still",
                        {
                            "eye": (bx + 1.5, by + 12.0 * side, 25.0),
                            "target": (bx, by, 0.0),
                            "fov": 0.9,
                        },
                    ),
                    (0.2, "shot", "webots_rfid_annulus"),
                ],
            )
        elif name == "MARKER_ACQUIRED":  # docking: give the camera back to the AUTO-DIRECTOR
            self.queue = [item for item in self.queue if item[3] != "staged"]
            self.busy_until = -1e9
            self.aligned = None
            if app.camera.script is not None and app.camera.script.get("kind") == "still":
                app.camera.end_script()
            if app.camera.mode != MODE_AUTO:
                self.handle("key", ord("5"), t)
            if app.scene.night:
                self.handle("key", ord("L"), t)
            while app.layers.map_layer != "none":
                self.handle("key", ord("M"), t)
        elif name == "PACKAGE_LOADED" and self.once("pickup"):
            self.plan(t, [(1.5, "shot", "webots_pickup")])  # the parcel lands on the deck
        elif name == "DELIVERED" and self.once("dropoff"):
            self.plan(t, [(1.7, "shot", "webots_delivery")])

    def ready(self, t: float) -> bool:
        """A staged shot may start: locked for a while, no other staged shot, not docking."""
        r = self.app.result
        return (
            self.locked_at is not None
            and t - self.locked_at > 3.0
            and t >= self.busy_until
            and r is not None
            and r.phase in (core.TO_PICKUP, core.TO_DOCK)
        )

    def stage(self, t: float, duration: float, steps) -> None:
        self.busy_until = t + duration
        self.plan(t, steps + [(duration, "restore", None)], tag="staged")

    def update(self, t: float) -> None:
        app = self.app
        r = app.result
        if r is not None and self.ready(t):
            x, y = app.robot_xy
            in_aisle = 15.0 < y < 34.0 and any(x1 + 1.0 < x < x2 - 1.0 for x1, x2 in AISLES_X)
            if r.phase == core.TO_PICKUP and in_aisle and self.once("chase"):
                self.stage(t, 2.6, [(0.0, "key", ord("1")), (2.4, "shot", "webots_chase_aisle")])
            elif r.phase == core.TO_PICKUP and "chase" in self.done and self.once("value"):
                self.stage(
                    t,
                    2.4,
                    [
                        (0.0, "key", ord("M")),
                        (
                            0.0,
                            "still",
                            {"eye": (18.0, -24.0, 52.0), "target": (52.0, 26.0, 0.0), "fov": 0.8},
                        ),
                        (2.1, "shot", "webots_value_function"),
                        (2.2, "key", ord("M")),
                        (2.25, "key", ord("M")),
                    ],
                )
            elif r.phase == core.TO_DOCK and t - self.phase_t0 > 5.0 and self.once("night"):
                # The chase camera settles, then the shot waits (at most 3 s) for the robot to drive
                # straight away from the camera, its headlights lighting the aisle ahead.
                self.stage(
                    t,
                    6.0,
                    [
                        (0.0, "key", ord("L")),
                        (0.0, "key", ord("1")),
                        (1.8, "aligned_shot", ("webots_night_shift", 3.5)),
                    ],
                )
        if self.aligned is not None:
            name, deadline = self.aligned
            travel, (hx, hy) = app.travel, app.camera.heading
            # The robot drives away from the camera and its eye (and headlights) look the same way.
            straight = (
                travel is not None
                and travel[0] * hx + travel[1] * hy > 0.97
                and app.camera.blend is None
                and math.cos(app.psi - math.atan2(hy, hx)) > 0.99
            )
            if straight or t >= deadline:
                self.aligned = None
                log(
                    f"screenshots: {name} {'aligned' if straight else 'NOT aligned (deadline)'}: "
                    f"eye "
                    f"{math.degrees(app.psi) % 360:.0f} deg, camera "
                    f"{math.degrees(math.atan2(hy, hx)) % 360:.0f} deg"
                )
                super().handle("shot", name, t)
                if app.scene.night:
                    self.handle("key", ord("L"), t)
                self.handle("restore", None, t)
                self.busy_until = t + 0.5
        super().update(t)

    def handle(self, kind: str, arg: Any, t: float) -> None:
        """Also ``still`` (a fixed camera shot), ``restore`` (back to the AUTO-DIRECTOR) and
        ``aligned_shot`` ((name, max wait): shoot once the robot drives straight away from the
        camera). Staged key presses leave no toast in the pictures."""
        camera = self.app.camera
        if kind == "key":
            self.app.on_key(arg)
            self.app.hud.toast("")
        elif kind == "aligned_shot":
            name, wait = arg
            self.aligned = (name, t + wait)
        elif kind == "still":
            camera.start_script("still", t, snap=True, **arg)
        elif kind == "restore":
            if camera.script is not None and camera.script.get("kind") == "still":
                camera.end_script()
            if camera.mode != MODE_AUTO:
                self.handle("key", ord("5"), t)
        else:
            super().handle(kind, arg, t)


# The aisles between the rack blocks (x ranges, m), for the docs planner's "in an aisle" test.
AISLES_X = [
    (core.SHELF_RECTS[k][2], core.SHELF_RECTS[k + 1][0]) for k in range(len(core.SHELF_RECTS) - 1)
]


###############################################################################
#                                 Entry point                                 #
###############################################################################


def main(controller_dir: Path | None = None) -> None:
    controller_dir = (
        Path(controller_dir) if controller_dir is not None else Path(__file__).resolve().parents[2]
    )
    app = App(parse_args(), controller_dir)
    try:
        app.run()
    finally:
        app._trace_files.close()
