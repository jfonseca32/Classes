"""Camera modes, scripted flyovers, and smooth transitions for the warehouse show.

Keys (DESIGN section 8): 1 CHASE, 2 OVERHEAD, 3 ROBOT POV, 4 CINEMATIC ORBIT,
5 AUTO-DIRECTOR, 0 FREE (the mouse owns the camera; nothing is written).

Every frame the active mode (or script) computes a *shot*: eye, target, and
field of view. Two mechanisms make the result smooth:

* **tracking**: a critically damped spring (:class:`util.SmoothDamp`) follows
  the shot of the current mode with a short, mode-specific smoothing time
  (ROBOT POV is rigid: it is bolted to the robot's eye);
* **transitions**: every cut (a mode switch, an AUTO-DIRECTOR shot change, the
  next shot of a script) blends from the view on screen to the new, live shot
  with an ease-in-out curve; long moves rise in a "crane" arc so that the
  camera never flies through the racks. During blends into or out of a steep
  shot (OVERHEAD) the steep in-between views take the blend's interpolated
  heading, so the picture turns smoothly to "north up" instead of spinning
  around a target that passes almost straight below the camera.

OVERHEAD frames the whole building for the current view shape (the HUD
measures the 3D view's aspect ratio): the building fits between the HUD's top
band and the bottom band, which holds the minimap overlay and, in this view,
the HUD's compact telemetry block. High shots hide the roof and the yard
lights (whose painted light pools would read as white blobs from above).
An experiment tour sets an *area* (its loop): OVERHEAD then frames that area
instead, and the AUTO-DIRECTOR stays on it, so the size of the particle cloud
is easy to compare between the sensor tests.

The AUTO-DIRECTOR (VIZ_SPEC 3.3 "Cameras"): OVERHEAD from deployment until
the first lock (+4 s), a low close-up while docking, OVERHEAD after a
kidnapping, a failed docking or a loss, an over-the-shoulder "reveal" of an
obstacle falling onto the route, and otherwise chase and orbit shots
alternating every 7 s.

Robot-following shots keep to free space: a ray from the robot toward the
camera stops at the rack blocks and the walls, so the camera comes closer
instead of clipping into the shelves (and flies above the racks only when
there is no room at all).

Webots Viewpoints look along their local +x axis with z up; see
:func:`util.look_at`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import suppress

from ..core import SHELF_RECTS
from . import util
from .scene import Prop, Scene

MODE_FREE, MODE_CHASE, MODE_OVERHEAD, MODE_POV, MODE_ORBIT, MODE_AUTO = 0, 1, 2, 3, 4, 5
MODE_NAMES = {
    MODE_FREE: "FREE",
    MODE_CHASE: "CHASE",
    MODE_OVERHEAD: "OVERHEAD",
    MODE_POV: "ROBOT POV",
    MODE_ORBIT: "ORBIT",
    MODE_AUTO: "AUTO",
}

DEFAULT_FOV = 0.785398  # horizontal field of view (rad) of the Viewpoint
# OVERHEAD: the building (floor and walls, m: x0, y0, x1, y1) must fit in this part of the view
# (fractions x0, x1, y0, y1): below the HUD's top band (name, banner, clock) and above the bottom
# band, where the HUD's compact overhead telemetry sits right of the minimap overlay (which starts
# at y = 0.72: it may hide a strip of the south wall, nothing of the floor).
BUILDING = (-2.0, -2.0, 101.0, 51.0)
OVERHEAD_SAFE = (0.012, 0.988, 0.118, 0.75)
STEEP = (
    1.1  # rad: a blend with an end steeper than this interpolates the heading (see _orientation)
)
RACK_CLEARANCE_Z = (
    7.4  # above the racks (~5.5 m) and the hanging signs (~7 m), below the lamps (8.4 m)
)
HIGH_SHOT_Z = 12.0  # m: scripted shots above this hide the roof and the yard lights
HEADING_TURN_RATE = math.radians(
    80.0
)  # rad/s: the fastest turn of the chase / POV cameras' heading
FREE_AERIAL_Z = 20.0  # m: a FREE camera the user flew above this height gets the same clean view
INSIDE = (0.2, 98.8, 0.2, 48.8)  # the camera stays inside the walls in robot-following shots
SHELVES = [tuple(float(v) for v in rect) for rect in SHELF_RECTS]

Vec3 = tuple[float, float, float]
Shot = tuple[Vec3, Vec3, float]


def overhead_shot(aspect: float, area: Sequence[float] | None = None) -> Shot:
    """The whole warehouse (or ``area``: x0, y0, x1, y1) from straight above, north up, fitted to
    OVERHEAD_SAFE for this view shape."""
    aspect = util.clamp(aspect, 0.8, 3.5)
    x0, x1, y0, y1 = OVERHEAD_SAFE
    bx0, by0, bx1, by1 = BUILDING if area is None else area
    width = max(
        (bx1 - bx0) / (x1 - x0), (by1 - by0) / (y1 - y0) * aspect
    )  # ground width in view (m)
    height = width / aspect
    cx = 0.5 * (bx0 + bx1) - (0.5 * (x0 + x1) - 0.5) * width
    cy = 0.5 * (by0 + by1) + (0.5 * (y0 + y1) - 0.5) * height
    z = 0.5 * width / math.tan(0.5 * DEFAULT_FOV)
    return (cx, cy, z), (cx, cy + 0.01 * z, 0.0), DEFAULT_FOV  # the target 1% north: "north up"


def _view_angles(eye: Sequence[float], target: Sequence[float]) -> tuple[float, float, float]:
    """(yaw, pitch, distance) of the view from ``eye`` to ``target`` (pitch > 0 looks down)."""
    dx, dy, dz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
    horizontal = math.hypot(dx, dy)
    return (
        math.atan2(dy, dx),
        math.atan2(-dz, horizontal),
        math.sqrt(horizontal * horizontal + dz * dz),
    )


def _nodes_of_type(supervisor, type_name: str) -> list:
    """The top-level nodes of the world whose (PROTO) type is ``type_name``."""
    try:
        children = supervisor.getRoot().getField("children")
        nodes = (children.getMFNode(i) for i in range(children.getCount()))
        return [node for node in nodes if node is not None and node.getTypeName() == type_name]
    except Exception:  # noqa: BLE001 - only a cosmetic tweak
        return []


def in_shelf(x: float, y: float, margin: float = 0.6) -> bool:
    for x0, y0, x1, y1 in SHELVES:
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
            return True
    return False


def constrain(eye: Sequence[float], margin: float = 0.7) -> Vec3:
    """Keep a camera inside the building, and above a rack block it would be inside of."""
    x = util.clamp(eye[0], INSIDE[0], INSIDE[1])
    y = util.clamp(eye[1], INSIDE[2], INSIDE[3])
    z = eye[2]
    if in_shelf(x, y, margin):
        z = max(z, RACK_CLEARANCE_Z)
    return (x, y, z)


def _blocked(px: float, py: float) -> bool:
    return not (INSIDE[0] <= px <= INSIDE[1] and INSIDE[2] <= py <= INSIDE[3]) or in_shelf(px, py)


def free_distance(
    x: float, y: float, dx: float, dy: float, max_d: float, step: float = 0.5
) -> float:
    """How far (up to ``max_d``) one can go from (x, y) along (dx, dy) without entering a rack or a
    wall.

    Marches in ``step`` increments, then refines the first blocked step by
    bisection, so the result varies continuously as the direction turns.
    """
    free, d = 0.0, step
    while d <= max_d + 1e-9:
        if _blocked(x + dx * d, y + dy * d):
            lo, hi = free, d
            for _ in range(5):
                mid = 0.5 * (lo + hi)
                if _blocked(x + dx * mid, y + dy * mid):
                    hi = mid
                else:
                    lo = mid
            return lo
        free = d
        d += step
    return max_d


class CameraInput:
    """What the camera needs from the app each frame."""

    __slots__ = ("robot", "travel", "dock")

    def __init__(self, robot=(0.0, 0.0, 0.0), travel=None, dock=None):
        self.robot = robot  # (x, y, z) of the robot origin
        self.travel = travel  # unit (dx, dy) of the current motion, or None when standing
        self.dock = dock  # StepResult.dock while docking, else None


class CameraDirector:
    """Drives the Viewpoint: modes, AUTO-DIRECTOR, scripts, transitions, roof hiding."""

    def __init__(self, supervisor, scene: Scene):
        self.sup = supervisor
        self.scene = scene
        self.vp = supervisor.getFromDef("VIEWPOINT")
        self.roof = supervisor.getFromDef("ROOF")
        # Hidden from the camera in the high shots: the roof, and the yard lights (from above, their
        # painted light pools read as big white blobs around the building).
        self.high_hidden = [
            node for node in [self.roof] + _nodes_of_type(supervisor, "YardLight") if node
        ]
        vp = self.vp
        self.p_pos = Prop(vp.getField("position") if vp else None, "vec3", 1e-5)
        self.p_ori = Prop(vp.getField("orientation") if vp else None, "rot", 1e-6)
        self.p_fov = Prop(vp.getField("fieldOfView") if vp else None, "float", 1e-4)
        self.mode = MODE_AUTO
        self.script: dict | None = None
        self.aspect = 16.0 / 9.0  # 3D view width / height (the app passes on the HUD's measurement)
        self.roof_hidden = False
        self.yaw = 0.0
        self.now = 0.0
        self.heading = (1.0, 0.0)  # smoothed travel direction (chase and POV cameras)
        self.orbit_angle = -2.2
        self.orbit_squeeze = (
            0.0  # 0 .. 1: how much a rack or a wall squeezes the orbit (it hurries past)
        )
        eye, target, fov = self._read_viewpoint()
        self.out: Shot = (eye, target, fov)  # the view on screen
        self.track_eye = util.SmoothDamp(eye)
        self.track_target = util.SmoothDamp(target)
        self.track_fov = util.SmoothDamp((fov,))
        self.retrack = True  # snap the trackers to the next shot (after a cut)
        self.blend: dict | None = None
        self.blend_heading: float | None = None  # interpolated heading of a blend with a steep end
        # AUTO-DIRECTOR state
        self.auto_shot = "chase"
        self.auto_until = 0.0
        self.auto_hold_docking = False
        self.auto_hold_localizing = False
        self.auto_alternate = "orbit"
        self.focus: tuple[float, float] | None = (
            None  # (x, y) the "reveal" shot looks at (an obstacle)
        )
        self.area: tuple[float, float, float, float] | None = (
            None  # what OVERHEAD frames (a tour's loop)
        )
        self.reveal_turn: float | None = None  # the reveal shot's side (rad), chosen once per shot
        self.debug = (
            (0.0, 0.0, 0.0),
            False,
            False,
        )  # (shot eye, retracked, blending) of the last frame
        # FREE mode: where the user put the camera (read from Webots at ~4 Hz; a getter is a round
        # trip).
        self.free_eye: Vec3 | None = None
        self.free_read_t = -1e9

    # ======================================================================
    # Public API
    # ======================================================================
    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.mode]

    @property
    def shot_name(self) -> str:
        """The effective shot (for the HUD and QA logs)."""
        if self.script is not None:
            return self.script["kind"].upper()
        if self.mode == MODE_AUTO:
            return "AUTO " + self.auto_shot.upper()
        return self.mode_name

    def set_area(self, area: Sequence[float] | None) -> None:
        """Frame ``area`` (x0, y0, x1, y1) in OVERHEAD, and keep the AUTO-DIRECTOR on it; None: the
        whole building.
        """
        self.area = None if area is None else tuple(float(v) for v in area)
        if self.area is not None and self.mode == MODE_AUTO:
            self.auto_shot = "overhead"

    @property
    def overhead_view(self) -> bool:
        """True while the OVERHEAD shot is (or is becoming) the view on screen."""
        if self.script is not None:
            return False
        return self.mode == MODE_OVERHEAD or (
            self.mode == MODE_AUTO and self.auto_shot == "overhead"
        )

    def set_mode(self, mode: int, t: float = 0.0) -> None:
        """Switch mode (keys 0-5). Cancels any script; blends smoothly to the new mode."""
        self.script = None
        if mode == self.mode:
            return
        if self.mode == MODE_FREE:
            self._sync_from_viewpoint()
        self.mode = mode
        if mode == MODE_FREE:  # the view stays where it is until the user moves it
            self.free_eye, self.free_read_t = self.out[0], -1e9
        self.cut()
        if mode == MODE_AUTO:
            self.auto_shot, self.auto_until = "chase", t + 7.0
            self.auto_hold_docking = self.auto_hold_localizing = False

    def cut(self, duration: float | None = None) -> None:
        """Blend from the view on screen to the next shot (duration: automatic from the distance; 0
        = snap).
        """
        self.retrack = True
        if duration is not None and duration <= 0.0:
            self.blend = None
            return
        eye, target, fov = self.out
        self.blend = {
            "t0": self.now,
            "duration": duration,
            "eye": eye,
            "target": target,
            "fov": fov,
        }

    def start_script(self, kind: str, t: float, snap: bool = False, **data) -> None:
        """``boot`` | ``briefing`` | ``countdown`` | ``celebrate`` | ``idle`` | ``still`` (eye=,
        target=).
        """
        self.now = t
        self.script = dict(data, kind=kind, t0=t)
        self.cut(0.0 if snap else None)

    def end_script(self, snap: bool = False) -> None:
        if self.script is not None:
            self.script = None
            self.cut(0.0 if snap else None)

    def auto_event(self, name: str, t: float, focus: Sequence[float] | None = None) -> None:
        """AUTO-DIRECTOR: pick a shot for a mission event (VIZ_SPEC 3.3 "Cameras").

        ``focus`` (x, y): where the event happens, for events shown with the
        "reveal" shot (an obstacle falling in front of the robot).
        """
        if (
            name == "DEPLOYED"
        ):  # global localization: smash cut to the whole particle cloud, until the lock
            self._auto("overhead", t, 15.0, snap=True)
            self.auto_hold_localizing = True
        elif name == "LOCKED":
            if self.auto_hold_localizing and self.auto_shot == "overhead":
                self.auto_until = t + 4.0  # 4 more seconds of the locked belief, then the chase
                self.auto_hold_localizing = False
            else:
                self._auto("overhead", t, 4.0)
        elif name in ("KIDNAPPED", "DOCKING_FAILED"):
            self._auto("overhead", t, 5.5)
        elif name == "LOST" and self.auto_shot != "overhead":
            self._auto("overhead", t, 4.0)
        elif (
            name == "VALUE_CHANGE"
        ):  # OBSTACLE TEST: the value-change layer is best seen from above
            self._auto("overhead", t, 6.0)
        elif name == "MARKER_ACQUIRED":
            self._auto("closeup", t, 0.0)
            self.auto_hold_docking = True
        elif name == "OBSTACLE_DROPPED" and not self.auto_hold_docking:
            # Over the robot's shoulder toward the barrier falling onto its route, then the re-plan.
            if focus is None and self.auto_shot == "reveal":
                return  # the engine's report of the drop the camera is already showing
            self.focus = (float(focus[0]), float(focus[1])) if focus is not None else None
            self.reveal_turn = None  # chosen on the shot's first frame (see _reveal)
            self._auto("reveal" if self.focus is not None else "chase", t, 4.5)

    # ======================================================================
    # Per frame
    # ======================================================================
    def update(self, dt: float, t: float, inp: CameraInput) -> None:
        self.now = t
        self._update_heading(dt, inp)
        if self.mode == MODE_FREE:  # the mouse owns the viewpoint: only read it now and then
            if t - self.free_read_t >= 0.25:
                self.free_read_t = t
                # If no viewpoint is available, keep the last value.
                with suppress(Exception):
                    self.free_eye = tuple(float(v) for v in self.p_pos.field.getSFVec3f())
            self._set_roof_hidden(self.free_eye is not None and self.free_eye[2] > FREE_AERIAL_Z)
            return
        if self.script is not None:
            shot, track, hide_roof = self._script_shot(dt, t, inp)
        else:
            shot, track, hide_roof = self._mode_shot(dt, t, inp)
        eye_d, target_d, fov_d = shot
        self.debug = (eye_d, self.retrack, self.blend is not None)  # (for --trace)
        if self.retrack or track <= 0.0:
            self.retrack = False
            self.track_eye.reset(eye_d)
            self.track_target.reset(target_d)
            self.track_fov.reset((fov_d,))
            eye, target, fov = eye_d, target_d, fov_d
        else:
            eye = self.track_eye.update(eye_d, track, dt)
            target = self.track_target.update(target_d, track, dt)
            fov = self.track_fov.update((fov_d,), track, dt)[0]
        self.blend_heading = None
        if self.blend is not None:
            eye, target, fov = self._blend(t, eye, target, fov)
        self.out = (eye, target, fov)
        rotation = self._orientation(eye, target)
        self.p_pos.set(eye)
        self.p_ori.set(rotation)
        self.p_fov.set(fov)
        self._set_roof_hidden(hide_roof and eye[2] > HIGH_SHOT_Z)

    # ======================================================================
    # Internals
    # ======================================================================
    def _blend(self, t: float, eye: Vec3, target: Vec3, fov: float) -> Shot:
        b = self.blend
        e0 = b["eye"]
        t0 = b["target"]
        yaw0, pitch0, _ = _view_angles(e0, t0)
        yaw1, pitch1, _ = _view_angles(eye, target)
        if b["duration"] is None:  # the first frame: the blend's length and style are chosen once
            # A style that changed from one frame to the next would make the camera jump.
            if max(pitch0, pitch1) > STEEP:
                b["style"] = "steep"
            elif util.dist2(t0, target) < 12.0:
                b["style"] = "polar"
            else:
                b["style"] = "linear"
            # Longer moves and wider turns take longer: turning the view (or swinging the camera
            # around the robot) half around takes ~2.4 s, so the picture never whips around
            # (smoothstep peaks at 1.5x the mean speed).
            turn = abs(util.wrap_angle(yaw1 - yaw0))
            if b["style"] == "polar":
                turn = max(
                    turn,
                    abs(
                        util.wrap_angle(
                            math.atan2(eye[1] - target[1], eye[0] - target[0])
                            - math.atan2(e0[1] - t0[1], e0[0] - t0[0])
                        )
                    ),
                )
            distance = math.sqrt(sum((a - c) ** 2 for a, c in zip(e0, eye, strict=False)))
            b["duration"] = util.clamp(0.9 + distance / 45.0 + 0.5 * turn, 0.9, 2.6)
            planar = util.dist2(e0, eye)
            b["arc"] = min(12.0, 0.2 * planar) if planar > 10.0 else 0.0  # crane over the racks
        w = (t - b["t0"]) / b["duration"]
        if w >= 1.0:
            self.blend = None
            return eye, target, fov
        e = util.smoothstep(w)
        blended_eye = (
            e0[0] + (eye[0] - e0[0]) * e,
            e0[1] + (eye[1] - e0[1]) * e,
            e0[2] + (eye[2] - e0[2]) * e + b["arc"] * math.sin(math.pi * e),
        )
        blended_target = (
            t0[0] + (target[0] - t0[0]) * e,
            t0[1] + (target[1] - t0[1]) * e,
            t0[2] + (target[2] - t0[2]) * e,
        )
        if b["style"] == "steep":  # into / out of OVERHEAD: see _orientation()
            # Unwrap the turn continuously from frame to frame: the shortest arc would switch sides
            # (a jump of the whole picture) when the live shot's heading passes behind the start's.
            turn = util.wrap_angle(yaw1 - yaw0)
            if "turn" in b:
                turn = b["turn"] + util.wrap_angle(turn - b["turn"])
            b["turn"] = turn
            self.blend_heading = yaw0 + turn * e
        elif b["style"] == "polar":
            # Both shots look at the robot: swing around it (polar blend about the moving target)
            # instead of cutting straight through it, which would flip the picture over its head.
            r0, r1 = util.dist2(e0, t0), util.dist2(eye, target)
            a0 = math.atan2(e0[1] - t0[1], e0[0] - t0[0])
            swing = util.wrap_angle(math.atan2(eye[1] - target[1], eye[0] - target[0]) - a0)
            if "swing" in b:  # unwrapped from frame to frame (see the heading above)
                swing = b["swing"] + util.wrap_angle(swing - b["swing"])
            b["swing"] = swing
            a = a0 + swing * e
            c, s = math.cos(a), math.sin(a)
            r = r0 + (r1 - r0) * e
            if min(r0, r1) > 1.3 and not _blocked(blended_target[0], blended_target[1]):
                # Stay in free space: come closer (never jump up over the racks).
                r = min(r, max(1.2, free_distance(blended_target[0], blended_target[1], c, s, r)))
            # The free distance jumps when the ray sweeps past a rack corner, so the radius changes
            # at a limited rate (the 0.6 m margin of in_shelf() keeps the racks out of view
            # meanwhile).
            if "r" not in b:  # the first frame starts from the view on screen
                b["r"], b["t_last"] = r0, t - 1.0 / 30.0
            step = max(t - b["t_last"], 0.0)
            r = util.clamp(r, b["r"] - 15.0 * step, b["r"] + 15.0 * step)
            b["r"], b["t_last"] = r, t
            h = (e0[2] - t0[2]) + ((eye[2] - target[2]) - (e0[2] - t0[2])) * e
            blended_eye = (
                blended_target[0] + r * c,
                blended_target[1] + r * s,
                blended_target[2] + h,
            )
        return blended_eye, blended_target, b["fov"] + (fov - b["fov"]) * e

    def _orientation(self, eye: Vec3, target: Vec3) -> list:
        """Viewpoint rotation looking from ``eye`` at ``target`` (updates ``self.yaw``).

        In a steep view the heading hardly moves the aim point but turns the
        whole picture, and the heading toward a target almost straight below
        swings wildly; so during a blend into or out of a steep shot, steep
        views take the smoothly interpolated heading of the blend instead.
        """
        dx, dy, dz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
        horizontal = math.hypot(dx, dy)
        pitch = math.atan2(-dz, horizontal) if (horizontal > 1e-9 or abs(dz) > 1e-9) else 0.0
        yaw = math.atan2(dy, dx) if horizontal > 1e-4 else self.yaw
        if self.blend_heading is not None:
            yaw = util.angle_lerp(yaw, self.blend_heading, util.smoothstep((pitch - 0.9) / 0.45))
        self.yaw = yaw
        return util.yaw_pitch(yaw, pitch)

    def _read_viewpoint(self) -> Shot:
        try:
            eye = tuple(float(v) for v in self.p_pos.field.getSFVec3f())
            rotation = self.p_ori.field.getSFRotation()
            fov = float(self.p_fov.field.getSFFloat())
        except Exception:  # noqa: BLE001 - no viewpoint (never in the real world)
            return (-24.0, -30.0, 36.0), (50.0, 24.0, 0.0), DEFAULT_FOV
        fwd = util.forward_from_axis_angle(rotation)
        target = (eye[0] + 10.0 * fwd[0], eye[1] + 10.0 * fwd[1], eye[2] + 10.0 * fwd[2])
        return eye, target, fov

    def _sync_from_viewpoint(self) -> None:
        """The user moved the camera in FREE mode: continue smoothly from there."""
        self.out = self._read_viewpoint()
        for prop in (self.p_pos, self.p_ori, self.p_fov):
            prop.forget()

    def _set_roof_hidden(self, hidden: bool) -> None:
        """Hide the roof (and the yard lights) from the viewpoint in the high shots."""
        if hidden == self.roof_hidden or self.vp is None:
            return
        self.roof_hidden = hidden
        for node in self.high_hidden:
            node.setVisibility(self.vp, not hidden)

    def _update_heading(self, dt: float, inp: CameraInput) -> None:
        """Smooth the travel direction as an angle: when the robot turns around, the chase and POV
        cameras swing around it (a vector average would shrink to nothing and flip)."""
        if inp.travel is None:
            return
        current = math.atan2(self.heading[1], self.heading[0])
        turn = util.wrap_angle(
            math.atan2(inp.travel[1], inp.travel[0]) - current
        ) * util.smooth_factor(dt, 0.8)
        limit = (
            HEADING_TURN_RATE * dt
        )  # at a corner the camera swings around the robot, never whips
        heading = current + util.clamp(turn, -limit, limit)
        self.heading = (math.cos(heading), math.sin(heading))

    def _auto(self, shot: str, t: float, hold: float, snap: bool = False) -> None:
        """Show ``shot`` for at least ``hold`` seconds (extends the current shot if it is the
        same).
        """
        same = shot == self.auto_shot
        if snap:
            self.cut(0.0)
        elif not same:
            self.cut()
        self.auto_until = max(self.auto_until, t + hold) if same else t + hold
        self.auto_shot = shot
        self.auto_hold_docking = False
        self.auto_hold_localizing = False

    # --- the shots of the modes ------------------------------------------------
    def _mode_shot(self, dt: float, t: float, inp: CameraInput):
        mode = self.mode
        if (
            mode == MODE_AUTO and self.area is not None
        ):  # an experiment tour: always its loop from above
            self.auto_shot = "overhead"
            shot = "overhead"
        elif mode == MODE_AUTO:
            if self.auto_hold_docking and inp.dock is None:
                self._auto("chase", t, 7.0)
            elif not self.auto_hold_docking and t >= self.auto_until:
                if inp.dock is not None:
                    self._auto("closeup", t, 0.0)
                    self.auto_hold_docking = True
                else:
                    shot = self.auto_alternate
                    self.auto_alternate = "chase" if shot == "orbit" else "orbit"
                    self._auto(shot, t, 7.0)
            shot = self.auto_shot
        else:
            shot = {
                MODE_CHASE: "chase",
                MODE_OVERHEAD: "overhead",
                MODE_POV: "pov",
                MODE_ORBIT: "orbit",
            }[mode]
        return self._named_shot(shot, dt, inp)

    def _named_shot(self, shot: str, dt: float, inp: CameraInput):
        x, y, z = inp.robot
        if shot == "overhead":
            return overhead_shot(self.aspect, self.area), 0.3, True
        if shot == "pov":
            ex, ey, ez = self.scene.eye_position()
            hx, hy = self.heading
            eye = (ex + 0.16 * hx, ey + 0.16 * hy, ez + 0.03)
            target = (eye[0] + 10.0 * hx, eye[1] + 10.0 * hy, eye[2] - 1.3)
            return (eye, target, 1.15), 0.0, False
        if shot == "orbit":
            self.orbit_angle += (
                0.12 * (1.0 + 3.0 * self.orbit_squeeze) * dt
            )  # hurry past racks and walls
            return self._orbit(x, y, z, 9.0, 0.5), 0.5, False
        if shot == "closeup" and inp.dock is not None:
            return self._closeup(inp.dock), 0.45, False
        if shot == "reveal" and self.focus is not None:
            return self._reveal(x, y, z, self.focus), 0.4, False
        return self._chase(x, y, z), 0.35, False  # chase (and the closeup / reveal fallback)

    def _chase(self, x: float, y: float, z: float) -> Shot:
        """Behind and above the robot along its (smoothed) travel direction (~6 m back, 3.5 m
        up).
        """
        hx, hy = self.heading
        back = free_distance(x, y, -hx, -hy, 6.0)
        if back >= 1.8:
            # A rack or a wall close behind: rise and look down more steeply (the robot stays
            # small).
            rise = 0.9 * max(0.0, 4.0 - back)
            eye = (x - back * hx, y - back * hy, 1.9 + 0.27 * back + rise + z)
            ahead = 0.33 * back  # closer camera, closer target: the robot stays in the frame
        else:  # no room behind: look down from above the racks
            eye = constrain((x - 6.0 * hx, y - 6.0 * hy, RACK_CLEARANCE_Z + z))
            ahead = 1.0
        return eye, (x + ahead * hx, y + ahead * hy, 0.6 + z), 0.85

    def _reveal(self, x: float, y: float, z: float, focus: tuple[float, float]) -> Shot:
        """Over the robot's shoulder toward ``focus`` (a falling obstacle): both in the frame.

        The camera stands behind the robot as seen from the obstacle, or (a wall
        or a rack too close behind) turned up to ~65 degrees to one side; the
        side is chosen once per shot so that the camera never jumps.
        """
        fx, fy = focus
        dx, dy = fx - x, fy - y
        d = math.hypot(dx, dy)
        if d < 1.0:
            return self._chase(x, y, z)
        ux, uy = dx / d, dy / d
        if self.reveal_turn is None:
            best = (0.0, -1.0)
            for turn in (0.0, 0.55, -0.55, 1.1, -1.1):
                c, s = math.cos(turn), math.sin(turn)
                room = free_distance(x, y, -(ux * c - uy * s), -(ux * s + uy * c), 7.0)
                if room >= 5.5:
                    best = (turn, room)
                    break
                if room > best[1]:
                    best = (turn, room)
            self.reveal_turn = best[0]
        c, s = math.cos(self.reveal_turn), math.sin(self.reveal_turn)
        bx, by = -(ux * c - uy * s), -(ux * s + uy * c)
        back = free_distance(x, y, bx, by, 7.0)
        if back >= 1.5:
            eye = (
                x + back * bx,
                y + back * by,
                2.6 + 0.35 * back + 0.08 * d + 0.8 * max(0.0, 4.0 - back) + z,
            )
        else:  # no room at all: from above the racks
            eye = constrain((x + 6.0 * bx, y + 6.0 * by, RACK_CLEARANCE_Z + z))
        return eye, (x + 0.4 * dx, y + 0.4 * dy, 0.6), 0.9

    def _orbit(self, x: float, y: float, z: float, radius: float, target_z: float) -> Shot:
        """A slow orbit that hugs the free space around the robot (an ellipse in an aisle)."""
        c, s = math.cos(self.orbit_angle), math.sin(self.orbit_angle)
        r = free_distance(x, y, c, s, radius)
        self.orbit_squeeze = util.clamp(1.0 - r / (0.6 * radius), 0.0, 1.0)
        if r >= 1.6:
            # Squeezed by a rack or a wall: rise and look down more steeply (the robot stays small).
            rise = 1.1 * max(0.0, 0.6 * radius - r)
            eye = (x + r * c, y + r * s, 1.4 + 0.4 * r + rise + z)
        else:  # docked right at a rack face: swing over the rack
            eye = constrain((x + radius * c, y + radius * s, RACK_CLEARANCE_Z + z))
        return eye, (x, y, target_z + z), 0.85

    def _closeup(self, dock: dict) -> Shot:
        """Low shot beside the robot looking at the rack slot / the dock door."""
        px, py = dock["pose"]
        fx, fy = dock["facing"]
        side = 1.0  # stand on the side of the robot the camera is already on
        ex, ey, _ = self.out[0]
        if (ex - px) * (-fy) + (ey - py) * fx < 0.0:
            side = -1.0
        qx, qy = -fy * side, fx * side
        if dock.get("target") == "dock":
            eye = (px - 4.9 * fx + 3.7 * qx, py - 4.9 * fy + 3.7 * qy, 2.7)
            return constrain(eye, margin=0.2), (px + 1.3 * fx, py + 1.3 * fy, 1.1), 0.8
        eye = (px - 2.75 * fx + 3.0 * qx, py - 2.75 * fy + 3.0 * qy, 1.75)
        return constrain(eye, margin=0.2), (px + 0.45 * fx, py + 0.45 * fy, 0.6), 0.75

    # --- scripts ------------------------------------------------------------------
    def _script_shot(self, dt: float, t: float, inp: CameraInput):
        sc = self.script
        u = t - sc["t0"]
        kind = sc["kind"]
        x, y, z = inp.robot
        if kind == "boot":
            # Establishing shot of the dark building from the south-west (the dusk sky behind the
            # SYSTEM CHECK text), then up over the south wall as the lamp rows power up.
            keys_eye = [
                (2.0, -40.0, 9.0),
                (18.0, -33.0, 13.0),
                (31.0, -24.0, 21.0),
                (38.0, -15.0, 29.0),
            ]
            keys_target = [
                (44.0, 12.0, 7.0),
                (48.0, 16.0, 4.0),
                (51.0, 21.0, 1.0),
                (52.0, 24.0, 0.0),
            ]
            s = util.smoothstep(u / sc.get("duration", 5.4))
            return (
                (util.catmull_rom(keys_eye, s), util.catmull_rom(keys_target, s), DEFAULT_FOV),
                0.25,
                False,
            )
        if kind == "briefing":
            holds = sc.get("holds", [])
            for start, end, shot in holds:
                if start <= u < end:
                    self._next_hold(sc, start)
                    return shot, 0.3, False
            self._next_hold(sc, holds[-1][1] if holds else 0.0)
            self.orbit_angle += 0.1 * dt
            # Aim above the robot: it sits in the lower part of the view, under the briefing card.
            return self._orbit(x, y, z, 7.0, 1.5), 0.5, False
        if kind == "still":  # a fixed shot (staff screenshots for the docs)
            return (
                (tuple(sc["eye"]), tuple(sc["target"]), sc.get("fov", DEFAULT_FOV)),
                0.3,
                sc["eye"][2] > 12.0,
            )
        if kind == "countdown":
            a = sc.setdefault("angle", self.orbit_angle)
            k = util.ease_in_out(u / 3.0)
            r = max(2.2, free_distance(x, y, math.cos(a), math.sin(a), util.lerp(5.6, 3.4, k)))
            eye = (x + r * math.cos(a), y + r * math.sin(a), util.lerp(2.4, 1.6, k) + z)
            return (eye, (x, y, 0.7 + z), 0.8), 0.3, False
        if kind == "celebrate":
            # The robot sits in the middle of the view, between the score table and the final score.
            facing = sc.get("facing")
            if facing is None:  # (no dock to show) a slow orbit
                self.orbit_angle += 0.18 * dt
                return self._orbit(x, y, z, 7.0, 1.25), 0.5, False
            # In front of the dock it just served: seen from behind, with the open door, the green
            # status light and the trailer beyond it; the camera sways slowly from side to side.
            fx, fy = facing
            a = math.atan2(-fy, -fx) + 0.2 + 0.4 * math.sin(0.25 * u)
            r = max(2.5, free_distance(x, y, math.cos(a), math.sin(a), 7.5))
            eye = (x + r * math.cos(a), y + r * math.sin(a), 1.2 + 0.36 * r + z)
            return (eye, (x + 1.2 * fx, y + 1.2 * fy, 1.3 + z), 0.85), 0.5, False
        # "idle": a calm, high orbit; the robot sits low in the view (10 degrees below its center,
        # also when a wall pushes the camera closer), under the fault screens' text and their
        # "[R] RETRY" prompt
        self.orbit_angle += 0.06 * dt
        c, s = math.cos(self.orbit_angle), math.sin(self.orbit_angle)
        eye = constrain((x + 9.0 * c, y + 9.0 * s, 7.0 + z))
        d = max(util.dist2(eye, (x, y)), 1.0)
        aim = math.atan2(eye[2] - (0.7 + z), d) - math.radians(10.0)  # the view center's pitch
        return (eye, (x, y, eye[2] - d * math.tan(aim)), 0.8), 0.8, False

    def _next_hold(self, sc: dict, start: float) -> None:
        if sc.get("hold") != start:
            if "hold" in sc:
                self.cut()
            sc["hold"] = start


def briefing_holds(slot, dock):
    """The briefing preview: the pickup slot, then the shipping dock (then an orbit of the robot).

    Returns ``[(start, end, shot), ...]`` with times relative to the start of
    the briefing.
    """
    sx, sy = slot.station
    fx = slot.facing[0]
    px, py = slot.parcel_xy
    # Down the aisle from the south, the rack face on one side; the camera aims high enough that
    # the glowing parcel (lowest rack level) and the goal marker sit below the briefing card.
    rack_eye = constrain((sx - 3.2 * fx, sy - 8.0, 3.6), margin=0.0)
    rack = (rack_eye, (0.55 * px + 0.45 * sx, sy, 2.0), 0.8)
    dx, dy = dock.station
    door = (constrain((dx - 5.5, dy - 6.0, 3.6), margin=0.0), (99.5, dy, 1.3), 0.8)
    return [(0.0, 3.8, rack), (3.8, 7.6, door)]
