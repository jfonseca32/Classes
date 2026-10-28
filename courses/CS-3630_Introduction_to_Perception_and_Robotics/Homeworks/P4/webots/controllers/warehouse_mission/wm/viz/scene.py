"""Every Webots node the warehouse show animates, and its 3D animations.

The :class:`Scene` owns all node and field handles (VIZ_SPEC section 2): the
robot's exposed PROTO fields (wheels, eye, deck, hologram plane, colors), the
lights, the RFID beacons, the shipping docks, the forklifts, the goal markers,
and the nodes imported at run time: the parcels and obstacles, and two glowing
meshes of the show itself, the ping shockwave ring (``DEF FX_RING``; the
world's ``PING_RING`` cannot change color) and the crisp 3D estimate reticle
of close camera views (``DEF HOLO_RETICLE``). Both are imported once and reused
by restarts (a copy left in a saved world is reused too).

Two rules keep it cheap (it runs every 32 ms frame):

* every write goes through :class:`Prop`, which remembers the last value and
  only calls Webots when the value changes (setters are batched by Webots
  until the next ``step``; getters would cost a round trip, so there are none
  per frame);
* time-based effects (beacon flashes, ping rings, doors, marker spin, falling
  obstacles, parcels riding a conveyor, the lights power-up) live here and are
  advanced by :meth:`Scene.update`; the app only triggers them.

All times are simulated seconds (``Supervisor.getTime()``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import suppress

from ..core import config as C
from . import util
from .util import Color

HOLO_CENTER = (49.5, 24.5, 0.02)  # world position of the 100 x 50 m hologram plane
HOLO_HIDDEN_Z = -5.0  # below the yard (z = -1.2): invisible from everywhere
EYE_OFFSET = (-0.4, 0.0, 1.18)  # EYE_PAN in the robot frame (the robot never rotates)
DECK_TOP = 0.64  # m, deck top at rest (a parcel's bottom rests here)
DECK_LIFT = 0.04  # m, deck raised while loaded
RACK_Z = C.PARCEL_DECK_HEIGHT  # 0.40 m, lowest rack level
CONVEYOR_Z = 0.64  # m, belt top
CONVEYOR_SPEED = 0.4  # m/s, the belt speed of LoadingDock (fixed at load time)
DOOR_CLEAR_X = 100.6  # m, a parcel beyond this x has passed the door
DOOR_OPEN_Z = 2.95
DOOR_SECONDS = 1.6  # full open/close
FORKLIFT_Z = 0.81
FORKLIFT_CENTER_OFFSET = 0.19  # the model's center is 0.19 m ahead of the PROTO origin
MARKER_HIDDEN_Z = -20.0
PING_PARK_Z = -2.0
FX_RING_DEF = "FX_RING"  # the ping ring (imported: the world's PING_RING cannot be colored)
BURST_SECONDS = 1.1  # MISSION COMPLETE: the goal beam launches out of the robot
BEACON_BASE_GLOW = 2.5
BEACON_FLASH_GLOW = 10.0
LAMP_NIGHT = 0.6
NIGHT_FACTOR = 0.15
FAULT_DIM = 0.4  # light level of the fault screens' emergency lighting
PANEL_DIM = (
    0.5  # light level while the help panel is open (the text reads better on a darker scene)
)
N_ORDERS_MAX = C.MAX_ORDERS

RETICLE_DEF = "HOLO_RETICLE"
RETICLE_Z = (
    0.045  # just above the floor hologram (z 0.02) and the goal markers' ground rings (0.04)
)
RETICLE_HIDDEN_Z = -5.0
GLOW_INTENSITY = (
    1.0  # emissive intensity of the imported glow meshes (brighter would saturate colors to white)
)

PICKUP_COLOR = (1.0, 0.82, 0.1)
DROPOFF_COLOR = (0.25, 1.0, 0.45)
DOCK_RED = (0.95, 0.08, 0.04)
DOCK_AMBER = (1.0, 0.5, 0.0)
DOCK_GREEN = (0.1, 1.0, 0.25)
PARCEL_GLOW = (1.0, 0.45, 0.0)

# robot_profile() colors -> (body, accent). GT gold with navy accents is the default.
PROFILE_COLORS: dict[str, tuple[Color, Color]] = {
    "gold": ((0.702, 0.639, 0.412), (0.0, 0.188, 0.341)),
    "navy": ((0.06, 0.17, 0.36), (0.702, 0.639, 0.412)),
    "orange": ((0.96, 0.46, 0.08), (0.08, 0.09, 0.12)),
    "cyan": ((0.08, 0.72, 0.86), (0.05, 0.08, 0.13)),
    "magenta": ((0.86, 0.2, 0.66), (0.08, 0.06, 0.13)),
    "lime": ((0.56, 0.86, 0.16), (0.07, 0.1, 0.08)),
    "white": ((0.92, 0.92, 0.9), (0.9, 0.35, 0.05)),
    "red": ((0.8, 0.1, 0.08), (0.09, 0.09, 0.12)),
}


class Prop:
    """One writable Webots field that is only written when its value changes.

    ``kind`` is ``"vec3" | "rot" | "color" | "float" | "bool"``. Values are
    converted to what the Python binding accepts (lists of floats, a float, a
    bool). A missing field (``None``) makes every write a no-op.
    """

    __slots__ = ("field", "kind", "last", "eps")

    def __init__(self, field, kind: str, eps: float = 1e-5):
        self.field = field
        self.kind = kind
        self.last = None
        self.eps = eps

    def set(self, value) -> None:
        field = self.field
        if field is None:
            return
        kind = self.kind
        if kind == "float":
            v = float(value)
            if self.last is not None and abs(v - self.last) <= self.eps:
                return
            field.setSFFloat(v)
        elif kind == "bool":
            v = bool(value)
            if v is self.last:
                return
            field.setSFBool(v)
        else:
            v = [float(x) for x in value]
            last = self.last
            if last is not None:
                eps = self.eps
                for a, b in zip(v, last, strict=False):
                    if abs(a - b) > eps:
                        break
                else:
                    return
            if kind == "vec3":
                field.setSFVec3f(v)
            elif kind == "rot":
                field.setSFRotation(v)
            else:
                field.setSFColor(v)
        self.last = v

    def forget(self) -> None:
        """Force the next write (after Webots may have changed the field itself)."""
        self.last = None


def _arc_band(
    points: list[str], faces: list[str], r0: float, r1: float, a0: float, a1: float, segments: int
) -> None:
    """Append a flat annulus sector (radii r0 < r1, angles a0 < a1) as quads, counterclockwise from
    above.
    """
    base = len(points)
    for i in range(segments + 1):
        a = a0 + (a1 - a0) * i / segments
        c, s = math.cos(a), math.sin(a)
        points.append(f"{r0 * c:.4f} {r0 * s:.4f} 0")
        points.append(f"{r1 * c:.4f} {r1 * s:.4f} 0")
    for i in range(segments):
        inner0, outer0 = base + 2 * i, base + 2 * i + 1
        faces.append(f"{inner0} {outer0} {outer0 + 2} {inner0 + 2} -1")


def _indexed_face_set(points: list[str], faces: list[str]) -> str:
    return (
        f"IndexedFaceSet {{ coord Coordinate {{ point [ {', '.join(points)} ] }} coordIndex "
        f"[ {' '.join(faces)} ] }}"
    )


def reticle_mesh(
    radius: float = 0.95,
    width: float = 0.07,
    gap_deg: float = 16.0,
    tick_in: float = 1.04,
    tick_out: float = 1.26,
    tick_width: float = 0.05,
    segments: int = 12,
) -> str:
    """The estimate reticle as a VRML ``IndexedFaceSet`` (flat, in the xy plane).

    Four arcs of an annulus (``radius`` +/- ``width`` / 2) separated by small
    gaps, and four radial ticks in the gaps: the crisp 3D counterpart of the
    floor hologram's estimate ring for close cameras. Faces are counterclockwise
    seen from above (the side Webots renders).
    """
    points: list[str] = []
    faces: list[str] = []
    gap = math.radians(gap_deg)
    for k in range(4):
        _arc_band(
            points,
            faces,
            radius - width / 2,
            radius + width / 2,
            k * math.pi / 2 + gap / 2,
            (k + 1) * math.pi / 2 - gap / 2,
            segments,
        )
    for k in range(4):  # ticks in the middle of the gaps, pointing outward
        a = k * math.pi / 2
        ux, uy = math.cos(a), math.sin(a)
        nx, ny = -uy * tick_width / 2, ux * tick_width / 2
        base = len(points)
        for rho, side in ((tick_in, -1.0), (tick_out, -1.0), (tick_out, 1.0), (tick_in, 1.0)):
            points.append(f"{rho * ux + side * nx:.4f} {rho * uy + side * ny:.4f} 0")
        faces.append(f"{base} {base + 1} {base + 2} {base + 3} -1")
    return _indexed_face_set(points, faces)


def ring_mesh(r0: float, r1: float, segments: int = 72) -> str:
    """A full flat annulus (radii ``r0`` < ``r1``) as a VRML ``IndexedFaceSet``."""
    points: list[str] = []
    faces: list[str] = []
    _arc_band(points, faces, r0, r1, 0.0, 2.0 * math.pi, segments)
    return _indexed_face_set(points, faces)


def drop_height(
    t: float, h0: float = 2.0, restitution: float = 0.3, bounces: int = 2
) -> tuple[float, int]:
    """Height of a box dropped from ``h0`` after ``t`` s with small bounces.

    Returns ``(z, impacts)``: the height and how many times it has hit the floor.
    """
    g = 9.81
    t_fall = math.sqrt(2.0 * h0 / g)
    if t < t_fall:
        return h0 - 0.5 * g * t * t, 0
    t -= t_fall
    v = restitution * g * t_fall
    for k in range(bounces):
        hop = 2.0 * v / g
        if t < hop:
            return v * t - 0.5 * g * t * t, k + 1
        t -= hop
        v *= restitution
    return 0.0, bounces + 1


class _Timed:
    """A running effect: start time and duration."""

    __slots__ = ("t0", "duration", "data")

    def __init__(self, t0: float, duration: float, data=None):
        self.t0, self.duration, self.data = t0, duration, data

    def s(self, t: float) -> float:
        return util.clamp((t - self.t0) / self.duration, 0.0, 1.0) if self.duration > 0 else 1.0


class ParcelVis:
    """An imported ParcelBox (one per order)."""

    def __init__(self, number: int, node, rack_xy: Sequence[float]):
        self.number = number
        self.node = node
        self.translation = Prop(node.getField("translation") if node else None, "vec3", 1e-4)
        self.glow = Prop(node.getField("glowColor") if node else None, "color", 2e-3)
        self.rack_xy = (float(rack_xy[0]), float(rack_xy[1]))
        self.pos = (self.rack_xy[0], self.rack_xy[1], RACK_Z)
        self.riding: tuple[float, int] | None = None  # (end x, dock index) while on a conveyor

    def move(self, xyz: Sequence[float]) -> None:
        self.pos = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self.translation.set(self.pos)


class ObstacleVis:
    """An imported SpillBarrier falling into place."""

    def __init__(self, oid: int, node, xy: Sequence[float], t0: float, animate: bool):
        self.id = oid
        self.node = node
        self.xy = (float(xy[0]), float(xy[1]))
        self.translation = Prop(node.getField("translation") if node else None, "vec3", 1e-4)
        self.flasher = Prop(node.getField("flasher") if node else None, "float", 0.05)
        self.t0 = t0
        self.animate = animate
        self.impacts = 0


class Scene:
    """Node handles and 3D animation of the warehouse world."""

    def __init__(self, supervisor, audio=None):
        self.sup = supervisor
        self.audio = audio
        self.missing: list[str] = []
        self.root_children = supervisor.getRoot().getField("children")

        # --- the robot (its exposed PROTO fields) ---------------------------
        robot = supervisor.getSelf()
        f = self._field
        self.p_translation = Prop(f(robot, "translation", "AMR"), "vec3", 1e-5)
        self.p_rotation = Prop(f(robot, "rotation", "AMR"), "rot")
        self.p_wheels = [Prop(f(robot, f"wheel{i}Rotation", "AMR"), "rot", 1e-4) for i in range(3)]
        self.p_eye = Prop(f(robot, "eyeRotation", "AMR"), "rot", 2e-4)
        self.p_deck = Prop(f(robot, "deckTranslation", "AMR"), "vec3", 1e-4)
        self.p_holo = Prop(f(robot, "holoTranslation", "AMR"), "vec3", 1e-5)
        self.p_body = Prop(f(robot, "bodyColor", "AMR"), "color")
        self.p_accent = Prop(f(robot, "accentColor", "AMR"), "color")
        self.p_led = Prop(f(robot, "ledColor", "AMR"), "color", 4e-3)
        self.p_iris = Prop(f(robot, "irisColor", "AMR"), "color", 4e-3)
        self.p_tag = Prop(f(robot, "tagColor", "AMR"), "color", 4e-3)
        self.p_headlights = Prop(f(robot, "headlightsOn", "AMR"), "bool")

        # --- lights ------------------------------------------------------------
        self.p_main = Prop(self._def_field("MAIN_LIGHT", "intensity"), "float", 2e-3)
        self.p_fill = Prop(self._def_field("FILL_LIGHT", "intensity"), "float", 2e-3)
        self.p_bg = Prop(self._def_field("BACKGROUND", "luminosity"), "float", 2e-3)
        self.base_main = self._read_float(self.p_main, 1.3)
        self.base_fill = self._read_float(self.p_fill, 0.45)
        self.base_bg = self._read_float(self.p_bg, 0.6)
        self.p_lamps: list[Prop] = []
        lamps = supervisor.getFromDef("CEILING_LAMPS")
        if lamps is None:
            self.missing.append("DEF CEILING_LAMPS")
        else:
            children = lamps.getField("children")
            for i in range(children.getCount()):
                self.p_lamps.append(
                    Prop(children.getMFNode(i).getField("intensity"), "float", 5e-3)
                )
        self.base_lamp = self._read_float(self.p_lamps[0], 6.0) if self.p_lamps else 6.0

        # --- beacons, ping ring, docks, forklifts, markers -----------------------
        self.p_beacon_glow = [
            Prop(self._def_field(f"BEACON_{k}", "glowIntensity"), "float", 0.02) for k in range(8)
        ]
        self.beacon_flash: dict[int, float] = {}  # beacon id -> flash start time
        self._init_fx_ring()  # the ping shockwave ring: p_ping_pos, p_ping_scale, p_ping_bands
        self.ping_fx: _Timed | None = None
        self.p_doors = {
            n: Prop(self._def_field(f"DOCK_{n}", "doorTranslation"), "vec3", 5e-4)
            for n in (1, 2, 3)
        }
        self.p_dock_light = {
            n: Prop(self._def_field(f"DOCK_{n}", "statusColor"), "color", 2e-3) for n in (1, 2, 3)
        }
        self.door_amount = {n: 0.0 for n in (1, 2, 3)}  # 0 closed .. 1 open
        self.door_want = {n: False for n in (1, 2, 3)}
        self.dock_party: dict[int, float] = {}  # dock -> time of a DELIVERED celebration
        self.p_forklift = [
            (
                Prop(self._def_field(f"FORKLIFT_{k}", "translation"), "vec3", 1e-4),
                Prop(self._def_field(f"FORKLIFT_{k}", "rotation"), "rot", 1e-4),
            )
            for k in range(2)
        ]
        self.markers = {}
        for kind, def_name, color in (
            ("pickup", "PICKUP_MARKER", PICKUP_COLOR),
            ("dropoff", "DROPOFF_MARKER", DROPOFF_COLOR),
        ):
            self.markers[kind] = {
                "pos": Prop(self._def_field(def_name, "translation"), "vec3", 1e-4),
                "rot": Prop(self._def_field(def_name, "rotation"), "rot", 1e-3),
                "color": Prop(self._def_field(def_name, "color"), "color", 2e-3),
                "base": color,
                "xy": None,
                "flash": None,
                "hide_at": None,
            }
        self.celebration: _Timed | None = None
        # The crisp 3D estimate reticle of close camera views (imported once, reused by restarts).
        self.reticle_shown = False
        self._init_reticle()

        # --- lighting state (boot power-up, night shift) ---------------------------
        self.power_t0: float | None = None  # power-up start (None = lights fully on)
        self.power_gap = 0.25  # s between two lamp rows during the power-up
        self.lights_dark = False
        self.night = False
        self.night_level = 0.0  # 0 day .. 1 night (animated)
        self.fault_dim = False
        self.fault_level = 0.0  # 0 normal .. 1 emergency lighting (animated)
        self.panel_dim = False
        self.panel_level = 0.0  # 0 normal .. 1 dimmed for the help panel (animated)

        # --- robot state -------------------------------------------------------------
        # Where the world put the robot: it stays there until a mission places it (e.g. when
        # project4.py cannot be imported), also when the hologram plane is hidden (place_robot).
        self.robot_xyz = (0.0, 0.0, 0.0)
        # If no translation field is available, keep the origin.
        with suppress(Exception):
            self.robot_xyz = tuple(float(v) for v in self.p_translation.field.getSFVec3f())
        self.phi = [0.0, 0.0, 0.0]
        self.holo_visible = False
        self.deck_z = 0.0

        # --- imported nodes ----------------------------------------------------------
        self.parcels: dict[int, ParcelVis] = {}
        self.obstacles: dict[int, ObstacleVis] = {}

        if self.missing:
            print(
                "warehouse_mission: WARNING, world contract problems (these effects are skipped): "
                + ", ".join(self.missing),
                flush=True,
            )

    # ======================================================================
    # Handles
    # ======================================================================
    def _field(self, node, name: str, owner: str):
        field = node.getField(name) if node is not None else None
        if field is None:
            self.missing.append(f"{owner}.{name}")
        return field

    def _def_field(self, def_name: str, name: str):
        node = self.sup.getFromDef(def_name)
        if node is None:
            self.missing.append(f"DEF {def_name}")
            return None
        return self._field(node, name, def_name)

    def _import_glow_shapes(self, def_name: str, geometries: Sequence[str], hidden_z: float):
        """Import (or reuse) ``DEF def_name Transform`` holding one unlit, emissive Shape per
        geometry.

        Returns ``(node, appearances)``: the Transform and the PBRAppearance of
        each Shape (whose ``emissiveColor`` / ``transparency`` are animated), or
        ``(None, [])`` if the import failed.
        """
        node = self.sup.getFromDef(def_name)
        if node is None:
            shapes = " ".join(
                "Shape { castShadows FALSE isPickable FALSE appearance PBRAppearance { "
                "baseColor 0 0 0 roughness 1 "
                f"metalness 0 emissiveColor 1 1 1 emissiveIntensity {GLOW_INTENSITY} "
                f"transparency 1 }} "
                f"geometry {geometry} }}"
                for geometry in geometries
            )
            node = self._import(
                def_name,
                f"DEF {def_name} Transform {{ translation 0 0 {hidden_z} children [ {shapes} ] }}",
            )
        appearances = []
        if node is not None:
            try:
                children = node.getField("children")
                appearances = [
                    children.getMFNode(i).getField("appearance").getSFNode()
                    for i in range(children.getCount())
                ]
            except Exception:  # noqa: BLE001 - a malformed leftover: no color changes
                appearances = []
        return node, appearances

    def _init_reticle(self) -> None:
        """The estimate reticle (``DEF HOLO_RETICLE``, imported once; a leftover in the world is
        reused).
        """
        node, appearances = self._import_glow_shapes(
            RETICLE_DEF, [reticle_mesh()], RETICLE_HIDDEN_Z
        )
        appearance = appearances[0] if appearances else None

        def field(owner, name):
            return owner.getField(name) if owner is not None else None

        self.p_reticle_pos = Prop(field(node, "translation"), "vec3", 1e-4)
        self.p_reticle_rot = Prop(field(node, "rotation"), "rot", 2e-3)
        self.p_reticle_scale = Prop(field(node, "scale"), "vec3", 2e-3)
        self.p_reticle_color = Prop(field(appearance, "emissiveColor"), "color", 3e-3)
        self.p_reticle_transparency = Prop(field(appearance, "transparency"), "float", 1e-2)
        self.p_reticle_pos.set([0.0, 0.0, RETICLE_HIDDEN_Z])
        self.reticle_shown = False

    def _init_fx_ring(self) -> None:
        """The expanding shockwave ring of pings (``DEF FX_RING``: a crisp band and a soft inner
        band).

        The world's ``PING_RING`` cannot be colored (its emissive texture
        overrides ``emissiveColor``), so the show imports its own ring mesh of
        unit radius, scaled like the world ring; ``PING_RING`` only serves as
        a white fallback if the import fails.
        """
        node, appearances = self._import_glow_shapes(
            FX_RING_DEF, [ring_mesh(0.93, 1.0), ring_mesh(0.8, 0.93)], PING_PARK_Z
        )

        def field(owner, name):
            return owner.getField(name) if owner is not None else None

        if node is not None and len(appearances) == 2:
            self.p_ping_pos = Prop(field(node, "translation"), "vec3", 1e-4)
            self.p_ping_scale = Prop(field(node, "scale"), "vec3", 1e-3)
            self.p_ping_bands = [
                (
                    Prop(field(a, "transparency"), "float", 3e-3),
                    Prop(field(a, "emissiveColor"), "color", 1e-3),
                )
                for a in appearances
            ]
        else:  # fallback: the world's (white) ring
            self.p_ping_pos = Prop(self._def_field("PING_RING", "translation"), "vec3", 1e-4)
            self.p_ping_scale = Prop(self._def_field("PING_RING", "scale"), "vec3", 1e-3)
            self.p_ping_bands = [
                (
                    Prop(self._def_field("PING_RING_MATERIAL", "transparency"), "float", 3e-3),
                    Prop(None, "color"),
                )
            ]
        # The world ring stays parked below the floor.
        world_ring = self.sup.getFromDef("PING_RING")
        if world_ring is not None and node is not None:
            Prop(world_ring.getField("translation"), "vec3").set([0.0, 0.0, PING_PARK_Z])

    @staticmethod
    def _read_float(prop: Prop, default: float) -> float:
        try:
            return float(prop.field.getSFFloat()) if prop.field is not None else default
        except Exception:  # noqa: BLE001
            return default

    # ======================================================================
    # Reset
    # ======================================================================
    def reset(self, start_xy: Sequence[float], color_name: str, forklift_routes) -> None:
        """Put everything back to the start of a mission (VIZ_SPEC 3.3 step 2)."""
        self.clear_dynamic()
        self.p_rotation.set([0.0, 0.0, 1.0, 0.0])
        self.phi = [0.0, 0.0, 0.0]
        for i in range(3):
            self.p_wheels[i].set([1.0, 0.0, 0.0, 0.0])
        self.set_eye(0.0)
        self.set_deck(0.0)
        self.set_profile(color_name)
        self.set_holo_visible(False)
        self.set_reticle(None)
        self.set_fault_lights(False)
        self.place_robot(start_xy[0], start_xy[1], 0.0)
        for kind in self.markers:
            self.hide_marker(kind, animate=False)
            self.markers[kind]["color"].set(self.markers[kind]["base"])
        self.celebration = None
        self.ping_fx = None
        self._park_ping()
        self.beacon_flash.clear()
        for prop in self.p_beacon_glow:
            prop.set(BEACON_BASE_GLOW)
        self.dock_party.clear()
        for n in (1, 2, 3):
            self.door_want[n] = False
            self.door_amount[n] = 0.0
            self.p_doors[n].set([0.0, 0.0, 0.0])
            self.p_dock_light[n].set(DOCK_RED)
        self.park_forklifts(forklift_routes)

    def park_forklifts(self, routes) -> None:
        """Forklifts at their depots (route vertex 0, facing vertex 1)."""
        for k, route in enumerate(routes[:2]):
            (x0, y0), (x1, y1) = route[0], route[1]
            self.set_forklift(k, x0, y0, math.atan2(y1 - y0, x1 - x0))

    def clear_dynamic(self) -> None:
        """Remove every imported parcel and obstacle (also leftovers from an earlier run)."""
        for vis in list(self.parcels.values()) + list(self.obstacles.values()):
            self._remove(vis.node)
        self.parcels.clear()
        self.obstacles.clear()
        for n in range(1, N_ORDERS_MAX + 1):
            self._remove(self.sup.getFromDef(f"ORDER_PARCEL_{n}"))
        for oid in range(C.MAX_OBSTACLES + 2):
            self._remove(self.sup.getFromDef(f"OBSTACLE_{oid}"))

    @staticmethod
    def _remove(node) -> None:
        if node is None:
            return
        # Removing an already deleted node is harmless.
        with suppress(Exception):
            node.remove()

    def _import(self, def_name: str, text: str):
        try:
            self.root_children.importMFNodeFromString(-1, text)
        except Exception as exc:  # noqa: BLE001
            print(f"warehouse_mission: could not import {def_name}: {exc}", flush=True)
            return None
        node = self.sup.getFromDef(def_name)
        if node is None:
            print(
                f"warehouse_mission: import of {def_name} failed (IMPORTABLE EXTERNPROTO missing?)",
                flush=True,
            )
        return node

    # ======================================================================
    # The robot
    # ======================================================================
    def set_profile(self, color_name: str) -> None:
        body, accent = PROFILE_COLORS.get(str(color_name), PROFILE_COLORS["gold"])
        self.p_body.set(body)
        self.p_accent.set(accent)

    def place_robot(self, x: float, y: float, z: float = 0.0) -> None:
        """Move the robot; keep the hologram plane fixed at world (49.5, 24.5, 0.02)."""
        self.robot_xyz = (x, y, z)
        self.p_translation.set([x, y, z])
        if self.holo_visible:
            self.p_holo.set([HOLO_CENTER[0] - x, HOLO_CENTER[1] - y, HOLO_CENTER[2] - z])
        else:
            self.p_holo.set([0.0, 0.0, HOLO_HIDDEN_Z - z])

    def set_holo_visible(self, visible: bool) -> None:
        self.holo_visible = bool(visible)
        x, y, z = self.robot_xyz
        self.place_robot(x, y, z)

    def roll(self, dx: float, dy: float) -> None:
        """Spin the wheels for a displacement (dx, dy): phi_i += (J_TRUE @ d)_i (rolling, no
        slip).
        """
        if dx == 0.0 and dy == 0.0:
            return
        J = C.J_TRUE
        for i in range(3):
            self.phi[i] = (self.phi[i] + J[i, 0] * dx + J[i, 1] * dy) % util.TAU
            self.p_wheels[i].set([1.0, 0.0, 0.0, self.phi[i]])

    def set_eye(self, psi: float) -> None:
        self.p_eye.set([0.0, 0.0, 1.0, util.wrap_angle(psi)])

    def set_deck(self, z: float) -> None:
        self.deck_z = float(z)
        self.p_deck.set([0.0, 0.0, self.deck_z])

    def set_led(self, rgb: Sequence[float], iris: Sequence[float] | None = None) -> None:
        self.p_led.set(rgb)
        self.p_iris.set(rgb if iris is None else iris)

    def set_tag(self, rgb: Sequence[float]) -> None:
        self.p_tag.set(rgb)

    def set_reticle(
        self,
        xy: Sequence[float] | None,
        color: Color = util.MAGENTA,
        alpha: float = 1.0,
        angle: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        """The crisp 3D estimate reticle at ``xy`` on the floor (``None`` or ``alpha`` ~0
        hides it).
        """
        if xy is None or alpha <= 0.02:
            if self.reticle_shown:
                self.p_reticle_pos.set([0.0, 0.0, RETICLE_HIDDEN_Z])
                self.reticle_shown = False
            return
        self.reticle_shown = True
        self.p_reticle_pos.set([float(xy[0]), float(xy[1]), RETICLE_Z])
        self.p_reticle_rot.set([0.0, 0.0, 1.0, angle % (math.pi / 2)])  # four-fold symmetric
        self.p_reticle_scale.set([scale, scale, 1.0])
        self.p_reticle_color.set(color)
        self.p_reticle_transparency.set(1.0 - 0.92 * util.clamp(alpha, 0.0, 1.0))

    def eye_position(self) -> tuple[float, float, float]:
        x, y, z = self.robot_xyz
        return (x + EYE_OFFSET[0], y + EYE_OFFSET[1], z + EYE_OFFSET[2])

    # ======================================================================
    # Lights
    # ======================================================================
    def lights_off(self) -> None:
        """Dark warehouse (start of the boot sequence)."""
        self.lights_dark = True
        self.power_t0 = None

    def power_up(self, t0: float, row_gap: float = 0.25) -> None:
        """Turn the lamp rows on one by one from ``t0`` (``row_gap`` s apart), fading the sun in.

        The warehouse stays dark until ``t0``.
        """
        self.lights_dark = False
        self.power_t0 = t0
        self.power_gap = max(0.05, float(row_gap))

    def lights_on(self) -> None:
        self.lights_dark = False
        self.power_t0 = None

    def set_night(self, on: bool) -> None:
        self.night = bool(on)

    def set_fault_lights(self, on: bool) -> None:
        """Fault screens: the warehouse dims to emergency lighting (the error text reads better)."""
        self.fault_dim = bool(on)

    def set_panel_dim(self, on: bool, instant: bool = False) -> None:
        """The help panel: dim the warehouse like a pause menu (the key list reads better)."""
        self.panel_dim = bool(on)
        if instant:
            self.panel_level = 1.0 if on else 0.0

    def power_up_done(self, t: float) -> bool:
        return (
            self.power_t0 is None
            or t - self.power_t0 > 0.6 + self.power_gap * len(self.p_lamps) + 0.5
        )

    def _update_lights(self, dt: float, t: float) -> None:
        target = 1.0 if self.night else 0.0
        self.night_level += (target - self.night_level) * util.smooth_factor(dt, 0.35)
        if abs(self.night_level - target) < 1e-3:
            self.night_level = target
        target = 1.0 if self.fault_dim else 0.0
        self.fault_level += (target - self.fault_level) * util.smooth_factor(dt, 0.5)
        if abs(self.fault_level - target) < 1e-3:
            self.fault_level = target
        target = 1.0 if self.panel_dim else 0.0
        self.panel_level += (target - self.panel_level) * util.smooth_factor(dt, 0.15)
        if abs(self.panel_level - target) < 1e-3:
            self.panel_level = target
        n = self.night_level
        dim = util.lerp(1.0, FAULT_DIM, self.fault_level) * util.lerp(
            1.0, PANEL_DIM, self.panel_level
        )
        lamp_on = util.lerp(self.base_lamp, LAMP_NIGHT, n) * dim
        if self.lights_dark:
            ambient, rows = 0.0, [0.0] * len(self.p_lamps)
        elif self.power_t0 is None:
            ambient, rows = 1.0, [1.0] * len(self.p_lamps)
        else:
            s = t - self.power_t0
            ambient = util.smoothstep((s - 0.3) / 2.0) if s > 0.0 else 0.0
            rows = []
            for k in range(len(self.p_lamps)):
                u = s - (0.6 + self.power_gap * k)
                if u < 0.0:
                    rows.append(0.0)
                elif u < 0.16:  # fluorescent flicker while the tube strikes
                    rows.append((0.7, 0.05, 0.45, 0.1)[int(u / 0.04)])
                else:
                    rows.append(1.0)
                if 0.0 <= u < dt + 1e-9 and self.audio is not None:
                    self.audio.play("ui_click", pitch=0.8 + 0.06 * k)
        for prop, r in zip(self.p_lamps, rows, strict=False):
            prop.set(r * lamp_on)
        day_night = util.lerp(1.0, NIGHT_FACTOR, n) * dim
        self.p_main.set(self.base_main * util.lerp(0.08, 1.0, ambient) * day_night)
        self.p_fill.set(self.base_fill * util.lerp(0.1, 1.0, ambient) * day_night)
        night_bg = NIGHT_FACTOR / max(self.base_bg, 1e-6)
        self.p_bg.set(self.base_bg * util.lerp(0.3, 1.0, ambient) * util.lerp(1.0, night_bg, n))
        self.p_headlights.set(n > 0.5)

    # ======================================================================
    # Beacons and the RFID ping ring
    # ======================================================================
    def flash_beacon(self, beacon_id: int, t: float) -> None:
        if 0 <= int(beacon_id) < len(self.p_beacon_glow):
            self.beacon_flash[int(beacon_id)] = t

    def ping(
        self,
        center: Sequence[float],
        radius: float,
        t: float,
        color: Color = util.CYAN,
        duration: float = 1.4,
        opacity: float = 0.55,
        priority: int = 0,
    ) -> None:
        """An expanding ring on the floor from ``center`` out to ``radius`` meters, in ``color``.

        ``opacity`` is the ring's starting opacity (it fades out while it
        expands). There is one ring: a ping never cuts short a running ring of
        higher ``priority`` (e.g. an RFID ping does not replace the LOCKED ring).
        """
        fx = self.ping_fx
        if fx is not None and priority < fx.data[4] and t - fx.t0 < fx.duration:
            return
        self.ping_fx = _Timed(
            t,
            duration,
            (
                float(center[0]),
                float(center[1]),
                max(0.3, float(radius)),
                util.clamp(opacity, 0.0, 1.0),
                int(priority),
            ),
        )
        peak = max(color)
        color = util.scale(color, 1.0 / peak) if peak > 0 else color
        for _, p_color in self.p_ping_bands:
            p_color.set(color)

    def _park_ping(self) -> None:
        self.p_ping_pos.set([0.0, 0.0, PING_PARK_Z])
        for p_transparency, _ in self.p_ping_bands:
            p_transparency.set(1.0)

    def _update_beacons(self, t: float) -> None:
        for bid in list(self.beacon_flash):
            u = t - self.beacon_flash[bid]
            if u > 1.2:
                del self.beacon_flash[bid]
                self.p_beacon_glow[bid].set(BEACON_BASE_GLOW)
                continue
            rise = util.clamp(u / 0.06, 0.0, 1.0)
            level = BEACON_BASE_GLOW + (BEACON_FLASH_GLOW - BEACON_BASE_GLOW) * rise * math.exp(
                -u / 0.32
            )
            self.p_beacon_glow[bid].set(level)

    def _update_ping(self, t: float) -> None:
        fx = self.ping_fx
        if fx is None:
            return
        s = (t - fx.t0) / fx.duration
        if s >= 1.0:
            self.ping_fx = None
            self._park_ping()
            return
        x, y, radius, opacity, _ = fx.data
        r = util.lerp(0.3, radius, util.ease_out(s / 0.6))
        self.p_ping_pos.set([x, y, 0.05])
        self.p_ping_scale.set([r, r, 1.0])
        alpha = opacity * (1.0 - util.smoothstep((s - 0.25) / 0.75))
        for k, (p_transparency, _) in enumerate(
            self.p_ping_bands
        ):  # the crisp band, then the soft one
            p_transparency.set(1.0 - alpha * (1.0 if k == 0 else 0.4))

    # ======================================================================
    # Docks
    # ======================================================================
    def set_dock_open(self, index: int, want_open: bool) -> None:
        if index in self.door_want:
            self.door_want[index] = bool(want_open)

    def dock_celebrate(self, index: int, t: float) -> None:
        if index in self.door_want:
            self.dock_party[index] = t

    def _update_docks(self, dt: float, t: float) -> None:
        for n in (1, 2, 3):
            amount = self.door_amount[n]
            goal = 1.0 if self.door_want[n] else 0.0
            if amount != goal:
                step = dt / DOOR_SECONDS
                amount = min(goal, amount + step) if goal > amount else max(goal, amount - step)
                self.door_amount[n] = amount
                self.p_doors[n].set([0.0, 0.0, DOOR_OPEN_Z * util.smoothstep(amount)])
            party = self.dock_party.get(n)
            if party is not None and t - party < 2.4:
                color = DOCK_GREEN if util.blink(t - party, 3.0) else util.scale(DOCK_GREEN, 0.25)
            elif 0.0 < amount < 1.0:
                color = DOCK_AMBER if util.blink(t, 2.5) else util.scale(DOCK_AMBER, 0.3)
            elif amount >= 1.0:
                color = DOCK_GREEN
            else:
                color = DOCK_RED
            self.p_dock_light[n].set(color)

    # ======================================================================
    # Forklifts
    # ======================================================================
    def set_forklift(self, index: int, x: float, y: float, heading: float) -> None:
        """Place a forklift so that its model is centered on the engine's box at (x, y)."""
        if not 0 <= index < len(self.p_forklift):
            return
        pos, rot = self.p_forklift[index]
        c, s = math.cos(heading), math.sin(heading)
        pos.set([x - FORKLIFT_CENTER_OFFSET * c, y - FORKLIFT_CENTER_OFFSET * s, FORKLIFT_Z])
        rot.set([0.0, 0.0, 1.0, util.wrap_angle(heading)])

    # ======================================================================
    # Goal markers
    # ======================================================================
    def show_marker(self, kind: str, xy: Sequence[float]) -> None:
        m = self.markers[kind]
        m["xy"] = (float(xy[0]), float(xy[1]))
        m["hide_at"] = None
        m["flash"] = None
        m["color"].set(m["base"])

    def hide_marker(self, kind: str, animate: bool = True, t: float = 0.0) -> None:
        m = self.markers[kind]
        if animate and m["xy"] is not None:
            m["flash"] = t  # flash white, then vanish ("marker acquired")
            m["hide_at"] = t + 0.7
            return
        m["xy"] = None
        m["flash"] = m["hide_at"] = None
        m["pos"].set([0.0, 0.0, MARKER_HIDDEN_Z])

    def marker_xy(self, kind: str):
        return self.markers[kind]["xy"]

    def _update_markers(self, t: float) -> None:
        celebrating = self.celebration is not None
        for kind, m in self.markers.items():
            if celebrating and kind == "dropoff":
                continue  # the celebration drives this marker
            if m["xy"] is None:
                continue
            if m["hide_at"] is not None and t >= m["hide_at"]:
                self.hide_marker(kind, animate=False)
                continue
            x, y = m["xy"]
            m["pos"].set([x, y, 0.0])
            spin = 1.4 * t if m["flash"] is None else 1.4 * t + 9.0 * (t - m["flash"])
            m["rot"].set([0.0, 0.0, 1.0, spin % util.TAU])
            if m["flash"] is not None:
                m["color"].set(
                    util.mix(m["base"], (1.0, 1.0, 1.0), util.pulse(t - m["flash"], 5.0))
                )

    # ======================================================================
    # Celebration (MISSION COMPLETE)
    # ======================================================================
    def celebrate(self, t: float, xy: Sequence[float]) -> None:
        """MISSION COMPLETE: a marker burst out of the robot, then calm green shockwave rings."""
        # data = (x, y, time of the last shockwave ring)
        self.celebration = _Timed(t, 1e9, (float(xy[0]), float(xy[1]), -1e9))
        m = self.markers["dropoff"]
        m["xy"] = (float(xy[0]), float(xy[1]))
        m["hide_at"] = m["flash"] = None

    def _update_celebration(self, t: float) -> None:
        fx = self.celebration
        if fx is None:
            return
        u = t - fx.t0
        x, y, _ = fx.data
        m = self.markers["dropoff"]
        if m["xy"] is not None:
            # The goal beam launches out of the robot like a rocket, spinning and cycling colors,
            # and
            # has left the frame (hidden) before the score screen appears.
            if u < BURST_SECONDS:
                m["pos"].set([x, y, 0.4 + 11.0 * util.ease_in(u / BURST_SECONDS)])
                m["rot"].set([0.0, 0.0, 1.0, (9.0 * u) % util.TAU])
                palette = (util.GREEN, util.CYAN, util.YELLOW, util.MAGENTA)
                k = int(u * 5.0)
                m["color"].set(util.mix(palette[k % 4], palette[(k + 1) % 4], (u * 5.0) % 1.0))
            else:
                self.hide_marker("dropoff", animate=False)
                m["color"].set(m["base"])
        # One green shockwave ring with the burst (the score screen then stays uncluttered).
        if fx.data[2] < -1e8:
            fx.data = (x, y, t)
            self.ping((x, y), 6.0, t, util.GREEN, duration=1.6, opacity=0.6, priority=1)

    def stop_celebration(self) -> None:
        self.celebration = None

    # ======================================================================
    # Parcels
    # ======================================================================
    def import_parcels(self, orders) -> None:
        """One ParcelBox per order on its rack slot (lowest level)."""
        for vis in self.parcels.values():
            self._remove(vis.node)
        self.parcels.clear()
        for order in orders:
            n = int(order.number)
            x, y = order.slot.parcel_xy
            name = f"ORDER_PARCEL_{n}"
            self._remove(self.sup.getFromDef(name))
            node = self._import(
                name,
                f"DEF {name} ParcelBox {{ translation {x:.4f} {y:.4f} {RACK_Z:.3f} "
                f'name "order parcel {n}" glowColor 0 0 0 }}',
            )
            vis = ParcelVis(n, node, (x, y))
            vis.move(vis.pos)
            self.parcels[n] = vis

    def parcel(self, number: int) -> ParcelVis | None:
        return self.parcels.get(int(number))

    def ride_conveyor(self, number: int, end_x: float, dock_index: int) -> None:
        vis = self.parcels.get(int(number))
        if vis is not None:
            vis.riding = (float(end_x), int(dock_index))

    def parcels_in_doorway(self) -> list[int]:
        """Docks whose door a riding parcel has not passed yet."""
        return [
            vis.riding[1]
            for vis in self.parcels.values()
            if vis.riding is not None and vis.pos[0] < DOOR_CLEAR_X
        ]

    def _update_parcels(self, dt: float) -> None:
        for vis in self.parcels.values():
            if vis.riding is None:
                continue
            end_x, _ = vis.riding
            x, y, z = vis.pos
            if x < end_x:
                vis.move((min(end_x, x + CONVEYOR_SPEED * dt), y, CONVEYOR_Z))
            else:
                vis.riding = None

    # ======================================================================
    # Obstacles
    # ======================================================================
    def drop_obstacle(self, info: dict, t: float, animate: bool = True) -> bool:
        """Import a SpillBarrier for an engine obstacle dict (idempotent by id)."""
        oid = int(info.get("id", len(self.obstacles)))
        if oid in self.obstacles:
            return False
        x, y = float(info["x"]), float(info["y"])
        z = 2.0 if animate else 0.0
        name = f"OBSTACLE_{oid}"
        node = self._import(
            name,
            f"DEF {name} SpillBarrier {{ translation {x:.3f} {y:.3f} {z:.3f} "
            f'rotation 0 0 1 {(oid % 4) * math.pi / 2:.4f} name "obstacle {oid}" flasher 3 }}',
        )
        vis = ObstacleVis(oid, node, (x, y), t, animate)
        vis.translation.last = [x, y, z]
        self.obstacles[oid] = vis
        return True

    def _update_obstacles(self, t: float) -> None:
        for vis in self.obstacles.values():
            u = t - vis.t0
            if vis.animate:
                z, impacts = drop_height(u)
                if impacts > vis.impacts:
                    if vis.impacts == 0 and self.audio is not None:
                        self.audio.play("obstacle_thud")
                    vis.impacts = impacts
                vis.translation.set([vis.xy[0], vis.xy[1], z])
                if impacts > 2:
                    vis.animate = False
            vis.flasher.set(4.0 if util.blink(u, 1.6) else 0.3)

    # ======================================================================
    # Per frame
    # ======================================================================
    def update(self, dt: float, t: float) -> None:
        self._update_lights(dt, t)
        self._update_beacons(t)
        self._update_celebration(t)
        self._update_ping(t)
        self._update_docks(dt, t)
        self._update_markers(t)
        self._update_parcels(dt)
        self._update_obstacles(t)
