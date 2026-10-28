"""The warehouse "truth": map, places, collisions, sensors, obstacles, forklifts.

This module is the engine's model of the real world. It is built directly from
the book's geometry (``gtbook.logistics``) and never reads the student's
constants, so a student who edits a provided constant cannot change reality.

Frames and units: world meters ``(x, y)`` with x east and y north, origin at
the center of cell (0, 0); grids are (50, 100) arrays indexed ``[i, j] =
[row = y, col = x]``; a point belongs to cell ``(round(y), round(x))``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import config as C
from .records import Sensors

###############################################################################
#                           Static map (the book's)                          #
###############################################################################

MAP_SHAPE = (C.MAP_HEIGHT, C.MAP_WIDTH)
_COLS, _ROWS = np.meshgrid(
    np.arange(C.MAP_WIDTH, dtype=float), np.arange(C.MAP_HEIGHT, dtype=float)
)
MAP_POINTS = np.column_stack((_COLS.ravel(), _ROWS.ravel()))  # (5000, 2) cell centers (x, y)
MAP_POINTS.flags.writeable = False


def _block_cols(k: int) -> tuple[int, int]:
    start = C.SHELF_FIRST_COL + k * C.SHELF_SPACING
    return start, start + C.SHELF_WIDTH


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def build_proximity_maps(obstacle: np.ndarray, band_sources: Sequence[tuple[int, int, int, int]]):
    """Proximity ON/OFF maps in the book's style.

    ``band_sources`` are cell rectangles ``(r0, r1, c0, c1)`` (half-open) that
    get a ``PROXIMITY_RANGE`` band; the walls always get one. Obstacle cells are
    neither ON nor OFF (the robot cannot be there).
    """
    rng = C.PROXIMITY_RANGE
    on = np.zeros(MAP_SHAPE)
    for r0, r1, c0, c1 in band_sources:
        on[max(0, r0 - rng) : max(0, r1 + rng), max(0, c0 - rng) : max(0, c1 + rng)] = 1
    on[:rng] = 1
    on[-rng:] = 1
    on[:, :rng] = 1
    on[:, -rng:] = 1
    on[obstacle != 0] = 0
    off = np.ones(MAP_SHAPE)
    off[on == 1] = 0
    off[obstacle != 0] = 0
    return on, off


SHELF_CELL_RECTS = tuple((C.SHELF_ROWS[0], C.SHELF_ROWS[1]) + _block_cols(k) for k in range(4))
BASE_MAP = np.zeros(MAP_SHAPE)
for _r0, _r1, _c0, _c1 in SHELF_CELL_RECTS:
    BASE_MAP[_r0:_r1, _c0:_c1] = 1
PROXIMITY_MAP_ON, PROXIMITY_MAP_OFF = build_proximity_maps(BASE_MAP, SHELF_CELL_RECTS)
for _grid in (BASE_MAP, PROXIMITY_MAP_ON, PROXIMITY_MAP_OFF):
    _read_only(_grid)

# Shelf blocks in meters: (x0, y0, x1, y1).
SHELF_RECTS = np.array(
    [[c0 - 0.5, r0 - 0.5, c1 - 0.5, r1 - 0.5] for r0, r1, c0, c1 in SHELF_CELL_RECTS]
)
SHELF_RECTS.flags.writeable = False

# Eight RFID beacons on the block ends: south ends (ids 0-3), north ends (4-7).
BEACONS = np.array(
    [[(SHELF_RECTS[k, 0] + SHELF_RECTS[k, 2]) / 2.0, SHELF_RECTS[k, 1]] for k in range(4)]
    + [[(SHELF_RECTS[k, 0] + SHELF_RECTS[k, 2]) / 2.0, SHELF_RECTS[k, 3]] for k in range(4)]
)
BEACONS.flags.writeable = False

# The zones (config.ZONE_NAMES): RECEIVING west of shelf A and SHIPPING east of shelf D, each cut
# into 10 m x 10 m bays (bands 1-5 from the south, west and east halves), the three aisles between
# the shelf blocks, and the cross aisles south and north of the blocks. Shelf cells are -1.
ZONE_MAP = np.full(MAP_SHAPE, -1, dtype=int)
_A0, _D1 = SHELF_CELL_RECTS[0][2], SHELF_CELL_RECTS[3][3]  # columns 20 and 80
_BAY = C.ZONE_BAY
for _b in range(C.MAP_HEIGHT // _BAY):
    for _h in range(2):
        ZONE_MAP[_b * _BAY : (_b + 1) * _BAY, _h * _BAY : (_h + 1) * _BAY] = (
            2 * _b + _h
        )  # RECEIVING: zones 0-9
        ZONE_MAP[_b * _BAY : (_b + 1) * _BAY, _D1 + _h * _BAY : _D1 + (_h + 1) * _BAY] = (
            15 + 2 * _b + _h
        )  # SHIPPING
for _k in range(3):  # AISLE 1-3: zones 10-12
    ZONE_MAP[
        C.SHELF_ROWS[0] : C.SHELF_ROWS[1], SHELF_CELL_RECTS[_k][3] : SHELF_CELL_RECTS[_k + 1][2]
    ] = 10 + _k
ZONE_MAP[: C.SHELF_ROWS[0], _A0:_D1] = 13  # CROSS AISLE S
ZONE_MAP[C.SHELF_ROWS[1] :, _A0:_D1] = 14  # CROSS AISLE N
_read_only(ZONE_MAP)
# Each zone is one rectangle (x0, y0, x1, y1) in meters (for renderers).
ZONE_RECTS = tuple(
    (
        float(np.min(np.nonzero(z == ZONE_MAP)[1])) - 0.5,
        float(np.min(np.nonzero(z == ZONE_MAP)[0])) - 0.5,
        float(np.max(np.nonzero(z == ZONE_MAP)[1])) + 0.5,
        float(np.max(np.nonzero(z == ZONE_MAP)[0])) + 0.5,
    )
    for z in range(len(C.ZONE_NAMES))
)


def zone_of(point) -> int:
    """The zone id of the cell containing ``point`` (-1: a shelf cell or off the map)."""
    cell = point_to_cell(point)
    return int(ZONE_MAP[cell]) if in_bounds(cell) else -1


# One free cell just outside the proximity band in front of every beacon (2.5 m away): the goals of
# the beacon search (config.BEACON_SEEK_AFTER).
_SEEK_ROWS = (
    C.SHELF_ROWS[0] - C.PROXIMITY_RANGE - 1,
    C.SHELF_ROWS[1] + C.PROXIMITY_RANGE,
)  # rows 7 and 42
BEACON_SEEK_CELLS = tuple(
    (_SEEK_ROWS[k // 4], int(round(BEACONS[k, 0]))) for k in range(len(BEACONS))
)


def point_to_cell(point) -> tuple[int, int]:
    """Cell ``(i, j)`` containing the point ``(x, y)`` (may be off the map)."""
    return int(np.round(point[1])), int(np.round(point[0]))


def cell_center(cell) -> np.ndarray:
    return np.array([float(cell[1]), float(cell[0])])


def in_bounds(cell) -> bool:
    return 0 <= cell[0] < C.MAP_HEIGHT and 0 <= cell[1] < C.MAP_WIDTH


###############################################################################
#                                   Places                                   #
###############################################################################


@dataclass(frozen=True)
class Slot:
    """A pickup slot on a rack face, e.g. ``B07-E``.

    ``station`` is the center of the goal cell (where docking starts),
    ``dock_pose`` where the robot parks to load, ``facing`` the unit vector
    from the station toward the rack, ``parcel_xy`` the parcel on the rack's
    lowest level (deck height ``config.PARCEL_DECK_HEIGHT``).
    """

    name: str
    block: int
    letter: str
    face: str
    index: int
    goal_cell: tuple[int, int]
    station: tuple[float, float]
    dock_pose: tuple[float, float]
    facing: tuple[float, float]
    face_x: float
    parcel_xy: tuple[float, float]


@dataclass(frozen=True)
class Dock:
    """A shipping dock on the east wall (``DOCK 1..3``)."""

    name: str
    index: int
    goal_cell: tuple[int, int]
    station: tuple[float, float]
    dock_pose: tuple[float, float]
    facing: tuple[float, float]
    conveyor_from: tuple[float, float]
    conveyor_to: tuple[float, float]


def _make_slots() -> tuple[Slot, ...]:
    slots = []
    for k, letter in enumerate(C.BLOCK_LETTERS):
        x_west, x_east = float(SHELF_RECTS[k, 0]), float(SHELF_RECTS[k, 2])
        for face in ("W", "E"):
            for s in range(1, C.SLOTS_PER_FACE + 1):
                row = C.SLOT_ROW_OFFSET + C.SLOT_ROW_STEP * s
                if face == "W":
                    col = C.WEST_FACE_GOAL_COL + C.SHELF_SPACING * k
                    face_x, sign = x_west, 1.0
                else:
                    col = C.EAST_FACE_GOAL_COL + C.SHELF_SPACING * k
                    face_x, sign = x_east, -1.0
                slots.append(
                    Slot(
                        name=f"{letter}{s:02d}-{face}",
                        block=k,
                        letter=letter,
                        face=face,
                        index=s,
                        goal_cell=(row, col),
                        station=(float(col), float(row)),
                        dock_pose=(face_x - sign * C.RACK_STANDOFF, float(row)),
                        facing=(sign, 0.0),
                        face_x=face_x,
                        parcel_xy=(face_x + sign * C.PARCEL_INSET, float(row)),
                    )
                )
    return tuple(slots)


def _make_docks() -> tuple[Dock, ...]:
    docks = []
    for n, row in enumerate(C.DOCK_ROWS, start=1):
        y = float(row)
        docks.append(
            Dock(
                name=f"DOCK {n}",
                index=n,
                goal_cell=(row, C.DOCK_GOAL_COL),
                station=(float(C.DOCK_GOAL_COL), y),
                dock_pose=(C.DOCK_POSE_X, y),
                facing=(1.0, 0.0),
                conveyor_from=(C.CONVEYOR_START_X, y),
                conveyor_to=(C.CONVEYOR_END_X, y),
            )
        )
    return tuple(docks)


SLOTS = _make_slots()
DOCKS = _make_docks()
SLOTS_BY_NAME: dict[str, Slot] = {slot.name: slot for slot in SLOTS}


###############################################################################
#                                  Geometry                                  #
###############################################################################


def rect_distance(points: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Distance from each point (M, 2) to the nearest axis-aligned rect (K, 4).

    Returns (M,), +inf when there are no rects, 0 for points inside a rect.
    """
    if len(rects) == 0:
        return np.full(len(points), np.inf)
    px, py = points[:, 0:1], points[:, 1:2]
    dx = np.maximum(np.maximum(rects[None, :, 0] - px, px - rects[None, :, 2]), 0.0)
    dy = np.maximum(np.maximum(rects[None, :, 1] - py, py - rects[None, :, 3]), 0.0)
    return np.min(np.hypot(dx, dy), axis=1)


def box_distance(
    points: np.ndarray, boxes: Sequence[tuple[float, float, float, float, float]]
) -> np.ndarray:
    """Distance from points (M, 2) to the nearest oriented box.

    A box is ``(cx, cy, heading, half_length, half_width)``; the length runs
    along the heading.
    """
    best = np.full(len(points), np.inf)
    for cx, cy, heading, half_length, half_width in boxes:
        c, s = math.cos(heading), math.sin(heading)
        rx, ry = points[:, 0] - cx, points[:, 1] - cy
        local_x = rx * c + ry * s
        local_y = -rx * s + ry * c
        dx = np.maximum(np.abs(local_x) - half_length, 0.0)
        dy = np.maximum(np.abs(local_y) - half_width, 0.0)
        best = np.minimum(best, np.hypot(dx, dy))
    return best


def wall_distance(points: np.ndarray) -> np.ndarray:
    """Signed distance from points to the nearest inner wall face (negative outside)."""
    return np.minimum.reduce(
        [
            points[:, 0] - C.FLOOR_X[0],
            C.FLOOR_X[1] - points[:, 0],
            points[:, 1] - C.FLOOR_Y[0],
            C.FLOOR_Y[1] - points[:, 1],
        ]
    )


def segment_samples(start: np.ndarray, end: np.ndarray, spacing: float = 0.1) -> np.ndarray:
    """Points along a segment, at most ``spacing`` apart, including both ends."""
    length = float(np.linalg.norm(end - start))
    n = max(2, int(math.ceil(length / spacing)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return start[None, :] * (1 - t) + end[None, :] * t


@dataclass
class SweepResult:
    end: np.ndarray
    fraction: float
    blocked_by: str | None  # None, "wall", "shelf", "obstacle", or "forklift"


###############################################################################
#                                  Obstacles                                 #
###############################################################################


@dataclass
class Obstacle:
    """A dropped obstacle filling exactly one grid cell (chaos key O)."""

    id: int
    cell: tuple[int, int]

    @property
    def xy(self) -> tuple[float, float]:
        return float(self.cell[1]), float(self.cell[0])

    @property
    def rect(self) -> tuple[float, float, float, float]:
        x, y = self.xy
        h = C.OBSTACLE_HALF_SIZE
        return x - h, y - h, x + h, y + h

    def as_dict(self) -> dict[str, object]:
        x, y = self.xy
        return {
            "id": self.id,
            "name": f"OBSTACLE_{self.id}",
            "x": x,
            "y": y,
            "cell": self.cell,
            "half_size": C.OBSTACLE_HALF_SIZE,
        }


###############################################################################
#                                  Forklifts                                 #
###############################################################################


class Forklift:
    """A forklift driving a closed loop route, yielding to the AMR.

    The route is a closed polyline; ``s`` is the arc length from vertex 0
    (the depot). ``direction`` is +1 (route order) or -1 (backing off after a
    standoff with the AMR). Traffic OFF sends the forklift back to its depot.
    """

    def __init__(self, index: int, route: Sequence[tuple[float, float]]):
        self.index = index
        self.name = f"FORKLIFT_{index}"
        self.route = np.array(route, dtype=float)
        self._seg_vec = np.roll(self.route, -1, axis=0) - self.route
        self._seg_len = np.linalg.norm(self._seg_vec, axis=1)
        self._cum = np.concatenate(([0.0], np.cumsum(self._seg_len)))
        self.length = float(self._cum[-1])
        self.s = 0.0
        self.direction = 1
        self.traffic = False  # traffic ON: drive the loop forever
        self.returning = False  # drive until the depot, then park
        self._laps_left = 0.0  # distance to drive while "making way" from the depot
        self.yielding = False
        self.moving = False

    # --- kinematics -------------------------------------------------------
    def _locate(self, s: float, direction: int) -> tuple[np.ndarray, float]:
        """Point and heading at arc length ``s`` when traveling in ``direction``.

        A vertex belongs to the segment the forklift is about to drive, so the
        heading at a corner is the new travel direction.
        """
        s = float(s) % self.length
        last = len(self._seg_len) - 1
        if direction >= 0:
            seg = int(np.searchsorted(self._cum, s, side="right")) - 1
        else:
            s = s if s > 0.0 else self.length  # vertex 0 == end of the last segment
            seg = int(np.searchsorted(self._cum, s, side="left")) - 1
        seg = min(max(seg, 0), last)
        t = (s - self._cum[seg]) / self._seg_len[seg]
        point = self.route[seg] + t * self._seg_vec[seg]
        vx, vy = self._seg_vec[seg] * (1 if direction >= 0 else -1)
        return point, math.atan2(vy, vx)

    def pose(self, s: float | None = None) -> tuple[float, float, float]:
        point, heading = self._locate(self.s if s is None else s, self.direction)
        return float(point[0]), float(point[1]), heading

    def box(self, s: float | None = None) -> tuple[float, float, float, float, float]:
        x, y, heading = self.pose(s)
        return x, y, heading, C.FORKLIFT_HALF_LENGTH, C.FORKLIFT_HALF_WIDTH

    @property
    def active(self) -> bool:
        return self.traffic or self.returning or self._laps_left > 0

    @property
    def parked(self) -> bool:
        return not self.active

    def _distance_to_depot(self) -> float:
        if self.direction > 0:
            return (self.length - self.s) % self.length
        return self.s % self.length

    def reverse(self) -> None:
        """Turn around (used to break a standoff with the AMR)."""
        self.direction = -self.direction

    def make_way(self, away_from: np.ndarray) -> None:
        """A parked forklift blocking the AMR drives one lap, away from the AMR."""
        ahead = np.array(self._locate(self.s + 2.0, 1)[0])
        behind = np.array(self._locate(self.s - 2.0, -1)[0])
        self.direction = (
            1 if np.linalg.norm(ahead - away_from) >= np.linalg.norm(behind - away_from) else -1
        )
        self._laps_left = self.length

    def set_traffic(self, on: bool) -> None:
        self.traffic = bool(on)
        if not on and self._distance_to_depot() > 1e-6:
            self.returning = True  # finish the lap, then park at the depot

    def planned_distance(self, dt: float) -> float:
        """How far the forklift wants to drive during the next ``dt`` seconds."""
        if not self.active:
            return 0.0
        ds = C.FORKLIFT_SPEED * dt
        if self._laps_left > 0:
            return min(ds, self._laps_left)
        if self.traffic:
            return ds
        remaining = self._distance_to_depot()
        if remaining <= 1e-6:
            self.returning = False  # already home: park
            return 0.0
        return min(ds, remaining)

    def advance(self, ds: float) -> None:
        self.s = (self.s + self.direction * ds) % self.length
        if self._laps_left > 0:
            self._laps_left = max(0.0, self._laps_left - ds)
            if self._laps_left <= 1e-9:
                self._laps_left = 0.0
                self.returning = not self.traffic and self._distance_to_depot() > 1e-6
        elif self.returning and not self.traffic and self._distance_to_depot() <= 1e-6:
            self.returning = False
        if self.parked and min(self.s, self.length - self.s) <= 1e-6:
            self.s = 0.0

    def as_dict(self) -> dict[str, object]:
        x, y, heading = self.pose()
        return {
            "name": self.name,
            "index": self.index,
            "x": x,
            "y": y,
            "heading": heading,
            "moving": self.moving,
            "yielding": self.yielding,
            "traffic": self.traffic,
            "parked": self.parked,
            "half_length": C.FORKLIFT_HALF_LENGTH,
            "half_width": C.FORKLIFT_HALF_WIDTH,
        }


###############################################################################
#                                  The world                                 #
###############################################################################


class World:
    """Truth maps (static + dropped obstacles), collisions, sensors, actors."""

    def __init__(self):
        self.obstacles: list[Obstacle] = []
        self.forklifts = [Forklift(i, route) for i, route in enumerate(C.FORKLIFT_ROUTES)]
        self.map_version = 0
        self.obstacle_map = BASE_MAP
        self.prox_on = PROXIMITY_MAP_ON
        self.prox_off = PROXIMITY_MAP_OFF

    # --- maps ---------------------------------------------------------------
    def _rebuild_maps(self) -> None:
        obstacle = np.array(BASE_MAP)
        sources = list(SHELF_CELL_RECTS)
        for ob in self.obstacles:
            i, j = ob.cell
            obstacle[i, j] = 1
            sources.append((i, i + 1, j, j + 1))
        on, off = build_proximity_maps(obstacle, sources)
        self.obstacle_map = _read_only(obstacle)
        self.prox_on = _read_only(on)
        self.prox_off = _read_only(off)
        self.map_version += 1

    def add_obstacle(self, cell: tuple[int, int]) -> Obstacle:
        obstacle = Obstacle(id=len(self.obstacles), cell=(int(cell[0]), int(cell[1])))
        self.obstacles.append(obstacle)
        self._rebuild_maps()
        return obstacle

    def free_cells(self, non_proximity: bool = True) -> np.ndarray:
        """(K, 2) array of cells that are free (and optionally not in proximity)."""
        mask = self.obstacle_map == 0
        if non_proximity:
            mask &= self.prox_on == 0
        return np.argwhere(mask)

    # --- collisions ---------------------------------------------------------
    def static_rects(self) -> np.ndarray:
        """Shelf blocks and dropped obstacles as (K, 4) rectangles ``(x0, y0, x1, y1)`` in m."""
        if not self.obstacles:
            return SHELF_RECTS
        return np.vstack((SHELF_RECTS, np.array([ob.rect for ob in self.obstacles])))

    def forklift_boxes(self) -> list[tuple[float, float, float, float, float]]:
        """Current forklift footprints ``(cx, cy, heading, half_length, half_width)``."""
        return [f.box() for f in self.forklifts]

    def clearance_parts(self, points: np.ndarray, forklifts: bool = True) -> dict[str, np.ndarray]:
        """Distance from the robot *disk* (radius 0.5) to each kind of thing."""
        r = C.ROBOT_RADIUS
        parts = {
            "wall": wall_distance(points) - r,
            "shelf": rect_distance(points, SHELF_RECTS) - r,
        }
        if self.obstacles:
            parts["obstacle"] = (
                rect_distance(points, np.array([ob.rect for ob in self.obstacles])) - r
            )
        if forklifts:
            parts["forklift"] = box_distance(points, self.forklift_boxes()) - r
        return parts

    def clearance(self, points: np.ndarray, forklifts: bool = True) -> np.ndarray:
        return np.minimum.reduce(
            list(self.clearance_parts(np.atleast_2d(points), forklifts).values())
        )

    def sweep(
        self, start: np.ndarray, displacement: np.ndarray, forklifts: bool = True
    ) -> SweepResult:
        """Move the robot disk along a straight segment; stop just before contact.

        Returns the reached position, the completed fraction of the segment,
        and what blocked it (the SAFETY STOP cause), if anything.
        """
        start = np.asarray(start, dtype=float)
        displacement = np.asarray(displacement, dtype=float)
        length = float(np.linalg.norm(displacement))
        if length < 1e-12:
            return SweepResult(start.copy(), 1.0, None)
        n = max(2, int(math.ceil(length / 0.05)) + 1)
        ts = np.linspace(0.0, 1.0, n)
        points = start[None, :] + ts[:, None] * displacement[None, :]
        parts = self.clearance_parts(points, forklifts)
        clearance = np.minimum.reduce(list(parts.values()))
        # Never block a move that starts in (near) contact and backs away.
        limit = min(C.SAFETY_MARGIN, float(clearance[0]) - 1e-9)
        bad = np.nonzero(clearance < limit)[0]
        if len(bad) == 0:
            return SweepResult(start + displacement, 1.0, None)
        first = int(bad[0])
        kinds = list(parts.keys())
        blocked_by = kinds[int(np.argmin([parts[k][first] for k in kinds]))]
        lo, hi = (ts[first - 1] if first > 0 else 0.0), ts[first]
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if self.clearance(start + mid * displacement, forklifts)[0] < limit:
                hi = mid
            else:
                lo = mid
        return SweepResult(start + lo * displacement, float(lo), blocked_by)

    # --- sensors (book Section 4.3) ------------------------------------------
    def sense_proximity(self, position: np.ndarray) -> bool:
        cell = point_to_cell(position)
        if not in_bounds(cell):
            return True
        return bool(self.prox_on[cell])

    @staticmethod
    def sense_rfid(position: np.ndarray, rng: np.random.Generator) -> tuple[int | None, float]:
        noise = float(rng.normal(0.0, C.RFID_NOISE_SIGMA))  # always drawn: fixed RNG use
        distances = np.linalg.norm(BEACONS - np.asarray(position, dtype=float)[None, :], axis=1)
        k = int(np.argmin(distances))
        if distances[k] <= C.RFID_MAX_RANGE:
            return k, max(0.0, float(distances[k]) + noise)
        return None, float("inf")

    @staticmethod
    def sense_zone(position: np.ndarray, rng: np.random.Generator) -> int:
        """The zone reader: the id of the robot's zone; within ``ZONE_BLUR`` of another zone (across
        open floor, not through a shelf) it hears that zone's beacon instead with probability
        ``ZONE_CONFUSION``."""
        u = rng.random(2)  # always drawn: fixed RNG use
        position = np.asarray(position, dtype=float)
        zone = zone_of(position)
        near = []
        for offset in (
            (-C.ZONE_BLUR, 0.0),
            (C.ZONE_BLUR, 0.0),
            (0.0, -C.ZONE_BLUR),
            (0.0, C.ZONE_BLUR),
        ):
            other = zone_of(position + offset)
            if other >= 0 and other != zone and other not in near:
                near.append(other)
        if zone < 0:  # (cannot happen: the robot's center never lies in a shelf cell)
            return near[0] if near else 0
        if near and u[0] < C.ZONE_CONFUSION:
            return near[min(int(u[1] * len(near)), len(near) - 1)]
        return zone

    @staticmethod
    def sense_fiducial(position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return C.FIDUCIAL_C @ np.asarray(position, dtype=float) + rng.normal(
            0.0, C.FIDUCIAL_SIGMA_CM, 2
        )

    def sense(
        self,
        position: np.ndarray,
        rng: np.random.Generator,
        zone: bool = True,
        proximity: bool = True,
        rfid: bool = True,
    ) -> Sensors:
        """All regular sensors at the true ``position``; a switched-off sensor reads None.

        Every reading's noise is drawn even when its sensor is off, so that
        switching sensors off does not change the rest of the world's random stream.
        """
        z_on = self.sense_proximity(position)
        z_rfid = self.sense_rfid(position, rng)
        z_zone = self.sense_zone(position, rng)
        return Sensors(
            z_on=z_on if proximity else None,
            z_rfid=z_rfid if rfid else None,
            z_zone=z_zone if zone else None,
        )

    # --- sampling -----------------------------------------------------------
    def sample_free_position(
        self,
        rng: np.random.Generator,
        far_from: np.ndarray | None = None,
        min_distance: float = 0.0,
        avoid_cells: Sequence[tuple[int, int]] = (),
        avoid_radius: int = 0,
    ) -> np.ndarray:
        """A uniformly random free, non-proximity cell (plus a little jitter).

        Avoids forklifts (with ``START_DEPOT_CLEARANCE``), cells within
        ``avoid_radius`` (Chebyshev) of ``avoid_cells``, and positions closer
        than ``min_distance`` to ``far_from``.
        """
        cells = self.free_cells(non_proximity=True)
        centers = cells[:, ::-1].astype(float)
        ok = np.ones(len(cells), dtype=bool)
        boxes = self.forklift_boxes()
        ok &= box_distance(centers, boxes) > C.ROBOT_RADIUS + C.START_DEPOT_CLEARANCE
        if far_from is not None and min_distance > 0:
            ok &= np.linalg.norm(centers - np.asarray(far_from)[None, :], axis=1) >= min_distance
        for cell in avoid_cells:
            near = (np.abs(cells[:, 0] - cell[0]) <= avoid_radius) & (
                np.abs(cells[:, 1] - cell[1]) <= avoid_radius
            )
            ok &= ~near
        candidates = cells[ok] if np.any(ok) else cells
        i, j = candidates[int(rng.integers(len(candidates)))]
        return np.array([float(j), float(i)]) + rng.uniform(-0.25, 0.25, 2)

    # --- forklifts ------------------------------------------------------------
    def set_traffic(self, on: bool) -> None:
        for forklift in self.forklifts:
            forklift.set_traffic(on)

    @property
    def traffic_on(self) -> bool:
        return any(f.traffic for f in self.forklifts)

    def forklift_blocking(self, start: np.ndarray, end: np.ndarray) -> int | None:
        """Index of a forklift that the AMR's straight move would come too close to.

        Moves that do not bring the robot closer to a forklift are always allowed,
        so a robot that is already near a forklift can back away.
        """
        samples = segment_samples(np.asarray(start, float), np.asarray(end, float))
        limit = C.ROBOT_RADIUS + C.AMR_FORKLIFT_MARGIN
        for forklift in self.forklifts:
            d = box_distance(samples, [forklift.box()])
            if np.min(d[1:]) < min(limit, float(d[0]) - 1e-6):
                return forklift.index
        return None

    def step_forklifts(self, dt: float, amr_start: np.ndarray, amr_end: np.ndarray) -> None:
        """Advance every active forklift by ``dt`` seconds, yielding to the AMR.

        A forklift stops (``yielding``) if driving on would bring it within
        ``FORKLIFT_CLEARANCE`` of any point of the AMR's segment for this step,
        looking ``FORKLIFT_LOOKAHEAD`` meters ahead. It never drives through the robot.
        """
        amr = segment_samples(np.asarray(amr_start, float), np.asarray(amr_end, float))
        limit = C.ROBOT_RADIUS + C.FORKLIFT_CLEARANCE
        for forklift in self.forklifts:
            forklift.moving = False
            forklift.yielding = False
            if dt <= 0:
                continue
            ds = forklift.planned_distance(dt)
            if ds <= 0:
                continue
            probes = np.linspace(0.25, ds + C.FORKLIFT_LOOKAHEAD, 8)
            boxes = [forklift.box(forklift.s + forklift.direction * p) for p in probes]
            ahead = float(np.min(box_distance(amr, boxes)))
            now = float(np.min(box_distance(amr, [forklift.box()])))
            if ahead < min(limit, now - 1e-6):
                forklift.yielding = True
                continue
            forklift.advance(ds)
            forklift.moving = True

    # --- layout for renderers ------------------------------------------------
    def layout(self) -> dict[str, object]:
        """Static layout (plus current obstacles) for renderers and docs."""
        return {
            "floor": (C.FLOOR_X[0], C.FLOOR_Y[0], C.FLOOR_X[1], C.FLOOR_Y[1]),
            "shelves": [
                {
                    "name": f"RACK_{C.BLOCK_LETTERS[k]}",
                    "letter": C.BLOCK_LETTERS[k],
                    "rect": tuple(float(v) for v in SHELF_RECTS[k]),
                }
                for k in range(4)
            ],
            "beacons": [tuple(float(v) for v in b) for b in BEACONS],
            "slots": list(SLOTS),
            "docks": list(DOCKS),
            "forklift_routes": [tuple(tuple(p) for p in route) for route in C.FORKLIFT_ROUTES],
            "forklift_half_size": (C.FORKLIFT_HALF_LENGTH, C.FORKLIFT_HALF_WIDTH),
            "obstacle_half_size": C.OBSTACLE_HALF_SIZE,
            "obstacles": [ob.as_dict() for ob in self.obstacles],
            "robot_radius": C.ROBOT_RADIUS,
            "wheel_radius": C.WHEEL_RADIUS,
            "wheel_angles_deg": C.WHEEL_ANGLES_DEG,
            "capture_radius": C.CAPTURE_RADIUS,
        }
