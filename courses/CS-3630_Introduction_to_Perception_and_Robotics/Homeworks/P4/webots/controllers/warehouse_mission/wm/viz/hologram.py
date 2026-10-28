"""Robot Vision: NumPy renderers for the floor hologram and the minimap (no Webots import).

The mission engine (``wm.core``) returns the robot's belief, plan and
sensor readings in every ``StepResult``; this module turns them into two
RGBA images that the Webots layer (``app.py``) pastes into the robot's
``Display`` devices:

* ``render_floor``: ``(500, 1000, 4)`` for ``Display "floor_holo"``, the
  texture of a 100 m x 50 m plane lying on the warehouse floor (10 px per
  meter). Its background is transparent, so the robot's "thoughts" appear
  painted on the real floor: particles, grid belief, estimate, planned path,
  goal, RFID annulus, the zone reading, value function / likelihood, obstacles.
* ``render_minimap``: ``(250, 500, 4)`` for ``Display "minimap"`` (5 px per
  meter), an opaque "robot vision" map shown as an overlay in the bottom-left
  corner of the 3D view (like the book's ``show_map`` figures, with the true
  robot next to its estimate).

Pixel mapping (verified in Webots with a calibration pattern): a world point
``(x, y)`` in meters lands at pixel column ``(x + 0.5) * s`` and row
``(49.5 - y) * s`` (north up; ``s`` = 10 floor, 5 minimap). Grid cell
``(i, j)`` (center ``x = j``, ``y = i``) covers columns ``[j s, (j + 1) s)``
and rows ``[(49 - i) s, (50 - i) s)``.

Images use straight (not premultiplied) alpha. Everything is vectorized
NumPy; the static parts are drawn once in ``HologramRenderer.__init__``.
Budget (DESIGN / VIZ_SPEC 3.1): ``render_floor`` <= 15 ms and
``render_minimap`` <= 8 ms per call on a laptop.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..core import BEACONS, DOCKS, SHELF_RECTS, SLOTS_BY_NAME, config
from ..core.world import BASE_MAP, PROXIMITY_MAP_ON, ZONE_RECTS
from . import font5x7

###############################################################################
#                                  Palette                                    #
###############################################################################

# The neon palette of the style guide (DESIGN section 7), shared with the HUD.
CYAN = (25, 230, 255)  # sensing / belief
MAGENTA = (255, 61, 220)  # estimate
YELLOW = (255, 210, 26)  # path / goal
GREEN = (57, 255, 136)  # success / docks
ORANGE = (255, 138, 0)  # cargo / RFID / forklifts
RED = (255, 59, 59)  # danger / obstacles
WHITE = (232, 244, 255)  # truth
ZONE_LIME = (170, 255, 80)  # the zone the zone reader names
NAVY = (7, 14, 26)  # minimap background
SHADOW = (6, 10, 20)  # dark underlay that keeps strokes readable on the bright concrete floor

MAP_H, MAP_W = config.MAP_HEIGHT, config.MAP_WIDTH  # 50 x 100 cells of 1 m
# Phases in which the floor shows no goal pulse and no path: the robot is docking at the goal
# (it would draw them right under the robot), or it is done.
_NO_FLOOR_GOAL_PHASES = ("DOCK_PICKUP", "DOCK_DROPOFF", "COMPLETE")


###############################################################################
#                            Layer switches + input                           #
###############################################################################


@dataclass
class HoloLayers:
    """What Robot Vision shows (keys V, B, M)."""

    vision: bool = True  # V: floor hologram visible (the minimap is always drawn)
    belief: str = "particles"  # B: "particles" | "off"
    map_layer: str = (
        "none"  # M: "none" | "value" | "change" (after an obstacle drop) | "likelihood" | "zones"
    )


@dataclass
class HoloFrame:
    """Everything the renderers need for one image (filled by the Webots layer)."""

    result: object | None  # latest wm.core StepResult (None before BOOT)
    robot_xy: tuple[float, float]  # current interpolated TRUE robot position (minimap truth marker)
    t: float  # simulation time (s), for pulsing/fading animations
    layers: HoloLayers = field(default_factory=HoloLayers)
    ping: tuple | None = (
        None  # (beacon_id, range_m, age_s) of the last RFID_PING (annulus fades over ~2 s)
    )
    zone_age: float = 0.0  # seconds since the last zone reading (the highlight fades a little)
    goal_xy: tuple | None = None  # current goal station (m)
    phase: str = ""  # mission phase
    show_truth: bool = True  # minimap: draw the true robot (white) as well as the estimate
    # 0..1: how much the particle weights show. The Webots layer animates it from 0 to 1 when a
    # fresh uniform particle set appears (deployment, LOST): first the prior's 2000 equal guesses
    # everywhere, then the first measurement weighs them (dead particles fade, survivors grow).
    weigh: float = 1.0
    # 0..1: how close the camera is (0 = the whole warehouse from above, 1 = a chase or close-up
    # shot). The floor texture has only 10 px per meter, and close cameras magnify it: there the
    # strokes get thinner and the glows softer, so the floor reads as light, not as blurry paint.
    near: float = 0.0
    # (dx, dy) m added to the belief layers (particles, estimate ring, ellipse). A result's belief
    # is the one at its END; the Webots layer slides it along with the robot while the result
    # plays (the motion the brain was told about), so the swarm travels with the robot.
    belief_shift: tuple = (0.0, 0.0)
    # False: the floor skips the estimate ring, its reticle and its center dot (the Webots layer
    # shows a crisp 3D reticle in close views instead); the 2-sigma ellipse stays.
    estimate_ring: bool = True
    # MISSION COMPLETE with a working TODO 17: the core's smooth_track() dict ("track", "truth",
    # "segments", "smooth_error", ...). The minimap draws the student's smoothed path (magenta) over
    # the true path (white); smoothed_reveal (0..1) is how much of it is drawn yet.
    smoothed: dict | None = None
    smoothed_reveal: float = 1.0
    # A sensor test: {"truth": [(x, y), ...], "belief": [(x, y), ...], "gap": bool}: the path the
    # robot
    # really drove (white) and the path it BELIEVES it drove (magenta: its estimate, step by step);
    # "gap" (at the end): a yellow line from the robot to its final estimate, labeled in meters
    # (floor);
    # "other" (a PENALTY TEST): the true path of the other test of the session (orange).
    trails: dict | None = None


###############################################################################
#                           Small color helpers                               #
###############################################################################


def _turbo_lut(n: int = 256) -> np.ndarray:
    """Google's Turbo colormap (polynomial fit), ``(n, 3)`` float32 in 0..255."""
    x = np.linspace(0.0, 1.0, n)
    r = 0.13572138 + x * (
        4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))
    )
    g = 0.09140261 + x * (
        2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))
    )
    b = 0.10667330 + x * (
        12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))
    )
    return (np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0) * 255.0).astype(np.float32)


def _ramp(stops: Sequence[tuple[float, Sequence[int]]], n: int = 256) -> np.ndarray:
    """A piecewise-linear color ramp through ``(position, rgb)`` stops, ``(n, 3)`` float32."""
    x = np.linspace(0.0, 1.0, n)
    pos = [p for p, _ in stops]
    return np.stack([np.interp(x, pos, [c[k] for _, c in stops]) for k in range(3)], axis=1).astype(
        np.float32
    )


def _finite(value, default: float = 0.0) -> float:
    """``value`` as a finite float, or ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mix(a: Sequence[int], b: Sequence[int], t: float) -> tuple[float, float, float]:
    t = min(1.0, max(0.0, t))
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))


def _lerp(far: float, near: float, k: float) -> float:
    """A style value between its overhead (``far``) and close-camera (``near``) setting."""
    return far + (near - far) * k


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _gauss_matrix(n: int, h: float) -> np.ndarray:
    """``(n, n)`` float32 matrix of a 1D Gaussian blur of ``h`` px (unnormalized, symmetric)."""
    i = np.arange(n, dtype=np.float32)
    return np.exp(np.float32(-0.5 / (h * h)) * (i[:, None] - i[None, :]) ** 2)


def _shift_of(frame) -> tuple[float, float]:
    """``frame.belief_shift`` as two finite floats (m), (0, 0) if missing or invalid."""
    try:
        dx, dy = float(frame.belief_shift[0]), float(frame.belief_shift[1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return 0.0, 0.0
    if not (math.isfinite(dx) and math.isfinite(dy)) or abs(dx) > 50.0 or abs(dy) > 50.0:
        return 0.0, 0.0
    return dx, dy


TURBO = _turbo_lut()
GRID_RAMP = _ramp(
    [(0.0, (190, 20, 170)), (0.5, MAGENTA), (1.0, (255, 215, 250))]
)  # Markov grid belief
LIKELIHOOD_RAMP = _ramp(
    [(0.0, (140, 40, 0)), (0.6, ORANGE), (1.0, (255, 232, 170))]
)  # sensor likelihood


###############################################################################
#                               Compositing                                   #
###############################################################################


def _over(region: np.ndarray, alpha: np.ndarray, rgb) -> None:
    """Straight-alpha "over": paint ``rgb`` with coverage ``alpha`` onto the uint8 RGBA ``region``
    in place.

    ``alpha`` has the region's shape ``(h, w)`` (or ``(n,)`` for gathered
    pixels) and ``rgb`` is ``(3,)`` or matches ``region[..., :3]``.
    """
    a_src = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
    a_dst = region[..., 3].astype(np.float32) * np.float32(1.0 / 255.0)
    a_out = a_src + a_dst * (1.0 - a_src)
    k = (a_src / np.maximum(a_out, 1e-6))[..., None]
    src = np.asarray(rgb, dtype=np.float32)
    region[..., :3] = (src * k + region[..., :3].astype(np.float32) * (1.0 - k) + 0.5).astype(
        np.uint8
    )
    region[..., 3] = (a_out * 255.0 + 0.5).astype(np.uint8)


def _over_cells(dst: np.ndarray | None, src: np.ndarray) -> np.ndarray:
    """ "Over" for small float RGBA grids (straight alpha, 0..255 colors, 0..1 alpha)."""
    if dst is None:
        return src
    a_s, a_d = src[..., 3:4], dst[..., 3:4]
    a_o = a_s + a_d * (1.0 - a_s)
    rgb = (src[..., :3] * a_s + dst[..., :3] * a_d * (1.0 - a_s)) / np.maximum(a_o, 1e-6)
    return np.concatenate([rgb, a_o], axis=-1)


class _Canvas:
    """An RGBA image of the warehouse at ``scale`` px per meter, with anti-aliased primitives.

    Coordinates of the primitives are continuous pixel coordinates
    ``(cx, cy)`` = (column, row); the pixel ``(r, c)`` covers
    ``[c, c + 1) x [r, r + 1)``, so its center is ``(c + 0.5, r + 0.5)``.
    """

    def __init__(self, img: np.ndarray, scale: int):
        self.img = img
        self.s = scale
        self.h, self.w = img.shape[:2]
        self.flat = img.reshape(-1, 4)

    # --- coordinates ------------------------------------------------------------
    def px(self, x: float, y: float) -> tuple[float, float]:
        """Continuous pixel coordinates of the world point (x, y) in meters."""
        return (x + 0.5) * self.s, (49.5 - y) * self.s

    def _box(self, cx: float, cy: float, rx: float, ry: float):
        if not all(math.isfinite(v) for v in (cx, cy, rx, ry)) or max(abs(cx), abs(cy)) > 1e6:
            return None
        c0, c1 = max(0, int(math.floor(cx - rx))), min(self.w, int(math.ceil(cx + rx)) + 1)
        r0, r1 = max(0, int(math.floor(cy - ry))), min(self.h, int(math.ceil(cy + ry)) + 1)
        return None if c0 >= c1 or r0 >= r1 else (r0, r1, c0, c1)

    @staticmethod
    def _offsets(box, cx: float, cy: float):
        r0, r1, c0, c1 = box
        dx = np.arange(c0, c1, dtype=np.float32) + np.float32(0.5 - cx)
        dy = np.arange(r0, r1, dtype=np.float32) + np.float32(0.5 - cy)
        return dx[None, :], dy[:, None]

    def paint(self, box, alpha: np.ndarray, rgb, shadow_alpha: np.ndarray | None = None) -> None:
        """Paint ``alpha`` coverage of ``rgb`` into the box, over an optional dark underlay."""
        r0, r1, c0, c1 = box
        region = self.img[r0:r1, c0:c1]
        if shadow_alpha is not None:
            _over(region, shadow_alpha, SHADOW)
        _over(region, alpha, rgb)

    @staticmethod
    def _band(dist: np.ndarray, width: float) -> np.ndarray:
        """Anti-aliased coverage of a stroke of ``width`` px from the distance to its center
        line.
        """
        return np.clip(np.float32(0.5 * width + 0.5) - dist, 0.0, 1.0)

    # --- primitives ---------------------------------------------------------------
    def ring(
        self,
        cx,
        cy,
        radius,
        width,
        rgb,
        alpha=1.0,
        glow=0.0,
        dashes=0,
        dash_phase=0.0,
        dash_fill=0.55,
        shadow=0.0,
    ) -> None:
        """Anti-aliased circle outline; optional soft glow, rotating dashes and a dark underlay."""
        if alpha <= 0.01 or radius <= 0:
            return
        spread = 0.8 * width + 1.5
        pad = width + (3.0 * spread if glow > 0 else 0.0) + 3.0
        box = self._box(cx, cy, radius + pad, radius + pad)
        if box is None:
            return
        dx, dy = self._offsets(box, cx, cy)
        dist = np.abs(np.sqrt(dx * dx + dy * dy) - np.float32(radius))
        a = self._band(dist, width)
        under = self._band(dist, width + 3.0) * np.float32(shadow * alpha) if shadow > 0 else None
        if glow > 0:
            a = np.maximum(a, np.float32(glow) * np.exp(-((dist / np.float32(spread)) ** 2)))
        if dashes:
            turn = np.arctan2(dy, dx) * np.float32(dashes / (2 * math.pi)) + np.float32(dash_phase)
            on = (turn % 1.0) < dash_fill
            a = a * on
            if under is not None:
                under = under * on
        self.paint(box, a * np.float32(alpha), rgb, under)

    def disk(self, cx, cy, radius, rgb, alpha=1.0, glow=0.0, shadow=0.0) -> None:
        """Filled anti-aliased disk with an optional soft halo and dark underlay."""
        if alpha <= 0.01:
            return
        pad = radius + (2.5 * radius + 3 if glow > 0 else 3.0)
        box = self._box(cx, cy, pad, pad)
        if box is None:
            return
        dx, dy = self._offsets(box, cx, cy)
        d = np.sqrt(dx * dx + dy * dy)
        a = np.clip(np.float32(radius + 0.5) - d, 0.0, 1.0)
        under = (
            np.clip(np.float32(radius + 2.0) - d, 0.0, 1.0) * np.float32(shadow * alpha)
            if shadow
            else None
        )
        if glow > 0:
            a = np.maximum(
                a,
                np.float32(glow)
                * np.exp(-(((d - radius) / np.float32(radius + 2.0)) ** 2))
                * (d > radius),
            )
        self.paint(box, a * np.float32(alpha), rgb, under)

    def line(self, x0, y0, x1, y1, width, rgb, alpha=1.0, shadow=0.0) -> None:
        """Anti-aliased thick segment with round caps (and an optional dark underlay)."""
        if alpha <= 0.01:
            return
        pad = 0.5 * width + 3.0
        box = self._box(
            (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2 + pad, abs(y1 - y0) / 2 + pad
        )
        if box is None:
            return
        px, py = self._offsets(box, x0, y0)
        vx, vy = np.float32(x1 - x0), np.float32(y1 - y0)
        length2 = max(float(vx * vx + vy * vy), 1e-6)
        t = np.clip((px * vx + py * vy) / np.float32(length2), 0.0, 1.0)
        d = np.sqrt((px - t * vx) ** 2 + (py - t * vy) ** 2)
        under = self._band(d, width + 3.0) * np.float32(shadow * alpha) if shadow > 0 else None
        self.paint(box, self._band(d, width) * np.float32(alpha), rgb, under)

    def ellipse(self, cx, cy, cov_px: np.ndarray, k: float, width, rgb, alpha=1.0) -> None:
        """Outline of the ``k``-sigma ellipse of the pixel-space covariance ``cov_px``."""
        if alpha <= 0.01:
            return
        try:
            inv = np.linalg.inv(cov_px)
        except np.linalg.LinAlgError:
            return
        hx = k * math.sqrt(max(cov_px[0, 0], 1e-9)) + width + 2
        hy = k * math.sqrt(max(cov_px[1, 1], 1e-9)) + width + 2
        box = self._box(cx, cy, hx, hy)
        if box is None:
            return
        dx, dy = self._offsets(box, cx, cy)
        a00, a01, a11 = (np.float32(v) for v in (inv[0, 0], inv[0, 1], inv[1, 1]))
        gx, gy = a00 * dx + a01 * dy, a01 * dx + a11 * dy  # = Sigma^-1 q
        m = np.sqrt(np.maximum(dx * gx + dy * gy, 0.0))  # Mahalanobis distance
        grad = np.sqrt(gx * gx + gy * gy) / np.maximum(m, 1e-6)  # |d m / d pixel|
        d = np.abs(m - np.float32(k)) / np.maximum(grad, 1e-6)
        self.paint(
            box, np.clip(np.float32(0.5 * width + 0.5) - d, 0.0, 1.0) * np.float32(alpha), rgb
        )

    def rect(self, x0, y0, x1, y1, rgb, alpha=1.0) -> None:
        """Fill the integer pixel rectangle ``[x0, x1) x [y0, y1)``."""
        if not all(math.isfinite(v) and abs(v) < 1e6 for v in (x0, y0, x1, y1)):
            return
        c0, c1 = max(0, int(x0)), min(self.w, int(x1))
        r0, r1 = max(0, int(y0)), min(self.h, int(y1))
        if c0 < c1 and r0 < r1 and alpha > 0.01:
            _over(self.img[r0:r1, c0:c1], np.full((r1 - r0, c1 - c0), alpha, np.float32), rgb)

    def frame_rect(self, x0, y0, x1, y1, width, rgb, alpha=1.0) -> None:
        """Outline of the integer pixel rectangle ``[x0, x1) x [y0, y1)``, ``width`` px inside
        it.
        """
        self.rect(x0, y0, x1, y0 + width, rgb, alpha)
        self.rect(x0, y1 - width, x1, y1, rgb, alpha)
        self.rect(x0, y0 + width, x0 + width, y1 - width, rgb, alpha)
        self.rect(x1 - width, y0 + width, x1, y1 - width, rgb, alpha)

    def box(self, cx, cy, heading, half_length, half_width, rgb, alpha=1.0) -> None:
        """Filled anti-aliased rectangle rotated by ``heading`` (radians, CCW in the world
        frame).
        """
        reach = math.hypot(half_length, half_width) + 1.5
        box = self._box(cx, cy, reach, reach)
        if box is None:
            return
        dx, dy = self._offsets(box, cx, cy)
        c, s = np.float32(math.cos(heading)), np.float32(math.sin(heading))
        u = dx * c - dy * s  # along the heading (image rows point south)
        v = dx * s + dy * c
        inside = np.minimum(np.float32(half_length) - np.abs(u), np.float32(half_width) - np.abs(v))
        self.paint(box, np.clip(inside + 0.5, 0.0, 1.0) * np.float32(alpha), rgb)

    def splat(
        self,
        flat_index: np.ndarray,
        weight: np.ndarray,
        rgb,
        cap: float = 1.0,
        hot=None,
        hot_start: float = 1.0,
        soft: bool = False,
    ) -> None:
        """Additively accumulate ``weight`` at flat pixel indices, then paint the sum (clipped at
        ``cap``).

        Where the sum exceeds ``hot_start`` the color shifts toward ``hot``
        (dense particle clusters glow whiter). ``soft`` saturates smoothly
        (``cap * (1 - exp(-sum / cap))``) instead of clipping, so a dense swarm
        reads as a glow that fades at its rim rather than a flat patch. Only the
        index range in use is accumulated, so a compact cluster costs little.
        """
        if flat_index.size == 0:
            return
        base = int(flat_index.min())
        acc = np.bincount(flat_index - base, weights=weight)
        idx = np.flatnonzero(acc)
        total = acc[idx].astype(np.float32)
        idx += base
        if soft:
            alpha = np.float32(cap) * (np.float32(1.0) - np.exp(-total / np.float32(cap)))
        else:
            alpha = np.minimum(total, np.float32(cap))
        color = np.asarray(rgb, np.float32)
        if hot is not None:
            t = np.clip((total - np.float32(hot_start)) / np.float32(hot_start + 1.0), 0.0, 1.0)[
                :, None
            ]
            color = color + (np.asarray(hot, np.float32) - color) * t
        dst = self.flat[idx]
        _over(dst, alpha, color)
        self.flat[idx] = dst

    def paint_indices(self, flat_index: np.ndarray, alpha: float, rgb) -> None:
        """Paint a constant ``alpha`` of ``rgb`` at flat pixel indices (duplicates are harmless)."""
        if flat_index.size == 0:
            return
        dst = self.flat[flat_index]
        _over(dst, np.full(len(flat_index), alpha, np.float32), rgb)
        self.flat[flat_index] = dst

    def stamp_indices(self, rows: np.ndarray, cols: np.ndarray, offsets: np.ndarray):
        """Flat indices of ``offsets`` (K, 2) (drow, dcol) around every (row, col), clipped to the
        image.
        """
        rr = rows[:, None] + offsets[None, :, 0]
        cc = cols[:, None] + offsets[None, :, 1]
        ok = (rr >= 0) & (rr < self.h) & (cc >= 0) & (cc < self.w)
        return (rr * self.w + cc)[ok], ok

    def text(self, text, x, y, rgb, scale=2, alpha=1.0, align="left", shadow=(0, 0, 0)):
        return font5x7.draw_text(
            self.img,
            text,
            x,
            y,
            rgb,
            scale=scale,
            alpha=alpha,
            align=align,
            shadow=shadow,
            shadow_alpha=0.8,
        )


def _disk_offsets(radius: float, soft: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A round sprite: ``(K, 2)`` integer (drow, dcol) offsets around a pixel and their ``(K,)``
    coverage.
    """
    r = int(math.ceil(radius + soft))
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    d = np.hypot(dx, dy)
    if soft > 0:
        w = np.exp(-((np.maximum(d - radius, 0.0) / soft) ** 2)) * (d <= radius + soft)
    else:
        w = np.clip(radius + 0.5 - d, 0.0, 1.0)
    keep = w > 0.02
    return np.column_stack([dy[keep], dx[keep]]).astype(np.int64), w[keep].astype(np.float64)


# Image directions of the path's 4-connected steps: (d col, d row); north is up (row - 1).
_DIRECTIONS = ((1, 0), (-1, 0), (0, -1), (0, 1))  # east, west, north, south


def _chevron_sprites(half_length: float, half_width: float, stroke: float):
    """Anti-aliased chevrons (">" shapes) pointing along each of ``_DIRECTIONS``, centered on a
    pixel.

    Returns one ``((K, 2) int (d row, d col) offsets, (K,) coverage)`` sprite per direction.
    The tip is ``half_length`` px ahead of the center, the two arms end ``half_length`` px
    behind it and ``half_width`` px to each side; strokes are ``stroke`` px wide.
    """
    r = int(math.ceil(max(half_length, half_width) + stroke + 1.0))
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1].astype(
        np.float64
    )  # pixel-center offsets (rows, cols)
    sprites = []
    for ux, uy in _DIRECTIONS:
        tip = np.array((ux * half_length, uy * half_length))
        dist = np.full(dx.shape, np.inf)
        for side in (-1.0, 1.0):
            end = np.array(
                (
                    -ux * half_length - side * uy * half_width,
                    -uy * half_length + side * ux * half_width,
                )
            )
            seg = end - tip
            t = np.clip(
                ((dx - tip[0]) * seg[0] + (dy - tip[1]) * seg[1]) / float(seg @ seg), 0.0, 1.0
            )
            dist = np.minimum(dist, np.hypot(dx - tip[0] - t * seg[0], dy - tip[1] - t * seg[1]))
        cover = np.clip(0.5 * stroke + 0.5 - dist, 0.0, 1.0)
        keep = cover > 0.02
        sprites.append((np.column_stack([dy[keep], dx[keep]]).astype(np.int64), cover[keep]))
    return sprites


###############################################################################
#                                The renderer                                 #
###############################################################################


class HologramRenderer:
    """Draws the Robot Vision layers into the floor hologram and the minimap.

    One instance per controller: ``__init__`` precomputes the static layers
    (shelves, beacons, docks, proximity bands, sprites); ``render_floor`` and
    ``render_minimap`` then only draw what changed with the mission.
    """

    FLOOR_W, FLOOR_H = 1000, 500
    MINI_W, MINI_H = 500, 250
    FLOOR_SCALE, MINI_SCALE = 10, 5  # pixels per meter

    def __init__(self):
        # Particle sprites: a soft glow, and for the dead particles (weight 0) and the size classes
        # 1-3 crisp dots of 1-4 px on the floor (10-40 cm) and 1-3 px on the minimap.
        self._halo = _disk_offsets(1.2, soft=2.2)
        self._dots = [_disk_offsets(r) for r in (0.4, 0.8, 1.25, 1.75)]
        self._near_dots = [
            _disk_offsets(r) for r in (0.4, 0.6, 0.9, 1.2)
        ]  # close cameras: finer grain
        self._mini_dots = [_disk_offsets(r) for r in (0.4, 0.5, 0.8, 1.1)]
        # The planned path on the floor: marching chevrons (unlike the floor's painted yellow
        # dashes).
        self._chevrons_far = _chevron_sprites(3.0, 4.3, 2.3)  # overhead: bolder
        self._chevrons_near = _chevron_sprites(2.6, 3.8, 1.6)  # close cameras: finer strokes
        # Obstacle hatch tile (one 1 m cell) for the floor.
        s = self.FLOOR_SCALE
        yy, xx = np.mgrid[0:s, 0:s]
        self._hatch = np.where(((xx + yy) % 5) < 2, 0.95, 0.35).astype(np.float32)
        self._hatch[[0, -1], :] = 1.0
        self._hatch[:, [0, -1]] = 1.0
        # Soft tile edges ("LED floor panels") for the cell layers of the floor: alpha x 256.
        edge = np.full(s, 256, np.uint16)
        edge[0] = edge[-1] = 150
        self._tile16 = np.minimum(edge[:, None], edge[None, :])
        self._floor_static = self._build_floor_static()
        self._mini_static = self._build_minimap_static()
        self._mini_chrome = self._build_minimap_chrome()
        # Per-result caches of the parts that do not animate: (result, layer key, image).
        self._floor_cache = None
        self._mini_cache = None
        # Free cells of the static map (the value layer only colors these).
        self._free = np.asarray(BASE_MAP) == 0

    # =====================================================================
    # Static layers (built once)
    # =====================================================================
    def _build_floor_static(self):
        """Faint map the robot "knows": proximity-band edges and beacon markers (sparse pixels)."""
        img = np.zeros((self.FLOOR_H, self.FLOOR_W, 4), np.uint8)
        cv = _Canvas(img, self.FLOOR_SCALE)
        s = self.FLOOR_SCALE
        # Outline of the proximity-ON band (book 4.3.1): a dashed 2 px line just outside the band.
        band = PROXIMITY_MAP_ON[::-1] > 0  # image row order
        up = np.repeat(np.repeat(band, s, axis=0), s, axis=1)
        grown = up.copy()
        for d in (1, 2):
            grown[d:, :] |= up[:-d, :]
            grown[:-d, :] |= up[d:, :]
            grown[:, d:] |= up[:, :-d]
            grown[:, :-d] |= up[:, d:]
        edge = grown & ~up
        dash = ((np.arange(self.FLOOR_W)[None, :] + np.arange(self.FLOOR_H)[:, None]) % 10) < 6
        edge &= dash
        img[edge] = (*CYAN, 64)
        # Beacon pads: a ring and a dot at every RFID beacon.
        for bx, by in BEACONS:
            cx, cy = cv.px(bx, by)
            cv.ring(cx, cy, 7.0, 2.0, CYAN, 0.85)
            cv.disk(cx, cy, 2.0, CYAN, 0.9)
        idx = np.flatnonzero(img[..., 3].ravel())
        flat = img.reshape(-1, 4)
        return idx, flat[idx].copy()

    def _build_minimap_static(self) -> np.ndarray:
        """Opaque floor plan: background, grid, proximity bands, shelves, walls, beacons, docks."""
        H, W, s = self.MINI_H, self.MINI_W, self.MINI_SCALE
        img = np.zeros((H, W, 4), np.uint8)
        # Background: navy with a gentle vertical gradient.
        ramp = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
        top, bottom = np.array((10, 20, 36), np.float32), np.array(NAVY, np.float32)
        img[..., :3] = (top + (bottom - top) * ramp).astype(np.uint8)
        img[..., 3] = 255
        cv = _Canvas(img, s)
        # Faint 5 m grid.
        for x in range(0, 100, 5):
            cv.rect(int((x + 0.5) * s), 0, int((x + 0.5) * s) + 1, H, (60, 110, 160), 0.18)
        for y in range(0, 50, 5):
            r = int((49.5 - y) * s)
            cv.rect(0, r, W, r + 1, (60, 110, 160), 0.18)
        # Proximity bands, very faint (where the proximity sensor fires).
        band = np.repeat(np.repeat(PROXIMITY_MAP_ON[::-1] > 0, s, axis=0), s, axis=1)
        region = img[band]
        region[:, :3] = (
            region[:, :3].astype(np.float32) * 0.8 + np.array((20, 60, 90), np.float32) * 0.2
        )
        img[band] = region
        # Charging pads on the west wall.
        for y in (20.5, 24.5, 28.5):
            cx, cy = cv.px(0.4, y)
            cv.rect(cx - 2, cy - 4, cx + 3, cy + 4, (57, 200, 255), 0.55)
        # Shelf blocks with letters.
        for k, (x0, y0, x1, y1) in enumerate(SHELF_RECTS):
            c0, r0 = cv.px(x0, y1)
            c1, r1 = cv.px(x1, y0)
            cv.rect(c0, r0, c1, r1, (38, 50, 66), 1.0)
            # Two back-to-back rack rows per block, separated by a thin flue.
            mid = (c0 + c1) / 2
            cv.rect(mid - 1, r0 + 2, mid + 1, r1 - 2, (24, 32, 44), 1.0)
            for yy in range(int(y0) + 3, int(y1), 3):  # slot rows (every 3 m)
                ry = cv.px(0, yy)[1]
                cv.rect(c0 + 2, ry, mid - 2, ry + 1, (52, 68, 88), 1.0)
                cv.rect(mid + 2, ry, c1 - 2, ry + 1, (52, 68, 88), 1.0)
            cv.frame_rect(c0, r0, c1, r1, 1, (98, 128, 160), 1.0)
            cv.text(
                config.BLOCK_LETTERS[k],
                (c0 + c1) / 2,
                (r0 + r1) / 2 - 7,
                (120, 150, 185),
                scale=2,
                alpha=0.9,
                align="center",
                shadow=(20, 28, 38),
            )
        # Walls.
        cv.frame_rect(0, 0, W, H, 3, (70, 96, 124), 1.0)
        # Docks on the east wall: a green tick and the dock number.
        for dock in DOCKS:
            cx, cy = cv.px(99.5, dock.station[1])
            cv.rect(W - 6, cy - 9, W - 1, cy + 9, GREEN, 0.9)
            cv.text(
                str(dock.index), W - 12, cy - 7, GREEN, scale=2, align="right", shadow=(0, 0, 0)
            )
        # RFID beacons.
        for bx, by in BEACONS:
            cx, cy = cv.px(bx, by)
            cv.disk(cx, cy, 2.2, CYAN, 0.9)
        return img

    # =====================================================================
    # Cell layers (value function, likelihood, grid belief)
    # =====================================================================
    def _cell_layers(
        self, frame: HoloFrame, obstacles: np.ndarray, value_alpha: float, heat_alpha: float
    ) -> np.ndarray | None:
        """Composite the per-cell layers into a (50, 100, 4) float RGBA grid (rows = cell rows
        i).
        """
        result, layers = frame.result, frame.layers
        belief = getattr(result, "belief", None)
        acc = None
        if layers.map_layer == "value" and getattr(result, "value", None) is not None:
            acc = _over_cells(
                acc,
                self._value_cells(
                    result.value, getattr(result, "goal_mask", None), obstacles, value_alpha
                ),
            )
        elif layers.map_layer == "change" and getattr(result, "value_change", None) is not None:
            acc = _over_cells(acc, self._change_cells(result.value_change, obstacles, value_alpha))
        elif (
            layers.map_layer == "likelihood"
            and belief is not None
            and belief.likelihood is not None
        ):
            acc = _over_cells(acc, self._likelihood_cells(belief.likelihood, heat_alpha))
        if layers.belief in ("grid", "both") and belief is not None and belief.grid is not None:
            acc = _over_cells(acc, self._grid_cells(belief.grid, heat_alpha))
        return acc

    def _value_cells(self, V, goal_mask, obstacles, alpha) -> np.ndarray:
        """Turbo value function on free cells (the book's Figure 2 of 4.5), robustly normalized.

        V is 100 next to the goal, decays with the discount factor, and is
        strongly negative in the proximity bands. Positive values are colored
        by rank (histogram equalization, so the whole cyan -> red half of the
        colormap shows the gradient toward the goal); negative values are
        squashed into the dark blues, most negative darkest.
        """
        V = np.asarray(V, dtype=np.float32)
        free = self._free & ~obstacles
        out = np.zeros((MAP_H, MAP_W, 4), np.float32)
        vals = V[free]
        if vals.size == 0 or not np.all(np.isfinite(vals)):
            return out
        t = np.zeros((MAP_H, MAP_W), np.float32)
        pos = free & (V >= 0)
        if np.any(pos):
            ranks = np.argsort(np.argsort(V[pos], kind="stable"), kind="stable").astype(np.float32)
            t[pos] = 0.30 + 0.70 * ranks / max(1.0, float(pos.sum() - 1))
        neg = free & (V < 0)
        if np.any(neg):
            lo = min(float(np.percentile(V[neg], 5.0)), -1e-6)
            t[neg] = 0.26 * (1.0 - np.clip(V[neg] / lo, 0.0, 1.0)) + 0.02
        out[..., :3] = TURBO[np.clip((t * 255).astype(np.int32), 0, 255)]
        out[..., 3] = np.where(free, alpha, 0.0)
        if goal_mask is not None:
            out[np.asarray(goal_mask, dtype=bool), 3] = min(1.0, alpha + 0.3)
        return out

    def _change_cells(self, dV, obstacles, alpha) -> np.ndarray:
        """How the value function changed after an obstacle drop (new minus old V): red where the
        value dropped, green where it rose, stronger for bigger changes (square root of the share
        of the largest change, so that the small changes far away still show)."""
        d = np.asarray(dV, dtype=np.float32)
        out = np.zeros((MAP_H, MAP_W, 4), np.float32)
        free = self._free & ~obstacles & np.isfinite(d)
        if not np.any(free):
            return out
        scale = float(np.max(np.abs(d[free])))
        if not (scale > 1e-6 and math.isfinite(scale)):
            return out
        drop = free & (d < -1e-6)
        rise = free & (d > 1e-6)
        out[drop, :3] = RED
        out[rise, :3] = GREEN
        strength = np.sqrt(np.clip(np.abs(d) / np.float32(scale), 0.0, 1.0))
        out[..., 3] = np.where(
            drop | rise, np.float32(min(1.0, alpha + 0.25)) * (0.25 + 0.75 * strength), 0.0
        )
        return out

    @staticmethod
    def _likelihood_cells(L, alpha) -> np.ndarray:
        """Orange heat of the last measurement's likelihood (normalized to its maximum)."""
        L = np.asarray(L, dtype=np.float32)
        out = np.zeros((MAP_H, MAP_W, 4), np.float32)
        top = float(L.max()) if L.size else 0.0
        if not math.isfinite(top) or top <= 0:
            return out
        n = np.clip(L / top, 0.0, 1.0) ** 0.7
        out[..., :3] = LIKELIHOOD_RAMP[(n * 255).astype(np.int32)]
        out[..., 3] = alpha * np.clip((n - 0.03) / 0.97, 0.0, 1.0)
        return out

    @staticmethod
    def _grid_cells(grid, alpha) -> np.ndarray:
        """Magenta heat of the Markov grid belief.

        The color follows p / max(p); the opacity also fades with the spread
        of the belief (effective number of cells) so that a uniform belief is
        a faint wash, not a solid magenta floor.
        """
        g = np.asarray(grid, dtype=np.float64)
        out = np.zeros((MAP_H, MAP_W, 4), np.float32)
        top = float(g.max()) if g.size else 0.0
        if not math.isfinite(top) or top <= 0:
            return out
        total = float(g.sum())
        n_eff = total * total / max(float((g * g).sum()), 1e-300)
        strength = min(1.0, max(0.3, 1.15 - 0.2 * math.log10(max(n_eff, 1.0))))
        n = np.sqrt(np.clip(g / top, 0.0, 1.0)).astype(np.float32)
        out[..., :3] = GRID_RAMP[(n * 255).astype(np.int32)]
        out[..., 3] = alpha * strength * np.clip((n - 0.08) / 0.92, 0.0, 1.0)
        return out

    @staticmethod
    def _obstacle_cells(result) -> np.ndarray:
        """(50, 100) bool mask of the dropped obstacles' cells."""
        mask = np.zeros((MAP_H, MAP_W), bool)
        for ob in getattr(result, "obstacles", None) or []:
            try:
                if "cell" in ob:
                    i, j = int(ob["cell"][0]), int(ob["cell"][1])
                else:
                    i, j = int(round(float(ob["y"]))), int(round(float(ob["x"])))
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                continue
            if 0 <= i < MAP_H and 0 <= j < MAP_W:
                mask[i, j] = True
        return mask

    # =====================================================================
    # The floor hologram
    # =====================================================================
    def render_floor(self, frame: HoloFrame) -> np.ndarray:
        """The Robot Vision floor texture: ``(500, 1000, 4) uint8`` RGBA, transparent background."""
        result = frame.result
        if result is None or not frame.layers.vision:
            return np.zeros((self.FLOOR_H, self.FLOOR_W, 4), np.uint8)
        obstacles = self._obstacle_cells(result)
        img = self._floor_base(frame, obstacles).copy()  # steps 1-3 (they do not animate)
        cv = _Canvas(img, self.FLOOR_SCALE)
        s = self.FLOOR_SCALE
        t = _finite(frame.t)
        k = min(1.0, max(0.0, _finite(getattr(frame, "near", 0.0))))  # 0 overhead .. 1 close camera
        shift = _shift_of(frame)
        belief = getattr(result, "belief", None)
        lost = bool(belief is not None and belief.relocalizing)

        # 3b. A pulsing red glow around every dropped obstacle.
        for i, j in zip(*np.nonzero(obstacles), strict=False):
            cx, cy = j * s + s / 2, (MAP_H - 1 - i) * s + s / 2
            cv.ring(
                cx,
                cy,
                0.95 * s,
                _lerp(2.0, 1.4, k),
                RED,
                0.45 + 0.3 * math.sin(4 * t),
                glow=0.35 * (1 - k),
            )

        # 4. Goal: the goal cell, a steady ring and expanding "sonar" rings (not under the docking
        # robot).
        at_goal = str(frame.phase) in _NO_FLOOR_GOAL_PHASES
        goal = self._goal_xy(frame)
        if goal is not None and not at_goal:
            self._draw_goal(cv, goal, t, big=True, near=k)

        # 5. Planned path: chevrons marching toward the goal along a faint guide line (none while
        # docking at it); finer strokes for close cameras, bolder ones for the overhead camera.
        path = list(getattr(result, "path", None) or [])
        if len(path) >= 2 and not at_goal:
            near_path = k >= 0.5
            self._draw_path(
                cv,
                path,
                t,
                width=1 if near_path else 2,
                under_width=5 if near_path else 7,
                period=16,
                dash=16,
                speed=12.0,
                under_alpha=_lerp(0.35, 0.22, k),
                chevrons=self._chevrons_near if near_path else self._chevrons_far,
                line_alpha=_lerp(0.45, 0.3, k),
            )

        # 5b. A sensor test: the true path (white) and the believed path (magenta).
        trails = getattr(frame, "trails", None)
        if trails:
            self._draw_trails(
                cv, trails, truth_width=_lerp(2.0, 1.2, k), belief_width=_lerp(3.0, 1.8, k)
            )

        # 6. RFID annulus of the last ping (book Figure 5 of 4.3): radius = measured range.
        self._draw_rfid(cv, frame, t, sigma_px=1.0 * s, width=_lerp(2.2, 1.3, k), near=k)

        # 7. The zone reading: the zone the reader names, outlined (it can be a neighbor near a
        # boundary).
        self._draw_zone(
            cv,
            result,
            _finite(frame.zone_age),
            width=_lerp(2.5, 1.5, k),
            fill=_lerp(0.07, 0.035, k),
        )

        # 8. Particles (every particle a dot, size ~ weight), then the estimate on top; both slide
        # with the robot (shift).
        if belief is not None and frame.layers.belief in ("particles", "both"):
            self._draw_particles_floor(cv, belief, t, lost, frame.weigh, near=k, shift=shift)
        if belief is not None:
            self._draw_estimate(
                cv,
                belief,
                t,
                lost,
                ring_r=0.95 * s,
                width=_lerp(2.0, 1.2, k),
                big=True,
                near=k,
                shift=shift,
                ring=bool(getattr(frame, "estimate_ring", True)),
            )
        # 9. The end of a sensor test: the gap between the robot and its final estimate, on top.
        if trails:
            self._draw_gap(cv, trails, width=_lerp(3.0, 1.8, k), label=2)
        return img

    def _floor_base(self, frame: HoloFrame, obstacles: np.ndarray) -> np.ndarray:
        """Steps 1-3 of the floor (they only change with the result and the layer choice), cached.

        The Webots layer re-renders the same result a few times per second
        for the animations; only the first call pays for the cell layers.
        """
        key = (frame.layers.belief, frame.layers.map_layer)
        cached = self._floor_cache
        if cached is not None and cached[0] is frame.result and cached[1] == key:
            return cached[2]
        img = np.zeros((self.FLOOR_H, self.FLOOR_W, 4), np.uint8)
        s = self.FLOOR_SCALE
        flat = img.reshape(-1, 4)
        # 1. Cell layers: value / likelihood map, Markov grid belief (tiled "LED panels").
        cells = self._cell_layers(frame, obstacles, value_alpha=0.66, heat_alpha=0.85)
        idx, rgba = self._floor_static
        if cells is None:
            flat[idx] = rgba  # 2. (on a transparent image) the static map overlay
        else:
            rgba_cells = cells[::-1]  # image rows run north -> south
            view = img.reshape(
                MAP_H, s, MAP_W, s, 4
            )  # (cell row, pixel row, cell col, pixel col, rgba)
            view[..., :3] = np.clip(rgba_cells[..., :3] + 0.5, 0, 255).astype(np.uint8)[
                :, None, :, None, :
            ]
            a8 = (np.clip(rgba_cells[..., 3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint16)
            view[..., 3] = ((a8[:, None, :, None] * self._tile16[None, :, None, :]) >> 8).astype(
                np.uint8
            )
            # 2. The static map overlay (proximity band edges, beacon pads).
            dst = flat[idx]
            _over(dst, rgba[:, 3].astype(np.float32) / 255.0, rgba[:, :3].astype(np.float32))
            flat[idx] = dst
        # 3. Dropped obstacles: red hatched cells.
        for i, j in zip(*np.nonzero(obstacles), strict=False):
            r0, c0 = (MAP_H - 1 - i) * s, j * s
            _over(img[r0 : r0 + s, c0 : c0 + s], self._hatch, RED)
        # 3c. The zones layer (M): every zone of the zone reader, tinted by kind, outlined and
        # named.
        if frame.layers.map_layer == "zones":
            self._draw_all_zones(_Canvas(img, s))
        self._floor_cache = (frame.result, key, img)
        return img

    # --- floor layers ----------------------------------------------------------
    @staticmethod
    def _goal_xy(frame: HoloFrame):
        if frame.goal_xy is not None:
            try:
                x, y = float(frame.goal_xy[0]), float(frame.goal_xy[1])
                return (x, y) if math.isfinite(x) and math.isfinite(y) else None
            except (TypeError, ValueError, IndexError):
                return None
        cell = getattr(frame.result, "goal_cell", None)
        if cell is not None:
            return float(cell[1]), float(cell[0])
        return None

    def _draw_goal(self, cv: _Canvas, goal, t: float, big: bool, near: float = 0.0) -> None:
        """The goal cell (the MDP's goal state) with a steady ring and two expanding "sonar" rings.

        ``near``: thinner rings for close cameras (the briefing preview looks at the goal from a few
        meters).
        """
        s = cv.s
        cx, cy = cv.px(goal[0], goal[1])
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 0.9 * t)
        c0, r0 = int(round(cx - s / 2)), int(round(cy - s / 2))
        cv.rect(c0 - 1, r0 - 1, c0 + s + 1, r0 + s + 1, SHADOW, _lerp(0.45, 0.25, near))
        cv.rect(c0, r0, c0 + s, r0 + s, YELLOW, _lerp(0.30, 0.2, near) + 0.25 * pulse)
        cv.frame_rect(c0, r0, c0 + s, r0 + s, 2 if big and near < 0.5 else 1, YELLOW, 1.0)
        period = 1.6
        for k in range(2):
            phase = ((t / period) + 0.5 * k) % 1.0
            radius = s * (0.8 + 1.9 * phase)
            cv.ring(
                cx,
                cy,
                radius,
                _lerp(1.8, 1.1, near) if big else 1.5,
                YELLOW,
                0.9 * (1.0 - phase) ** 1.3,
                shadow=_lerp(0.3, 0.1, near) if big else 0.0,
            )
        cv.ring(
            cx,
            cy,
            1.25 * s,
            _lerp(2.2, 1.3, near) if big else 1.2,
            YELLOW,
            0.6 + 0.35 * pulse,
            shadow=_lerp(0.35, 0.12, near) if big else 0.0,
        )

    def _draw_path(
        self,
        cv: _Canvas,
        path,
        t: float,
        width: int,
        under_width: int,
        period: int,
        dash: int,
        speed: float,
        under_alpha: float,
        chevrons=None,
        line_alpha: float = 0.97,
    ) -> None:
        """The planned path over a dark lane, with a big chevron at the goal end.

        ``path`` is the greedy rollout ``[(i, j), ...]`` of the student's
        policy (4-connected cells). Without ``chevrons`` (the minimap) it is a
        line of yellow dashes (``dash`` px out of every ``period`` px); with
        ``chevrons`` (sprites from :func:`_chevron_sprites`, the floor) it is a
        faint ``width`` px guide line (``line_alpha``) with one chevron every
        ``period`` px, which reads as a route and never as the floor's painted
        yellow dashes. Both march toward the goal at ``speed`` px/s.
        """
        s = cv.s
        cells = np.asarray(path, dtype=np.int64)
        rows = (MAP_H - 1 - cells[:, 0]) * s + s // 2
        cols = cells[:, 1] * s + s // 2
        # One sample per pixel along the polyline (a step is one cell = s pixels; robust to
        # repeats).
        seg_r, seg_c = np.diff(rows), np.diff(cols)
        steps = np.maximum(np.maximum(np.abs(seg_r), np.abs(seg_c)), 1)
        seg_id = np.repeat(np.arange(len(steps)), steps)
        start = np.concatenate([[0], np.cumsum(steps)[:-1]])
        frac = (np.arange(int(steps.sum())) - start[seg_id]) / steps[seg_id]
        pr = np.round(rows[:-1][seg_id] + seg_r[seg_id] * frac).astype(np.int64)
        pc = np.round(cols[:-1][seg_id] + seg_c[seg_id] * frac).astype(np.int64)
        length = np.hypot(seg_r, seg_c)[seg_id] / steps[seg_id]
        arc = np.cumsum(length) - length
        # Stop short of the goal cell so the goal marker stays clean.
        keep = arc <= arc[-1] + length[-1] - 0.9 * s
        pr, pc, arc, seg_id = pr[keep], pc[keep], arc[keep], seg_id[keep]
        if pr.size == 0:
            return
        # Dark lane under the path (readable on the bright floor).
        half = under_width // 2
        uy, ux = np.mgrid[-half : under_width - half, -half : under_width - half]
        ui, _ = cv.stamp_indices(pr, pc, np.column_stack([uy.ravel(), ux.ravel()]))
        cv.paint_indices(ui, under_alpha, SHADOW)
        # The dashes (or, with chevrons, the continuous guide line).
        on = ((arc - speed * t) % period) < dash
        half = width // 2
        dy, dx = np.mgrid[-half : width - half, -half : width - half]
        di, _ = cv.stamp_indices(pr[on], pc[on], np.column_stack([dy.ravel(), dx.ravel()]))
        cv.paint_indices(di, line_alpha, YELLOW)
        if chevrons is not None:
            # One chevron per period, pointing along its step of the path (4-connected: E, W, N, S).
            mark = ((arc - speed * t) % period) < 1.0
            step_c, step_r = seg_c[seg_id], seg_r[seg_id]
            horizontal = np.abs(step_c) >= np.abs(step_r)
            classes = np.where(horizontal, np.where(step_c > 0, 0, 1), np.where(step_r < 0, 2, 3))
            mark &= (step_c != 0) | (step_r != 0)  # (a repeated cell has no direction)
            idx_all, w_all = [], []
            for d, (offsets, cover) in enumerate(chevrons):
                sel = mark & (classes == d)
                if np.any(sel):
                    ci, cok = cv.stamp_indices(pr[sel], pc[sel], offsets)
                    idx_all.append(ci)
                    w_all.append(np.broadcast_to(cover[None, :], cok.shape)[cok])
            if idx_all:
                cv.splat(np.concatenate(idx_all), np.concatenate(w_all), YELLOW, cap=1.0)
        # Chevron pointing along the last direction, just before the goal.
        k = len(pr) - 1
        back = max(0, k - int(1.2 * s))
        vr, vc = float(pr[k] - pr[back]), float(pc[k] - pc[back])
        norm = math.hypot(vr, vc)
        if norm > 0:
            vr, vc = vr / norm, vc / norm
            tip_c, tip_r = pc[k] + 0.5, pr[k] + 0.5
            size = 0.9 * s
            for sign in (-1, 1):
                ec = tip_c - size * vc + sign * size * 0.8 * vr
                er = tip_r - size * vr - sign * size * 0.8 * vc
                cv.line(
                    tip_c, tip_r, ec, er, max(2.0, width - 1.0), YELLOW, 1.0, shadow=under_alpha
                )

    @staticmethod
    def _ping(frame: HoloFrame):
        """(beacon id, range m, age s) of a recent, valid RFID ping, or None."""
        try:
            beacon_id, rng, age = (
                int(frame.ping[0]),
                float(frame.ping[1]),
                max(0.0, _finite(frame.ping[2])),
            )
        except (TypeError, ValueError, IndexError):
            return None
        if (
            not (0 <= beacon_id < len(BEACONS))
            or not (0.0 <= rng <= 3 * config.RFID_MAX_RANGE)
            or age > 2.4
        ):
            return None
        return beacon_id, rng, age

    def _draw_rfid(
        self,
        cv: _Canvas,
        frame: HoloFrame,
        t: float,
        sigma_px: float,
        width: float,
        near: float = 0.0,
    ) -> None:
        ping = self._ping(frame)
        if ping is None:
            return
        beacon_id, rng, age = ping
        fade = max(0.0, 1.0 - age / 2.4) ** 1.2
        cx, cy = cv.px(*BEACONS[beacon_id])
        radius = max(rng * cv.s, 1.0)
        # Soft annulus: the book's RFID likelihood exp(-(r - d)^2 / 2 sigma^2) with sigma = 1 m
        # (fainter
        # in close views, where the 2 m wide band would wash the floor around the robot orange) ...
        pad = 3.0 * sigma_px
        box = cv._box(cx, cy, radius + pad, radius + pad)
        if box is not None:
            dx, dy = cv._offsets(box, cx, cy)
            dist = np.sqrt(dx * dx + dy * dy) - np.float32(radius)
            a = np.exp(-0.5 * (dist / np.float32(sigma_px)) ** 2) * np.float32(
                _lerp(0.24, 0.11, near) * fade
            )
            cv.paint(box, a, ORANGE)
        # ... with a crisp ring at the measured range and a short "ping" wave from the beacon.
        cv.ring(cx, cy, radius, width, ORANGE, 0.95 * fade, glow=0.2 * (1.0 - near))
        if age < 0.6:
            cv.ring(cx, cy, radius * age / 0.6, 2.0, (255, 200, 120), 0.8 * (1 - age / 0.6))
        cv.disk(cx, cy, 0.35 * cv.s, ORANGE, fade, glow=0.5)

    @staticmethod
    def _draw_all_zones(cv: _Canvas) -> None:
        """Every zone: a tint by kind (RECEIVING blue, aisles green, cross aisles violet, SHIPPING
        orange,
        neighboring bays alternating), a white outline, and the name in the middle.
        """
        for zone, (x0, y0, x1, y1) in enumerate(ZONE_RECTS):
            name = config.ZONE_NAMES[zone]
            kind = name.split()[0]
            color = {
                "RECEIVING": (90, 160, 255),
                "SHIPPING": (255, 160, 60),
                "AISLE": (90, 230, 110),
                "CROSS": (190, 140, 255),
            }[kind]
            checker = kind in ("RECEIVING", "SHIPPING") and (int(round(x0 + y0)) // 10) % 2 == 1
            (ax, ay), (bx, by) = cv.px(x0, y0), cv.px(x1, y1)
            left, right, top, bottom = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
            cv.rect(left, top, right, bottom, color, 0.42 if checker else 0.3)
            for p0, p1 in (
                ((left, top), (right, top)),
                ((right, top), (right, bottom)),
                ((right, bottom), (left, bottom)),
                ((left, bottom), (left, top)),
            ):
                cv.line(p0[0], p0[1], p1[0], p1[1], 3.0, WHITE, 0.95, shadow=0.5)
            cx, cy = (left + right) / 2, (top + bottom) / 2
            if kind in (
                "RECEIVING",
                "SHIPPING",
            ):  # the bay, e.g. "3W" (the floor paint names the area)
                cv.text(name.split()[1], cx, cy - 10, WHITE, scale=3, align="center")
            elif kind == "AISLE":
                cv.text("AISLE", cx, cy - 22, WHITE, scale=2, align="center")
                cv.text(name.split()[1], cx, cy - 2, WHITE, scale=4, align="center")
            else:
                cv.text(name, cx, cy - 10, WHITE, scale=3, align="center")
        # The RFID beacons on the shelf ends, with their ids and their 8 m range.
        for k, (bx, by) in enumerate(BEACONS):
            px, py = cv.px(float(bx), float(by))
            cv.ring(px, py, config.RFID_MAX_RANGE * cv.s, 2.0, CYAN, 0.55, dashes=48)
            cv.disk(px, py, 0.6 * cv.s, CYAN, 1.0, glow=0.5, shadow=0.6)
            south = by < MAP_H / 2
            cv.text(f"RFID {k}", px, py + (8 if south else -28), CYAN, scale=2, align="center")

    def _draw_zone(self, cv: _Canvas, result, age: float, width: float, fill: float = 0.07) -> None:
        """The zone the zone reader names: a light fill, an outline, and its name in the corner."""
        sensors = getattr(result, "sensors", None)
        zone = getattr(sensors, "z_zone", None) if sensors is not None else None
        if zone is None or not 0 <= int(zone) < len(ZONE_RECTS):
            return
        fade = max(0.6, 1.0 - max(age, 0.0) / 0.9)
        x0, y0, x1, y1 = ZONE_RECTS[int(zone)]
        (ax, ay), (bx, by) = cv.px(x0, y0), cv.px(x1, y1)
        left, right, top, bottom = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        cv.rect(left, top, right, bottom, ZONE_LIME, fill * fade)
        for p0, p1 in (
            ((left, top), (right, top)),
            ((right, top), (right, bottom)),
            ((right, bottom), (left, bottom)),
            ((left, bottom), (left, top)),
        ):
            cv.line(p0[0], p0[1], p1[0], p1[1], width, ZONE_LIME, 0.75 * fade, shadow=0.35)
        cv.text(config.ZONE_NAMES[int(zone)], left + 4, top + 4, ZONE_LIME, scale=1, alpha=fade)

    def _particle_style(self, belief, t: float, lost: bool, weigh: float = 1.0):
        """Per-particle size class (0: dead, 1..3), relative weight (1 = average) and the swarm's
        color.

        ``weigh`` < 1 blends every relative weight toward 1 (the equal weights of the prior).
        Returns ``(particles, size, rel, color, gain)``.
        """
        p = np.asarray(belief.particles, dtype=np.float64)
        w = np.asarray(belief.weights, dtype=np.float64)
        n = max(len(w), 1)
        rel = np.where(np.isfinite(w), w, 0.0) * n
        weigh = min(1.0, max(0.0, _finite(weigh, 1.0)))
        if weigh < 1.0:
            rel = 1.0 + (rel - 1.0) * weigh
        size = np.select([rel <= 1e-9, rel < 0.6, rel < 2.5], [0, 1, 2], 3)
        if lost:
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.3 * t)
            color = _mix(RED, MAGENTA, pulse)
            gain = 0.65 + 0.35 * pulse
        else:
            color, gain = CYAN, 1.0
        return p, size, rel, color, gain

    def _draw_particles_floor(
        self,
        cv: _Canvas,
        belief,
        t: float,
        lost: bool,
        weigh: float = 1.0,
        near: float = 0.0,
        shift: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Particles as crisp "voxels" of 1-4 px (10-40 cm on the floor), bigger and brighter with
        weight.

        The floor texture has only 10 px per meter and the chase camera sees it
        up close, so the swarm stays light: a faint glow under the heavier
        particles only, and dense clusters saturate to a paler cyan, not white.
        Close cameras (``near``) get a finer grain and a more translucent swarm.
        Every camera shows the particles themselves (the samples), never a
        density estimate drawn from them.
        """
        p, size, rel, color, gain = self._particle_style(belief, t, lost, weigh)
        if len(p) == 0 or p.ndim != 2 or p.shape[1] < 2 or len(rel) != len(p):
            return
        s = cv.s
        p = p[:, :2]
        ok = np.all(np.isfinite(p) & (np.abs(p) < 1e4), axis=1)
        xy = p[ok] + np.asarray(shift, dtype=np.float64)
        size = size[ok]
        cols = np.floor((xy[:, 0] + 0.5) * s).astype(np.int64)
        rows = np.floor((49.5 - xy[:, 1]) * s).astype(np.int64)
        # Once LOCKED the 600 particles pile up within a meter of the robot: draw them as a
        # softer, translucent swarm (the global swarm stays bright).
        calm = bool(belief.locked) and not lost
        if calm:
            halo, halo_cap = _lerp(0.045, 0.0, near), 0.15
            strengths = tuple(
                _lerp(a, b, near)
                for a, b in zip((0.22, 0.5, 0.62, 0.7), (0.14, 0.3, 0.38, 0.44), strict=False)
            )
            cap = _lerp(0.62, 0.42, near)
        else:
            halo, halo_cap = _lerp(0.07, 0.03, near), 0.24
            strengths = tuple(
                _lerp(a, b, near)
                for a, b in zip((0.3, 0.75, 0.95, 1.0), (0.25, 0.6, 0.8, 0.9), strict=False)
            )
            cap = _lerp(0.92, 0.85, near)
        if halo > 0.005:
            heavy = size >= 2
            offsets, cover = self._halo
            hi, hok = cv.stamp_indices(rows[heavy], cols[heavy], offsets)
            cv.splat(
                hi,
                np.broadcast_to(cover[None, :] * halo, hok.shape)[hok] * gain,
                color,
                cap=halo_cap,
            )
        sprites = self._near_dots if near >= 0.5 else self._dots
        self._splat_sprites(
            cv,
            rows,
            cols,
            size,
            sprites,
            strengths,
            color,
            gain,
            hot=None if lost else (150, 245, 255),
            hot_start=2.5,
            cap=cap,
            soft=calm,
        )

    @staticmethod
    def _splat_sprites(
        cv: _Canvas,
        rows,
        cols,
        size,
        sprites,
        strengths,
        color,
        gain,
        hot,
        hot_start,
        cap: float = 1.0,
        soft: bool = False,
    ) -> None:
        """Stamp each particle with the sprite of its size class (0 = dead, 1..3) in one additive
        splat.

        ``sprites`` has four entries: the dead particles' sprite, then sizes 1-3.
        """
        idx_all, w_all = [], []
        for cls, (sprite, strength) in enumerate(zip(sprites, strengths, strict=False)):
            sel = size == cls
            if not np.any(sel):
                continue
            offsets, cover = sprite
            di, dok = cv.stamp_indices(rows[sel], cols[sel], offsets)
            idx_all.append(di)
            w_all.append(np.broadcast_to(cover[None, :] * strength, dok.shape)[dok])
        if idx_all:
            cv.splat(
                np.concatenate(idx_all),
                np.concatenate(w_all) * gain,
                color,
                cap=cap,
                hot=hot,
                hot_start=hot_start,
                soft=soft,
            )

    def _draw_estimate(
        self,
        cv: _Canvas,
        belief,
        t: float,
        lost: bool,
        ring_r: float,
        width: float,
        big: bool,
        near: float = 0.0,
        shift: tuple[float, float] = (0.0, 0.0),
        ring: bool = True,
    ) -> None:
        """The student's estimate: a magenta ring at the mean, its 2-sigma ellipse, and a lock
        reticle.

        ``shift`` (m) slides everything with the robot (see ``HoloFrame.belief_shift``); ``near``
        thins the strokes for close cameras; ``ring=False`` leaves the ring, its reticle and the
        center dot to the Webots layer's 3D reticle (the ellipse stays).
        """
        mean = np.asarray(belief.mean, dtype=np.float64)
        if mean.shape != (2,) or not np.all(np.isfinite(mean)):
            return
        cx, cy = cv.px(mean[0] + shift[0], mean[1] + shift[1])
        s = cv.s
        color = RED if lost else MAGENTA
        shadow = _lerp(0.3, 0.12, near) if big else 0.0
        # 2-sigma ellipse of the student's covariance, faded out while the belief is huge. Without
        # the
        # ring (close cameras: the Webots layer's 3D reticle, radius ~1 m, marks the estimate) it
        # also
        # fades out while it is about the reticle's size: two nested rings would only look blurry.
        cov = np.asarray(belief.cov, dtype=np.float64)
        if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
            sigma = math.sqrt(max(float(np.max(np.linalg.eigvalsh((cov + cov.T) / 2))), 0.0))
            fade = min(1.0, max(0.0, (14.0 - sigma) / 6.0))
            if not ring:
                fade *= min(1.0, max(0.0, (2.0 * sigma - 1.2) / 0.6))
            if fade > 0 and sigma > 0.05:
                cov_px = np.array([[cov[0, 0], -cov[0, 1]], [-cov[1, 0], cov[1, 1]]]) * (s * s)
                cov_px += np.eye(2) * 1e-3
                cv.ellipse(
                    cx,
                    cy,
                    cov_px,
                    2.0,
                    _lerp(1.4, 1.0, near) if big else 1.2,
                    color,
                    (_lerp(0.6, 0.38, near) if big else 0.9) * fade,
                )
        if ring and lost:
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.3 * t)
            cv.ring(
                cx,
                cy,
                ring_r,
                width + 0.4,
                color,
                0.55 + 0.4 * pulse,
                dashes=6,
                dash_phase=-0.4 * t,
                shadow=shadow,
            )
            return
        if ring and belief.locked:
            cv.ring(
                cx,
                cy,
                ring_r,
                width,
                color,
                0.95,
                glow=0.18 * (1.0 - near) if big else 0.0,
                shadow=shadow,
            )
            # Four thin reticle ticks slowly turning around the lock.
            spin = 0.6 * t
            tick = _lerp(1.6, 1.0, near) if big else 1.2
            for k in range(4):
                ang = spin + k * math.pi / 2
                c, sn = math.cos(ang), math.sin(ang)
                cv.line(
                    cx + c * (ring_r + 0.3 * s),
                    cy + sn * (ring_r + 0.3 * s),
                    cx + c * (ring_r + 0.6 * s),
                    cy + sn * (ring_r + 0.6 * s),
                    tick,
                    color,
                    0.95,
                    shadow=shadow,
                )
        elif ring:
            # Still localizing: a ring with four gaps, turning (the robot is "scanning").
            cv.ring(
                cx,
                cy,
                ring_r,
                width + 0.4,
                color,
                0.95,
                dashes=4,
                dash_phase=0.3 * t,
                dash_fill=0.78,
                shadow=shadow,
            )
        if (
            not lost and ring
        ):  # (the 3D reticle marks the center well enough; up close the dot is a blob)
            cv.disk(cx, cy, 0.12 * s + 0.6, (255, 225, 250), 1.0, shadow=0.4 if big else 0.0)

    # =====================================================================
    # The minimap
    # =====================================================================
    def render_minimap(self, frame: HoloFrame) -> np.ndarray:
        """The minimap overlay: ``(250, 500, 4) uint8`` RGBA with an opaque dark background."""
        result = frame.result
        img = self._mini_base(frame).copy()
        cv = _Canvas(img, self.MINI_SCALE)
        s = self.MINI_SCALE
        t = _finite(frame.t)
        belief = getattr(result, "belief", None) if result is not None else None
        lost = bool(belief is not None and belief.relocalizing)
        shift = _shift_of(frame)

        if result is not None:
            self._draw_orders_minimap(cv, result, t)
            goal = self._goal_xy(frame)
            if goal is not None:
                self._draw_goal(cv, goal, t, big=False)
            path = list(getattr(result, "path", None) or [])
            if len(path) >= 2:
                self._draw_path(
                    cv,
                    path,
                    t,
                    width=2,
                    under_width=4,
                    period=8,
                    dash=5,
                    speed=6.0,
                    under_alpha=0.5,
                )
            self._draw_rfid_minimap(cv, frame)
            trails = getattr(frame, "trails", None)
            if trails:
                self._draw_trails(cv, trails, truth_width=1.2, belief_width=1.8)
            if belief is not None and frame.layers.belief in ("particles", "both"):
                self._draw_particles_minimap(cv, belief, t, lost, frame.weigh, shift)
            self._draw_forklifts(cv, result)
            smoothed = getattr(frame, "smoothed", None)
            if smoothed:
                self._draw_smoothed(
                    cv, smoothed, _finite(getattr(frame, "smoothed_reveal", 1.0), 1.0)
                )
            if belief is not None:
                self._draw_estimate(
                    cv, belief, t, lost, ring_r=1.1 * s, width=2.0, big=False, shift=shift
                )
            if trails:
                self._draw_gap(cv, trails, width=1.6, label=0)
        # The true robot (white): where the robot really is.
        if frame.show_truth and frame.robot_xy is not None:
            x, y = _finite(frame.robot_xy[0], math.nan), _finite(frame.robot_xy[1], math.nan)
            if math.isfinite(x) and math.isfinite(y):
                cx, cy = cv.px(x, y)
                cv.ring(cx, cy, 0.5 * s + 1.8, 1.4, WHITE, 1.0, shadow=0.8)
                cv.disk(cx, cy, 1.7, WHITE, 1.0, shadow=0.8)
        self._draw_minimap_title(cv, frame, belief, lost)
        return img

    def _mini_base(self, frame: HoloFrame) -> np.ndarray:
        """The static map + cell layers + obstacles of this result and layer choice (cached)."""
        key = (frame.layers.belief, frame.layers.map_layer)
        cached = self._mini_cache
        if cached is not None and cached[0] is frame.result and cached[1] == key:
            return cached[2]
        img = self._mini_static.copy()
        s = self.MINI_SCALE
        obstacles = self._obstacle_cells(frame.result)
        cells = None
        if frame.result is not None:
            cells = self._cell_layers(frame, obstacles, value_alpha=0.6, heat_alpha=0.9)
        if cells is not None:
            # The background is opaque, so "over" is a plain blend, done on a (cell, pixel) view.
            rgba = cells[::-1]
            view = img.reshape(MAP_H, s, MAP_W, s, 4)
            bg = view[..., :3].astype(np.float32)
            a = rgba[..., 3][:, None, :, None, None]
            view[..., :3] = (bg + (rgba[..., :3][:, None, :, None, :] - bg) * a + 0.5).astype(
                np.uint8
            )
        cv = _Canvas(img, s)
        for i, j in zip(*np.nonzero(obstacles), strict=False):
            r0, c0 = (MAP_H - 1 - i) * s, j * s
            cv.rect(c0 - 1, r0 - 1, c0 + s + 1, r0 + s + 1, RED, 0.95)
            cv.line(c0 + 0.5, r0 + 0.5, c0 + s - 0.5, r0 + s - 0.5, 1.2, (60, 0, 0), 0.9)
            cv.line(c0 + s - 0.5, r0 + 0.5, c0 + 0.5, r0 + s - 0.5, 1.2, (60, 0, 0), 0.9)
        idx, rgba = self._mini_chrome  # legend and frame on top of the map layers
        dst = img.reshape(-1, 4)[idx]
        _over(dst, rgba[:, 3].astype(np.float32) / 255.0, rgba[:, :3].astype(np.float32))
        img.reshape(-1, 4)[idx] = dst
        self._mini_cache = (frame.result, key, img)
        return img

    # --- minimap layers ---------------------------------------------------------
    def _draw_orders_minimap(self, cv: _Canvas, result, t: float) -> None:
        """The current order's pickup slot (orange parcel) and destination dock (green)."""
        order = getattr(result, "order", None) or {}
        parcel = getattr(result, "parcel", None) or {}
        slot = SLOTS_BY_NAME.get(order.get("pickup", ""))
        state = parcel.get("state", "rack")
        if slot is not None and state == "rack":
            cx, cy = cv.px(*slot.parcel_xy)
            cv.rect(cx - 2, cy - 2, cx + 3, cy + 3, ORANGE, 1.0)
        dock_name = order.get("dock", "")
        for dock in DOCKS:
            if dock.name == dock_name:
                cx, cy = cv.px(99.5, dock.station[1])
                pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 0.9 * t)
                cv.rect(cv.w - 8, cy - 11, cv.w - 1, cy + 11, (200, 255, 220), 0.35 + 0.5 * pulse)

    def _draw_particles_minimap(
        self,
        cv: _Canvas,
        belief,
        t: float,
        lost: bool,
        weigh: float = 1.0,
        shift: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        p, size, _, color, gain = self._particle_style(belief, t, lost, weigh)
        if len(p) == 0:
            return
        s = cv.s
        ok = np.all(np.isfinite(p), axis=1)
        cols = np.floor((p[ok, 0] + shift[0] + 0.5) * s).astype(np.int64)
        rows = np.floor((49.5 - p[ok, 1] - shift[1]) * s).astype(np.int64)
        self._splat_sprites(
            cv,
            rows,
            cols,
            size[ok],
            self._mini_dots,
            (0.3, 0.9, 1.0, 1.0),
            color,
            gain,
            hot=(235, 255, 255),
            hot_start=2.0,
        )

    def _draw_rfid_minimap(self, cv: _Canvas, frame: HoloFrame) -> None:
        ping = self._ping(frame)
        if ping is None:
            return
        beacon_id, rng, age = ping
        fade = max(0.0, 1.0 - age / 2.4)
        cx, cy = cv.px(*BEACONS[beacon_id])
        cv.ring(cx, cy, max(rng * cv.s, 1.0), 1.6, ORANGE, 0.9 * fade, glow=0.25)
        cv.disk(cx, cy, 3.0, (200, 255, 255), fade, glow=0.6)

    def _draw_forklifts(self, cv: _Canvas, result) -> None:
        for actor in getattr(result, "actors", None) or []:
            try:
                x, y, heading = float(actor["x"]), float(actor["y"]), float(actor["heading"])
                hl = float(actor.get("half_length", config.FORKLIFT_HALF_LENGTH)) * cv.s
                hw = float(actor.get("half_width", config.FORKLIFT_HALF_WIDTH)) * cv.s
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x, y, heading, hl, hw)):
                continue
            cx, cy = cv.px(x, y)
            cv.box(cx, cy, heading, hl + 1, hw + 1, (40, 16, 0), 0.9)
            cv.box(cx, cy, heading, hl, hw, ORANGE, 1.0)
            # Forks: a darker tip toward the heading.
            fx, fy = cx + math.cos(heading) * hl * 0.7, cy - math.sin(heading) * hl * 0.7
            cv.disk(fx, fy, 1.2, (90, 40, 0), 1.0)

    @staticmethod
    def _draw_smoothed(cv: _Canvas, smoothed: dict, reveal: float) -> None:
        """TODO 17: the true path (white) and the student's Kalman-smoothed track (magenta), segment
        by
        segment (a kidnapping starts a new one), the first ``reveal`` fraction of the steps.
        """
        track = np.asarray(smoothed.get("track") or [], dtype=float).reshape(-1, 2)
        truth = np.asarray(smoothed.get("truth") or [], dtype=float).reshape(-1, 2)
        n = len(track)
        if n < 2 or truth.shape != track.shape:
            return
        shown = int(round(min(1.0, max(0.0, reveal)) * n))
        start = 0
        for length in list(smoothed.get("segments") or [n]):
            stop = min(start + int(length), shown)
            for points, rgb, width, alpha in (
                (truth, WHITE, 2.0, 0.85),
                (track, MAGENTA, 3.4, 1.0),
            ):
                # The dark underlay of the whole polyline first, then the line: per-segment shadows
                # would darken every joint and make the line look dotted.
                for color, w, a in ((SHADOW, width + 3.0, 0.6 * alpha), (rgb, width, alpha)):
                    for k in range(start, stop - 1):
                        x0, y0 = cv.px(points[k, 0], points[k, 1])
                        x1, y1 = cv.px(points[k + 1, 0], points[k + 1, 1])
                        cv.line(x0, y0, x1, y1, w, color, a)
            start += int(length)

    @staticmethod
    def _draw_trails(cv: _Canvas, trails: dict, truth_width: float, belief_width: float) -> None:
        """A test: the true path (white) and the believed path (magenta) on top; a PENALTY TEST
        (no believed path): its route, bolder, over the other test's route (orange, "other")."""
        route_only = not trails.get("belief")
        if route_only:
            truth_width = belief_width
        layers = (
            ("other", ORANGE, belief_width, 0.85),
            ("truth", WHITE, truth_width, 1.0 if route_only else 0.85),
            ("belief", MAGENTA, belief_width, 1.0),
        )
        for key, rgb, width, alpha in layers:
            points = np.asarray(trails.get(key) or [], dtype=float).reshape(-1, 2)
            pts = [cv.px(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
            # The dark underlay of the whole polyline first, then the line: per-segment shadows
            # would darken every joint and make the line look dashed.
            for color, w, a in ((SHADOW, width + 3.0, 0.5 * alpha), (rgb, width, alpha)):
                for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=False):
                    cv.line(x0, y0, x1, y1, w, color, a)

    @staticmethod
    def _draw_gap(cv: _Canvas, trails: dict, width: float, label: int) -> None:
        """The end of a sensor test: a yellow line from the robot to its final estimate, and its
        length.
        """
        truth = np.asarray(trails.get("truth") or [], dtype=float).reshape(-1, 2)
        belief = np.asarray(trails.get("belief") or [], dtype=float).reshape(-1, 2)
        if not (trails.get("gap") and len(truth) and len(belief)):
            return
        (tx, ty), (bx, by) = cv.px(*truth[-1]), cv.px(*belief[-1])
        cv.line(tx, ty, bx, by, width, YELLOW, 1.0, shadow=0.7)
        gap = float(np.hypot(*(belief[-1] - truth[-1])))
        if label:  # beside the line, on the side away from it
            right = max(tx, bx)
            cv.text(
                f"{gap:.1f} M",
                right + 6 * label + 4,
                0.5 * (ty + by) - 4 * label,
                YELLOW,
                scale=label,
            )

    def _draw_minimap_title(self, cv: _Canvas, frame: HoloFrame, belief, lost: bool) -> None:
        """Title and live status in the north aisle: "ROBOT VISION  MCL 2000"."""
        smoothed = getattr(frame, "smoothed", None)
        if smoothed and smoothed.get("smooth_error") is not None:  # MISSION COMPLETE with TODO 17
            x1 = cv.text("KALMAN SMOOTHER", 9, 7, MAGENTA, scale=2)[2]
            cv.text(
                f"PATH ERROR {float(smoothed['smooth_error']):.2f} M", x1 + 12, 7, WHITE, scale=2
            )
            return
        x1 = cv.text("ROBOT VISION", 9, 7, CYAN, scale=2)[2]
        layer = {"value": "VALUE", "likelihood": "LIKELIHOOD", "zones": "ZONES"}.get(
            frame.layers.map_layer
        )
        if belief is None:
            status, color = "STANDBY", (150, 175, 200)
        elif lost:
            status, color = "LOST - RELOCALIZING", RED
        else:
            mode = {"grid": "GRID", "both": "MCL+GRID"}.get(frame.layers.belief, "MCL")
            status = f"{mode} {belief.n_particles}"
            color = GREEN if belief.locked else WHITE
            if belief.locked and not layer:
                status += "  LOCKED"  # (with a map layer tag the green color alone says LOCKED)
        x2 = cv.text(status, x1 + 12, 7, color, scale=2)[2]
        if layer and x2 + 12 + font5x7.text_size(layer, 2)[0] <= cv.w - 28:
            cv.text(layer, cv.w - 28, 7, (255, 190, 90), scale=2, align="right")

    def _build_minimap_chrome(self):
        """Legend (between the two forklift depots of the south aisle) and the cyan frame (sparse
        pixels).
        """
        W, H = self.MINI_W, self.MINI_H
        img = np.zeros((H, W, 4), np.uint8)
        cv = _Canvas(img, self.MINI_SCALE)
        items = (
            ("particles", "BELIEF"),
            ("estimate", "ESTIMATE"),
            ("path", "PATH"),
            ("truth", "TRUE"),
        )
        widths = [16 + font5x7.text_size(label, 2)[0] for _, label in items]
        gap = 16
        total = sum(widths) + gap * (len(items) - 1)
        x, y = (W - total) // 2, H - 21
        cv.rect(x - 8, y - 4, x + total + 8, y + 17, (4, 9, 16), 0.78)
        for (kind, label), width in zip(items, widths, strict=False):
            cx, cy = x + 5, y + 7
            if kind == "particles":
                for dx, dy in ((-3, -2), (2, -3), (0, 2), (4, 2), (-3, 3)):
                    cv.rect(cx + dx, cy + dy, cx + dx + 2, cy + dy + 2, CYAN, 1.0)
            elif kind == "estimate":
                cv.ring(cx + 0.5, cy + 0.5, 4.5, 1.6, MAGENTA, 1.0)
            elif kind == "path":
                cv.rect(cx - 5, cy, cx - 1, cy + 2, YELLOW, 1.0)
                cv.rect(cx + 2, cy, cx + 6, cy + 2, YELLOW, 1.0)
            else:
                cv.ring(cx + 0.5, cy + 0.5, 4.0, 1.2, WHITE, 1.0)
                cv.disk(cx + 0.5, cy + 0.5, 1.5, WHITE, 1.0)
            cv.text(label, x + 16, y + 1, (200, 215, 230), scale=2)
            x += width + gap
        # Frame: thin cyan border with brighter corner brackets.
        cv.frame_rect(0, 0, W, H, 2, CYAN, 0.75)
        arm = 18
        for x0, y0, sx, sy in ((0, 0, 1, 1), (W, 0, -1, 1), (0, H, 1, -1), (W, H, -1, -1)):
            xa, xb = sorted((x0, x0 + sx * arm))
            ya, yb = sorted((y0, y0 + sy * 4))
            cv.rect(xa, ya, xb, yb, CYAN, 1.0)
            xa, xb = sorted((x0, x0 + sx * 4))
            ya, yb = sorted((y0, y0 + sy * arm))
            cv.rect(xa, ya, xb, yb, CYAN, 1.0)
        idx = np.flatnonzero(img[..., 3].ravel())
        return idx, img.reshape(-1, 4)[idx].copy()
