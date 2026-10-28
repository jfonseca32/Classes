"""Small math helpers for the Webots presentation layer (no Webots import).

Conventions: world frame = book frame (x east, y north, z up, meters).
A Webots Viewpoint (or Camera) looks along its local +x axis with local +z up,
so a view direction is described by a yaw (about world z, 0 = looking east)
and a pitch (> 0 looks down). Rotations are axis-angle lists
``[ax, ay, az, angle]``, the format of ``Field.setSFRotation``.

Everything here is plain ``math`` on floats and tuples: these functions run
every frame, and NumPy's per-call overhead is larger than the work.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Color = tuple[float, float, float]

TAU = 2.0 * math.pi


###############################################################################
#                             Scalars and easing                              #
###############################################################################


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_vec(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, ...]:
    return tuple(x + (y - x) * t for x, y in zip(a, b, strict=False))


def smoothstep(t: float) -> float:
    """C1 ease in/out on [0, 1] (clamped)."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t: float) -> float:
    """Cubic ease in/out on [0, 1] (clamped)."""
    t = clamp(t, 0.0, 1.0)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_out(t: float) -> float:
    """Cubic ease out on [0, 1] (clamped): fast start, soft landing."""
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def ease_in(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * t


def smooth_factor(dt: float, tau: float) -> float:
    """Blend factor of an exponential low-pass filter with time constant ``tau`` (s)."""
    if tau <= 1e-9:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def pulse(t: float, hz: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Sinusoidal pulse between ``lo`` and ``hi`` at ``hz``."""
    return lo + (hi - lo) * (0.5 + 0.5 * math.sin(TAU * hz * t))


def blink(t: float, hz: float, duty: float = 0.5) -> bool:
    """Square wave: True during the first ``duty`` fraction of each period."""
    return (t * hz) % 1.0 < duty


###############################################################################
#                                  Angles                                     #
###############################################################################


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, TAU)
    if a <= 0.0:
        a += TAU
    return a - math.pi


def angle_lerp(a: float, b: float, t: float) -> float:
    """Interpolate from angle ``a`` toward ``b`` along the shorter arc."""
    return a + wrap_angle(b - a) * t


###############################################################################
#                                 Vectors                                     #
###############################################################################


def add3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def dist2(a: Sequence[float], b: Sequence[float]) -> float:
    """Planar distance between the (x, y) parts of two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit2(dx: float, dy: float) -> Vec2 | None:
    """Unit vector of (dx, dy), or None for a (near) zero vector."""
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    return (dx / n, dy / n)


def catmull_rom(points: Sequence[Sequence[float]], s: float) -> tuple[float, ...]:
    """Uniform Catmull-Rom spline through ``points`` at ``s`` in [0, 1] (ends clamped)."""
    n = len(points)
    if n == 1:
        return tuple(points[0])
    s = clamp(s, 0.0, 1.0) * (n - 1)
    i = min(int(s), n - 2)
    u = s - i
    p0 = points[max(i - 1, 0)]
    p1, p2 = points[i], points[i + 1]
    p3 = points[min(i + 2, n - 1)]
    u2, u3 = u * u, u * u * u
    return tuple(
        0.5
        * (
            2.0 * b
            + (-a + c) * u
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * u2
            + (-a + 3.0 * b - 3.0 * c + d) * u3
        )
        for a, b, c, d in zip(p0, p1, p2, p3, strict=False)
    )


###############################################################################
#                         Viewpoint orientation                               #
###############################################################################


def yaw_pitch(yaw: float, pitch: float) -> list[float]:
    """Axis-angle of Rz(yaw) * Ry(pitch): a camera looking along +x, pitch > 0 looks down.

    Composed as a quaternion product (qz * qy) and converted to axis-angle,
    which avoids the numerically delicate matrix -> axis-angle step.
    """
    cz, sz = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    cy, sy = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    w = cz * cy
    x, y, z = -sz * sy, cz * sy, cy * sz
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return [0.0, 0.0, 1.0, 0.0]
    return [x / n, y / n, z / n, 2.0 * math.atan2(n, w)]


def look_at(
    eye: Sequence[float], target: Sequence[float], yaw_hint: float = 0.0
) -> tuple[list[float], float]:
    """Orientation looking from ``eye`` at ``target`` (z up, no roll).

    Returns ``(axis_angle, yaw)``. When the view is (almost) vertical the yaw is
    undefined and ``yaw_hint`` (for example the previous yaw) is used instead,
    so that a camera looking straight down keeps its heading.
    """
    dx, dy, dz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
    horizontal = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx) if horizontal > 1e-4 else yaw_hint
    pitch = math.atan2(-dz, horizontal) if (horizontal > 1e-9 or abs(dz) > 1e-9) else 0.0
    return yaw_pitch(yaw, pitch), yaw


def forward_from_axis_angle(rotation: Sequence[float]) -> Vec3:
    """The world direction of a viewpoint's local +x axis (its view direction)."""
    kx, ky, kz, angle = (float(v) for v in rotation)
    n = math.sqrt(kx * kx + ky * ky + kz * kz)
    if n < 1e-12:
        return (1.0, 0.0, 0.0)
    kx, ky, kz = kx / n, ky / n, kz / n
    c, s = math.cos(angle), math.sin(angle)
    # Rodrigues' formula applied to v = (1, 0, 0).
    return (c + kx * kx * (1.0 - c), kz * s + kx * ky * (1.0 - c), -ky * s + kx * kz * (1.0 - c))


###############################################################################
#                                  Colors                                     #
###############################################################################


def hex_rgb(value: int) -> Color:
    """0xRRGGBB -> (r, g, b) in [0, 1]."""
    return (((value >> 16) & 255) / 255.0, ((value >> 8) & 255) / 255.0, (value & 255) / 255.0)


def mix(a: Sequence[float], b: Sequence[float], t: float) -> Color:
    t = clamp(t, 0.0, 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def scale(c: Sequence[float], k: float) -> Color:
    return (c[0] * k, c[1] * k, c[2] * k)


# The neon palette of the style guide (DESIGN section 7, VIZ_SPEC section 3.2).
CYAN = hex_rgb(0x19E6FF)
MAGENTA = hex_rgb(0xFF3DDC)
YELLOW = hex_rgb(0xFFD21A)
GREEN = hex_rgb(0x39FF88)
ORANGE = hex_rgb(0xFF8A00)
RED = hex_rgb(0xFF3B3B)
WHITE = hex_rgb(0xE8F4FF)
BLACK = (0.0, 0.0, 0.0)


class SmoothDamp:
    """Critically damped spring toward a moving goal (the camera smoother).

    Unlike an exponential filter it starts and stops gently, so camera moves
    ease in and out. ``smooth_time`` is roughly the time to reach the goal.
    Works on tuples of any length.
    """

    def __init__(self, value: Sequence[float]):
        self.value = tuple(float(v) for v in value)
        self.velocity = tuple(0.0 for _ in self.value)

    def reset(self, value: Sequence[float]) -> None:
        self.value = tuple(float(v) for v in value)
        self.velocity = tuple(0.0 for _ in self.value)

    def update(self, goal: Sequence[float], smooth_time: float, dt: float) -> tuple[float, ...]:
        smooth_time = max(1e-4, smooth_time)
        omega = 2.0 / smooth_time
        x = omega * dt
        decay = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        value, velocity = [], []
        for cur, vel, tgt in zip(self.value, self.velocity, goal, strict=False):
            change = cur - tgt
            temp = (vel + omega * change) * dt
            velocity.append((vel - omega * temp) * decay)
            value.append(tgt + (change + temp) * decay)
        self.value, self.velocity = tuple(value), tuple(velocity)
        return self.value
