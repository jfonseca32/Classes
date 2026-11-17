"""
dynamic_restaurant.py: geometry for the dynamic simulation (STAFF PROVIDED).

In the dynamic simulation customers walk in, move stools and walk out while the
robot works. This module is shared by the restaurant_crowd supervisor (which
picks where a customer moves a stool) and by rrt_controller (which needs the
seat cards' new positions). It has no Webots dependencies except read_stool_poses.

All coordinates are Webots world coordinates in meters (x east, y north).
"""

import math
import random

import numpy as np
import restaurant_map as M

# Clearances used when a customer picks a new spot for a stool [m]
STOOL_GAP = 0.005  # stool to any other obstacle (home stools sit 9 mm from the counter)
LANE_GAP = M.CUSTOMER_RADIUS + 0.01  # stool to a customer walkway
VIEW_GAP = M.ROBOT_RADIUS + M.PLANNER_MARGIN + 0.01  # every viewing pose to any obstacle
ROBOT_GAP = 0.15  # stool to the robot, so a stool never lands on it


def rot(yaw, x, y):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


def stool_polygon(pose):
    """Planner footprint of a stool at pose (x, y, yaw)."""
    x, y, yaw = pose
    return [(x + px, y + py) for px, py in (rot(yaw, *p) for p in M.STOOL_FOOTPRINT)]


def card_from_stool(pose):
    """Card center and the unit normal out of the card face, for a stool at pose."""
    x, y, yaw = pose
    dx, dy = rot(yaw, M.CARD_LOCAL_X, 0.0)
    nx, ny = rot(yaw, -1.0, 0.0)
    return np.array([x + dx, y + dy]), np.array([nx, ny])


def viewing_pose_from_stool(pose):
    """(x, y, yaw) where the robot has to stand to read the card of a stool at pose."""
    c, n = card_from_stool(pose)
    d = M.VIEWING_DISTANCE + M.CARD_THICKNESS / 2.0
    yaw = math.atan2(-n[1], -n[0])
    return float(c[0] + n[0] * d), float(c[1] + n[1] * d), yaw


def fixed_obstacles():
    """Polygons of everything that never moves (counter island and props)."""
    return [poly for name, poly in M.OBSTACLES if not name.startswith("stool")]


def all_obstacles(stool_poses):
    """Fixed obstacles plus every stool at its current pose."""
    return fixed_obstacles() + [stool_polygon(p) for _, p in sorted(stool_poses.items())]


# ---------------------------------------------------------------------------
# Distances between points, segments and polygons
# ---------------------------------------------------------------------------
def _seg_dist(p, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / max(ax * ax + ay * ay, 1e-12)
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - a[0] - t * ax, p[1] - a[1] - t * ay)


def _inside(p, poly):
    inside = False
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        if (y0 > p[1]) != (y1 > p[1]) and p[0] < (x1 - x0) * (p[1] - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def point_polygon_dist(p, poly):
    if _inside(p, poly):
        return 0.0
    return min(_seg_dist(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def _segments_cross(a, b, c, d):
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def segment_polygon_dist(a, b, poly):
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    if _inside(a, poly) or any(_segments_cross(a, b, c, d) for c, d in edges):
        return 0.0
    return min(
        [_seg_dist(v, a, b) for v in poly]
        + [min(_seg_dist(a, c, d), _seg_dist(b, c, d)) for c, d in edges]
    )


def polygon_polygon_dist(p, q):
    if _inside(p[0], q) or _inside(q[0], p):
        return 0.0
    return min(segment_polygon_dist(p[i], p[(i + 1) % len(p)], q) for i in range(len(p)))


def point_clearance(p, polygons):
    """Distance from p to the nearest polygon or wall of the room."""
    best = min(p[0] - M.ROOM_X[0], M.ROOM_X[1] - p[0], p[1] - M.ROOM_Y[0], M.ROOM_Y[1] - p[1])
    for poly in polygons:
        best = min(best, point_polygon_dist(p, poly))
    return best


# ---------------------------------------------------------------------------
# Where may a customer leave a stool?
# ---------------------------------------------------------------------------
def stool_pose_is_valid(seat, pose, stool_poses, robot_xy):
    """robot_xy: a point, or a list of points, the stool must stay ROBOT_GAP away from."""
    return _stool_pose_is_valid(
        seat,
        pose,
        stool_poses,
        []
        if robot_xy is None
        else ([robot_xy] if isinstance(robot_xy[0], (int, float)) else list(robot_xy)),
    )


def _stool_pose_is_valid(seat, pose, stool_poses, keep_clear):
    """
    True if the stool of `seat` can stand at `pose`: it does not touch anything,
    keeps the customer walkways open, keeps clear of the robot, and every seat
    card can still be read from a collision-free viewing pose.
    """
    poly = stool_polygon(pose)
    others = dict(stool_poses)
    others[seat] = pose
    if any(point_clearance(v, []) < STOOL_GAP for v in poly):  # inside the room
        return False
    for obstacle in fixed_obstacles() + [stool_polygon(p) for s, p in others.items() if s != seat]:
        if polygon_polygon_dist(poly, obstacle) < STOOL_GAP:
            return False
    for a, b in M.CROWD_LANES:
        if segment_polygon_dist(a, b, poly) < LANE_GAP:
            return False
    if any(point_polygon_dist(p, poly) < ROBOT_GAP for p in keep_clear):
        return False
    everything = all_obstacles(others)
    for p in others.values():
        vx, vy, _ = viewing_pose_from_stool(p)
        if point_clearance((vx, vy), everything) < VIEW_GAP:
            return False
    return True


def sample_stool_pose(seat, stool_poses, robot_xy, rng=random, tries=40):
    """A random valid new pose for the stool of `seat` near its home pose, or None."""
    hx, hy, hyaw = M.STOOL_HOME_POSES[seat]
    outward = 1.0 if M.STOOL_ROWS[seat] == "N" else -1.0
    for _ in range(tries):
        push = rng.uniform(-0.04, M.STOOL_MAX_EXTRA_PUSH)
        pose = (
            hx + rng.uniform(-0.04, 0.04),
            hy + outward * push,
            hyaw
            + math.radians(rng.uniform(-M.STOOL_MAX_EXTRA_TURN_DEG, M.STOOL_MAX_EXTRA_TURN_DEG)),
        )
        if stool_pose_is_valid(seat, pose, stool_poses, robot_xy):
            return pose
    return None


def read_stool_poses(supervisor):
    """Current (x, y, yaw) of every stool, read from the Webots world (Supervisor only)."""
    poses = {}
    for seat in M.STOOL_HOME_POSES:
        node = supervisor.getFromDef(f"SEAT_{seat}")
        x, y, _ = node.getPosition()
        r = node.getOrientation()
        poses[seat] = (x, y, math.atan2(r[3], r[0]))
    return poses
