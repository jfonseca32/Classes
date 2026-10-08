"""Pure-Python tables and helpers for the CS 3630 Project 3 Webots world.

This module deliberately does NOT import the Webots ``controller`` package so
it can be unit-tested with plain pytest and imported by the world generator.
Everything geometric about the Freiburg-lab world lives here: room centers,
doorway positions, adjacency, the light-level thresholds, BFS routing between
rooms, the waypoint list the robot follows, the seeded transition sampler,
the controller's command-line parser, and the console line formats that
students copy into their report.

Coordinates are metres in the world frame (z-up). The layout matches the
Fall 2025 map of the COLD-Freiburg lab used in the notebook:

    corridor 2 is the hub; 1, 3, 4 are north of it; 9, 5, 6 south of it;
    7 is reachable only through 6 and 8 (bathroom) only through 9 (stairs).
"""

import argparse
from collections import deque

# ---------------------------------------------------------------------------
# Constants shared with the student's main.py (same order, same strings).
# ---------------------------------------------------------------------------
ROOMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
ACTIONS = [
    "2->1",
    "1->2",
    "2->3",
    "3->2",
    "2->4",
    "4->2",
    "2->5",
    "5->2",
    "2->6",
    "6->7",
    "7->6",
    "6->2",
    "2->9",
    "9->8",
    "8->9",
    "9->2",
]
LIGHT_LEVELS = ["dark", "medium", "light"]
N = 8

ROOM_NAMES = {
    "1": "printer area",
    "2": "corridor",
    "3": "kitchen",
    "4": "large office",
    "5": "two-person office",
    "6": "one-person office",
    "7": "one-person office",
    "8": "bathroom",
    "9": "stairs area",
}

# Room center: where the robot parks when it "is in" that room.
ROOM_CENTERS = {
    "1": (-5.0, 3.2),
    "2": (0.0, 0.0),
    "3": (-1.0, 3.2),
    "4": (4.0, 3.2),
    "5": (-1.5, -3.2),
    "6": (2.5, -3.2),
    "7": (5.5, -3.2),
    "8": (-8.5, -3.2),
    "9": (-5.5, -3.2),
}

# (xmin, xmax, ymin, ymax) of each room's floor.
ROOM_BOUNDS = {
    "1": (-7.0, -3.0, 1.2, 5.2),
    "2": (-7.0, 7.0, -1.0, 1.0),
    "3": (-3.0, 1.0, 1.2, 5.2),
    "4": (1.0, 7.0, 1.2, 5.2),
    "5": (-4.0, 1.0, -5.2, -1.2),
    "6": (1.0, 4.0, -5.2, -1.2),
    "7": (4.0, 7.0, -5.2, -1.2),
    "8": (-10.0, -7.0, -5.2, -1.2),
    "9": (-7.0, -4.0, -5.2, -1.2),
}

# Doorway (1.0 m wall gap) between two rooms. The key order is the DEF name
# used in the world file: DOOR_a_b. Use doorway(a, b) to look up either order.
DOORWAYS = {
    ("1", "2"): (-5.0, 1.1),
    ("3", "2"): (-1.0, 1.1),
    ("4", "2"): (4.0, 1.1),
    ("9", "2"): (-5.5, -1.1),
    ("5", "2"): (-1.5, -1.1),
    ("6", "2"): (3.3, -1.1),
    ("6", "7"): (4.0, -1.8),
    ("9", "8"): (-7.0, -3.2),
}

# The robot approaches every doorway perpendicular to the wall: it first
# drives to a point this far from the door on its own side, crosses the gap,
# and continues to the mirror point on the other side before heading to the
# next room center. Without these points the diagonal corridor legs would clip
# the wall corners visually.
DOOR_APPROACH = 0.5

# Normalized LightSensor reading r is dark if r < 0.33, light if r >= 0.66,
# medium otherwise.
LIGHT_THRESHOLDS = (0.33, 0.66)

# Robot geometry (iRobot Create) used for visual wheel spin.
WHEEL_RADIUS = 0.031
ROBOT_Z = 0.044

# Kinematic motion parameters.
DEFAULT_SPEED = 1.2  # m/s while translating
TURN_RATE_DEG = 180.0  # deg/s while rotating in place
STAY_BUMP = 0.6  # m the robot nudges toward a door on a "stay"
DWELL_SECONDS = 1.0  # time spent sampling the light sensor per room

DEFAULT_SEED = 3630
DEFAULT_START = {"POLICY": "8", "CUSTOM": "8", "TAPE": "3", "MPE": "3", "LIGHT": "8"}
MODES = ("POLICY", "CUSTOM", "TAPE", "LIGHT", "MPE")

# Used by --auto-demo (no student code): kitchen and small-office loop.
DEMO_ROUTE = ["8", "9", "2", "3", "2", "6", "7", "6", "2"]
# The notebook's custom control tape cycle (create_custom_action_sequence).
CUSTOM_TAPE_CYCLE = ["3->2", "2->6", "6->7", "7->6", "6->2", "2->3"]


def _build_adjacency():
    adjacency = {room: set() for room in ROOMS}
    for a, b in DOORWAYS:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return {room: sorted(neighbors) for room, neighbors in adjacency.items()}


ADJACENCY = _build_adjacency()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def action_parts(action):
    """Split "2->1" into ("2", "1")."""
    source, destination = action.split("->")
    return source, destination


def action_available(action, room):
    """True if the action's source room is the room the robot is in."""
    return action_parts(action)[0] == room


def door_key(a, b):
    """Return the DOORWAYS key (and DEF-name order) joining rooms a and b."""
    if (a, b) in DOORWAYS:
        return (a, b)
    if (b, a) in DOORWAYS:
        return (b, a)
    raise KeyError(f"rooms {a} and {b} are not joined by a doorway")


def doorway(a, b):
    """Doorway position (x, y) between adjacent rooms a and b."""
    return DOORWAYS[door_key(a, b)]


def door_def_name(a, b):
    """DEF name of the threshold tile between rooms a and b."""
    return "DOOR_{}_{}".format(*door_key(a, b))


def room_of_action_target(action):
    return action_parts(action)[1]


def discretize(raw):
    """Map a normalized light reading to one of LIGHT_LEVELS."""
    dark_max, light_min = LIGHT_THRESHOLDS
    if raw < dark_max:
        return "dark"
    if raw < light_min:
        return "medium"
    return "light"


def route(a, b):
    """Shortest room-to-room path [a, ..., b] through doorways (BFS)."""
    if a not in ROOMS or b not in ROOMS:
        raise KeyError(f"unknown room in route({a!r}, {b!r})")
    if a == b:
        return [a]
    previous = {a: None}
    queue = deque([a])
    while queue:
        room = queue.popleft()
        for neighbor in ADJACENCY[room]:
            if neighbor in previous:
                continue
            previous[neighbor] = room
            if neighbor == b:
                path = [b]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return path[::-1]
            queue.append(neighbor)
    raise ValueError(f"no route from room {a} to room {b}")


# Doorways whose wall runs along y (the robot crosses them along x).
VERTICAL_DOORS = {("6", "7"), ("9", "8")}


def door_is_vertical(a, b):
    return door_key(a, b) in VERTICAL_DOORS


def _approach_point(key, toward):
    """Point DOOR_APPROACH metres from doorway `key` along the door's normal,
    on the side of `toward` (a room center)."""
    door = DOORWAYS[key]
    if key in VERTICAL_DOORS:
        sign = 1.0 if toward[0] > door[0] else -1.0
        return (round(door[0] + sign * DOOR_APPROACH, 6), door[1])
    sign = 1.0 if toward[1] > door[1] else -1.0
    return (door[0], round(door[1] + sign * DOOR_APPROACH, 6))


def waypoints(a, b):
    """World-frame points the robot drives through to get from room a to b.

    center(a) -> [approach, door, approach] per doorway on the BFS route ->
    center(b). The first element is always center(a) and the last center(b);
    waypoints(a, a) == [center(a)].
    """
    path = route(a, b)
    points = [ROOM_CENTERS[path[0]]]
    for here, there in zip(path, path[1:], strict=False):
        key = door_key(here, there)
        points.append(_approach_point(key, ROOM_CENTERS[here]))
        points.append(DOORWAYS[key])
        points.append(_approach_point(key, ROOM_CENTERS[there]))
        points.append(ROOM_CENTERS[there])
    return points


def stay_bump_points(room, action):
    """Points for a "stayed" outcome: nudge toward the intended door and back."""
    center = ROOM_CENTERS[room]
    source, destination = action_parts(action)
    door = doorway(source, destination)
    dx = door[0] - center[0]
    dy = door[1] - center[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return [center]
    scale = min(STAY_BUMP, length) / length
    return [(center[0] + dx * scale, center[1] + dy * scale), center]


def sample_next_room(rng, T, x, a):
    """Sample the next room index from T[x, a] with a numpy Generator.

    Returns (next_index, probability). The row is renormalized so a
    hand-written prob_spec that does not sum to exactly one still works.
    """
    import numpy as np

    row = np.asarray(T[x, a], dtype=float).reshape(-1)
    if row.shape[0] != len(ROOMS) or not np.all(np.isfinite(row)) or np.any(row < 0):
        raise ValueError(f"T[{x}, {a}] is not a valid probability row: {row}")
    total = float(row.sum())
    if total <= 0.0:
        raise ValueError(f"T[{x}, {a}] sums to zero; cannot sample a next room")
    row = row / total
    next_index = int(rng.choice(len(ROOMS), p=row))
    return next_index, float(row[next_index])


def path_length(points):
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _room_argument(value):
    value = str(value).strip()
    if value not in ROOMS:
        raise argparse.ArgumentTypeError(
            f"start room must be one of {', '.join(ROOMS)}; got {value!r}"
        )
    return value


def parse_key_script(text):
    """Parse "R@6,5@8,P@10" into [(6.0, "R"), (8.0, "5"), (10.0, "P")], sorted by time."""
    if not text:
        return []
    presses = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, when = item.partition("@")
        key = key.strip()
        if len(key) != 1 or not when.strip():
            raise ValueError(f"bad key script entry {item!r}; expected KEY@SECONDS such as R@6")
        presses.append((float(when), key.upper()))
    return sorted(presses)


def parse_args(argv):
    """Parse controllerArgs / CS3630_P3_CONTROLLER_ARGS. Unknown flags are ignored.

    Returns an argparse.Namespace with: mode (upper-case or None), start (room
    string or None), seed, steps, speed, student_code, auto_demo,
    integration_test, calibrate, export_image.
    """
    parser = argparse.ArgumentParser(prog="project3_vacuum", add_help=False)
    parser.add_argument("--mode", default=None, type=lambda s: s.strip().upper())
    parser.add_argument("--start", default=None, type=_room_argument)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--steps", default=N, type=int)
    parser.add_argument("--speed", default=DEFAULT_SPEED, type=float)
    parser.add_argument("--student-code", dest="student_code", default=None)
    parser.add_argument("--auto-demo", dest="auto_demo", action="store_true")
    parser.add_argument("--integration-test", dest="integration_test", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--export-image", dest="export_image", default=None)
    # Staff/test option: press keys at scripted simulation times, e.g. "R@6,5@8,P@10" presses R at
    # 6 s, then 5, then P. Lets the restart / start-room / mode-switch paths run headlessly.
    parser.add_argument("--keys", default=None)
    args, _unknown = parser.parse_known_args(list(argv))
    if args.mode is not None and args.mode not in MODES:
        raise ValueError(
            f"--mode must be one of {', '.join(m.lower() for m in MODES)}; got "
            f"{args.mode.lower()!r}"
        )
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    return args


# ---------------------------------------------------------------------------
# Console line formats (students paste these into the report; keep stable)
# ---------------------------------------------------------------------------
def room_label(room):
    return f"room {room} ({ROOM_NAMES[room]})"


def format_step(
    mode, k, n, room, action, intended, sampled, p, raw, level, reward=None, total=None
):
    """One rollout step.

    [POLICY] step 3/8 | room 2 (corridor) | action 2->3 | intended 3 | sampled 3 (p=0.88) | light
    raw=0.68 level=light | reward +10 | total 10
    [TAPE] step 2/7 | room 2 (corridor) | action 2->6 | intended 6 | sampled 6 (p=0.93) | light
    raw=0.49 level=medium
    """
    text = (
        f"[{mode}] step {k}/{n} | {room_label(room)} | action {action} | intended {intended} | "
        f"sampled {sampled} (p={p:.2f}) | light raw={raw:.2f} level={level}"
    )
    if reward is not None:
        text += f" | reward {reward:+.0f} | total {total:.0f}"
    return text


def format_light(room, raw, level):
    """[LIGHT] room 9 (stairs area) | raw=0.19 | level=dark"""
    return f"[LIGHT] {room_label(room)} | raw={raw:.2f} | level={level}"


def format_mpe_step(k, n, action, z):
    """[MPE] step 4/8 | action 6->7 | z=medium | true room hidden"""
    return f"[MPE] step {k}/{n} | action {action} | z={z} | true room hidden"


def format_policy_line(source, policy, start, value):
    """POLICY from policy_iteration(): 1:1->2 2:2->3 ... 9:9->2   V(8)=31.42"""
    pairs = " ".join(f"{room}:{ACTIONS[int(a)]}" for room, a in zip(ROOMS, policy, strict=False))
    value_text = f"V({start})={value:.2f}" if value is not None else f"V({start})=n/a"
    return f"POLICY from {source}: {pairs}   {value_text}"


def format_done(mode, seed, start, steps, rooms, total_reward=None):
    """POLICY DONE seed=3630 start=8 steps=8 total_reward=30 rooms=8,9,2,3,2,3,2,3,2
    TAPE DONE seed=3631 final_room=7 rooms=3,2,6,7,7,6,2,3"""
    room_text = ",".join(rooms)
    if mode in ("POLICY", "CUSTOM"):
        return (
            f"{mode} DONE seed={seed} start={start} steps={steps} total_reward={total_reward:.0f} "
            f"rooms={room_text}"
        )
    return f"{mode} DONE seed={seed} final_room={rooms[-1]} rooms={room_text}"
