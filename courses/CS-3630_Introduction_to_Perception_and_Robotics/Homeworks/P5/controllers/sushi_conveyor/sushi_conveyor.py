"""
sushi_conveyor.py: makes the conveyor belt "revolve" (STAFF PROVIDED, decoration only).

This supervisor moves every SushiPlate that is a child of the CONVEYOR robot node
along the stadium-shaped belt centerline of the counter island. It does not touch
the waiter robot and has no effect on the project code.

controllerArgs (optional): halfStraight radius height speed
"""

import math
import sys

from controller import Supervisor

args = [float(a) for a in sys.argv[1:]]
HALF_STRAIGHT, BELT_RADIUS, BELT_Z, SPEED = (args + [0.78, 0.17, 0.1062, 0.035][len(args) :])[:4]

STRAIGHT = 2.0 * HALF_STRAIGHT
ARC = math.pi * BELT_RADIUS
LOOP = 2.0 * STRAIGHT + 2.0 * ARC


def belt_pose(s):
    """Point and heading at arc length s along the belt (counterclockwise seen from above)."""
    s %= LOOP
    if s < STRAIGHT:  # south side, heading east
        return -HALF_STRAIGHT + s, -BELT_RADIUS, 0.0
    s -= STRAIGHT
    if s < ARC:  # east half circle
        a = -math.pi / 2 + s / BELT_RADIUS
        return HALF_STRAIGHT + BELT_RADIUS * math.cos(a), BELT_RADIUS * math.sin(a), a + math.pi / 2
    s -= ARC
    if s < STRAIGHT:  # north side, heading west
        return HALF_STRAIGHT - s, BELT_RADIUS, math.pi
    s -= STRAIGHT
    a = math.pi / 2 + s / BELT_RADIUS  # west half circle
    return -HALF_STRAIGHT + BELT_RADIUS * math.cos(a), BELT_RADIUS * math.sin(a), a + math.pi / 2


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
children = robot.getSelf().getField("children")
plates = []
for i in range(children.getCount()):
    node = children.getMFNode(i)
    field = node.getField("translation") if node is not None else None
    if field is not None:
        plates.append((node.getField("translation"), node.getField("rotation")))

offsets = [i * LOOP / max(len(plates), 1) for i in range(len(plates))]
travelled = 0.0
while robot.step(timestep) != -1:
    travelled += SPEED * timestep / 1000.0
    for (t_field, r_field), offset in zip(plates, offsets, strict=False):
        x, y, heading = belt_pose(travelled + offset)
        t_field.setSFVec3f([x, y, BELT_Z])
        r_field.setSFRotation([0, 0, 1, heading])
