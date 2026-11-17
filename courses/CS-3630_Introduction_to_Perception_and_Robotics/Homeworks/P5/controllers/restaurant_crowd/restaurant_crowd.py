"""
restaurant_crowd.py: customers for the dynamic simulation (STAFF PROVIDED).

This supervisor does nothing unless rrt_controller runs the dynamic simulation
(it writes "dynamic;target=<seat>" or "static;target=<seat>" into SUSHI_BOT's
customData; keys 1 and 2 switch). In dynamic mode, every few seconds a customer
walks in through the glass doors,
follows the walkway lanes to a stool, moves that stool to a new random spot
(pushed out and/or turned), and walks back out.

Customers are polite but not smart: they stop and wait when the robot is close
in front of them, they never move a stool onto the robot, and they never touch
the stool of the seat the robot is currently heading to. Everything else is up
to the robot: it has to notice them (and the moved stools) with its proximity
sensors and replan.
"""

import math
import os
import random
import sys

from controller import Supervisor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rrt_controller"))
import dynamic_restaurant as D  # noqa: E402
import restaurant_map as M  # noqa: E402

PARK_Z = -1.0  # customers wait under the floor when not in the restaurant
ARRIVAL_GAP_S = (5.0, 12.0)  # time between two customers coming in
SIT_S = (1.5, 3.0)  # how long a customer stays at the stool before moving it
STOOL_SLIDE_S = 1.5  # how long moving a stool takes
YIELD_DIST = 0.10  # customer waits when the robot center is this close and ahead of it
GIVE_UP_S = 4.0  # a customer blocked by the robot this long steps back to let it pass
STEP_BACK = 0.12  # how far back along their own way they step
STOOL_ROBOT_DIST = 0.12  # a sliding stool pauses while the robot is this close
VIEW_GAP = (
    M.CUSTOMER_RADIUS + 0.03
)  # customers keep out of the robot's line of sight to its target card
PERSON_GAP = 2 * M.CUSTOMER_RADIUS + 0.01  # closest two customers' centers get
PASS_GAP = PERSON_GAP + 0.06  # a customer who stepped aside waits until the other is this far away
YIELD_PEOPLE_S = 3.0  # blocked by another customer this long: step aside even if not head-on
ASIDE_WAIT_S = 6.0  # give up waiting at the side after this long


class Customer:
    def __init__(self, node):
        self.t_field = node.getField("translation")
        self.r_field = node.getField("rotation")
        self.state = "parked"
        self.route = []
        self.visited = []  # waypoints already reached, to walk back out the same way
        self.seat = None
        self.timer = 0.0
        self.blocked = 0.0
        self.people_blocked = 0.0
        self.yield_for = None  # (other customer, side spot, time left) when giving way
        self.slide = None
        self.pos = tuple(self.t_field.getSFVec3f()[:2])

    def place(self, x, y, yaw=None, z=0.0):
        self.pos = (x, y)
        self.t_field.setSFVec3f([x, y, z])
        if yaw is not None:
            self.r_field.setSFRotation([0, 0, 1, yaw])


def side_step(c, ux, uy, others):
    """
    A spot just to the left or right of c's way that is clear of obstacles, people and the robot.
    """
    fixed = D.fixed_obstacles() + [D.stool_polygon(p) for p in stool_poses.values()]
    for sign in (1, -1):
        x = c.pos[0] - sign * uy * (PERSON_GAP + 0.01)
        y = c.pos[1] + sign * ux * (PERSON_GAP + 0.01)
        if (
            D.point_clearance((x, y), fixed) > M.CUSTOMER_RADIUS + 0.005
            and all(math.hypot(x - o.pos[0], y - o.pos[1]) > PERSON_GAP for o in others)
            and math.hypot(x - robot_xy[0], y - robot_xy[1]) > YIELD_DIST
        ):
            return (x, y)
    return None


def route_to(seat, stool_pose):
    """Waypoints from the doors to the spot behind the stool of `seat`."""
    lane_y = M.CROWD_LANE_Y[M.STOOL_ROWS[seat]]
    x_max = M.CROWD_LANE_X_MAX[M.STOOL_ROWS[seat]]
    lane_x = min(max(stool_pose[0], M.CROWD_HALL_X), x_max)
    route = [M.CROWD_ENTRANCE, (M.CROWD_HALL_X, 0.0), (M.CROWD_HALL_X, lane_y), (lane_x, lane_y)]
    card, normal = D.card_from_stool(stool_pose)
    behind = (card[0] + 0.075 * normal[0], card[1] + 0.075 * normal[1])
    # Step off the lane to stand right behind the stool, if nothing is in the way.
    others = D.fixed_obstacles() + [D.stool_polygon(p) for s, p in stool_poses.items() if s != seat]
    if all(D.segment_polygon_dist(route[-1], behind, poly) > M.CUSTOMER_RADIUS for poly in others):
        route.append(behind)
    return route


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0
rng = random.Random()

children = robot.getSelf().getField("children")
customers = [Customer(children.getMFNode(i)) for i in range(children.getCount())]
waiter = robot.getFromDef("SUSHI_BOT")
waiter_data = waiter.getField("customData") if waiter is not None else None
stools = {seat: robot.getFromDef(f"SEAT_{seat}") for seat in M.STOOL_HOME_POSES}
stool_poses = D.read_stool_poses(robot)

# Start every run with an empty restaurant and every stool at home, even if the
# world was saved (or reset) in the middle of a dynamic simulation.
for c in customers:
    c.place(M.CROWD_ENTRANCE[0], 0.0, z=PARK_Z)
for seat, (x, y, yaw) in M.STOOL_HOME_POSES.items():
    stools[seat].getField("translation").setSFVec3f([x, y, 0])
    stools[seat].getField("rotation").setSFRotation([0, 0, 1, yaw])
    stools[seat].resetPhysics()
    stool_poses[seat] = (x, y, yaw)

next_arrival = rng.uniform(*ARRIVAL_GAP_S) / 2
announced = False

while robot.step(timestep) != -1:
    data = waiter_data.getSFString() if waiter_data is not None else ""
    closing = not data.startswith("dynamic")
    if closing and all(c.state == "parked" for c in customers):
        announced = False
        continue  # static simulation: the restaurant stays still
    if closing:  # switched back to static: everyone heads out
        for c in customers:
            if c.state in ("walking_in", "sitting"):
                c.state, c.route = "walking_out", list(reversed(c.visited))
    elif not announced:
        print("restaurant_crowd: dynamic simulation, customers are coming in.")
        announced = True
    target = None
    for part in data.split(";")[1:]:
        if part.startswith("target="):
            target = int(part[len("target=") :]) if part[len("target=") :].isdigit() else None
    rx, ry, _ = waiter.getPosition()
    robot_xy = (rx, ry)
    view = None  # segment from the target viewing pose to its card
    if target in stool_poses:
        vx, vy, _ = D.viewing_pose_from_stool(stool_poses[target])
        card, _ = D.card_from_stool(stool_poses[target])
        view = ((vx, vy), (float(card[0]), float(card[1])))

    # --- a new customer comes in ---
    next_arrival -= dt
    if next_arrival <= 0 and not closing:
        next_arrival = rng.uniform(*ARRIVAL_GAP_S)
        free = [c for c in customers if c.state == "parked"]
        busy = {c.seat for c in customers if c.state != "parked"}
        seats = [s for s in M.CROWD_SEATS if s != target and s not in busy]
        door_clear = all(
            math.hypot(c.pos[0] - M.CROWD_ENTRANCE[0], c.pos[1] - M.CROWD_ENTRANCE[1]) > PASS_GAP
            for c in customers
            if c.state != "parked"
        )
        if free and seats and door_clear:
            c = free[0]
            c.seat = rng.choice(seats)
            c.route = route_to(c.seat, stool_poses[c.seat])
            c.visited = []
            c.state = "walking_in"
            c.blocked = 0.0
            c.place(*M.CROWD_ENTRANCE, yaw=0.0)

    for k, c in enumerate(customers):
        if c.state in ("walking_in", "walking_out"):
            if c.yield_for is not None and c.yield_for[1] not in c.route:
                other, side, left = (
                    c.yield_for
                )  # standing aside: wait until the other one has passed
                left -= dt
                passed = (
                    other.state == "parked"
                    or math.hypot(other.pos[0] - c.pos[0], other.pos[1] - c.pos[1]) > PASS_GAP
                )
                c.yield_for = None if passed or left <= 0 else (other, side, left)
                if c.yield_for is not None:
                    continue
            if not c.route:
                if c.state == "walking_in":
                    c.state, c.timer = "sitting", rng.uniform(*SIT_S)
                else:
                    c.state, c.seat = "parked", None
                    c.place(M.CROWD_ENTRANCE[0], 0.0, z=PARK_Z)
                continue
            gx, gy = c.route[0]
            dx, dy = gx - c.pos[0], gy - c.pos[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                c.visited.append(c.route.pop(0))
                continue
            ux, uy = dx / dist, dy / dist
            to_robot = (robot_xy[0] - c.pos[0], robot_xy[1] - c.pos[1])
            if math.hypot(*to_robot) < YIELD_DIST and to_robot[0] * ux + to_robot[1] * uy > 0:
                c.blocked += dt  # the robot is in the way: wait for it
                if c.blocked > GIVE_UP_S:  # still there: step back the way we came, then go on
                    print("restaurant_crowd: a customer steps back to let the robot pass.")
                    c.blocked = 0.0
                    back = (c.pos[0] - ux * STEP_BACK, c.pos[1] - uy * STEP_BACK)
                    c.route = [back, c.pos] + c.route
                continue
            c.blocked = 0.0
            step = min(dist, M.CUSTOMER_SPEED * dt)
            nxt = (c.pos[0] + ux * step, c.pos[1] + uy * step)
            if view is not None and D._seg_dist(nxt, *view) < VIEW_GAP <= D._seg_dist(c.pos, *view):
                continue  # would walk into the robot's view of its card: wait
            # Other customers: never walk into one. Head-on in a lane, the later arrival steps
            # aside.
            others = [o for o in customers if o is not c and o.state != "parked"]
            in_way = [
                o
                for o in others
                if math.hypot(nxt[0] - o.pos[0], nxt[1] - o.pos[1]) < PERSON_GAP
                and math.hypot(nxt[0] - o.pos[0], nxt[1] - o.pos[1])
                < math.hypot(c.pos[0] - o.pos[0], c.pos[1] - o.pos[1])
            ]
            if in_way:
                o = in_way[0]
                c.people_blocked += dt
                o_walking = o.state in ("walking_in", "walking_out") and o.route
                head_on = (
                    o_walking
                    and (o.route[0][0] - o.pos[0]) * ux + (o.route[0][1] - o.pos[1]) * uy < 0
                )
                lower = customers.index(o) < k
                if o_walking and ((head_on and lower) or c.people_blocked > YIELD_PEOPLE_S):
                    side = side_step(c, ux, uy, others)
                    if side is not None:
                        c.route = [side] + c.route
                        c.yield_for = (o, side, ASIDE_WAIT_S)
                        c.people_blocked = 0.0
                continue
            c.people_blocked = 0.0
            c.place(c.pos[0] + ux * step, c.pos[1] + uy * step, yaw=math.atan2(uy, ux))
            if step >= dist:
                c.visited.append(c.route.pop(0))

        elif c.state == "sitting":
            c.timer -= dt
            if c.timer > 0:
                continue
            keep_clear = [robot_xy] + ([view[0]] if view is not None else [])
            new_pose = (
                None
                if c.seat == target
                else D.sample_stool_pose(c.seat, stool_poses, keep_clear, rng)
            )
            if new_pose is None:  # nowhere to put it (or the robot needs this seat): leave
                c.state = "walking_out"
                c.route = list(reversed(c.visited))
            else:
                c.state, c.timer = "moving_stool", 0.0
                c.slide = (stool_poses[c.seat], new_pose)

        elif c.state == "moving_stool":
            (x0, y0, a0), (x1, y1, a1) = c.slide
            node = stools[c.seat]
            if c.seat == target:  # the robot is now headed to this seat: leave the stool be
                c.state, c.route = "walking_out", list(reversed(c.visited))
                continue
            if (
                D.point_polygon_dist(robot_xy, D.stool_polygon(stool_poses[c.seat]))
                < STOOL_ROBOT_DIST
            ):
                continue  # never slide a stool into the robot
            c.timer = min(1.0, c.timer + dt / STOOL_SLIDE_S)
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            pose = (x0 + (x1 - x0) * c.timer, y0 + (y1 - y0) * c.timer, a0 + da * c.timer)
            node.getField("translation").setSFVec3f([pose[0], pose[1], 0])
            node.getField("rotation").setSFRotation([0, 0, 1, pose[2]])
            stool_poses[c.seat] = pose
            if c.timer >= 1.0:
                print(f"restaurant_crowd: a customer moved stool {c.seat}.")
                c.state = "walking_out"
                c.route = list(reversed(c.visited))
