"""
dynamic_navigation.py: live obstacle avoidance for the dynamic simulation (STAFF PROVIDED).

With SIMULATION_MODE = "dynamic", rrt_controller hands the robot these helpers:

- ProximitySensing reads the e-puck's eight infra-red proximity sensors (ps0-ps7,
  range about 7 cm) and turns each reading into a point in the world.
- DynamicNavigator is called by RobotController.execute_path on every step. It
  ignores points that belong to obstacles already on the map, and when a point
  that is NOT on the map (a customer, a stool that was moved) gets in the way of
  the path ahead, it stops the robot, adds that obstacle to the map, and replans
  the rest of the leg with RRT* from where the robot is.
- SimulationModeSwitch lets you switch while the mission runs: press 1 for the
  static simulation and 2 for the dynamic one (click the 3D view first so Webots
  sends it the keys). Going back to static sends the customers out again.
- DynamicGoalSequence replaces the fixed goal list: at the start of every leg it
  reads where the target stool is now (the "seating chart" the host keeps up to
  date), refreshes the stools on the planner map, and computes the viewing pose.

Everything that changes DURING a leg is only known through the proximity sensors.
"""

import math

import dynamic_restaurant as D
import gtsam
import numpy as np
import restaurant_map as M

# Sensor mounts from E-puck.proto: (x, y) on the robot body. Each sensor looks
# straight out from the robot center.
SENSOR_MOUNTS = [
    (0.03, -0.01),
    (0.022, -0.025),
    (0.0, -0.031),
    (-0.03, -0.015),
    (-0.03, 0.015),
    (0.0, 0.031),
    (0.022, 0.025),
    (0.03, 0.01),
]
FRONT_SENSORS = (0, 1, 6, 7)
REAR_SENSORS = (3, 4)
SENSOR_MAX_RANGE = 0.055  # readings farther than this are noise
KNOWN_TOLERANCE = 0.02  # a point this close to a mapped obstacle belongs to it
PATH_CLEARANCE = M.ROBOT_RADIUS + 0.006  # an unknown point this close to the path ahead blocks it
LOOKAHEAD_METERS = 0.25  # how far ahead along the path to check
TOO_CLOSE = 0.06  # back up before replanning if something is this close in front (any
# front reading): gives the replanned path room to leave the obstacle's margin
REVERSE_S = 1.0  # backing up for this long moves the robot about 4 cm
SENSED_RADIUS = 0.03  # size of the obstacle added to the map for one sensed point
SENSED_LIFETIME_S = 6.0  # forget sensed obstacles after this long (people move)
REPLAN_ITERATIONS = 700
WAIT_S = 1.0  # wait this long before trying again when no path is found
STUCK_S = 3.0  # not moving 1 cm in this long while driving means something unseen holds the robot
NO_CLOSER_S = 15.0  # not getting 2 cm closer to the goal in this long (circling): replan from here


class ProximitySensing:
    """The e-puck's eight infra-red proximity sensors, as points in the world."""

    def __init__(self, robot, timestep):
        self.sensors = []
        for i in range(8):
            sensor = robot.getDevice(f"ps{i}")
            sensor.enable(timestep)
            self.sensors.append(sensor)
        table = np.asarray(self.sensors[0].getLookupTable(), dtype=float).reshape(-1, 3)
        # lookup table rows: distance [m], value, noise; values fall as the distance grows
        order = np.argsort(table[:, 1])
        self._values, self._distances = table[order, 1], table[order, 0]

    def distances(self):
        """Distance [m] seen by each sensor, or None when nothing is in range."""
        out = []
        for sensor in self.sensors:
            d = float(np.interp(sensor.getValue(), self._values, self._distances))
            out.append(d if d < SENSOR_MAX_RANGE else None)
        return out

    def world_points(self, pose):
        """[(sensor index, distance, (x, y) of the hit, (ux, uy) ray direction)] in the world."""
        c, s = math.cos(pose.theta()), math.sin(pose.theta())
        hits = []
        for i, d in enumerate(self.distances()):
            if d is None:
                continue
            mx, my = SENSOR_MOUNTS[i]
            norm = math.hypot(mx, my)
            ux, uy = c * mx / norm - s * my / norm, s * mx / norm + c * my / norm
            sx, sy = pose.x() + c * mx - s * my, pose.y() + s * mx + c * my
            hits.append((i, d, (sx + ux * d, sy + uy * d), (ux, uy)))
        return hits


class DynamicNavigator:
    """Watches the proximity sensors while a path is followed and replans around new obstacles."""

    def __init__(self, robot, controller, make_environment, plan_path, world_to_planner):
        self.robot = robot
        self.controller = controller
        self.make_environment = make_environment  # world polygons -> planner environment
        self.plan_path = plan_path  # (start, goal, env, name) -> (N, 2) path or empty
        self.world_to_planner = world_to_planner
        self.sensing = ProximitySensing(robot, controller.timestep)
        self.dt = controller.timestep / 1000.0
        self.known = []  # world polygons on the map for this leg
        self.sensed = []  # [(time, (x, y))] obstacles found by the sensors
        self.goal = None
        self.leg_name = "leg"
        self.replans = 0
        self.wait_steps = 0
        self.reverse_steps = 0
        self.progress = None  # (time, x, y) of the last time the robot had moved 1 cm
        self.closest = (
            None  # (time, distance) of the last time the robot got 2 cm closer to the goal
        )

    def start_leg(self, polygons, goal, leg_name):
        self.known = polygons
        self.sensed = []
        self.goal = goal
        self.leg_name = leg_name
        self.replans = 0
        self.progress = None
        self.closest = None

    def _now(self):
        return self.robot.getTime()

    def _is_known(self, p):
        if D.point_clearance(p, []) < KNOWN_TOLERANCE:  # a wall
            return True
        if any(D.point_polygon_dist(p, poly) < KNOWN_TOLERANCE for poly in self.known):
            return True
        return any(
            math.hypot(p[0] - q[0], p[1] - q[1]) < SENSED_RADIUS + KNOWN_TOLERANCE
            for _, q in self.sensed
        )

    def _path_ahead(self, path, index):
        ahead, travelled = [], 0.0
        for k in range(max(index, 0), len(path)):
            if ahead:
                travelled += float(np.hypot(*(path[k] - ahead[-1])))
                if travelled > LOOKAHEAD_METERS:
                    break
            ahead.append(path[k])
        return np.asarray(ahead)

    def __call__(self, pose, path, index):
        """
        Called by RobotController.execute_path every step. Returns ("continue", None),
        ("wait", None), ("reverse", None) or ("replanned", new_path).
        """
        if self.reverse_steps > 0 or self.wait_steps > 0:
            self.progress = None
            self.closest = None
        if self.reverse_steps > 0:
            self.reverse_steps -= 1
            behind = [
                d
                for i, d in enumerate(self.sensing.distances())
                if i in REAR_SENSORS and d is not None
            ]
            if behind and min(behind) < 0.025:  # something behind too: stop backing up
                self.reverse_steps = 0
            return ("reverse", None) if self.reverse_steps > 0 else self._replan(pose)
        if self.wait_steps > 0:
            self.wait_steps -= 1
            if self.wait_steps > 0:
                return "wait", None
            # Done waiting: forget what was sensed and look again. If it is still in the
            # way it is sensed (and replanned around) on the next step; if not, carry on.
            self.sensed = []
            return "continue", None

        now = self._now()
        self.sensed = [(t, q) for t, q in self.sensed if now - t < SENSED_LIFETIME_S]
        all_hits = self.sensing.world_points(pose)

        # Watchdog: driving but not moving means something holds the robot that the map says
        # is known (or the sensors miss). Treat everything the sensors see as an obstacle.
        if (
            self.progress is None
            or math.hypot(pose.x() - self.progress[1], pose.y() - self.progress[2]) > 0.01
        ):
            self.progress = (now, pose.x(), pose.y())
        elif now - self.progress[0] > STUCK_S:
            self.progress = None
            for _i, _d, p, (ux, uy) in all_hits:
                self.sensed.append((now, (p[0] + ux * SENSED_RADIUS, p[1] + uy * SENSED_RADIUS)))
            print(
                "Not making progress: backing up and replanning "
                "around everything the sensors see...",
                flush=True,
            )
            self.reverse_steps = int(REVERSE_S / self.dt)
            return "reverse", None

        to_goal = math.hypot(self.goal[0] - pose.x(), self.goal[1] - pose.y())
        if self.closest is None or to_goal < self.closest[1] - 0.02:
            self.closest = (now, to_goal)
        elif now - self.closest[0] > NO_CLOSER_S:
            self.closest = None
            print("Not getting any closer to the goal: replanning from here...", flush=True)
            return self._replan(pose)

        hits = [h for h in all_hits if not self._is_known(h[2])]
        if not hits:
            return "continue", None

        ahead = self._path_ahead(path, index)
        blocking = []
        for i, d, p, u in hits:
            near_path = len(ahead) and np.min(np.hypot(*(ahead - np.asarray(p)).T)) < PATH_CLEARANCE
            in_front = i in FRONT_SENSORS and d < 0.035
            if near_path or in_front:
                blocking.append((i, d, p, u))
        if not blocking:
            return "continue", None

        for _i, _d, p, (
            ux,
            uy,
        ) in blocking:  # the obstacle's center is a bit behind the surface we see
            self.sensed.append((now, (p[0] + ux * SENSED_RADIUS, p[1] + uy * SENSED_RADIUS)))
        i, d, p, _ = min(blocking, key=lambda b: b[1])
        print(
            "Proximity sensors: something that is not on the map is in the way "
            f"at ({p[0]:+.2f}, {p[1]:+.2f}) "
            f"({d * 100:.1f} cm from sensor ps{i}). Replanning...",
            flush=True,
        )
        return self._back_up_or_replan(pose)

    def _back_up_or_replan(self, pose):
        close = [
            d
            for i, d in enumerate(self.sensing.distances())
            if i in FRONT_SENSORS and d is not None
        ]
        if close and min(close) < TOO_CLOSE:
            self.reverse_steps = int(REVERSE_S / self.dt)
            return "reverse", None
        return self._replan(pose)

    def _replan(self, pose):
        discs = [
            [
                (q[0] + SENSED_RADIUS * math.cos(a), q[1] + SENSED_RADIUS * math.sin(a))
                for a in np.linspace(0, 2 * math.pi, 10, endpoint=False)
            ]
            for _, q in self.sensed
        ]
        polygons = self.known + discs
        env = self.make_environment(polygons)
        start = self._free_start(env, pose)
        if start is None:
            print("  The robot is boxed in. Waiting for the way to clear...", flush=True)
            self.wait_steps = int(WAIT_S / self.dt)
            return "wait", None
        self.replans += 1
        path = self.plan_path(start, self.goal, env, f"{self.leg_name}_replan_{self.replans}")
        if path.size == 0:
            # The planner works on positions only (the robot can turn in place), so there is no
            # better heading to try: wait for the way to clear.
            print("  No path around it right now. Waiting for the way to clear...", flush=True)
            self.wait_steps = int(WAIT_S / self.dt)
            return "wait", None
        return "replanned", path

    def _free_start(self, env, pose):
        """Find the robot position or a free point within 3 cm of an inflated obstacle."""
        for r in (0.0, 0.01, 0.02, 0.03):
            for a in np.linspace(0, 2 * math.pi, 12, endpoint=False) if r else [0.0]:
                x, y = pose.x() + r * math.cos(a), pose.y() + r * math.sin(a)
                if env.is_free(*self.world_to_planner(x, y)):
                    return (x, y, pose.theta())
        return None


class SimulationModeSwitch:
    """
    The path monitor RobotController calls every step. Reads the keyboard (1 = static,
    2 = dynamic), tells the restaurant_crowd supervisor the mode through the robot's
    customData, and only lets the DynamicNavigator react to the sensors when it matters.
    """

    def __init__(self, robot, navigator, mode):
        self.robot = robot
        self.navigator = navigator
        self.mode = mode
        self.target = None
        self.ever_dynamic = mode == "dynamic"
        self.keyboard = robot.getKeyboard()
        self.keyboard.enable(navigator.controller.timestep)
        print(
            f"Simulation mode: {self.mode}. Press 1 for static, 2 for dynamic "
            f"(click the 3D view first so it gets the keys).",
            flush=True,
        )
        self.publish()

    def publish(self, target=None):
        if target is not None:
            self.target = target
        self.robot.setCustomData(
            self.mode + (f";target={self.target}" if self.target is not None else "")
        )

    def poll(self):
        key = self.keyboard.getKey()
        while key != -1:
            mode = {ord("1"): "static", ord("2"): "dynamic"}.get(key)
            if mode is not None and mode != self.mode:
                self.mode = mode
                self.ever_dynamic |= mode == "dynamic"
                print(
                    f"*** Switched to the {mode} simulation"
                    + (
                        ": customers are coming in. ***"
                        if mode == "dynamic"
                        else ": customers are leaving. ***"
                    ),
                    flush=True,
                )
                self.publish()
            key = self.keyboard.getKey()

    def __call__(self, pose, path, index):
        self.poll()
        # After a dynamic stretch, keep watching: customers may still be on their way out.
        if not self.ever_dynamic:
            return "continue", None
        return self.navigator(pose, path, index)


class DynamicGoalSequence:
    """
    Stands in for the fixed list of (seat, goal pose) in the dynamic simulation.
    Each goal is computed when the robot starts that leg, from where the stool is now.
    """

    def __init__(self, robot, env, order, navigator, make_polygons, mode_switch):
        self.robot = robot
        self.mode_switch = mode_switch
        self.env = env
        self.order = list(order)
        self.navigator = navigator
        self.make_polygons = make_polygons  # stool poses -> world polygons of the map

    def __len__(self):
        return len(self.order)

    def refresh_map(self, target=None):
        """Updates the planner map with where every stool is right now; returns the polygons."""
        stool_poses = D.read_stool_poses(self.robot)
        polygons = self.make_polygons(stool_poses)
        fresh = self.navigator.make_environment(polygons)
        self.env.obstacles, self.env.occupancy = fresh.obstacles, fresh.occupancy
        self.mode_switch.poll()
        self.mode_switch.publish(target)
        return stool_poses, polygons

    def leave_obstacle_margin(self):
        """
        If the robot stands inside an obstacle's safety margin (a stool was moved next to it),
        the planner cannot start there: turn toward the nearest free spot and creep onto it.
        """
        nav, ctrl = self.navigator, self.navigator.controller
        pose = ctrl.get_current_pose()
        to_planner = nav.world_to_planner
        if self.env.is_free(*to_planner(pose.x(), pose.y())):
            return
        for r in np.arange(0.01, 0.10, 0.01):
            spots = [
                (pose.x() + r * math.cos(a), pose.y() + r * math.sin(a))
                for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)
            ]
            spots = [p for p in spots if self.env.is_free(*to_planner(*p))]
            if spots:
                break
        else:
            return
        tx, ty = spots[0]
        print(
            "The robot is inside an obstacle's safety margin: "
            f"moving {r * 100:.0f} cm to a free spot first.",
            flush=True,
        )
        for _ in range(int(6.0 / nav.dt)):
            pose = ctrl.get_current_pose()
            dist = math.hypot(tx - pose.x(), ty - pose.y())
            if dist < 0.005:
                break
            err = ctrl._normalize_angle(math.atan2(ty - pose.y(), tx - pose.x()) - pose.theta())
            if abs(err) > 0.1:
                ctrl.set_motor_speeds(*ctrl._calculate_docking_velocities(err))
            else:
                speed = min(0.3, dist / 0.02) * ctrl.MAX_SPEED_RADS
                ctrl.set_motor_speeds(speed, speed)
            if self.robot.step(ctrl.timestep) == -1:
                break
            ctrl.update_odometry()
        ctrl.stop_motors()

    def __iter__(self):
        for i, seat in enumerate(self.order):
            stool_poses, polygons = self.refresh_map(target=seat)
            self.leave_obstacle_margin()
            x, y, yaw = D.viewing_pose_from_stool(stool_poses[seat])
            self.navigator.start_leg(polygons, (x, y, yaw), f"leg_{i}_seat_{seat}")
            yield seat, gtsam.Pose2(x, y, yaw)
        # last leg: back to the drink station
        _, polygons = self.refresh_map()
        self.leave_obstacle_margin()
        self.navigator.start_leg(polygons, M.RETURN_POSE, "leg_return")
