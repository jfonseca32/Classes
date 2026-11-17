import math
from unittest.mock import MagicMock

import gtsam
import numpy as np

try:
    from controller import Robot
except ImportError as e:
    print(f"ERROR: Could not import controller module: {e}")
    # Mock the controller module to avoid import errors when running outside of webots
    mock_controller_module = MagicMock()
    mock_robot_class = MagicMock()
    mock_device = MagicMock()
    mock_controller_module.Robot = mock_robot_class
    mock_controller_module.GPS = MagicMock()
    mock_controller_module.Compass = MagicMock()
    mock_robot_instance = mock_robot_class()
    mock_robot_instance.getDevice.return_value = mock_device


class RobotController:
    """
    Manages the low-level hardware and control loops for the e-puck robot.

    The robot only drives forward. It follows the smooth parts of a path with
    pure pursuit, and wherever the path turns sharply (and at the start of a path
    that leaves behind it) it stops and rotates in place, like the "turn toward
    the target, then drive" steering of Section 5.5.5 of the textbook. Every
    motion, including the rotations, goes through the inverse kinematics.

    While it drives, it also integrates its wheel encoders (odometry, Section 5.2)
    next to the GPS and compass, and reports how far the two drift apart.
    """

    def __init__(self, robot_instance: Robot, timestep: int):
        self.robot = robot_instance
        self.timestep = timestep

        # Robot & Controller parameters in meters, radians, seconds. (see the e-puck spec sheet)
        self.WHEEL_DISTANCE = 0.053
        self.WHEEL_RADIUS = 0.0205
        self.MAX_SPEED_RADS = 6.28
        self.LOOKAHEAD_DISTANCE = 0.08  # short lookahead: the restaurant walkways are narrow
        self.FORWARD_SPEED = 0.1  # m/s: the robot is careful, to avoid knocking over things
        self.POSITION_TOLERANCE = 0.02
        self.ANGLE_TOLERANCE = 0.02

        # Rotation in place: omega = K_TURN * heading error (rad/s per rad). 4.86 makes a
        # 1 rad error spin the wheels at full speed, MAX_SPEED_RADS * 2 * r / L.
        self.K_TURN = 4.86

        # A path corner that turns the heading by more than this is a "sharp corner":
        # the robot drives to it, stops and rotates in place instead of cutting it.
        self.CORNER_ANGLE = 0.6

        # During pure pursuit, a heading error larger than this (for example at the
        # start of a path) is first removed by rotating in place.
        self.ROTATE_IN_PLACE_ANGLE = 0.6

        # When docking, turn to face a point this far ahead of the goal pose (the seat card's
        # center), not just to the goal yaw: a stop that is ~1 cm off to the side then still
        # puts the card in the middle of the 32x32 image
        self.AIM_DISTANCE = 0.152
        self.MAX_STEPS_PER_PATH = 6000  # safety net: give up on a path after ~3 simulated minutes

        # Dynamic simulation only: called every step with (pose, path, path index); it can
        # make the robot wait, back up, or switch to a replanned path (see dynamic_navigation.py)
        self.path_monitor = None

        # True: drive on the odometry estimate instead of the GPS and compass (an experiment)
        self.DRIVE_ON_ODOMETRY = False

        # Odometry: two estimates, one reset to the GPS pose at the start of every path
        # (drift over one leg), one never reset (drift over the whole mission)
        self.odom_leg = None
        self.odom_mission = None
        self._last_wheel_angles = None
        self._last_odom_time = None

        try:
            # Get the GPS, compass, and wheel motors and encoders (the robot's devices)
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.timestep)
            self.compass = self.robot.getDevice("compass")
            self.compass.enable(self.timestep)
            self.leftMotor = self.robot.getDevice("left wheel motor")
            self.rightMotor = self.robot.getDevice("right wheel motor")
            self.leftMotor.setPosition(float("inf"))
            self.rightMotor.setPosition(float("inf"))
            self.leftEncoder = self.robot.getDevice("left wheel sensor")
            self.rightEncoder = self.robot.getDevice("right wheel sensor")
            self.leftEncoder.enable(self.timestep)
            self.rightEncoder.enable(self.timestep)
            self.stop_motors()
        except Exception as e:
            print(f"ERROR: Could not get devices in RobotController. {e}")
            raise

    def get_sensor_pose(self) -> gtsam.Pose2:
        """
        Reads the GPS and compass and returns the robot's true pose
        as a gtsam.Pose2 object.
        """
        current_pos = self.gps.getValues()
        current_x, current_y = current_pos[0], current_pos[1]

        north = self.compass.getValues()
        current_theta = (
            math.atan2(north[0], north[1])
            if north[1] != 0.00
            else (1.57 if north[0] > 0.00 else -1.57)
        )

        return gtsam.Pose2(current_x, current_y, current_theta)

    def get_current_pose(self) -> gtsam.Pose2:
        """
        The pose the controller drives on: the GPS and compass, or the odometry
        estimate when DRIVE_ON_ODOMETRY is set.
        """
        if self.DRIVE_ON_ODOMETRY and self.odom_mission is not None:
            return self.odom_mission
        return self.get_sensor_pose()

    def set_motor_speeds(self, left_rads: float, right_rads: float):
        """
        Sets the robot's wheel velocities. If one wheel would exceed the motor
        limit, BOTH wheels are scaled down by the same factor, which keeps the
        ratio v / w (the curvature of the arc the robot drives) unchanged.
        """
        largest = max(abs(left_rads), abs(right_rads))
        if largest > self.MAX_SPEED_RADS:
            scale = self.MAX_SPEED_RADS / largest
            left_rads *= scale
            right_rads *= scale
        # Clamp too: the scaled value can come out a hair above the limit after rounding
        left_rads = min(max(left_rads, -self.MAX_SPEED_RADS), self.MAX_SPEED_RADS)
        right_rads = min(max(right_rads, -self.MAX_SPEED_RADS), self.MAX_SPEED_RADS)
        self.leftMotor.setVelocity(left_rads)
        self.rightMotor.setVelocity(right_rads)

    def stop_motors(self):
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

    # ------------------------------------------------------------------
    # Odometry (staff code around your _forward_kinematics and _update_odometry)
    # ------------------------------------------------------------------
    def update_odometry(self):
        """
        Reads the wheel encoders and advances both odometry estimates by one step.
        Call it once per simulation step while the robot moves.
        """
        angles = (self.leftEncoder.getValue(), self.rightEncoder.getValue())
        now = self.robot.getTime()

        if not all(
            isinstance(a, (int, float)) and math.isfinite(a) for a in angles
        ) or not isinstance(now, (int, float)):
            return  # encoders not ready yet (first step)

        if self.odom_mission is None:
            self.odom_mission = self.get_sensor_pose()

        if self.odom_leg is None:
            self.odom_leg = self.get_sensor_pose()

        if (
            self._last_wheel_angles is not None
            and self._last_odom_time is not None
            and now > self._last_odom_time
        ):
            dt = now - self._last_odom_time
            phidot_left = (angles[0] - self._last_wheel_angles[0]) / dt
            phidot_right = (angles[1] - self._last_wheel_angles[1]) / dt
            v, w = self._forward_kinematics(phidot_left, phidot_right)
            self.odom_leg = self._update_odometry(self.odom_leg, v, w, dt)
            self.odom_mission = self._update_odometry(self.odom_mission, v, w, dt)

        self._last_wheel_angles, self._last_odom_time = angles, now

    def odometry_drift(self, estimate: gtsam.Pose2) -> tuple[float, float]:
        """(position error in m, heading error in rad) between an odometry estimate and
        the GPS/compass.
        """
        truth = self.get_sensor_pose()
        return (
            math.hypot(estimate.x() - truth.x(), estimate.y() - truth.y()),
            abs(self._normalize_angle(estimate.theta() - truth.theta())),
        )

    def report_odometry(self, label: str = "leg"):
        if self.odom_leg is None or self.odom_mission is None:
            return
        leg_d, leg_a = self.odometry_drift(self.odom_leg)
        all_d, all_a = self.odometry_drift(self.odom_mission)
        print(
            f"Odometry drift: this {label} {leg_d * 100:.1f} cm / {math.degrees(leg_a):.1f} deg, "
            f"whole mission {all_d * 100:.1f} cm / {math.degrees(all_a):.1f} deg",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Path following
    # ------------------------------------------------------------------
    def _find_sharp_corners(self, path_np: np.ndarray) -> list[int]:
        """Indices of the path points where the path turns by more than CORNER_ANGLE."""
        corners = []
        last_heading: float | None = None

        for i in range(1, len(path_np)):
            d = path_np[i] - path_np[i - 1]
            if math.hypot(d[0], d[1]) < 1e-6:
                continue

            heading = math.atan2(d[1], d[0])
            if (
                last_heading is not None
                and abs(self._normalize_angle(heading - last_heading)) > self.CORNER_ANGLE
            ):
                corners.append(i - 1)

            last_heading = heading

        return corners

    def _set_path(self, path_np: np.ndarray):
        self._path = np.asarray(path_np, dtype=float)
        self._path_index = 0  # progress along the path (see _find_lookahead_point)
        self._corners = self._find_sharp_corners(self._path)

    def execute_path(self, path_np: np.ndarray, final_goal_pose: gtsam.Pose2):
        """
        Executes the main control loop to follow a path.
        This function blocks until the final pose is reached.
        Returns True if the final pose was reached.
        """
        final_goal_x = final_goal_pose.x()
        final_goal_y = final_goal_pose.y()
        final_goal_theta = final_goal_pose.theta()
        aim_point = np.array(
            [
                final_goal_x + self.AIM_DISTANCE * math.cos(final_goal_theta),
                final_goal_y + self.AIM_DISTANCE * math.sin(final_goal_theta),
            ]
        )
        self._set_path(path_np)
        self.odom_leg = None  # restarts from the GPS pose on the next update
        steps = 0

        while self.robot.step(self.timestep) != -1:
            steps += 1
            self.update_odometry()
            if steps > self.MAX_STEPS_PER_PATH:
                print("WARNING: path following timed out (is the robot stuck on a chair?)")
                self.stop_motors()
                return False

            current_pose = self.get_current_pose()

            final_dist = math.sqrt(
                (final_goal_x - current_pose.x()) ** 2 + (final_goal_y - current_pose.y()) ** 2
            )
            final_angle_error = self._heading_error(current_pose, aim_point)

            if final_dist < self.POSITION_TOLERANCE:
                if abs(final_angle_error) < self.ANGLE_TOLERANCE:
                    self.stop_motors()
                    self.report_odometry()
                    return True
                else:
                    # Turn in place to correct orientation
                    (left_rads, right_rads) = self._calculate_docking_velocities(final_angle_error)

            else:
                if self.path_monitor is not None:
                    action, new_path = self.path_monitor(current_pose, self._path, self._path_index)
                    if action == "wait":
                        self.stop_motors()
                        continue
                    if action == "reverse":
                        self.set_motor_speeds(
                            -0.3 * self.MAX_SPEED_RADS, -0.3 * self.MAX_SPEED_RADS
                        )
                        continue
                    if action == "turn":  # new_path holds the heading error to turn through
                        self.set_motor_speeds(*self._calculate_docking_velocities(new_path))
                        continue
                    if action == "replanned":
                        self._set_path(new_path)

                # --- Pure Pursuit Phase (rotate in place first if the path is far off our heading)
                target_point = self._find_lookahead_point(
                    current_pose.x(), current_pose.y(), self._path
                )
                heading_error = self._heading_error(current_pose, target_point)
                if abs(heading_error) > self.ROTATE_IN_PLACE_ANGLE:
                    (left_rads, right_rads) = self._calculate_docking_velocities(heading_error)
                else:
                    (linear_v, angular_v) = self._calculate_pure_pursuit_velocities(
                        current_pose, target_point
                    )
                    (left_rads, right_rads) = self._inverse_kinematics(linear_v, angular_v)

            self.set_motor_speeds(left_rads, right_rads)
        return False

    def _normalize_angle(self, angle: float) -> float:
        """
        Constrains an angle to the [-pi, pi] range
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _find_lookahead_point(self, robot_x, robot_y, path_points):
        """
        Finds the lookahead point on the path for Pure Pursuit: the farthest
        path point that is still within LOOKAHEAD_DISTANCE of the robot.

        Only a window of the path ahead of the robot's current progress is
        searched. In the restaurant the path often comes back close to itself
        (around a pushed-out chair); searching the whole path would let the robot
        "jump" to a later part of the path and cut straight through the chair in
        between. The window also ends at the next sharp corner, so the robot drives
        all the way to the corner before it turns.
        """
        n = len(path_points)
        start = min(getattr(self, "_path_index", 0), n - 1)
        end = min(n, start + 40)
        corner = next((c for c in getattr(self, "_corners", []) if c > start), None)
        if corner is not None:
            end = min(end, corner + 1)
        window = path_points[start:end]
        d = np.hypot(window[:, 0] - robot_x, window[:, 1] - robot_y)

        # Progress: the closest point of the window becomes the new start.
        closest = int(np.argmin(d))
        if (
            corner is not None
            and start + closest == corner
            and d[closest] > self.POSITION_TOLERANCE
        ):
            closest = max(closest - 1, 0)  # not at the corner yet: keep it ahead of us
        self._path_index = start + closest

        # Walk forward from the closest point while we stay inside the lookahead circle.
        k = closest
        while k + 1 < len(window) and d[k + 1] <= self.LOOKAHEAD_DISTANCE:
            k += 1
        return path_points[start + k]

    def _calculate_docking_velocities(self, final_angle_error: float) -> tuple[float, float]:
        """
        Wheel speeds for turning in place: no forward motion (v = 0) and an angular
        velocity proportional to the heading error, through the inverse kinematics.
        """
        return self._inverse_kinematics(0.0, self.K_TURN * final_angle_error)

    def _heading_error(self, current_pose: gtsam.Pose2, target_point: np.ndarray) -> float:
        """Angle the robot has to turn through to face target_point: the direction of
        the target as seen from the robot, minus the robot's heading theta.

        Parameters
        ----------
        current_pose : gtsam.Pose2
            The current pose of the robot as a gtsam.Pose2 object, containing .x(), .y(),
            .theta(), etc.
        target_point : np.ndarray, size 2 (x, y)
            The desired point (x, y) the robot should be moving towards.

        Returns
        -------
        float
            heading error in radians, in [-pi, pi]. Positive means the target
            is to the LEFT (turn counter-clockwise), negative to the right.
        """
        if target_point.size != 2:
            raise ValueError("target_point not size 2 (x, y).")

        dx = target_point[0] - current_pose.x()
        dy = target_point[1] - current_pose.y()

        # Angle in radians between positive x-axis and target_point
        target_angle = math.atan2(dy, dx)

        return self._normalize_angle(target_angle - current_pose.theta())

    def _calculate_pure_pursuit_velocities(
        self, current_pose: gtsam.Pose2, target_point: np.ndarray
    ) -> tuple[float, float]:
        """Calculate the required linear (v) and angular (w) velocities
        to drive towards the target_point.

        Parameters
        ----------
        current_pose : gtsam.Pose2
            The current pose of the robot as a gtsam.Pose2 object, containing .x(), .y(),
            .theta(), etc.
        target_point : np.ndarray, size 2 (x, y)
            The desired point (x, y) the robot should be moving towards.

        Returns
        -------
        tuple[float, float], (linear_velocity, angular_velocity)
            Desired linear velocity (m/s) and desired angular velocity (rad/s) to
            reach the target point.

        Notes
        -----
        Pure pursuit steers along a circular arc toward the lookahead point:

            w = v * curvature = 2 * v * sin(alpha) / d

        The distance d is current_lookahead_dist and
        clamped to at least 0.01 m to avoid division by zero.

        Linear velocity v is in m/s; angular velocity w is in rad/s.
        Positive w turns counter-clockwise (left); negative w turns right.
        When alpha is zero, w is zero and the robot drives straight.
        """
        target_x, target_y = target_point[0], target_point[1]

        current_lookahead_dist = math.sqrt(
            (target_x - current_pose.x()) ** 2 + (target_y - current_pose.y()) ** 2
        )
        if current_lookahead_dist < 0.01:
            current_lookahead_dist = 0.01

        linear_velocity = self.FORWARD_SPEED
        alpha = self._heading_error(current_pose=current_pose, target_point=target_point)
        angular_velocity = 2 * linear_velocity * math.sin(alpha) / current_lookahead_dist

        return (linear_velocity, angular_velocity)

    def _inverse_kinematics(
        self, linear_velocity: float, angular_velocity: float
    ) -> tuple[float, float]:
        """You need to calculate the left and right wheel speeds (in rad/s)
        from a desired linear (v) and angular (w) velocity in the robot's body frame.

        Parameters
        ----------
        linear_velocity : float
            Desired linear velocity of the robot in m/s along the robot's heading.
        angular_velocity : float
            Desired angular velocity of the robot in rad/s counter-clockwise positive.

        Returns
        -------
        tuple[float, float], (left_speed, right_speed)
            Left and right wheel speeds in rad/s to get desired linear and angular
            velocities.

        Notes
        -----
        The wheel speeds are computed using the inverse kinematics matrix:

        .. math::

            \\begin{bmatrix} \\dot{\\phi}_L \\\\ \\dot{\\phi}_R \\end{bmatrix}
            = \\frac{1}{r}
            \\begin{bmatrix} 1 & -\\frac{L}{2} \\\\ 1 & \\frac{L}{2} \\end{bmatrix}
            \\begin{bmatrix} v \\\\ \\omega \\end{bmatrix}

        where :math:`r` is the wheel radius and :math:`L` is the wheel distance.
        """
        left_speed = (
            1 / self.WHEEL_RADIUS * (linear_velocity - self.WHEEL_DISTANCE / 2 * angular_velocity)
        )
        right_speed = (
            1 / self.WHEEL_RADIUS * (linear_velocity + self.WHEEL_DISTANCE / 2 * angular_velocity)
        )
        return (left_speed, right_speed)

    def _forward_kinematics(
        self, left_speed_rads: float, right_speed_rads: float
    ) -> tuple[float, float]:
        """The inverse of _inverse_kinematics: the robot's linear (v) and angular (w)
        velocities from the speeds of its two wheels in the robot's body frame.

        Parameters
        ----------
        left_speed_rads : float
            The left wheel speed of the robot in rad/s.
        right_speed_rads : float
            The right wheel speed of the robot in rad/s.

        Returns
        -------
        tuple[float, float], (linear_velocity, angular_velocity)
            The linear velocity of the robot in m/s along it's heading and angular velocity
            of the robot counter-clockwise positive.

        Notes
        -----
        The body velocities are computed using the forward kinematics matrix:

        .. math::

            \\begin{bmatrix} v \\\\ \\omega \\end{bmatrix}
            = r
            \\begin{bmatrix} \\frac{1}{2} & \\frac{1}{2} \\\\ -\\frac{1}{L} & \\frac{1}{L}
            \\end{bmatrix}
            \\begin{bmatrix} \\dot{\\phi}_L \\\\ \\dot{\\phi}_R \\end{bmatrix}

        where :math:`r` is the wheel radius and :math:`L` is the wheel distance.
        """
        linear_velocity = self.WHEEL_RADIUS / 2 * (right_speed_rads + left_speed_rads)
        angular_velocity = (
            self.WHEEL_RADIUS
            / 2
            * (
                2 / self.WHEEL_DISTANCE * right_speed_rads
                - 2 / self.WHEEL_DISTANCE * left_speed_rads
            )
        )
        return (linear_velocity, angular_velocity)

    def _update_odometry(
        self, pose: gtsam.Pose2, linear_velocity: float, angular_velocity: float, dt: float
    ) -> gtsam.Pose2:
        """Dead reckoning: the pose of the robot after driving with the body-frame
        velocity (v, w) for dt seconds, starting from `pose`.

        Parameters
        ----------
        pose : gtsam.Pose2
            The previous pose of the robot as a gtsam.Pose2 robot, contaning .x() and .y() in the
            world frame.
        linear_velocity : float
            The linear velocity of the robot in m/s in the robot's body frame.
        angular_velocity : float
            The angular velocity of the robot in rad/s in the robot's body frame.
        dt : float
            Infinitesimal timestep in s.

        Returns
        -------
        gtsam.Pose2
            the new pose of the robot as a gtsam.Pose2 object.

        Notes
        -----
        Uses one Euler step of the world-frame velocity of equation:
            x     <- x + v cos(theta) dt
            y     <- y + v sin(theta) dt
            theta <- theta + w dt
        with the heading theta at the START of the step, and wraps the new theta
        to [-pi, pi].
        """
        x = pose.x()
        y = pose.y()
        theta = pose.theta()

        x_new = x + linear_velocity * math.cos(theta) * dt
        y_new = y + linear_velocity * math.sin(theta) * dt
        theta_new = self._normalize_angle(theta + angular_velocity * dt)

        return gtsam.Pose2(x=x_new, y=y_new, theta=theta_new)
