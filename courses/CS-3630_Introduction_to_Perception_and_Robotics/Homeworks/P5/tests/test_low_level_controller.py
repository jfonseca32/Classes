import math
import os
import sys
from unittest.mock import MagicMock, patch

import gtsam
import numpy as np

# add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock the entire 'controller' module to avoid import errors when running
# outside of the Webots environment.
mock_controller_module = MagicMock()
mock_robot_class = MagicMock()
mock_device = MagicMock()
mock_controller_module.Robot = mock_robot_class
mock_controller_module.GPS = MagicMock()
mock_controller_module.Compass = MagicMock()
mock_robot_instance = mock_robot_class()
mock_robot_instance.getDevice.return_value = mock_device

# Patch our mocks in when importing RobotController
with patch.dict("sys.modules", {"controller": mock_controller_module}):
    from controllers.rrt_controller.low_level_controller import RobotController


def _controller():
    return RobotController(robot_instance=mock_robot_instance, timestep=0)


def test_inverse_kinematics():
    """
    Inverse kinematics (Equation 5.12) with the e-puck's wheels:
    straight driving, and turning in place (Equation 5.11).
    """
    controller = _controller()
    r, L = controller.WHEEL_RADIUS, controller.WHEEL_DISTANCE
    try:
        left, right = controller._inverse_kinematics(0.1, 0.0)
        assert abs(left - 0.1 / r) < 1e-9 and abs(right - 0.1 / r) < 1e-9, (
            f"driving straight: expected both wheels at {0.1 / r:.3f} rad/s, got {left}, {right}"
        )
        left, right = controller._inverse_kinematics(0.0, 1.0)
        assert abs(left + (L / 2) / r) < 1e-9 and abs(right - (L / 2) / r) < 1e-9, (
            f"turning in place: expected {-(L / 2) / r:.3f} and {(L / 2) / r:.3f} rad/s, "
            f"got {left}, {right}"
        )
        return True
    except AssertionError as e:
        print(e)
        return False


def test_forward_kinematics():
    """
    Forward kinematics (Equation 5.13) undoes the inverse kinematics.
    """
    controller = _controller()
    try:
        for v, w in [(0.1, 0.0), (0.0, 2.0), (0.08, -1.3)]:
            v2, w2 = controller._forward_kinematics(*controller._inverse_kinematics(v, w))
            assert abs(v2 - v) < 1e-9 and abs(w2 - w) < 1e-9, (
                f"expected ({v}, {w}), got ({v2}, {w2})"
            )
        return True
    except AssertionError as e:
        print(e)
        return False


def test_odometry():
    """
    One Euler step of Equation (5.14), and the wrap of theta to [-pi, pi].
    """
    controller = _controller()
    try:
        p = controller._update_odometry(gtsam.Pose2(1.0, 2.0, math.pi / 2), 0.1, 0.0, 2.0)
        assert np.allclose([p.x(), p.y(), p.theta()], [1.0, 2.2, math.pi / 2]), (
            f"driving north from (1, 2) for 0.2 m: got ({p.x():.3f}, {p.y():.3f}, {p.theta():.3f})"
        )
        p = controller._update_odometry(gtsam.Pose2(0.0, 0.0, 3.0), 0.0, 1.0, 0.5)
        assert abs(p.theta() - (3.5 - 2 * math.pi)) < 1e-6, (
            f"expected theta {3.5 - 2 * math.pi:.3f}, got {p.theta():.3f}"
        )
        return True
    except AssertionError as e:
        print(e)
        return False


def test_heading_error():
    """
    The heading error is positive to the left and wraps around +-pi.
    """
    controller = _controller()
    try:
        err = controller._heading_error(gtsam.Pose2(0, 0, 0), np.array([0.1, 0.1]))
        assert abs(err - math.pi / 4) < 1e-9, (
            f"target 45 deg to the left: expected {math.pi / 4:.3f}, got {err}"
        )
        pose = gtsam.Pose2(0, 0, math.radians(170))
        target = np.array([math.cos(math.radians(-170)), math.sin(math.radians(-170))])
        err = controller._heading_error(pose, target)
        assert abs(err - math.radians(20)) < 1e-9, (
            f"facing 170 deg, target at -170 deg: expected +20 deg, got {math.degrees(err):.1f}"
        )
        return True
    except AssertionError as e:
        print(e)
        return False


def test_pure_pursuit():
    """
    Tests the pure pursuit calculation when the target is directly ahead and to the left.
    """
    controller = _controller()
    controller.FORWARD_SPEED = 0.1
    try:
        linear_v, angular_v = controller._calculate_pure_pursuit_velocities(
            gtsam.Pose2(0, 0, 0), np.array([0.2, 0.0])
        )
        assert abs(linear_v - 0.1) < 1e-9 and abs(angular_v) < 1e-9, (
            f"straight ahead: got ({linear_v}, {angular_v})"
        )
        linear_v, angular_v = controller._calculate_pure_pursuit_velocities(
            gtsam.Pose2(0, 0, 0), np.array([0.1, 0.1])
        )
        expected = 2 * 0.1 * math.sin(math.pi / 4) / math.hypot(0.1, 0.1)
        assert abs(angular_v - expected) < 1e-9, (
            f"target to the left: expected w = {expected:.3f}, got {angular_v}"
        )
        return True
    except AssertionError as e:
        print(e)
        return False


if __name__ == "__main__":
    print("Passed test_inverse_kinematics:", test_inverse_kinematics())
    print("Passed test_forward_kinematics:", test_forward_kinematics())
    print("Passed test_odometry:", test_odometry())
    print("Passed test_heading_error:", test_heading_error())
    print("Passed test_pure_pursuit:", test_pure_pursuit())
