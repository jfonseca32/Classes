import math
import os
import sys

import numpy as np

# add the project root and the rrt_controller folder to python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "controllers", "rrt_controller"))

from environment import StaticEnvironment  # noqa: E402 - requires the project path above
from rrt import fillet_corner, smooth_path  # noqa: E402 - requires the project path above


def test_fillet_corner():
    """
    A 90 deg left turn at (10, 0) with radius 4: the arc goes from (6, 0) to
    (10, 4) around the center (6, 4).
    """
    try:
        points, used = fillet_corner((0, 0), (10, 0), (10, 10), 4.0, 0.5)
        points = np.asarray(points, dtype=float)
        assert abs(used - 4.0) < 1e-9, f"expected radius 4, got {used}"
        assert np.allclose(points[0], (6, 0)), f"the arc must start at (6, 0), got {points[0]}"
        assert np.allclose(points[-1], (10, 4)), f"the arc must end at (10, 4), got {points[-1]}"
        assert np.allclose(np.linalg.norm(points - (6, 4), axis=1), 4.0), (
            "arc points must be 4 away from (6, 4)"
        )
        assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() <= 0.5 + 1e-9, (
            "arc points too far apart"
        )

        points, used = fillet_corner((0, 0), (10, 0), (10, -10), 4.0, 0.5)
        assert np.allclose(np.asarray(points)[-1], (10, -4)), (
            "right turn: the arc must end at (10, -4)"
        )

        points, used = fillet_corner((0, 0), (3, 0), (3, 20), 4.0, 0.5)
        assert abs(used - 1.5) < 1e-9, f"the first segment only has room for radius 1.5, got {used}"
        return True
    except AssertionError as e:
        print(e)
        return False


def test_smooth_path():
    """
    Gentle (45 deg) corners are rounded, the sharp (90 deg) one is kept, the
    endpoints do not move and the result is collision free.
    """
    try:
        env = StaticEnvironment((100, 60), [[(52, 0), (60, 0), (60, 6), (52, 6)]])
        waypoints = np.array([(10, 10), (50, 10), (70, 30), (70, 50), (40, 50)], dtype=float)
        out = np.asarray(smooth_path(waypoints, env, radius=8.0, sharp_angle=math.radians(60)))

        def d(q):
            return np.linalg.norm(out - np.asarray(q, float), axis=1).min()

        assert np.allclose(out[0], waypoints[0]) and np.allclose(out[-1], waypoints[-1]), (
            "endpoints moved"
        )
        assert env.is_path_free(out), "smoothed path collides"
        assert d((50, 10)) > 0.5 and d((70, 30)) > 0.5, "45 deg corners should be rounded"
        assert d((70, 50)) < 1e-9, "the 90 deg corner should be kept"
        return True
    except AssertionError as e:
        print(e)
        return False


if __name__ == "__main__":
    print("Passed test_fillet_corner:", test_fillet_corner())
    print("Passed test_smooth_path:", test_smooth_path())
