import os
import sys

# add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from controllers.perception_controller.CONSTANTS import (
    CAMERA_PIXELS,
    CARD_SIZE_METERS,
    CARD_THICKNESS_METERS,
    FOCAL_LENGTH_PIXELS,
    SEAT_INFO,
)
from controllers.perception_controller.pinhole_camera import (
    find_viewing_distance,
    find_viewing_location_world,
)


def test_find_viewing_distance():
    target = 0.15
    result = find_viewing_distance(CARD_SIZE_METERS, FOCAL_LENGTH_PIXELS, CAMERA_PIXELS)
    print(f"Result: {result}, Target: {target}")
    try:
        np.testing.assert_allclose(result, target, rtol=1e-7, atol=1e-10)
        return True
    except AssertionError:
        return False


def test_find_viewing_location_world():
    translation, normal_vector = SEAT_INFO["0"]  # seat 0: card facing north (+y)
    viewing_distance = 0.15
    viewing_location_world, yaw = find_viewing_location_world(
        translation, normal_vector, viewing_distance, CARD_THICKNESS_METERS
    )
    target_viewing_location_world = np.array([-0.36, 0.464])
    target_yaw = -np.pi / 2
    print(f"Viewing Location World: {viewing_location_world}, Yaw: {yaw}")
    print(
        f"Target Viewing Location World: {target_viewing_location_world}, Target Yaw: {target_yaw}"
    )
    try:
        np.testing.assert_allclose(
            viewing_location_world, target_viewing_location_world, rtol=1e-7, atol=1e-10
        )
        np.testing.assert_allclose(yaw, target_yaw, rtol=1e-7, atol=1e-10)
    except AssertionError:
        return False

    # seat 4: the stool was pushed out at an angle, so the card normal is not axis aligned
    translation, normal_vector = SEAT_INFO["4"]
    viewing_location_world, yaw = find_viewing_location_world(
        translation, normal_vector, viewing_distance, CARD_THICKNESS_METERS
    )
    target_viewing_location_world = translation + normal_vector * (
        viewing_distance + CARD_THICKNESS_METERS / 2
    )
    target_yaw = np.arctan2(-normal_vector[1], -normal_vector[0])
    print(f"Seat 4 Viewing Location World: {viewing_location_world}, Yaw: {yaw}")
    try:
        np.testing.assert_allclose(
            viewing_location_world, target_viewing_location_world, rtol=1e-7, atol=1e-10
        )
        np.testing.assert_allclose(yaw, target_yaw, rtol=1e-7, atol=1e-10)
        return True
    except AssertionError:
        return False
