"""
perception_controller.py: a "teleporting" test controller for Part 4 (STAFF PROVIDED).

Instead of driving, SUSHI-BOT is teleported (it is a Supervisor) to the viewing
pose of every seat card computed with your pinhole_camera.py, grabs one camera
frame there, classifies it with your ImageClassifier, and saves the frame to
inference_test_frames/.

To use it, set the `controller` field of SUSHI_BOT (WaiterEpuck) in
worlds/sushi_restaurant.wbt to "perception_controller". Switch it back to
"rrt_controller" for Part 5.
"""

import os

import cv2
import numpy as np
from controller import Supervisor
from networks import ImageClassifier
from pinhole_camera import get_all_viewing_locations_world

model_path = "Final_LeNet.pt"

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("seat_camera")
camera.enable(timestep)

viewing_locations_world = get_all_viewing_locations_world()
image_classifier = ImageClassifier(model_path=model_path)

output_folder = "inference_test_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

node = robot.getFromDef("SUSHI_BOT")
t_field = node.getField("translation")
r_field = node.getField("rotation")

for seat_index in viewing_locations_world:
    viewing_location_world, yaw = viewing_locations_world[seat_index]
    TARGET_X, TARGET_Y, TARGET_YAW = viewing_location_world[0], viewing_location_world[1], yaw

    # Teleport: set desired (x, y, yaw) and keep current z
    pos = t_field.getSFVec3f()
    pos[0], pos[1] = TARGET_X, TARGET_Y
    t_field.setSFVec3f(pos)
    r_field.setSFRotation([0, 0, 1, TARGET_YAW])
    node.resetPhysics()

    # allow camera/physics to settle at new pose
    for _ in range(3):
        robot.step(timestep)

    # grab one frame, run inference, and save the frame
    image = camera.getImage()
    w, h = camera.getWidth(), camera.getHeight()
    frame = np.frombuffer(image, np.uint8).reshape((h, w, 4))[:, :, :3]
    print(f"Seat {seat_index}: ", end="")
    image_classifier.inference(frame)
    cv2.imwrite(
        f"{output_folder}/inference_test_frame_{seat_index}.png", frame[:, :, ::-1]
    )  # to BGR for OpenCV

# keep the simulation alive (and the robot still) after the test
while robot.step(timestep) != -1:
    pass
