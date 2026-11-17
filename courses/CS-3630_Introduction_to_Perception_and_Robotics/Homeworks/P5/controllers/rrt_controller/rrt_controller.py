"""
rrt_controller.py: SUSHI-BOT's high-level controller. Handles mission planning,
RRT* path generation, and perception tasks.

- Defines the static environment (the restaurant map) for the planner.
- Defines the sequence of goal poses for the mission (one per seat to serve).
- Calls the RRT* planner to generate a path for each leg.
- Runs image classification inference on the seat card upon reaching each goal.
- Delegates path execution to the low_level_controller.
"""

import math
import os
import random
import sys
import time

import cv2
import gtsam
import numpy as np
from controller import Supervisor

# --- Path Setup ---
CONTROLLERS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # .../project5/controllers
if CONTROLLERS_DIR not in sys.path:
    sys.path.insert(0, CONTROLLERS_DIR)

try:
    from perception_controller.networks import ImageClassifier
except ImportError as e:
    print(f"ERROR: Could not import ImageClassifier: {e}")
    raise

try:
    import restaurant_map
    from environment import StaticEnvironment
    from rrt import RRTStar, smooth_path
except ImportError as e:
    print(f"ERROR: Could not import required library for the planner: {e}")
    raise

try:
    from low_level_controller import RobotController
except ImportError as e:
    print(f"ERROR: Could not import low_level_controller.py file: {e}")
    raise

# --- Perception Setup ---
path_to_model = "../perception_controller/Final_LeNet.pt"
image_classifier = ImageClassifier(model_path=path_to_model)
output_folder = "inference_test_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Mission Setup ---
# This is the mode the mission starts in. While it runs, press 1 (static) or 2
# (dynamic) with the 3D view selected to switch.
# "static":  the restaurant never changes while the robot works (the default).
# "dynamic": customers walk in and out and move stools during the run. The robot
#            watches its proximity sensors, and when something that is not on its
#            map gets in the way it stops and replans the rest of the leg with RRT*.
#            Each leg's goal is computed from where the target stool is when the
#            leg starts. See dynamic_navigation.py and controllers/restaurant_crowd.
SIMULATION_MODE = "static"
# The seats are visited in a new random order on every run, so the mission is
# not always 0, 1, ..., 9. Set SEAT_ORDER_SEED to the seed printed at the start
# of a run to replay that exact order, or SHUFFLE_SEAT_ORDER = False for 0 to 9.
SHUFFLE_SEAT_ORDER = True
SEAT_ORDER_SEED = None
# Experiment (Part 6): drive on the wheel-encoder odometry instead of the GPS and
# compass. The drift printed after every leg tells you how far off it gets.
DRIVE_ON_ODOMETRY = False

# --- Planner Setup ---
# The planner works in its own coordinate system ("planner units"): 50 units per
# meter, with the south-west corner of the dining room at (0, 0) so that every
# coordinate is positive. RRT* plans straight segments for the robot's position;
# smooth_path then rounds every corner that turns by less than SHARP_CORNER_ANGLE
# with an arc of radius FILLET_RADIUS (4 units = 8 cm). Sharper corners are kept:
# the robot stops there and rotates in place.
PLANNER_SCALE = 50.0
PLANNER_ORIGIN = np.array([restaurant_map.ROOM_X[0], restaurant_map.ROOM_Y[0]])
RRT_STAR_ITERATIONS = 1500
FILLET_RADIUS = 4.0
SHARP_CORNER_ANGLE = math.radians(60)
SAVE_PLANNER_PLOTS = True  # saves planner_plots/leg_<i>.png (tree + path) for your report
plot_folder = "planner_plots"


def world_to_planner(x, y):
    """Webots world coordinates (m) -> planner coordinates."""
    return ((x - PLANNER_ORIGIN[0]) * PLANNER_SCALE, (y - PLANNER_ORIGIN[1]) * PLANNER_SCALE)


def planner_to_world(points):
    """(N, 2) planner coordinates -> Webots world coordinates (m)."""
    return np.asarray(points, dtype=float) / PLANNER_SCALE + PLANNER_ORIGIN


def define_environment(polygons=None):
    """
    Creates and returns the static environment for the RRT* planner.
    Obstacles (counter island, stools, props) are defined in Webots coordinates
    in restaurant_map.py and then shifted/scaled into planner coordinates. Every
    obstacle is inflated by the robot radius plus a safety margin.
    polygons: obstacle polygons in Webots coordinates to use instead of the
    restaurant_map ones (the dynamic simulation passes the current map).
    """
    if polygons is None:
        polygons = [polygon for _, polygon in restaurant_map.OBSTACLES]
    width = (restaurant_map.ROOM_X[1] - restaurant_map.ROOM_X[0]) * PLANNER_SCALE
    height = (restaurant_map.ROOM_Y[1] - restaurant_map.ROOM_Y[0]) * PLANNER_SCALE
    shifted_obstacles = [[world_to_planner(x, y) for x, y in polygon] for polygon in polygons]
    inflation = (restaurant_map.ROBOT_RADIUS + restaurant_map.PLANNER_MARGIN) * PLANNER_SCALE
    env = StaticEnvironment(
        dimensions=(width, height), obstacle_vertices_list=shifted_obstacles, inflation=inflation
    )
    return env


def get_goal_sequence():
    """
    Returns the list of (seat number, gtsam.Pose2 goal) for the mission: the
    viewing pose of each seat card, in delivery order (shuffled on every run,
    see SHUFFLE_SEAT_ORDER). These are the poses you computed with the pinhole
    camera model in Part 2.
    """
    order = list(restaurant_map.MISSION_ORDER)
    if SHUFFLE_SEAT_ORDER:
        seed = SEAT_ORDER_SEED if SEAT_ORDER_SEED is not None else random.randrange(10**6)
        random.Random(seed).shuffle(order)
        print(f"Seat order this run: {order} (SEAT_ORDER_SEED = {seed})", flush=True)

    goal_poses_list = []
    for seat in order:
        x, y, theta = restaurant_map.SEAT_VIEWING_POSES[seat]
        goal_poses_list.append((seat, gtsam.Pose2(x, y, theta)))
    return goal_poses_list


def generate_rrt_path(start_pose, goal_pose, env, leg_name="leg", iterations=None):
    """
    Runs the RRT* planner to generate a path from a given start to a goal.
    start_pose and goal_pose are (x, y, theta) tuples in Webots coordinates
    (the planner only uses x and y: the robot turns in place to any heading);
    the returned path is the smoothed (N, 2) array of Webots (x, y) points.
    iterations defaults to RRT_STAR_ITERATIONS (replanning uses fewer).
    """
    iterations = RRT_STAR_ITERATIONS if iterations is None else iterations
    x1, y1 = start_pose[:2]
    x2, y2 = goal_pose[:2]

    shifted_start = world_to_planner(x1, y1)
    shifted_goal = world_to_planner(x2, y2)

    print(
        f"RRT*: planning {leg_name} ({iterations} iterations, this takes a few seconds)...",
        flush=True,
    )
    tic = time.time()
    rrt = RRTStar(env)
    rrt.set_start(shifted_start)
    rrt.run(goal=shifted_goal, nb_iteration=iterations, goal_rate=0.1)
    toc = time.time() - tic

    waypoints = rrt.extract_waypoints()
    if len(waypoints) == 0:
        print(f"Could not find a path ({len(rrt.nodes)} nodes, {toc:.1f} s).")
        return np.array([])

    first = next((i for i, c in enumerate(rrt.cost_history) if c < float("inf")), None)
    print(
        f"RRT*: {len(rrt.nodes)} nodes in {toc:.1f} s | first solution "
        f"{rrt.cost_history[first] / PLANNER_SCALE:.2f} m (iteration {first + 1}) -> "
        f"final {rrt.path_cost() / PLANNER_SCALE:.2f} m"
    )

    if tuple(waypoints[-1]) != tuple(shifted_goal):
        waypoints = np.vstack(
            [waypoints, shifted_goal]
        )  # make sure the path ends exactly on the goal
    smoothed = smooth_path(waypoints, env, radius=FILLET_RADIUS, sharp_angle=SHARP_CORNER_ANGLE)

    if SAVE_PLANNER_PLOTS:
        save_planner_plot(rrt, env, leg_name, smoothed)

    return planner_to_world(smoothed)


def save_planner_plot(rrt, env, leg_name, smoothed=None):
    """
    Saves a picture of the RRT* tree, its path (red) and the smoothed path (green), in planner
    coordinates.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(plot_folder, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        env.plot(ax)
        rrt.plot(ax, smoothed=smoothed)
        ax.set_title(f"{leg_name}: RRT* path length {rrt.path_cost() / PLANNER_SCALE:.2f} m")
        fig.savefig(os.path.join(plot_folder, f"{leg_name}.png"), dpi=90, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"(could not save planner plot: {e})")


def setup_simulation(robot_instance, controller, env, goal_poses_list):
    """
    Sets up the static/dynamic switch (keys 1 and 2, starting in SIMULATION_MODE).
    In the dynamic simulation the customers come in and the robot watches its
    proximity sensors while it drives. Returns a goal sequence whose goals are
    computed from where each stool is when its leg starts (in the static
    simulation the stools never move, so these are the usual viewing poses).
    """
    import dynamic_restaurant
    from dynamic_navigation import (
        REPLAN_ITERATIONS,
        DynamicGoalSequence,
        DynamicNavigator,
        SimulationModeSwitch,
    )

    def plan_path(start, goal, environment, name):
        return generate_rrt_path(
            start, goal, environment, leg_name=name, iterations=REPLAN_ITERATIONS
        )

    navigator = DynamicNavigator(
        robot_instance, controller, define_environment, plan_path, world_to_planner
    )
    mode_switch = SimulationModeSwitch(robot_instance, navigator, SIMULATION_MODE)
    controller.path_monitor = mode_switch
    return DynamicGoalSequence(
        robot_instance,
        env,
        [seat for seat, _ in goal_poses_list],
        navigator,
        dynamic_restaurant.all_obstacles,
        mode_switch,
    )


def run_inference_task(robot_instance, timestep, frame_index):
    """
    Activates the camera, captures a frame of the seat card, runs inference,
    and saves the image. Returns the predicted character.
    """
    try:
        camera = robot_instance.getDevice("seat_camera")
        camera.enable(timestep)

        # Allow camera to stabilize
        for _ in range(3):
            robot_instance.step(timestep)

        image = camera.getImage()
        w, h = camera.getWidth(), camera.getHeight()
        frame = np.frombuffer(image, np.uint8).reshape((h, w, 4))[:, :, :3]

        # Run inference (logic is inside the classifier class)
        character = image_classifier.inference(frame)

        # Save image (convert from RGB to BGR for OpenCV)
        save_path = f"{output_folder}/inference_test_frame_{frame_index}.png"
        cv2.imwrite(save_path, frame[:, :, ::-1])
        print(f"Inference frame saved to {save_path}")

        camera.disable()
        return character

    except Exception as e:
        print(f"ERROR during inference task: {e}")
        return None


def run_mission(robot_instance):
    """
    Main function to initialize and run the robot's mission: starting at the
    drink station, deliver a drink to every seat (reading the seat card at each
    stop), then return to the drink station.
    """
    timestep = int(robot_instance.getBasicTimeStep())

    try:
        controller = RobotController(robot_instance, timestep)
        controller.DRIVE_ON_ODOMETRY = DRIVE_ON_ODOMETRY
    except Exception as e:
        print(f"ERROR: Could not initialize RobotController. {e}")
        return

    env = define_environment()
    goal_poses_list = get_goal_sequence()
    goal_poses_list = setup_simulation(robot_instance, controller, env, goal_poses_list)

    robot_instance.step(timestep)
    current_start_pose_gtsam = controller.get_current_pose()

    # --- main loop over all goal poses ---
    for i, (seat, goal_pose_gtsam) in enumerate(goal_poses_list):
        print(f"\n--- Delivery {i + 1} / {len(goal_poses_list)}: seat {seat} ---")
        # print(f"Start: ({current_start_pose_gtsam.x():.2f}, {current_start_pose_gtsam.y():.2f},
        # {current_start_pose_gtsam.theta():.2f})")
        # print(f"Goal:  ({goal_pose_gtsam.x():.2f}, {goal_pose_gtsam.y():.2f},
        # {goal_pose_gtsam.theta():.2f})")

        # Plan this leg with RRT*, drive it, then read the seat card
        start_config = (
            current_start_pose_gtsam.x(),
            current_start_pose_gtsam.y(),
            current_start_pose_gtsam.theta(),
        )
        goal_config = (goal_pose_gtsam.x(), goal_pose_gtsam.y(), goal_pose_gtsam.theta())

        path = generate_rrt_path(start_config, goal_config, env, leg_name=f"leg_{i}_seat_{seat}")

        if path.size == 0:
            print(f"Stopping controller. No path was generated for path {i + 1}")
            break

        controller.execute_path(path, goal_pose_gtsam)

        # print(f"Seat {seat} Goal Reached!")
        run_inference_task(robot_instance, timestep, frame_index=seat)
        current_start_pose_gtsam = controller.get_current_pose()

    # --- back to the drink station for the next round ---
    print("\n--- All seats served. Returning to the drink station. ---")
    start = controller.get_current_pose()
    home = gtsam.Pose2(*restaurant_map.RETURN_POSE)
    path = generate_rrt_path(
        (start.x(), start.y(), start.theta()),
        (home.x(), home.y(), home.theta()),
        env,
        leg_name="leg_return",
    )
    if path.size > 0:
        controller.execute_path(path, home)

    print("\n--- All Poses Completed. Stopping Robot. ---")
    controller.stop_motors()


def reset_robot_pose(robot_instance):
    """Puts SUSHI_BOT back at its start pose at the drink station, wherever the world was saved."""
    node = robot_instance.getSelf()
    x, y, yaw = restaurant_map.HOME_POSE
    node.getField("translation").setSFVec3f([x, y, 0])
    node.getField("rotation").setSFRotation([0, 0, 1, yaw])
    node.resetPhysics()


if __name__ == "__main__":
    my_robot = Supervisor()  # SUSHI_BOT is a Supervisor (the dynamic simulation reads the stools)
    reset_robot_pose(my_robot)
    run_mission(my_robot)
