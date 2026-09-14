"""Supplied Webots visualization controller for CS 3630 Project 2.

Students do not modify this file. The controller imports their completed
Project 2 module, passes each simulated item's sensor readings through the
student's inference and decision functions, and routes the item accordingly.
"""

import importlib.util
import math
import sys
from pathlib import Path

from controller import Supervisor

ITEM_NAMES = (
    "3D printed product",
    "structural bracing",
    "PCB board",
    "wing component",
    "motor",
)

# Sensor encoding matches the assignment enums:
# thickness: thin=0, medium=1, thick=2
# rigidity: rigid=0, flexible=1
# material: plastic=0, metal=1, ceramic=2
SENSOR_OBSERVATIONS = (
    (150.0, 0, 1, 0),
    (820.0, 2, 0, 1),
    (50.0, 1, 0, 2),
    (250.0, 1, 1, 0),
    (600.0, 1, 1, 1),
)

CHANNEL_NAMES = {
    0: "PRINTED BIN",
    1: "STRUCTURAL BIN",
    2: "BOARD BIN",
    3: "WING BIN",
    4: "MOTOR BIN",
}

# ABB IRB 4600/40 poses, ordered A through F. They were solved against the
# world-space pickup and bin coordinates with the claw kept vertically down.
HOME_POSE = (2.0087, -1.0407, 0.4332, 0.0822, 2.0900, 0.0464)
PICK_POSE = (1.1933, -0.5862, 1.0336, 0.7428, 1.2297, -0.2958)
LIFT_POSE = (0.9129, -0.9421, 0.6235, 0.6367, 1.8280, 0.1872)
CHANNEL_POSES = {
    0: (-0.5557, -0.0426, 0.6774, -0.3097, 0.9575, 0.1811),
    1: (-0.2127, -0.3029, 0.8743, -0.1424, 1.0030, 0.0766),
    2: (0.1197, -0.3069, 0.8739, 0.0806, 1.0042, -0.0431),
    3: (0.4509, -0.0877, 0.7124, 0.2635, 0.9612, -0.1524),
    4: (0.7778, 0.2394, 0.3703, 0.3211, 0.9832, -0.1811),
}
CHANNEL_LANDINGS = {
    0: [0.92, -1.65, 0.38],
    1: [1.00, -1.10, 0.38],
    2: [1.04, -0.55, 0.38],
    3: [1.00, 0.00, 0.38],
    4: [0.92, 0.55, 0.38],
}

PICKUP_X = -0.36
BELT_SPEED = 0.34
QUEUE_GAP = 0.57
OPEN_CLAW = (0.42, -0.42)
CLOSED_CLAW = (0.05, -0.05)


def argument_value(flag):
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f"{flag} requires a file path.")
    return sys.argv[index + 1]


def find_student_code():
    """Return an explicit module path or Project 2's student-facing main.py."""
    configured = argument_value("--student-code")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    student_module = Path(__file__).resolve().parents[3] / "main.py"
    if not student_module.is_file():
        raise RuntimeError(
            f"Project 2 main.py was not found beside the assignment notebooks: {student_module}."
        )
    return student_module


def import_student_code(path):
    """Import the same student work used by the Jupyter notebook."""
    if not path.is_file():
        raise RuntimeError(f"Student module does not exist: {path}")
    module_folder = str(path.parent)
    if module_folder not in sys.path:
        # Keep sibling imports available if the assignment module uses helpers.
        sys.path.insert(0, module_folder)
    spec = importlib.util.spec_from_file_location("cs3630_project2_student", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import specification for {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = ("bayes_given_all_sensors", "make_decision")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"{path.name} is missing required callable(s): {', '.join(missing)}")
    return module


class SortingDemo:
    def __init__(self):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        self.arm_motors = [self.robot.getDevice(f"{letter} motor") for letter in "ABCDEF"]
        for motor in self.arm_motors:
            motor.setVelocity(2.0)
        self.left_claw = self.robot.getDevice("left_claw_motor")
        self.right_claw = self.robot.getDevice("right_claw_motor")

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        self.self_node = self.robot.getSelf()
        link6_joint = self.self_node.getFromProtoDef("LINK6_TRANSFORM")
        if link6_joint is None:
            raise RuntimeError("The ABB arm is missing its LINK6_TRANSFORM node.")
        self.link6_node = link6_joint.getField("endPoint").getSFNode()
        self.items = [self.robot.getFromDef(f"ITEM_{i}") for i in range(5)]
        if any(item is None for item in self.items):
            raise RuntimeError("The Project 2 world is missing one or more ITEM nodes.")

        self.translation_fields = [item.getField("translation") for item in self.items]
        self.initial_positions = [field.getSFVec3f() for field in self.translation_fields]
        self.current_index = 0
        self.state = "BELT"
        self.state_started = self.robot.getTime()
        self.selected_action = None
        self.selected_channel = None
        self.carrying = False
        self.auto_demo = "--auto-demo" in sys.argv
        self.integration_test = "--integration-test" in sys.argv
        self.manual_mode = "--manual" in sys.argv
        self.student_code = None
        self.student_code_path = None
        self.integration_error = None
        self.last_observation = None
        self.last_posteriors = None
        self.last_joint_pose = None

        if not self.auto_demo and not self.manual_mode:
            try:
                self.student_code_path = find_student_code()
                self.student_code = import_student_code(self.student_code_path)
            except Exception as error:
                self.integration_error = str(error)

        self.command_pose(HOME_POSE)
        self.set_claw(opened=True)
        self.update_overlay("Conveyor starting")
        print("CS 3630 Project 2 Webots visualization ready.")
        if self.student_code is not None:
            print(f"Loaded student Project 2 code: {self.student_code_path}")
            print("Webots will call bayes_given_all_sensors() and make_decision() for each item.")
        elif self.integration_error:
            print(f"STUDENT CODE ERROR: {self.integration_error}")
            print("Manual keys 0-4 remain available for troubleshooting.")
        elif self.manual_mode:
            print("Manual mode enabled. Enter actions with keys 0 through 4.")
        print("Press R at any time to reset the simulation.")
        if self.auto_demo:
            print("Self-test mode enabled: expected channels will be selected automatically.")
        elif self.integration_test:
            print(
                "Integration-test mode enabled: the temporary student module will drive all routes."
            )

    def command_pose(self, pose):
        for motor, value in zip(self.arm_motors, pose, strict=False):
            motor.setPosition(value)

    def set_claw(self, opened):
        targets = OPEN_CLAW if opened else CLOSED_CLAW
        self.left_claw.setPosition(targets[0])
        self.right_claw.setPosition(targets[1])

    def transition(self, state):
        self.state = state
        self.state_started = self.robot.getTime()

    def elapsed(self):
        return self.robot.getTime() - self.state_started

    def current_item_name(self):
        if self.current_index >= len(ITEM_NAMES):
            return "queue complete"
        return ITEM_NAMES[self.current_index]

    def update_overlay(self, _detail):
        """Retain state-transition call sites without drawing viewport text."""
        return

    def move_queue(self):
        # Advance the unsorted portion of the queue together. The current item
        # stops at the pickup station while later items preserve their spacing.
        current_position = self.translation_fields[self.current_index].getSFVec3f()
        next_x = min(PICKUP_X, current_position[0] + BELT_SPEED * self.dt)
        delta = next_x - current_position[0]
        for index in range(self.current_index, len(self.items)):
            position = self.translation_fields[index].getSFVec3f()
            position[0] += delta
            self.translation_fields[index].setSFVec3f(position)

        if next_x >= PICKUP_X:
            self.last_observation = SENSOR_OBSERVATIONS[self.current_index]
            self.last_posteriors = None
            self.transition("WAITING")
            self.update_overlay("Item at pickup; preparing sensor readings for the student module")
            print(f"{self.current_item_name()} is ready at the sensor station.")

    def run_student_decision(self):
        if self.student_code is None:
            self.transition("ERROR")
            detail = self.integration_error or "Student-code integration is unavailable."
            self.update_overlay(f"STUDENT CODE ERROR: {detail}")
            return

        observation = SENSOR_OBSERVATIONS[self.current_index]
        try:
            raw_posteriors = self.student_code.bayes_given_all_sensors(*observation)
            posteriors = [float(value) for value in raw_posteriors]
            if len(posteriors) != 5 or not all(math.isfinite(value) for value in posteriors):
                raise ValueError("bayes_given_all_sensors() must return five finite values")
            action = int(self.student_code.make_decision(posteriors))
            if action not in CHANNEL_NAMES:
                raise ValueError(
                    f"make_decision() returned {action}; expected an integer from 0 through 4"
                )
        except Exception as error:
            self.integration_error = f"{type(error).__name__}: {error}"
            self.transition("ERROR")
            self.update_overlay(f"STUDENT CODE ERROR: {self.integration_error}")
            print(
                f"STUDENT CODE ERROR while processing {self.current_item_name()}: "
                f"{self.integration_error}"
            )
            return

        self.last_posteriors = posteriors
        posterior_text = ", ".join(f"{value:.3f}" for value in posteriors)
        print(
            f"Sensors for {self.current_item_name()}: weight={observation[0]:.1f}, "
            f"thickness={observation[1]}, rigidity={observation[2]}, material={observation[3]}"
        )
        print(f"Student posteriors: [{posterior_text}]")
        self.select_action(action, decision_detail=f"student posteriors [{posterior_text}]")

    def read_keyboard(self):
        key = self.keyboard.getKey()
        while key != -1:
            if key in (ord("R"), ord("r")):
                self.robot.simulationReset()
                return
            if self.state in ("WAITING", "ERROR") and key in tuple(
                ord(str(value)) for value in range(5)
            ):
                self.select_action(key - ord("0"))
                return
            key = self.keyboard.getKey()

    def select_action(self, action, decision_detail=None):
        self.selected_action = action
        # Channel labels deliberately match the notebook's action values.
        self.selected_channel = action
        pose = CHANNEL_POSES[self.selected_channel]
        self.last_joint_pose = pose
        self.update_overlay(
            f"Action {action} -> Channel {self.selected_channel} "
            f"({CHANNEL_NAMES[self.selected_channel]})"
        )
        print(
            f"Received notebook action {action}; routing to matching "
            f"Channel {self.selected_channel} "
            f"({CHANNEL_NAMES[self.selected_channel]}) with ABB joints A-F="
            f"[{', '.join(f'{value:+.2f}' for value in pose)}] rad."
        )
        if decision_detail:
            print(f"Decision trace: {decision_detail}")
        self.command_pose(PICK_POSE)
        self.transition("MOVE_TO_PICK")

    def update_carried_item(self):
        if not self.carrying or self.link6_node is None:
            return
        wrist_position = self.link6_node.getPosition()
        wrist_orientation = self.link6_node.getOrientation()
        # The ABB hand slot and claw extend 0.24 m along link 6's local X axis.
        tip_position = [wrist_position[row] + 0.24 * wrist_orientation[3 * row] for row in range(3)]
        if self.auto_demo or self.integration_test:
            if not all(math.isfinite(value) for value in tip_position):
                raise RuntimeError(
                    f"Self-test detected a non-finite gripper position: {tip_position}"
                )
            if any(abs(value) > 10.0 for value in tip_position):
                raise RuntimeError(
                    f"Self-test detected an out-of-bounds gripper position: {tip_position}"
                )
        tip_position[2] -= 0.06
        self.translation_fields[self.current_index].setSFVec3f(tip_position)

    def verify_completed_routes(self):
        for index, field in enumerate(self.translation_fields):
            expected = CHANNEL_LANDINGS[index]
            actual = field.getSFVec3f()
            if any(
                abs(observed - target) > 1e-6
                for observed, target in zip(actual, expected, strict=False)
            ):
                raise RuntimeError(
                    f"Self-test route mismatch for {ITEM_NAMES[index]}: "
                    f"expected {expected}, observed {actual}"
                )
        print("PROJECT 2 WEBOTS SELF-TEST PASS: all five items reached their expected channels.")

    def run_manipulator(self):
        if self.state == "MOVE_TO_PICK" and self.elapsed() > 1.35:
            self.set_claw(opened=False)
            self.carrying = True
            self.transition("GRASP")
            self.update_overlay("Claw closing around the item")

        elif self.state == "GRASP" and self.elapsed() > 0.55:
            self.command_pose(LIFT_POSE)
            self.transition("LIFT")
            self.update_overlay("ABB shoulder and elbow lifting the item")

        elif self.state == "LIFT" and self.elapsed() > 1.05:
            self.command_pose(CHANNEL_POSES[self.selected_channel])
            self.transition("MOVE_TO_CHANNEL")
            self.update_overlay(
                f"Six-axis arm moving toward Channel {self.selected_channel}: "
                f"{CHANNEL_NAMES[self.selected_channel]}"
            )

        elif self.state == "MOVE_TO_CHANNEL" and self.elapsed() > 1.45:
            self.carrying = False
            self.translation_fields[self.current_index].setSFVec3f(
                CHANNEL_LANDINGS[self.selected_channel]
            )
            self.set_claw(opened=True)
            self.transition("RELEASE")
            self.update_overlay(f"Item released into Channel {self.selected_channel}")

        elif self.state == "RELEASE" and self.elapsed() > 0.70:
            self.command_pose(HOME_POSE)
            self.transition("RETURN_HOME")
            self.update_overlay("Arm returning home; next item advancing")

        elif self.state == "RETURN_HOME" and self.elapsed() > 1.20:
            self.current_index += 1
            self.selected_action = None
            self.selected_channel = None
            self.last_observation = None
            self.last_posteriors = None
            if self.current_index >= len(self.items):
                self.transition("COMPLETE")
                self.update_overlay("Sorting run complete")
                if self.auto_demo or self.integration_test:
                    self.verify_completed_routes()
                print("All five Project 2 items have been routed. Press R to restart.")
            else:
                # Re-establish the intended queue gap after the preceding item
                # was moved away from the belt by the manipulator.
                lead_x = self.translation_fields[self.current_index].getSFVec3f()[0]
                if lead_x > PICKUP_X - QUEUE_GAP:
                    position = self.translation_fields[self.current_index].getSFVec3f()
                    position[0] = PICKUP_X - QUEUE_GAP
                    self.translation_fields[self.current_index].setSFVec3f(position)
                self.transition("BELT")
                self.update_overlay("Next item moving toward the pickup station")

    def run(self):
        while self.robot.step(self.time_step) != -1:
            self.read_keyboard()
            if self.state == "BELT":
                self.move_queue()
            elif self.state == "WAITING" and self.auto_demo and self.elapsed() > 0.35:
                self.select_action(self.current_index)
            elif self.state == "WAITING" and self.elapsed() > 0.35:
                self.run_student_decision()
            elif self.state not in ("WAITING", "COMPLETE", "ERROR"):
                self.run_manipulator()
            self.update_carried_item()


if __name__ == "__main__":
    SortingDemo().run()
