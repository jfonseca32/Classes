"""Supplied Webots visualization controller for CS 3630 Project 3.

Students do not modify this file. The Supervisor imports their completed
Project 3 module (main.py beside the notebook), asks it for the control tape,
the policy, or the factor-graph MPE, and drives the iRobot Create through the
nine-room Freiburg lab accordingly. Every transition is sampled from the
student's own prob_spec with a seeded numpy generator so a run can be
reproduced from the seed shown in the HUD and in the console.

Modes (keyboard first, then controllerArgs / CS3630_P3_CONTROLLER_ARGS):
    P  POLICY  roll out the policy returned by policy_iteration()
    C  CUSTOM  roll out get_custom_policy()
    T  TAPE    roll out create_custom_action_sequence()
    L  LIGHT   park in a room and print live light-sensor readings
    M  MPE     drive the tape with a hidden truth, then recover it with GTSAM
    1-9 start room (POLICY / CUSTOM / LIGHT), S next seed, R soft restart

Flags (controllerArgs in the world, or the CS3630_P3_CONTROLLER_ARGS
environment variable so tests never edit the shipped world):
    --mode policy|custom|tape|light|mpe   --start 1..9   --seed 3630
    --steps 8 (POLICY/CUSTOM)   --speed 1.2 (m/s)   --student-code PATH
    --auto-demo (no student code; drives 8->9->2->3->2->6->7->6->2 and
        cross-checks the world's DOOR_*/ROOM_* DEF nodes; exits 0/1)
    --integration-test (runs --mode to completion, checks the result, exits)
    --calibrate (visits every room and prints a light-level histogram)
    --export-image PATH (staff: save the 3D view when the run completes)
"""

import importlib.util
import math
import os
import shlex
import sys
import traceback
from pathlib import Path

from controller import Supervisor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vacuum_logic as vl  # noqa: E402

# Tests never edit the shipped world: extra arguments come from the environment.
sys.argv.extend(shlex.split(os.environ.get("CS3630_P3_CONTROLLER_ARGS", "")))

HUD_FONT = "Lucida Console"
HUD_SIZE = 0.06
HUD_COLOR = 0x101820
AUTO_START_DELAY = 5.0
WHEEL_MAX_VELOCITY = 16.0
LIGHT_REPORT_PERIOD = 1.0

# Per-mode names the student module must provide (checked when a run starts,
# reported per mode, never fatal for the world).
MODE_CONTRACT = {
    "POLICY": (
        "ROOMS",
        "ACTIONS",
        "prob_spec",
        "get_transition_prob",
        "generate_reward_table",
        "reward_function",
        "get_test_policy",
        "policy_iteration",
    ),
    "CUSTOM": (
        "ROOMS",
        "ACTIONS",
        "prob_spec",
        "get_transition_prob",
        "generate_reward_table",
        "reward_function",
        "get_custom_policy",
    ),
    "TAPE": (
        "ROOMS",
        "ACTIONS",
        "prob_spec",
        "get_transition_prob",
        "create_action_series",
        "create_custom_action_sequence",
    ),
    "LIGHT": (),
    "MPE": (
        "ROOMS",
        "ACTIONS",
        "LIGHT_LEVELS",
        "prob_spec",
        "get_transition_prob",
        "create_action_series",
        "create_custom_action_sequence",
        "create_factor_graph",
        "GTSAM_MPE",
        "get_kitchen_start_prior",
    ),
}
CALLABLE_NAMES = {
    "get_transition_prob",
    "generate_reward_table",
    "reward_function",
    "get_test_policy",
    "policy_iteration",
    "get_custom_policy",
    "create_action_series",
    "create_custom_action_sequence",
    "create_factor_graph",
    "GTSAM_MPE",
    "get_kitchen_start_prior",
}

# Recorded seed-3630 results for --integration-test, taken against
# main_sols.py in Webots R2025a (WorldInfo.randomSeed 0 makes the light-sensor
# noise reproducible, numpy default_rng(3630) makes the transitions
# reproducible). A None entry means "not recorded": the test then checks
# structural properties only (and MPE correct >= 6/8) and prints the observed
# sequence so it can be pasted here. Seed 3630's MPE run is deliberately kept:
# the true robot takes the 2 % branch of 6->2 into room 7 at step 6 and the
# tape's next actions are unavailable there, so only 5/8 rooms are
# recoverable; it is the worked example for the report question.
INTEGRATION_EXPECTED = {
    "POLICY": {"rooms": "8,9,2,3,3,3,3,3,3", "total_reward": 60.0},
    "CUSTOM": None,
    "TAPE": {"rooms": "3,2,6,7,6,7,7,7"},
    "MPE": {"rooms": "3,2,6,7,6,7,7,7", "inferred": "3,2,6,7,6,2,3,2", "correct": 5},
}


def log(text):
    print(text, flush=True)


def argument_value(flag):
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f"{flag} requires a file path.")
    return sys.argv[index + 1]


def find_student_code():
    """Return an explicit module path or Project 3's student-facing main.py."""
    configured = argument_value("--student-code")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    student_module = Path(__file__).resolve().parents[3] / "main.py"
    if not student_module.is_file():
        raise RuntimeError(
            f"Project 3 main.py was not found beside the assignment notebook: {student_module}."
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
    spec = importlib.util.spec_from_file_location("cs3630_project3_student", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import specification for {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def missing_contract(module, mode):
    """Names from MODE_CONTRACT[mode] that the module lacks (or are not callable)."""
    missing = []
    for name in MODE_CONTRACT[mode]:
        value = getattr(module, name, None)
        if value is None or (name in CALLABLE_NAMES and not callable(value)):
            missing.append(name)
    return missing


class StudentCodeError(RuntimeError):
    """Raised when a student function fails or returns an invalid value."""


class SelfTestFailure(RuntimeError):
    """Raised when --auto-demo / --integration-test / --calibrate checks fail."""


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class VacuumDemo:
    def __init__(self):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        self.args_error = None
        try:
            self.args = vl.parse_args(sys.argv[1:])
        except (ValueError, SystemExit) as error:
            self.args_error = str(error)
            self.args = vl.parse_args([])
        self.auto_demo = self.args.auto_demo
        self.integration_test = self.args.integration_test
        self.calibrate = self.args.calibrate
        self.self_test = self.auto_demo or self.integration_test or self.calibrate
        self.key_script = vl.parse_key_script(self.args.keys)
        self.scripted_keys = bool(self.key_script)
        self.speed = self.args.speed
        self.turn_rate = math.radians(vl.TURN_RATE_DEG)

        self.light_sensor = self.robot.getDevice("light sensor")
        self.light_sensor.enable(self.time_step)
        self.pen = self.robot.getDevice("pen")
        self.pen.write(False)
        self.wheels = [
            self.robot.getDevice("left wheel motor"),
            self.robot.getDevice("right wheel motor"),
        ]
        for wheel in self.wheels:
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0.0)
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        self.self_node = self.robot.getSelf()
        self.translation_field = self.self_node.getField("translation")
        self.rotation_field = self.self_node.getField("rotation")

        # Kinematic pose mirrored in Python and written to the fields each step.
        start = self.translation_field.getSFVec3f()
        self.position = (start[0], start[1])
        self.heading = 0.0
        self.path = []
        self.spin_remaining = 0.0
        self.moving = False

        # Light sensing.
        self.light_raw = 0.0
        self.dwell_samples = []

        # Run configuration and bookkeeping.
        self.mode = None
        self.pending_mode = self.args.mode or "POLICY"
        # Seeds: every mode has its own sequence base, base+1, base+2, ... so the first TAPE run is
        # always 3630, the second 3631, and MPE / POLICY also start at 3630 regardless of what ran
        # before. S (staff) shifts the whole sequence; R replays the last run with the same seed.
        self.base_seed = self.args.seed
        self.seed = self.base_seed
        self.seed_offset = 0
        self.runs_started = {name: 0 for name in vl.MODES}
        self.start_room = self.args.start or vl.DEFAULT_START[self.pending_mode]
        self.start_overridden = self.args.start is not None
        self.room = self.nearest_room(self.position)
        self.rng = None
        self.T = None
        self.policy = None
        self.values = None
        self.tape = []
        self.n_steps = 0
        self.k = 0
        self.rooms_visited = []
        self.measurements = []
        self.true_rooms = []
        self.total_reward = 0.0
        self.step_info = None
        self.after_sense = None
        self.light_last_report = 0.0
        self.mpe_summary = ""
        self.status = ""
        self.done_line = ""

        self.state = "LOADING"
        self.state_started = 0.0
        # A run starts by itself only when --mode was given (tests, staff); students press a key.
        self.auto_start_armed = self.args.mode is not None and not self.self_test
        self.auto_start_delay = 0.5
        self.script = None
        self.export_done = False

        self.student_code = None
        self.student_code_path = None
        self.integration_error = None
        if not self.auto_demo and not self.calibrate:
            try:
                self.student_code_path = find_student_code()
                self.student_code = import_student_code(self.student_code_path)
            except Exception as error:
                self.integration_error = f"{type(error).__name__}: {error}"

        code_text = (
            str(self.student_code_path) if self.student_code_path else "none (self-test mode)"
        )
        log(
            f"P3 WEBOTS READY | student code: {code_text} | mode={self.pending_mode} "
            f"seed={self.seed} start={self.start_room} "
            f"steps={self.run_length(self.pending_mode)}"
        )
        if self.student_code is not None:
            log(
                "Webots will call your main.py for the control tape, policy, transition model "
                "and MPE."
            )
        elif self.integration_error:
            log(f"STUDENT CODE ERROR: {self.integration_error}")
            log("LIGHT mode (L) still works without student code; fix main.py and press R.")
        if self.args_error:
            log(f"CONTROLLER ARGUMENT ERROR: {self.args_error} (using defaults)")
        if self.auto_demo:
            log("Self-test mode enabled: driving the demo route without student code.")
        elif self.integration_test:
            log(
                f"Integration-test mode enabled: running {self.pending_mode} with seed {self.seed}."
            )
        elif self.calibrate:
            log("Calibration mode enabled: visiting every room to histogram the light sensor.")
        else:
            log(
                "Keys: P policy  C custom policy  T tape  L light  M mpe  1-9 start room  R "
                "replay  (a mode key again = next seed)"
            )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    # Supervisor.simulationReset() (used by R to wipe the Pen trail) rewinds the simulation
    # clock; keep our own clock monotonic so every elapsed-time check survives a restart.
    _time_offset = 0.0
    _last_raw_time = 0.0
    reset_pending = False
    after_reset_mode = None
    keys_down = frozenset()  # keys reported by Webots on the previous step (for edge detection)
    replay_seed = None  # set by R: the next run reuses this seed instead of the next one
    last_mode = None

    def now(self):
        raw = self.robot.getTime()
        if raw < self._last_raw_time:
            self._time_offset += self._last_raw_time - raw
        self._last_raw_time = raw
        return raw + self._time_offset

    def transition(self, state):
        self.state = state
        self.state_started = self.now()

    def elapsed(self):
        return self.now() - self.state_started

    def quit(self, code):
        """Quit Webots after a few extra steps so the console output is drained."""
        self.update_hud()
        for _ in range(10):
            if self.robot.step(self.time_step) == -1:
                break
        self.robot.simulationQuit(code)

    def run_length(self, mode):
        if mode in ("POLICY", "CUSTOM"):
            return self.args.steps
        if mode in ("TAPE", "MPE"):
            return self.student_n() - 1
        return 0

    def student_n(self):
        value = getattr(self.student_code, "N", None) if self.student_code is not None else None
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = vl.N
        return value if value >= 2 else vl.N

    @staticmethod
    def nearest_room(position):
        return min(vl.ROOMS, key=lambda room: math.dist(vl.ROOM_CENTERS[room], position))

    def write_pose(self):
        self.translation_field.setSFVec3f([self.position[0], self.position[1], vl.ROBOT_Z])
        self.rotation_field.setSFRotation([0.0, 0.0, 1.0, self.heading])

    def teleport(self, room, heading=0.0):
        self.position = tuple(vl.ROOM_CENTERS[room])
        self.heading = heading
        self.room = room
        self.path = []
        self.spin_remaining = 0.0
        self.set_moving(False)
        self.write_pose()

    def set_wheels(self, left, right):
        self.wheels[0].setVelocity(max(-WHEEL_MAX_VELOCITY, min(WHEEL_MAX_VELOCITY, left)))
        self.wheels[1].setVelocity(max(-WHEEL_MAX_VELOCITY, min(WHEEL_MAX_VELOCITY, right)))

    def set_moving(self, moving):
        if moving != self.moving:
            self.moving = moving
            self.pen.write(bool(moving))
        if not moving:
            self.set_wheels(0.0, 0.0)

    # ------------------------------------------------------------------
    # Kinematic motion
    # ------------------------------------------------------------------
    def begin_motion(self, points):
        """Follow `points` (world x, y) from the current position."""
        self.path = [tuple(point) for point in points]
        self.spin_remaining = 0.0

    def begin_spin(self, radians):
        self.path = []
        self.spin_remaining = float(radians)

    def motion_active(self):
        return bool(self.path) or self.spin_remaining > 0.0

    def update_motion(self):
        wheel_speed = self.speed / vl.WHEEL_RADIUS
        if self.spin_remaining > 0.0:
            step = min(self.spin_remaining, self.turn_rate * self.dt)
            self.heading = wrap_angle(self.heading + step)
            self.spin_remaining -= step
            self.write_pose()
            self.set_wheels(-wheel_speed, wheel_speed)
            if self.spin_remaining <= 0.0:
                self.set_moving(False)
            return
        if not self.path:
            self.set_moving(False)
            return
        target = self.path[0]
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            self.path.pop(0)
            if not self.path:
                self.set_moving(False)
            return
        desired = math.atan2(dy, dx)
        error = wrap_angle(desired - self.heading)
        if abs(error) > 1e-3:
            step = max(-self.turn_rate * self.dt, min(self.turn_rate * self.dt, error))
            self.heading = wrap_angle(self.heading + step)
            self.write_pose()
            self.set_wheels(
                -wheel_speed if step > 0 else wheel_speed, wheel_speed if step > 0 else -wheel_speed
            )
            return
        self.heading = desired
        advance = self.speed * self.dt
        if advance >= distance:
            self.position = target
            self.path.pop(0)
        else:
            self.position = (
                self.position[0] + advance * math.cos(desired),
                self.position[1] + advance * math.sin(desired),
            )
        self.set_moving(True)
        self.write_pose()
        self.set_wheels(wheel_speed, wheel_speed)
        if not self.path:
            self.set_moving(False)

    # Generator helpers used by the scripted self-tests.
    def drive(self, points):
        self.begin_motion(points)
        while self.motion_active():
            self.update_motion()
            yield

    def dwell(self, seconds):
        self.dwell_samples = []
        started = self.now()
        while self.now() - started < seconds:
            self.dwell_samples.append(self.light_raw)
            yield

    # ------------------------------------------------------------------
    # Light sensing
    # ------------------------------------------------------------------
    def sample_light(self):
        self.light_raw = float(self.light_sensor.getValue())

    def dwell_mean(self):
        if not self.dwell_samples:
            return self.light_raw
        return sum(self.dwell_samples) / len(self.dwell_samples)

    def finish_measurement(self):
        """One measurement z = the last sample of the dwell (one noisy read)."""
        raw = self.dwell_samples[-1] if self.dwell_samples else self.light_raw
        return raw, vl.discretize(raw)

    # ------------------------------------------------------------------
    # Student-code calls
    # ------------------------------------------------------------------
    def call_student(self, name, *args, **kwargs):
        function = getattr(self.student_code, name, None)
        if not callable(function):
            raise StudentCodeError(f"{name}() is missing from main.py")
        try:
            return function(*args, **kwargs)
        except Exception as error:
            raise StudentCodeError(f"{name}() raised {type(error).__name__}: {error}") from error

    def validate_constants(self):
        rooms = list(getattr(self.student_code, "ROOMS", []))
        actions = list(getattr(self.student_code, "ACTIONS", []))
        if rooms != vl.ROOMS:
            raise StudentCodeError(f"ROOMS must be {vl.ROOMS}; main.py has {rooms}")
        if actions != vl.ACTIONS:
            raise StudentCodeError(
                f"ACTIONS order differs from the canonical Project 3 order: {actions}"
            )

    def load_transition_model(self):
        import numpy as np

        prob_spec = self.student_code.prob_spec
        if not isinstance(prob_spec, str) or "FILL IN" in prob_spec:
            raise StudentCodeError(
                "prob_spec in main.py still contains the 'FILL IN THESE TWO' placeholder: fill "
                "in the two entries for actions 2->1 and 1->2 from room 1 (notebook Section "
                "3.2)"
            )
        T = self.call_student("get_transition_prob", prob_spec)
        try:
            T = np.asarray(T, dtype=float)
        except (TypeError, ValueError) as error:
            raise StudentCodeError(
                f"get_transition_prob() must return a numeric array: {error}"
            ) from error
        if T.shape != (len(vl.ROOMS), len(vl.ACTIONS), len(vl.ROOMS)):
            raise StudentCodeError(
                f"get_transition_prob() must return shape (9, 16, 9); got {T.shape}"
            )
        if not np.all(np.isfinite(T)) or np.any(T < 0):
            raise StudentCodeError("get_transition_prob() contains negative or non-finite entries")
        self.T = T

    def load_reward_table(self):
        import numpy as np

        R = self.call_student("generate_reward_table", self.student_code.prob_spec)
        try:
            R = np.asarray(R, dtype=float)
        except (TypeError, ValueError) as error:
            raise StudentCodeError(
                f"generate_reward_table() must return a numeric array: {error}"
            ) from error
        if R.shape != (len(vl.ROOMS), len(vl.ACTIONS), len(vl.ROOMS)) or not np.all(np.isfinite(R)):
            raise StudentCodeError(
                f"generate_reward_table() must return a finite (9, 16, 9) array; got shape "
                f"{R.shape}"
            )
        return R

    @staticmethod
    def validate_policy(policy, source):
        try:
            policy = [int(a) for a in policy]
        except (TypeError, ValueError) as error:
            raise StudentCodeError(f"{source} must return a list of 9 action indices") from error
        if len(policy) != len(vl.ROOMS) or any(a < 0 or a >= len(vl.ACTIONS) for a in policy):
            raise StudentCodeError(f"{source} must return 9 action indices in 0..15; got {policy}")
        return policy

    def setup_policy_mode(self):
        import numpy as np

        self.load_transition_model()
        R = self.load_reward_table()
        if self.mode == "POLICY":
            initial = self.validate_policy(
                self.call_student("get_test_policy"), "get_test_policy()"
            )
            result = self.call_student("policy_iteration", R, self.T, pi=initial)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise StudentCodeError("policy_iteration() must return (policy, values)")
            policy, values = result
            self.policy = self.validate_policy(policy, "policy_iteration()")
            try:
                values = np.asarray(values, dtype=float).reshape(-1)
            except (TypeError, ValueError) as error:
                raise StudentCodeError("policy_iteration() values must be 9 numbers") from error
            if values.shape[0] != len(vl.ROOMS) or not np.all(np.isfinite(values)):
                raise StudentCodeError("policy_iteration() values must be 9 finite numbers")
            self.values = values
            source = "policy_iteration()"
        else:
            self.policy = self.validate_policy(
                self.call_student("get_custom_policy"), "get_custom_policy()"
            )
            self.values = None
            evaluate = getattr(self.student_code, "calculate_value_function", None)
            if callable(evaluate):
                try:
                    self.values = np.asarray(evaluate(self.policy, R, self.T), dtype=float).reshape(
                        -1
                    )
                except Exception:
                    self.values = None
            source = "get_custom_policy()"
        start_index = vl.ROOMS.index(self.start_room)
        value = float(self.values[start_index]) if self.values is not None else None
        log(vl.format_policy_line(source, self.policy, self.start_room, value))

    def setup_tape_mode(self):
        self.load_transition_model()
        n = self.student_n()
        A = self.call_student("create_action_series", "A", range(1, n))
        sequence = self.call_student("create_custom_action_sequence", A, n - 1)
        if not isinstance(sequence, dict) or len(sequence) != n - 1:
            raise StudentCodeError(
                f"create_custom_action_sequence(A, {n - 1}) must return a dict with {n - 1} entries"
            )
        try:
            tape = [sequence[A[k]] for k in range(1, n)]
        except (KeyError, TypeError):
            tape = list(sequence.values())
        bad = [a for a in tape if a not in vl.ACTIONS]
        if bad:
            raise StudentCodeError(
                f"create_custom_action_sequence() contains non-ACTIONS entries: {bad}"
            )
        self.tape = [str(a) for a in tape]
        log(f"TAPE from create_custom_action_sequence(): {' '.join(self.tape)}")
        if self.mode == "MPE":
            prior = self.call_student("get_kitchen_start_prior")
            if not isinstance(prior, str) or len(prior.split("/")) != len(vl.ROOMS):
                raise StudentCodeError(
                    "get_kitchen_start_prior() must return a 9-number spec string like "
                    "0/0/1/0/0/0/0/0/0"
                )
            self.prior = prior

    def step_reward(self, room, action, next_room):
        reward = self.call_student("reward_function", room, action, next_room)
        try:
            reward = float(reward)
        except (TypeError, ValueError) as error:
            raise StudentCodeError(
                f"reward_function({room!r}, {action!r}, {next_room!r}) must return a number"
            ) from error
        if not math.isfinite(reward):
            raise StudentCodeError("reward_function() returned a non-finite value")
        return reward

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------
    def start_run(self, mode):
        self.mode = mode
        self.pending_mode = mode
        if mode in ("TAPE", "MPE"):
            # The tape's first action is 3->2 and the MPE prior is get_kitchen_start_prior(), so
            # these modes always start in the kitchen, whatever start room was chosen.
            if self.start_room != "3":
                log(
                    f"{mode} mode always starts in the kitchen (room 3): the control tape "
                    f"begins with 3->2 and the MPE prior is the kitchen prior. Start room 1-9 "
                    f"applies to POLICY, CUSTOM and LIGHT."
                )
            self.start_room = "3"
        elif (
            not self.start_overridden
            and mode in ("POLICY", "CUSTOM")
            and self.start_room == "3"
            and self.args.start is None
        ):
            self.start_room = vl.DEFAULT_START[mode]
        self.k = 0
        self.total_reward = 0.0
        self.rooms_visited = []
        self.measurements = []
        self.true_rooms = []
        self.step_info = None
        self.mpe_summary = ""
        self.done_line = ""
        self.status = ""
        self.set_moving(False)

        if mode == "LIGHT":
            self.n_steps = 0
            self.last_mode = mode
            log(f"P3 RUN START | mode=LIGHT start={self.start_room}")
            self.light_last_report = -1.0
            points = (
                vl.waypoints(self.room, self.start_room)[1:] if self.room != self.start_room else []
            )
            self.begin_motion(points)
            self.transition("LIGHT")
            return

        if self.student_code is None:
            self.fail_run(self.integration_error or "student code is not loaded")
            return
        missing = missing_contract(self.student_code, mode)
        if missing:
            self.fail_run(f"{mode} mode needs {', '.join(missing)} in main.py")
            return

        import numpy as np

        if self.replay_seed is not None:
            self.seed, self.replay_seed = self.replay_seed, None
        else:
            self.seed = self.base_seed + self.seed_offset + self.runs_started[mode]
            self.runs_started[mode] += 1
        self.last_mode = mode
        self.rng = np.random.default_rng(self.seed)
        try:
            self.validate_constants()
            if mode in ("POLICY", "CUSTOM"):
                self.setup_policy_mode()
            else:
                self.setup_tape_mode()
        except StudentCodeError as error:
            self.fail_run(str(error))
            return

        self.n_steps = self.run_length(mode)
        if mode in ("TAPE", "MPE"):
            self.n_steps = len(self.tape)
        log(
            f"P3 RUN START | mode={mode} seed={self.seed} start={self.start_room} "
            f"steps={self.n_steps}"
        )
        self.teleport(self.start_room)
        self.rooms_visited = [self.start_room]
        self.true_rooms = [self.start_room]
        self.after_sense = "DECIDE"
        self.dwell_samples = []
        self.transition("SENSE")

    def fail_run(self, detail):
        self.integration_error = detail
        self.set_moving(False)
        log(f"STUDENT CODE ERROR: {detail}")
        self.transition("ERROR")
        if self.self_test:
            raise SelfTestFailure(detail)

    def decide(self):
        if self.k >= self.n_steps:
            self.finish_run()
            return
        room = self.room
        x = vl.ROOMS.index(room)
        if self.mode in ("POLICY", "CUSTOM"):
            action = vl.ACTIONS[self.policy[x]]
        else:
            action = self.tape[self.k]
        a = vl.ACTIONS.index(action)
        intended = vl.room_of_action_target(action)
        try:
            sampled_index, probability = vl.sample_next_room(self.rng, self.T, x, a)
        except ValueError as error:
            self.fail_run(f"get_transition_prob(): {error}")
            return
        sampled = vl.ROOMS[sampled_index]

        if not vl.action_available(action, room):
            probability = float(self.T[x, a, x]) if self.T[x, a].sum() > 0 else 1.0
            sampled = room
            if self.mode in ("POLICY", "CUSTOM"):
                # Under the notebook's model an action that is not a doorway from the current room
                # leaves the robot where it is; in the kitchen that is the optimal thing to do.
                again = " and collects the kitchen reward again" if room == "3" else ""
                log(
                    f"policy chose {action} in room {room}: not a doorway from room {room}, so "
                    f"the robot stays{again} (p={probability:.2f})"
                )
            else:
                log(
                    f"tape action {action} is not available from room {room}: robot stays "
                    f"(p={probability:.2f})"
                )
            self.begin_spin(2.0 * math.pi)
        elif sampled == room:
            log(f"attempted {action} but stayed in {room} (p={probability:.2f})")
            self.begin_motion(vl.stay_bump_points(room, action))
        else:
            self.begin_motion(vl.waypoints(room, sampled)[1:])
        self.step_info = {
            "k": self.k + 1,
            "room": room,
            "action": action,
            "intended": intended,
            "sampled": sampled,
            "p": probability,
        }
        self.transition("MOVING")

    def complete_step(self):
        raw, level = self.finish_measurement()
        info = self.step_info
        self.room = info["sampled"]
        self.k = info["k"]
        self.rooms_visited.append(self.room)
        self.true_rooms.append(self.room)
        self.measurements.append(level)
        info["raw"] = raw
        info["level"] = level
        if self.mode in ("POLICY", "CUSTOM"):
            try:
                reward = self.step_reward(info["room"], info["action"], self.room)
            except StudentCodeError as error:
                self.fail_run(str(error))
                return
            self.total_reward += reward
            info["reward"] = reward
            log(
                vl.format_step(
                    self.mode,
                    self.k,
                    self.n_steps,
                    info["room"],
                    info["action"],
                    info["intended"],
                    info["sampled"],
                    info["p"],
                    raw,
                    level,
                    reward,
                    self.total_reward,
                )
            )
        elif self.mode == "TAPE":
            log(
                vl.format_step(
                    "TAPE",
                    self.k,
                    self.n_steps,
                    info["room"],
                    info["action"],
                    info["intended"],
                    info["sampled"],
                    info["p"],
                    raw,
                    level,
                )
            )
        else:
            log(vl.format_mpe_step(self.k + 1, self.n_steps + 1, info["action"], level))
        self.transition("DECIDE")

    def finish_run(self):
        if self.mode == "MPE":
            self.transition("INFER")
            return
        rooms = self.rooms_visited
        if self.mode in ("POLICY", "CUSTOM"):
            self.done_line = vl.format_done(
                self.mode, self.seed, self.start_room, self.n_steps, rooms, self.total_reward
            )
        else:
            self.done_line = vl.format_done("TAPE", self.seed, self.start_room, self.n_steps, rooms)
        log(self.done_line)
        self.transition("COMPLETE")

    def infer(self):
        n = len(self.measurements)
        try:
            try:
                result = self.call_student(
                    "create_factor_graph",
                    n,
                    measurements=list(self.measurements),
                    actions=list(self.tape),
                    prior=self.prior,
                )
            except StudentCodeError as error:
                if "TypeError" in str(error) and "argument" in str(error):
                    raise StudentCodeError(
                        "create_factor_graph(N, measurements=..., actions=..., prior=...) "
                        "rejected keyword arguments; MPE mode needs the Fall 2026 signature, "
                        "not the old create_factor_graph(N)"
                    ) from error
                raise
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise StudentCodeError("create_factor_graph() must return (graph, X)")
            graph, X = result
            mpe = self.call_student("GTSAM_MPE", graph)
            inferred = []
            for k in range(1, n + 1):
                try:
                    inferred.append(vl.ROOMS[int(mpe[X[k][0]])])
                except Exception as error:
                    raise StudentCodeError(
                        f"could not decode ROOMS[mpe[X[{k}][0]]]: {type(error).__name__}: {error}"
                    ) from error
        except StudentCodeError as error:
            self.fail_run(str(error))
            return
        truth = self.true_rooms
        correct = sum(1 for t, i in zip(truth, inferred, strict=False) if t == i)
        header = (
            f"MPE RESULT seed={self.seed} | correct {correct}/{n} | true={','.join(truth)} "
            f"inferred={','.join(inferred)}"
        )
        # Model probability of both trajectories under the student's own factor graph, so the
        # prior-versus-likelihood trade-off behind a "wrong" inference is visible in one line.
        ratio_text = ""
        try:
            p_inferred = self.trajectory_probability(graph, X, inferred)
            p_true = self.trajectory_probability(graph, X, truth)
            ratio = f" ({p_inferred / p_true:.0f}x)" if p_true > 0 and p_inferred >= p_true else ""
            header += f" | P(inferred)={p_inferred:.2e} P(true)={p_true:.2e}{ratio}"
            ratio_text = f" | P(inferred)/P(true)={p_inferred / p_true:.0f}x" if p_true > 0 else ""
        except Exception as error:  # never let the diagnostic break the result
            log(f"(could not evaluate trajectory probabilities: {type(error).__name__}: {error})")
        log(header)
        log("[MPE]  k  action  z        true  inferred")
        for k in range(n):
            action = "start" if k == 0 else self.tape[k - 1]
            mark = "" if truth[k] == inferred[k] else "  <- mismatch"
            log(
                f"[MPE] {k + 1:>2}  {action:<6}  {self.measurements[k]:<7}  {truth[k]:<4}  "
                f"{inferred[k]}{mark}"
            )
        self.mpe_summary = (
            f"true {','.join(truth)} | inferred {','.join(inferred)} | correct "
            f"{correct}/{n}{ratio_text}"
        )
        self.mpe_correct = correct
        self.mpe_inferred = inferred
        self.done_line = header
        self.transition("COMPLETE")

    @staticmethod
    def trajectory_probability(graph, X, rooms):
        """
        Unnormalized model probability of a room sequence: the product of all factors in `graph`.
        """
        import gtsam

        values = gtsam.DiscreteValues()
        for k, room in enumerate(rooms, start=1):
            values[X[k][0]] = vl.ROOMS.index(room)
        return float(graph(values))

    def soft_restart(self, next_mode=None):
        """Stop the current run, wipe the floor trail, park in the start room.

        simulationReset() cleans the Pen's painted textures and rewinds the clock at the end of
        this step without restarting this controller; it also snaps the robot back to its
        world-file pose, so the teleport to the start room (and an optional mode start) is done
        on the next step, in the RESETTING state.
        """
        self.set_moving(False)
        self.mode = None
        self.auto_start_armed = False
        self.step_info = None
        self.mpe_summary = ""
        self.done_line = ""
        self.status = ""
        self.k = 0
        self.n_steps = 0
        self.total_reward = 0.0
        self.rooms_visited = []
        self.after_reset_mode = next_mode
        self.reset_pending = True
        self.robot.simulationReset()
        self.transition("RESETTING")

    def finish_restart(self):
        self.reset_pending = False
        self.teleport(self.start_room)
        self.transition("IDLE")
        if self.after_reset_mode is not None:
            mode, self.after_reset_mode = self.after_reset_mode, None
            self.start_run(mode)
        else:
            self.status = (
                f"Restarted in room {self.start_room}. Press P, C, T, L or M (1-9 changes the "
                f"start room)."
            )
            log(
                f"Restarted: trail cleared, robot parked in room {self.start_room}. Choose a "
                f"mode (P, C, T, L, M)."
            )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def read_keyboard(self):
        # Webots reports a key on every step while it is held, so a normal press would arrive
        # three or four times. Act only on keys that were not down on the previous step.
        down = set()
        key = self.keyboard.getKey()
        while key != -1:
            down.add(key & 0xFFFF)
            key = self.keyboard.getKey()
        for key in sorted(down - set(self.keys_down)):
            self.handle_key(key)
        self.keys_down = frozenset(down)

    def press_scripted_keys(self):
        while self.key_script and self.now() >= self.key_script[0][0]:
            _, key = self.key_script.pop(0)
            log(f"[KEYS] pressing {key} at t={self.now():.1f}s")
            self.handle_key(ord(key), scripted=True)

    def handle_key(self, key, scripted=False):
        if self.state in ("LOADING", "RESETTING", "SCRIPT") or (self.self_test and not scripted):
            return
        selectable = self.state in ("IDLE", "COMPLETE", "ERROR", "LIGHT")
        self.auto_start_armed = False  # any key means the user is choosing
        if key in (ord("R"), ord("r")):
            replay = self.mode or self.last_mode
            if replay is None or replay == "LIGHT":
                self.soft_restart(next_mode=replay)
            else:
                log(f"Replaying {replay} with seed {self.seed}.")
                self.replay_seed = self.seed
                self.soft_restart(next_mode=replay)
            return
        if key in (ord("S"), ord("s")):
            # Staff option: shift every mode's seed sequence by one.
            self.seed_offset += 1
            self.status = (
                f"Seed sequence shifted by +{self.seed_offset} (next new run of each mode uses "
                f"base+offset+count)"
            )
            log(f"Seed sequence shifted: offset is now +{self.seed_offset}")
            return
        if ord("1") <= key <= ord("9"):
            room = chr(key)
            self.start_room = room
            self.start_overridden = True
            if self.state == "LIGHT":
                self.begin_motion(vl.waypoints(self.room, room)[1:])
                log(f"[LIGHT] driving to room {room} ({vl.ROOM_NAMES[room]})")
            elif selectable:
                self.teleport(room)
                self.status = f"Start room set to {room} ({vl.ROOM_NAMES[room]}). Press P, C or L."
                log(f"Start room set to {room} ({vl.ROOM_NAMES[room]}).")
            else:
                # A run is in progress: stop it, clear the trail, park in the chosen room.
                log(f"Start room set to {room} ({vl.ROOM_NAMES[room]}); stopping the current run.")
                self.soft_restart()
            return
        mode = {
            ord("P"): "POLICY",
            ord("C"): "CUSTOM",
            ord("T"): "TAPE",
            ord("L"): "LIGHT",
            ord("M"): "MPE",
        }.get(key if key < ord("a") else key - 32)
        if mode is None:
            return
        if not selectable:
            # A run is in progress: restart straight into the requested mode.
            log(f"Stopping the current run and starting {mode} mode.")
            self.soft_restart(next_mode=mode)
            return
        self.start_run(mode)

    # ------------------------------------------------------------------
    # Scripted self-tests
    # ------------------------------------------------------------------
    def check_world_tables(self):
        for (a, b), (x, y) in vl.DOORWAYS.items():
            node = self.robot.getFromDef(f"DOOR_{a}_{b}")
            if node is None:
                raise SelfTestFailure(f"world is missing DEF DOOR_{a}_{b}")
            actual = node.getField("translation").getSFVec3f()
            if abs(actual[0] - x) > 1e-6 or abs(actual[1] - y) > 1e-6:
                raise SelfTestFailure(
                    f"DOOR_{a}_{b} is at {actual[:2]} but vacuum_logic says {(x, y)}"
                )
        for room, (x, y) in vl.ROOM_CENTERS.items():
            node = self.robot.getFromDef(f"ROOM_{room}")
            if node is None:
                raise SelfTestFailure(f"world is missing DEF ROOM_{room}")
            actual = node.getField("translation").getSFVec3f()
            if abs(actual[0] - x) > 1e-6 or abs(actual[1] - y) > 1e-6:
                raise SelfTestFailure(
                    f"ROOM_{room} is at {actual[:2]} but vacuum_logic says {(x, y)}"
                )
        log("World DEF nodes match vacuum_logic tables (8 doorways, 9 room centers).")

    def auto_demo_script(self):
        self.status = "SELF-TEST MODE (no student code)"
        self.check_world_tables()
        current = vl.DEMO_ROUTE[0]
        self.teleport(current)
        log(f"[DEMO] route {'->'.join(vl.DEMO_ROUTE)}")
        for room in vl.DEMO_ROUTE[1:]:
            yield from self.drive(vl.waypoints(current, room)[1:])
            self.room = room
            current = room
            yield from self.dwell(vl.DWELL_SECONDS)
            raw, level = self.finish_measurement()
            log(
                f"[DEMO] arrived {vl.room_label(room)} | light raw={raw:.2f} level={level} | "
                f"dwell mean={self.dwell_mean():.2f}"
            )
        actual = self.translation_field.getSFVec3f()
        expected = vl.ROOM_CENTERS[vl.DEMO_ROUTE[-1]]
        if (
            abs(actual[0] - expected[0]) > 1e-9
            or abs(actual[1] - expected[1]) > 1e-9
            or abs(actual[2] - vl.ROBOT_Z) > 1e-9
        ):
            raise SelfTestFailure(
                f"final translation {actual} != room {vl.DEMO_ROUTE[-1]} center {expected}"
            )
        for wheel_name in ("left wheel motor", "right wheel motor"):
            if self.robot.getDevice(wheel_name) is None:
                raise SelfTestFailure(f"robot device {wheel_name} is missing")
        log(
            "PROJECT 3 WEBOTS SELF-TEST PASS: demo route driven, final pose exact, world "
            "tables consistent."
        )
        self.export_image_if_requested()
        self.quit(0)

    def calibrate_script(self):
        self.status = "CALIBRATE MODE"
        order = ["8", "9", "2", "1", "3", "4", "5", "6", "7"]
        offsets = [(0.0, 0.0), (0.6, 0.0), (-0.6, 0.0), (0.0, 0.6), (0.0, -0.6)]
        results = {}
        current = self.room
        self.teleport(current)
        for room in order:
            yield from self.drive(vl.waypoints(current, room)[1:])
            current = room
            self.room = room
            cx, cy = vl.ROOM_CENTERS[room]
            samples = []
            for ox, oy in offsets:
                yield from self.drive([(cx + ox, cy + oy)])
                yield from self.dwell(vl.DWELL_SECONDS)
                samples.extend(self.dwell_samples)
            yield from self.drive([(cx, cy)])
            results[room] = samples
        log("CALIBRATION per-sample light readings (center + 4 offsets of 0.6 m, 1 s dwell each)")
        log("room  name               n    mean   std    min    max   dark  medium  light  modal")
        for room in vl.ROOMS:
            samples = results[room]
            n = len(samples)
            mean = sum(samples) / n
            std = (sum((s - mean) ** 2 for s in samples) / n) ** 0.5
            counts = {level: 0 for level in vl.LIGHT_LEVELS}
            for s in samples:
                counts[vl.discretize(s)] += 1
            modal = max(vl.LIGHT_LEVELS, key=lambda level: counts[level])
            log(
                f"{room:<4}  {vl.ROOM_NAMES[room]:<17} {n:>4}  {mean:.3f}  {std:.3f}  "
                f"{min(samples):.3f}  {max(samples):.3f}  {counts['dark']:>4}  "
                f"{counts['medium']:>6}  {counts['light']:>5}  {modal}"
            )
        log("CALIBRATION DONE")
        self.export_image_if_requested()
        self.quit(0)

    def verify_integration(self):
        rooms = self.rooms_visited
        mode = self.mode
        if not rooms or rooms[0] != self.start_room or any(r not in vl.ROOMS for r in rooms):
            raise SelfTestFailure(f"room sequence {rooms} does not start at {self.start_room}")
        if len(rooms) != self.n_steps + 1:
            raise SelfTestFailure(
                f"expected {self.n_steps + 1} rooms, observed {len(rooms)}: {rooms}"
            )
        for here, there in zip(rooms, rooms[1:], strict=False):
            if here != there and there not in vl.ADJACENCY[here]:
                # non-adjacent outcomes are legal (prob_spec allows 3->1); just make sure a route
                # exists
                vl.route(here, there)
        if mode in ("POLICY", "CUSTOM"):
            kitchen_entries = sum(1 for r in rooms[1:] if r == "3")
            if abs(self.total_reward - 10.0 * kitchen_entries) > 1e-6:
                raise SelfTestFailure(
                    f"total reward {self.total_reward} != 10 x {kitchen_entries} kitchen entries"
                )
        # Recorded expectations assume the default seed and start room; a scripted-key run
        # (restarts, start-room changes) gets structural checks only.
        expected = (
            INTEGRATION_EXPECTED.get(mode)
            if self.seed == vl.DEFAULT_SEED and not self.scripted_keys
            else None
        )
        observed = ",".join(rooms)
        if expected is None:
            if mode == "MPE" and self.mpe_correct < 6 and not self.scripted_keys:
                raise SelfTestFailure(
                    f"MPE recovered only {self.mpe_correct}/{len(rooms)} rooms (need >= 6/8)"
                )
            detail = f"rooms={observed}"
            if mode == "MPE":
                detail += f" inferred={','.join(self.mpe_inferred)} correct={self.mpe_correct}"
            log(
                f"INTEGRATION {mode} seed={self.seed} {detail} (no recorded expectation for "
                f"this seed; structural checks only)"
            )
        else:
            if expected["rooms"] != observed:
                raise SelfTestFailure(
                    f"{mode} seed {self.seed}: expected rooms {expected['rooms']}, observed "
                    f"{observed}"
                )
            if (
                "total_reward" in expected
                and abs(self.total_reward - expected["total_reward"]) > 1e-6
            ):
                raise SelfTestFailure(
                    f"{mode} seed {self.seed}: expected total reward "
                    f"{expected['total_reward']}, observed {self.total_reward}"
                )
            if mode == "MPE":
                inferred = ",".join(self.mpe_inferred)
                if inferred != expected["inferred"] or self.mpe_correct != expected["correct"]:
                    raise SelfTestFailure(
                        f"MPE seed {self.seed}: expected inferred {expected['inferred']} "
                        f"({expected['correct']} correct), observed {inferred} "
                        f"({self.mpe_correct} correct)"
                    )
            log(f"INTEGRATION {mode} seed={self.seed} matches the recorded expectation.")
        log(f"PROJECT 3 WEBOTS INTEGRATION-TEST PASS ({mode})")

    def export_image_if_requested(self):
        """Save the 3D view (staff tool). Rendering is forced on for a moment so
        the export also works when the run itself used --mode=fast."""
        if self.args.export_image and not self.export_done:
            self.export_done = True
            self.update_hud()
            self.robot.simulationSetMode(self.robot.SIMULATION_MODE_REAL_TIME)
            for _ in range(15):
                if self.robot.step(self.time_step) == -1:
                    return
            self.robot.exportImage(self.args.export_image, 95)
            log(f"Exported viewport image to {self.args.export_image}")

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def hud(self, line_id, text):
        self.robot.setLabel(
            line_id, text, 0.01, 0.02 + 0.04 * line_id, HUD_SIZE, HUD_COLOR, 0.0, HUD_FONT
        )

    def update_hud(self):
        mode = self.mode or self.pending_mode
        steps = self.n_steps if self.mode else self.run_length(mode)
        self.hud(
            0,
            (
                f"P3 VACUUM | mode {mode} | seed {self.seed} | start {self.start_room} | step "
                f"{self.k}/{steps} | {self.state}"
            ),
        )
        info = self.step_info
        if info:
            self.hud(
                1,
                (
                    f"room {info['room']} | action {info['action']} | intended {info['intended']} "
                    f"| sampled {info['sampled']} (p={info['p']:.2f})"
                ),
            )
        else:
            self.hud(1, f"{vl.room_label(self.room)}")
        last = self.measurements[-1] if self.measurements else vl.discretize(self.light_raw)
        self.hud(
            2,
            (
                f"light raw={self.light_raw:.2f} level={vl.discretize(self.light_raw)} | last "
                f"measurement={last} | dwell avg={self.dwell_mean():.2f}"
            ),
        )
        if mode in ("POLICY", "CUSTOM"):
            value = ""
            if self.values is not None and self.mode is not None:
                value = (
                    f" | "
                    f"V({self.start_room})={float(self.values[vl.ROOMS.index(self.start_room)]):.2f}"
                )
            self.hud(3, f"reward total {self.total_reward:.0f}{value}")
        elif mode == "MPE":
            self.hud(
                3,
                self.mpe_summary
                or "true room hidden from the inference | inferred: (after the run)",
            )
        elif mode == "TAPE":
            self.hud(
                3, f"rooms so far: {','.join(self.rooms_visited) if self.rooms_visited else '-'}"
            )
        else:
            self.hud(3, "LIGHT mode: keys 1-9 park the robot in another room")
        self.hud(
            4,
            (
                "keys: P policy  C custom  T tape  L light  M mpe  1-9 start room  R replay  (a "
                "mode key again = next seed)"
            ),
        )
        if self.state == "IDLE" and self.auto_start_armed:
            status = f"starting {self.pending_mode} from room {self.start_room} (--mode given)"
        elif self.state == "ERROR":
            status = f"STUDENT CODE ERROR: {self.integration_error}"
        elif self.state == "COMPLETE":
            status = self.done_line
        else:
            status = self.status
        self.hud(5, status[:160])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def tick(self):
        self.sample_light()
        self.press_scripted_keys()
        self.read_keyboard()

        if self.state == "LOADING":
            self.teleport(self.room)
            if self.auto_demo:
                self.script = self.auto_demo_script()
            elif self.calibrate:
                self.script = self.calibrate_script()
            elif not self.self_test and not self.auto_start_armed:
                self.status = (
                    "Ready. Press P, C, T, L or M (1-9 picks the start room for POLICY, CUSTOM and "
                    "LIGHT)."
                )
            self.transition("IDLE")
        elif self.state == "RESETTING":
            # simulationReset() has been applied at the end of the previous step.
            self.finish_restart()
        elif self.state == "IDLE":
            if self.script is not None:
                self.transition("SCRIPT")
            elif self.self_test:
                self.start_run(self.pending_mode)
            elif self.auto_start_armed and self.elapsed() >= self.auto_start_delay:
                self.auto_start_armed = False
                self.start_run(self.pending_mode)
        elif self.state == "SCRIPT":
            try:
                next(self.script)
            except StopIteration:
                self.transition("COMPLETE")
        elif self.state == "SENSE":
            self.dwell_samples.append(self.light_raw)
            if self.elapsed() >= vl.DWELL_SECONDS:
                if self.after_sense == "DECIDE":
                    raw, level = self.finish_measurement()
                    self.measurements.append(level)
                    if self.mode == "MPE":
                        log(vl.format_mpe_step(1, self.n_steps + 1, "start", level))
                    self.transition("DECIDE")
                else:
                    self.complete_step()
        elif self.state == "DECIDE":
            self.decide()
        elif self.state == "MOVING":
            self.update_motion()
            if not self.motion_active():
                self.after_sense = "STEP"
                self.dwell_samples = []
                self.transition("SENSE")
        elif self.state == "INFER":
            self.infer()
        elif self.state == "LIGHT":
            if self.motion_active():
                self.update_motion()
                if not self.motion_active():
                    self.room = self.nearest_room(self.position)
                    self.dwell_samples = []
                    self.light_last_report = self.now()
            else:
                self.dwell_samples.append(self.light_raw)
                if self.now() - self.light_last_report >= LIGHT_REPORT_PERIOD:
                    self.light_last_report = self.now()
                    raw, level = self.finish_measurement()
                    self.measurements.append(level)
                    log(vl.format_light(self.room, raw, level))
                    self.dwell_samples = []
                    if (
                        self.integration_test
                        and len(self.measurements) >= 5
                        and not self.key_script
                    ):
                        log("PROJECT 3 WEBOTS INTEGRATION-TEST PASS (LIGHT)")
                        self.export_image_if_requested()
                        self.quit(0)
                    if self.args.export_image and len(self.measurements) >= 3:
                        self.export_image_if_requested()
        elif self.state == "COMPLETE":
            if self.elapsed() == 0.0 or not getattr(self, "_complete_handled", False):
                self._complete_handled = True
                if self.integration_test and self.key_script:
                    log(
                        f"Run complete; {len(self.key_script)} scripted key press(es) still "
                        f"pending."
                    )
                elif self.integration_test:
                    self.verify_integration()
                    self.export_image_if_requested()
                    self.quit(0)
                elif self.script is not None:
                    self.quit(0)
                else:
                    self.export_image_if_requested()
                    log(
                        "Run complete. Press the same mode key for another run with the next "
                        "seed, R to replay this run, or another mode key."
                    )
        elif self.state == "ERROR":
            pass

        if self.state != "COMPLETE":
            self._complete_handled = False
        self.update_hud()

    def run(self):
        while self.robot.step(self.time_step) != -1:
            try:
                self.tick()
            except SelfTestFailure as error:
                log(f"PROJECT 3 WEBOTS SELF-TEST FAIL: {error}")
                self.quit(1)
                return
            except Exception as error:
                traceback.print_exc()
                self.integration_error = f"{type(error).__name__}: {error}"
                log(f"CONTROLLER ERROR: {self.integration_error}")
                self.set_moving(False)
                if self.self_test:
                    self.quit(1)
                    return
                self.transition("ERROR")


if __name__ == "__main__":
    VacuumDemo().run()
