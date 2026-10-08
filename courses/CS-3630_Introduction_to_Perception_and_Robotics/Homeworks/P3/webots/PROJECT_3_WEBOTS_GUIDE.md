# Project 3 Webots visualization: technical guide

For students who want to know what the simulation does, and for course staff. Students should
follow `STUDENT_SETUP_AND_NAVIGATION.md` for installation and use. **This component adds no
programming TODOs and does not replace or modify any GTSAM, NumPy, test, or autograder code.**

## What it is

A Webots R2025a world of the nine rooms of the COLD-Freiburg lab used in the notebook, with an
iRobot Create (the Roomba the assignment is about) driven by a Supervisor controller. The controller
imports the student's `main.py` and turns three things the notebook computes into motion:

| Mode | Notebook object | How it becomes motion |
|---|---|---|
| TAPE | the control tape from `create_custom_action_sequence` | at each step the next room is sampled from the student's `prob_spec` (via `get_transition_prob`) with a seeded NumPy generator; the robot drives to the sampled room |
| POLICY / CUSTOM | the policy from `policy_iteration` / `get_custom_policy` | as TAPE, but the action at each step is `policy[current room]`; the kitchen reward `R[x,a,x']` accumulates |
| MPE | the factor graph from `create_factor_graph` and `GTSAM_MPE` | the robot drives the tape with the true rooms hidden, the light sensor produces one reading per room, the controller builds the graph from those readings and actions and displays inferred versus true rooms |
| LIGHT | the sensor model `sensor_spec` | no student code; the robot's real light sensor under per-room lighting is printed as raw value and discretized level |

## The world

- Rooms are laid out to match the notebook's topology: corridor 2 is the hub connected to 1, 3, 4
  (north) and 5, 6, 9 (south); 7 is reached only through 6, and 8 only through 9. Each doorway
  is a 1 m gap with an orange threshold tile (`DEF DOOR_a_b`), each room has a label tile at its
  center (`DEF ROOM_k`). Room centers and doorway coordinates live in
  `controllers/project3_vacuum/vacuum_logic.py`; the world is generated from them by
  `instructor/tools/make_world.py`, and the `--auto-demo` self-test cross-checks the two.
- Lighting: one `CeilingLight` per room (four in the corridor) with intensities tuned so that the
  robot's `LightSensor` (lookup table 0..1.5 W/m2 to 0..1, 10 % noise, occlusion on) reads
  dark below 0.33, medium up to 0.66, light above. Calibrated modal levels: room 9 dark; every
  other room medium, with rooms 1 and 3 reading light about 38 % of the time and room 8 reading
  dark about 29 % of the time, which mirrors the shape of `sensor_spec`. Readings vary within a
  room with position and noise, which is the point of report question 3.1b.
- Robot: the official iRobot Create PROTO, vendored under `protos/` with its physics removed so the
  Supervisor can move it kinematically along authored waypoints (center, door approach, door,
  next center); see `assets/ATTRIBUTION.md` for the exact edits. A `LightSensor` and a `Pen` are
  added through the Create's `bodySlot` in the world file; the pen draws the cleaned trail on the
  parquet floors. The wheels spin while it moves.
- Decor is a handful of official Webots models (desks, chairs, monitors, fridge, oven, toilet,
  sink, stairs), placed by the generator with clearance from every room center and door path.

## The controller

`controllers/project3_vacuum/project3_vacuum.py` (Supervisor) and `vacuum_logic.py` (pure
Python: tables, BFS routing, waypoints, discretization, argument parsing, console formats).

- **Finding student code.** Same as Project 2: `Path(__file__).resolve().parents[3] / "main.py"`,
  overridable with `--student-code /path/to/module.py`. The module is loaded with
  `importlib` under the name `cs3630_project3_student`, with its folder on `sys.path`.
- **Contract per mode.** Checked at import (names must exist and be callable):
  POLICY: `ROOMS, ACTIONS, prob_spec, get_transition_prob, generate_reward_table, reward_function,
  get_test_policy, policy_iteration`. CUSTOM: same with `get_custom_policy`. TAPE:
  `create_action_series, create_custom_action_sequence, get_transition_prob`. MPE: TAPE's plus
  `LIGHT_LEVELS, create_factor_graph, GTSAM_MPE, get_kitchen_start_prior`. LIGHT: none.
  A missing name disables that mode with a message; the others keep working.
- **Failure handling.** Import errors are caught so the world still runs; every student call is
  wrapped, and a raised exception or an invalid return (wrong length, non-finite values, action
  index out of range) puts the run in `ERROR` with the message on the HUD and console. `R`
  re-imports `main.py`.
- **Determinism.** Transitions are sampled with `numpy.random.default_rng(seed)` from the row
  `T[x, a, :]` (renormalized) of `get_transition_prob(prob_spec)`, never with GTSAM's sampler, so a
  given seed and `prob_spec` reproduce a route. `WorldInfo.randomSeed 0` makes the light-sensor
  noise reproducible too. The seed is printed in `P3 RUN START` and every `... DONE` line.
  Each mode has its own seed sequence `base + n` for its n-th run (base 3630, or `--seed`), so the
  first TAPE, MPE and POLICY runs all use 3630 whatever ran before; `R` replays the last run with
  its seed; `S` (staff) shifts every sequence by one. Keys are edge-detected: Webots reports a held
  key on every step, so only the transition from up to down counts as a press.
- **Motion.** Kinematic: rotate in place toward the next waypoint, then translate at 1.2 m/s
  (`--speed`). Doorways are crossed perpendicularly through approach points 0.5 m either side.
  A sampled "stay" bumps 0.6 m toward the intended door and returns
  (`attempted 2->1 but stayed in 2 (p=0.04)`); an action that does not exist from the current room
  spins in place (`tape action 2->3 is not available from room 7: robot stays (p=1.00)`, or in
  POLICY/CUSTOM `policy chose 2->1 in room 3: not a doorway from room 3, so the robot stays and
  collects the kitchen reward again (p=1.00)`); a sampled room that is not adjacent is reached
  through BFS routing. After each move the robot dwells 1 s;
  the measurement used by TAPE/POLICY/MPE lines is the **last** sensor sample of the dwell (one
  noisy read, so P(Z|X) stays non-degenerate); the HUD also shows the dwell average.
- **Restart.** `R`, a digit during a run, or a mode key during a run call
  `Supervisor.simulationReset()`, which wipes the Pen's painted trail and rewinds the simulation
  clock without restarting the controller. The controller keeps a monotonic clock of its own, and
  performs the teleport to the start room (and any requested mode start) on the step after the
  reset, since the reset also snaps the robot back to its world-file pose. That pose is placed
  below the floor (`make_world.py`, `robot_node`) so the one frame between reset and teleport
  shows no robot instead of a jump to the bathroom; it also means the robot is not visible in a
  freshly opened, still-paused world until Run is pressed. `setVisibility` cannot be used for
  this because a supervisor reset resets node visibility as well.
- **Rewards.** POLICY/CUSTOM sum `R[x, a, x']` over all 8 steps. Note the notebook's provided
  `rollout_reward` sums `range(1, horizon)` with `horizon = N-1`, i.e. 6 of 7 transitions; the two
  are therefore not directly comparable, and the HUD prints `V(start)` from
  `calculate_value_function` for the discounted comparison question 5.3b asks for.

### Console formats

```
P3 WEBOTS READY | student code: /abs/path/main.py | mode=POLICY seed=3630 start=8 steps=8
P3 RUN START | mode=POLICY seed=3630 start=8 steps=8
POLICY from policy_iteration(): 1:1->2 2:2->3 3:2->1 ... 9:9->2   V(8)=77.63
[POLICY] step k/8 | room r (name) | action a | intended i | sampled s (p=..) | light raw=.. level=.. | reward +.. | total ..
POLICY DONE seed=3630 start=8 steps=8 total_reward=60 rooms=8,9,2,3,3,3,3,3,3
TAPE from create_custom_action_sequence(): 3->2 2->6 6->7 7->6 6->2 2->3 3->2
[TAPE] step k/7 | room r (name) | action a | intended i | sampled s (p=..) | light raw=.. level=..
TAPE DONE seed=3630 final_room=7 rooms=3,2,6,7,6,7,7,7
[LIGHT] room 9 (stairs area) | raw=0.23 | level=dark
[MPE] step k/8 | action a | z=level | true room hidden
MPE RESULT seed=3630 | correct 5/8 | true=3,2,6,7,6,7,7,7 inferred=3,2,6,7,6,2,3,2 | P(inferred)=3.92e-03 P(true)=1.08e-04 (36x)
[MPE]  k  action  z        true  inferred        (one row per step; "<- mismatch" marks a difference)
CUSTOM mode mirrors POLICY with [CUSTOM] / CUSTOM DONE / POLICY from get_custom_policy():
```

### Command-line and environment options (staff, tests)

`controllerArgs` on the `VACUUM` node, or the environment variable `CS3630_P3_CONTROLLER_ARGS`
(appended to the arguments, so the shipped world never needs editing):

| Option | Meaning |
|---|---|
| `--mode policy\|custom\|tape\|light\|mpe` | start this mode immediately |
| `--start 8`, `--seed 3630`, `--steps 8`, `--speed 1.2` | run parameters |
| `--student-code PATH` | import this module instead of `../../../main.py` (e.g. the solution) |
| `--auto-demo` | drive a fixed route with no student code, check world tables, print `PROJECT 3 WEBOTS SELF-TEST PASS`, quit |
| `--integration-test` | run the mode against recorded seed-3630 expectations and quit with 0/1 |
| `--calibrate` | visit every room center and four offsets, print per-room light statistics, quit |
| `--export-image PATH` | save a screenshot after a few steps (used for the docs) |
| `--keys "R@6,5@8,P@10"` | press keys at scripted simulation times (headless test of restart, start-room and mode-switch handling; with `--integration-test`, structural checks only) |

Headless self-test (macOS path shown; Webots must know the `cs3630_p3` interpreter, via
Preferences or a temporary `runtime.ini` next to the controller with `[python]` /
`COMMAND = /path/to/python`):

```bash
W=/Applications/Webots.app/Contents/MacOS/webots
CS3630_P3_CONTROLLER_ARGS="--auto-demo" $W --batch --minimize --stdout --stderr --mode=fast webots/worlds/project3_vacuum.wbt
CS3630_P3_CONTROLLER_ARGS="--integration-test --mode mpe --student-code $PWD/main_sols.py" $W --batch --minimize --stdout --stderr --mode=fast webots/worlds/project3_vacuum.wbt
python -m pytest instructor/tools/test_vacuum_logic.py -q
```

## Files

| Path | Role |
|---|---|
| `worlds/project3_vacuum.wbt` | the world (generated by `instructor/tools/make_world.py`; do not hand-edit coordinates) |
| `protos/Create.proto`, `protos/textures/` | vendored, kinematic iRobot Create (see `assets/ATTRIBUTION.md`) |
| `controllers/project3_vacuum/project3_vacuum.py` | Supervisor controller |
| `controllers/project3_vacuum/vacuum_logic.py` | tables, routing, formats (unit-tested) |
| `assets/room_labels/room_1..9.png` | label tiles (generated by `instructor/tools/make_room_labels.py`) |
| `assets/policy_mode_example.jpg` | example screenshot |
| `assets/ATTRIBUTION.md` | licenses and the exact edits to the vendored PROTO |
| `STUDENT_SETUP_AND_NAVIGATION.md` | student instructions |
