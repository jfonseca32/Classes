"""Every tunable constant of the warehouse mission engine, in one place.

Units: meters, seconds, radians; the dock marker fix in centimeters.
Frames: the book's warehouse frame (x east, y north, origin at the center of
grid cell (0, 0)); grids are (50, 100) arrays indexed [i, j] = [row=y, col=x].

Students: you never need to edit this file. It is part of the supplied
mission engine, not of the assignment. Staff: after changing a knob, re-run
the robustness sweep in ``instructor/engine_tests`` (the integration test
assumes that a correct solution completes its fixed missions).
"""

from __future__ import annotations

import numpy as np

###############################################################################
#                    Warehouse geometry (the book's map)                     #
###############################################################################

MAP_HEIGHT = 50  # rows (y), 1 m cells
MAP_WIDTH = 100  # columns (x), 1 m cells
FLOOR_X = (-0.5, 99.5)  # inner wall faces, meters
FLOOR_Y = (-0.5, 49.5)

SHELF_WIDTH = 9  # cells
SHELF_GAP = 8  # cells
SHELF_FIRST_COL = 20  # (100 - 4 * 9 - 3 * 8) // 2
SHELF_SPACING = SHELF_WIDTH + SHELF_GAP  # 17
SHELF_ROWS = (10, 40)  # rows [10, 40) are shelving
BLOCK_LETTERS = "ABCD"  # west -> east
PROXIMITY_RANGE = 2  # cells: the proximity band around shelves, walls, obstacles

###############################################################################
#                      The robot (book Section 4.2)                          #
###############################################################################

ROBOT_RADIUS = 0.5  # m, circular robot
WHEEL_RADIUS = 0.2  # m
WHEEL_ANGLES_DEG = (0.0, 120.0, 240.0)  # omni wheels on the rim
STEP_DT = 0.5  # s per control step
ROBOT_SPEED = 2.0  # m/s -> one 1 m cell per step
TRUE_MOTION_SIGMA = 0.2  # m per step: the real robot's motion noise (book value)
SAFETY_MARGIN = 0.02  # m: the safety layer stops this far before contact

# The book robot's true wheel Jacobian (rad/s per m/s), rows = wheels 0, 1, 2:
# row i = [-sin(theta_i), cos(theta_i)] / r (Section 4.2.3). Hard-coded so that
# the engine never needs a student function to know how its own wheels are
# mounted.
J_TRUE = np.array(
    [
        [0.0, 5.0],
        [-4.330127018922193, -2.5],
        [4.330127018922193, -2.5],
    ]
)
JACOBIAN_MATCH_TOL = 1e-3  # |J_student - J_TRUE| above this -> KINEMATICS MISMATCH

###############################################################################
#                        Sensors (book Section 4.3)                          #
###############################################################################

RFID_MAX_RANGE = 8.0  # m
RFID_NOISE_SIGMA = 0.3  # m, truth noise on the measured range (the model assumes 1 m)
RFID_PING_REPEAT_STEPS = 4  # re-announce a beacon every 4 steps while in range

# The zone reader: an IR zone beacon in every named area of the warehouse broadcasts that area's
# id (IR does not pass through shelves or walls), so the robot knows which zone it is in, not
# where in it. Zone ids index ZONE_NAMES (= ZONE_NAMES of project4.py; world.ZONE_MAP).
ZONE_NAMES = (
    tuple(f"RECEIVING {b}{side}" for b in range(1, 6) for side in "WE")
    + ("AISLE 1", "AISLE 2", "AISLE 3", "CROSS AISLE S", "CROSS AISLE N")
    + tuple(f"SHIPPING {b}{side}" for b in range(1, 6) for side in "WE")
)
# cells: RECEIVING and SHIPPING are cut into 10 m x 10 m bays: bands 1-5 from the south, W/E halves
ZONE_BAY = 10
ZONE_BLUR = 1.0  # m: within this distance of another zone (across open floor) ...
ZONE_CONFUSION = 0.3  # ... the reader hears that zone's beacon instead with this probability


FIDUCIAL_C = 100.0 * np.eye(2)  # m -> cm: the dock marker fix z = 100 x + w
FIDUCIAL_SIGMA_CM = 10.0  # the precise dock marker fix
FIDUCIAL_R = FIDUCIAL_SIGMA_CM**2 * np.eye(2)
CAPTURE_RADIUS = 2.5  # m: the marker is visible only this close to the station

###############################################################################
#                    System identification (Section 4.6)                     #
###############################################################################

SYSID_SAMPLES = 400
SYSID_CONTROL_SIGMA = 1.5  # m, random calibration controls ~ N(0, 1.5^2 I)
SYSID_TRUE_SIGMA = TRUE_MOTION_SIGMA
SIGMA_CLAMP = (0.05, 1.0)  # the brain's motion sigma is clamped to this range
SIGMA_FALLBACK = 0.2  # used if estimate_motion_model is missing or invalid

###############################################################################
#                   The brain: MCL, grid filter, planning                    #
###############################################################################

N_PARTICLES_GLOBAL = 2000  # uniform global localization (book 4.4.4.4)
N_PARTICLES_TRACKING = 600  # after LOCKED (and in the sensor tests)
TOUR_PATH_PREVIEW = 12  # cells of a sensor test's route shown ahead of the robot
TOUR_RESET_SIGMA = 0.3  # m: a sensor test starts from this good a belief around the truth
TOUR_BLOCKED_SKIP = (
    1  # safety stops in a row before a sensor test's carrot moves on anyway (never stuck)
)
EARLY_STEPS = 12  # START TEST: the "first 6 s" error (an unknown start is being found)
EXACT_PARTICLES = 100  # PENALTY TEST (planning only): the tight belief at the true position
EXACT_SIGMA = 0.05  # m
CONFIDENCE_RADIUS = 1.0  # m: confidence = weight fraction this close to the mean
LOCK_CONFIDENCE = 0.8
LOCK_STREAK = 2  # consecutive confident steps -> LOCKED
# ... but never before this many measurement updates since the particles were
# (re)seeded: right after global initialization the particle set collapses onto
# a handful of survivors and can look confident while being meters off.
LOCK_MIN_UPDATES = 8
LOST_CONFIDENCE = 0.3  # below this while locked -> LOST
# Without a GPS-like sensor the belief can stay wide for a long time (a zone is up to 20 m x 50 m).
# A robot that is not LOCKED after this many measurement updates, and has heard no RFID beacon since
# its particles were (re)seeded, first drives toward the nearest beacon (value iteration with the
# BEACON_SEEK_CELLS as goals); once it hears one, it resumes its order.
BEACON_SEEK_AFTER = 12
DEGENERATE_MEAN_WEIGHT = 1e-12  # mean *unnormalized* weight below this ...
DEGENERATE_STREAK = 2  # ... for this many steps -> LOST (kidnapped)
# A zone reading only scales every particle in the wrong zone by the same small factor, so after a
# kidnapping the weights never degenerate. Instead: if less than ZONE_MISS_SUPPORT of the particles
# lie in the zone the reader names, ZONE_MISS_STEPS steps in a row -> LOST (kidnapped). (Near a zone
# boundary the reader sometimes names the next zone, but rarely 3 times in a row.)
ZONE_MISS_SUPPORT = 0.02
ZONE_MISS_STEPS = 3
INVALID_WEIGHTS_FAULT_STREAK = 3  # NaN/negative weights this many times in a row -> FAULT
# When every particle of a LOCKED belief gets zero weight (the book's 0/1
# proximity and RFID-range likelihoods are unforgiving at band edges), the
# robot first retries the step with a wider motion model before giving up.
RECOVERY_SIGMA = 1.0  # m

MARKER_PARTICLES = 2000  # uniform samples in the capture disk for the marker fix
DOCK_MOTION_SIGMA = 0.03  # m: marker-guided docking is precise

GRID_SIGMA = 0.5  # m: Markov (grid) filter motion sigma
GRID_MAX_SECONDS = 0.5  # one grid update slower than this disables the grid filter

GAMMA = 0.975  # discount factor (book Figure 1)
ACTION_NAMES = ("WEST", "EAST", "SOUTH", "NORTH")  # book Q_value order
ACTION_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # (di, dj) per action
ACTION_VECTORS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])  # (dx, dy) m

# Anti-dither rule: while the robot is not LOCKED *and* its belief is still
# spread out (largest standard deviation above HOLD_MIN_SPREAD), a voted action
# is held for ANTI_DITHER_HOLD steps unless the safety layer stops it. Modes in
# mirror-image aisles otherwise vote WEST, EAST, WEST, ... and the robot never
# moves far enough to learn which mode is right. Once the belief is compact the
# robot re-votes every step: holding then only makes it overshoot its goal or
# swing across the edge of a proximity band that its route follows.
ANTI_DITHER_HOLD = 3
HOLD_MIN_SPREAD = 2.0  # m

# Arrival: the robot starts docking when its estimate is in the goal cell and
# either it is LOCKED, or at least DOCK_READY_CONFIDENCE of the belief weight
# lies within DOCK_READY_RADIUS of the station (so the dock marker, visible
# within CAPTURE_RADIUS, should be in view). Along the middle of a 30 m rack
# face there are no RFID beacons and the zone reader only says "in this aisle":
# the belief along the aisle only grows with the motion noise there (about
# 0.8 m): good enough to find the marker, not always good enough to LOCK.
DOCK_READY_RADIUS = 2.0  # m
DOCK_READY_CONFIDENCE = 0.9

# In the book's Q_value an obstacle cell has V = 0, while cells deep in the
# proximity band far from the goal have V < 0, so the greedy policy there would
# rather drive *into* the shelf. When the engine asks the student's
# greedy_action / vote_action for an action it passes a copy of V whose
# obstacle cells hold this value instead: actions into known obstacles are
# masked (the displayed value function is the student's, unchanged).
OBSTACLE_ACTION_VALUE = -1.0e4
PATH_PREVIEW_MAX = 200  # cells in the greedy path preview
SLOW_CALL_SECONDS = 1.0  # a student call slower than this -> SLOW_CODE warning (once per function)

###############################################################################
#                           Mission and timing                               #
###############################################################################

MOVE_DURATION = STEP_DT  # s, one control step
WAIT_DURATION = 0.5  # s, waiting for a forklift
SCAN_DURATION = 1.0  # s, first global scan after deployment
COUNTDOWN_SECONDS = 3  # 3, 2, 1 (1 s each; not counted in the mission time)
DOCK_DURATION = 2.0  # s, drive from the station to the rack/dock
PICKUP_DURATION = 2.5  # s, parcel slides onto the deck
DROPOFF_DURATION = 3.0  # s, parcel rides the conveyor out through the door
UNDOCK_DURATION = 1.5  # s, back out to the station
DOCK_FAIL_DURATION = 1.5  # s, "MARKER NOT FOUND"
TELEPORT_DURATION = 1.0  # s, kidnapping animation
MAX_STEPS_PER_ORDER = 600  # control steps (moves + waits) allowed per order
# Failsafe: a mission that produced this many results per order without
# finishing is declared FAILED (it cannot happen with the rules above).
MAX_RESULTS_PER_ORDER = 3 * MAX_STEPS_PER_ORDER
MAX_ORDERS = 9

###############################################################################
#                                  Places                                    #
###############################################################################

SLOTS_PER_FACE = 10  # slot s = 1..10 at row y = 8 + 3 s
SLOT_ROW_OFFSET = 8
SLOT_ROW_STEP = 3
WEST_FACE_GOAL_COL = 17  # + 17 k
EAST_FACE_GOAL_COL = 31  # + 17 k
RACK_STANDOFF = 0.65  # m: robot center to rack face when docked
PARCEL_INSET = 0.35  # m: parcel center inside the rack face
PARCEL_DECK_HEIGHT = 0.40  # m: lowest rack level (for the Webots layer)

DOCK_ROWS = (10, 25, 40)  # DOCK 1..3 on the east wall
DOCK_GOAL_COL = 96
DOCK_POSE_X = 98.1  # m: robot center when docked at a shipping dock
CONVEYOR_START_X = 98.8  # m
CONVEYOR_END_X = 103.0  # m: outside the east wall, in the trailer

###############################################################################
#                             Chaos: forklifts                               #
###############################################################################

FORKLIFT_SPEED = 1.4  # m/s
FORKLIFT_HALF_LENGTH = 1.6  # m, body + forks
FORKLIFT_HALF_WIDTH = 0.65  # m
FORKLIFT_CLEARANCE = 0.5  # m: a forklift stops before coming this close to the AMR
FORKLIFT_LOOKAHEAD = 1.5  # m: ... looking this far ahead along its route
AMR_FORKLIFT_MARGIN = 0.3  # m: the AMR waits instead of driving this close
FORKLIFT_DEADLOCK_STEPS = 2  # both waiting this long -> the forklift backs off
AMR_MAX_WAIT_STEPS = 8  # then the AMR tries anyway (the safety layer still protects)

# Closed loop routes; vertex 0 is the forklift's depot, where it parks while
# traffic is OFF (corners the robot's routes almost never visit).
FORKLIFT_ROUTES = (
    ((4.0, 1.8), (12.0, 1.8), (12.0, 45.0), (4.0, 45.0)),  # FORKLIFT_0: receiving (west)
    ((93.0, 1.8), (86.0, 1.8), (86.0, 45.5), (93.0, 45.5)),  # FORKLIFT_1: shipping (east)
)

###############################################################################
#                             Chaos: obstacles                               #
###############################################################################

OBSTACLE_HALF_SIZE = 0.5  # m: a dropped obstacle fills exactly one 1 m cell
OBSTACLE_MIN_AHEAD = 4  # cells along the planned path
OBSTACLE_MAX_AHEAD = 16
OBSTACLE_GOAL_CLEARANCE = 4  # cells (Chebyshev) from any remaining goal cell
OBSTACLE_ROBOT_CLEARANCE = 3.0  # m from the robot
OBSTACLE_LANE_CLEARANCE = 2.2  # m from any forklift route
MAX_OBSTACLES = 6

KIDNAP_MIN_DISTANCE = 25.0  # m: far enough that the belief clearly degenerates
START_DEPOT_CLEARANCE = 1.5  # m: starts and kidnaps avoid parked forklifts

###############################################################################
#                           Scoring (DESIGN 5.5)                             #
###############################################################################

# DESIGN 5.5 first suggested 40 points per second and asks to tune the constants
# so that a clean staff run scores about 8000-9000. With 40 the median clean
# 1-order mission (about 80 s) scores about 7600, so the time term is 30.
SCORE_BASE = 10000
SCORE_PER_SECOND = 30  # per second of mission time (per order)
SCORE_PER_SAFETY_STOP = 300
SCORE_PER_FAILED_DOCKING = 750
SCORE_PER_RELOCALIZATION = 400
SCORE_TRACKING_BONUS = 2000  # * max(0, 1 - mean tracking error in m)
STAR_THRESHOLDS = (8000, 6000)  # 3 stars, 2 stars; else 1
TITLES = {3: "WAREHOUSE LEGEND", 2: "LOGISTICS PRO", 1: "ROOKIE ROVER", 0: "MISSION FAILED"}

###############################################################################
#                              Robot profile                                 #
###############################################################################

ROBOT_COLORS = ("gold", "navy", "orange", "cyan", "magenta", "lime", "white", "red")
DEFAULT_PROFILE = {"name": "RB-3630", "color": "gold"}  # the starter's robot_profile()
PROFILE_NAME_MAX = 16
