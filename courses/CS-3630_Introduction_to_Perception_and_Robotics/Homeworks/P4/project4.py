"""CS 3630 Project 4: Autonomous Warehouse Challenge.

Meet RB-3630, a small three-omni-wheel robot on the night shift of a 100 m x
50 m fulfillment center, the warehouse of Chapter 4. Every mission starts the
same way: the robot wakes up somewhere on the floor with no idea where it is,
and an order comes in. Your job is to write its brain: eight short functions.
Part 1 is a particle filter (motion model, sensor likelihoods, prediction, update, and the
estimate); Part 2 is value iteration. Put them together and the robot
localizes itself, drives to a storage rack, picks up a parcel, and delivers it
to a shipping dock, live in a 3D Webots warehouse. Everything else is
provided: the map and helpers, the proximity sensor model, the policy
extraction (``greedy_action``, ``vote_action``), and the staff engine
(physics, graphics, HUD) that calls your code.

Conventions (the same as the book's ``gtbook.logistics``)
---------------------------------------------------------
* A position is a point ``(x, y)`` in meters: x runs east along the 100 m
  side and y runs north along the 50 m side.
* Grid maps (``BASE_MAP``, the proximity maps, beliefs, value functions) are
  arrays of shape ``(50, 100)`` indexed ``[i, j] = [row, col] = [y, x]``.
  Cell ``(i, j)`` is centered at the point ``(x=j, y=i)``, and a point
  ``(x, y)`` lies in cell ``(round(y), round(x))``. Points are (x, y) but
  cells are (row, col) = (y, x); mixing them up is the number one bug in this
  project!
* A batch of points or particles is an array of shape ``(N, 2)`` with one
  ``(x, y)`` per row.
* Actions move one cell: 0 = WEST (j - 1), 1 = EAST (j + 1), 2 = SOUTH
  (i - 1), 3 = NORTH (i + 1), in the same order as the book's ``Q_value``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

###############################################################################
#                     PROVIDED CONSTANTS                                      #
###############################################################################


MAP_HEIGHT = 50  # rows = y cells (1 m each)
MAP_WIDTH = 100  # columns = x cells (1 m each)

# MAP_COORDS[i, j] == (x, y) == (j, i): the center of every grid cell.
_X, _Y = np.meshgrid(np.arange(MAP_WIDTH, dtype=float), np.arange(MAP_HEIGHT, dtype=float))
MAP_COORDS = np.dstack((_X, _Y))

# Four blocks of storage shelves, exactly as in gtbook.logistics.base_map.
SHELF_WIDTH = 9
SHELF_GAP = 8
_SPACING = SHELF_WIDTH + SHELF_GAP
_SIDE = (MAP_WIDTH - SHELF_WIDTH * 4 - SHELF_GAP * 3) // 2  # 20

BASE_MAP = np.zeros((MAP_HEIGHT, MAP_WIDTH))

# Add the 4 shelves and 3 isles between them to BASE_MAP (occupied spaces = 1)
for _k in range(4):
    BASE_MAP[10:40, _SIDE + _k * _SPACING : _SIDE + SHELF_WIDTH + _k * _SPACING] = 1

# Eight RFID beacons at both ends of every shelf block (gtbook.logistics.beacons).
BEACONS = np.array(
    [[SHELF_WIDTH / 2 + _SIDE + _k * _SPACING - 0.5, 9.5] for _k in range(4)]
    + [[SHELF_WIDTH / 2 + _SIDE + _k * _SPACING - 0.5, 39.5] for _k in range(4)]
)

# Proximity sensor maps (gtbook.logistics.proximity_map_on / _off).
PROXIMITY_RANGE = 2  # cells
PROXIMITY_MAP_ON = np.zeros((MAP_HEIGHT, MAP_WIDTH))
PROXIMITY_MAP_OFF = np.ones((MAP_HEIGHT, MAP_WIDTH))

# Mark cells near shelves and walls where the proximity sensor reads ON.
for _k in range(4):
    PROXIMITY_MAP_ON[
        10 - PROXIMITY_RANGE : 40 + PROXIMITY_RANGE,
        _SIDE + _k * _SPACING - PROXIMITY_RANGE : _SIDE
        + SHELF_WIDTH
        + _k * _SPACING
        + PROXIMITY_RANGE,
    ] = 1
PROXIMITY_MAP_ON[0:PROXIMITY_RANGE] = 1
PROXIMITY_MAP_ON[-PROXIMITY_RANGE:] = 1
PROXIMITY_MAP_ON[:, -PROXIMITY_RANGE:] = 1
PROXIMITY_MAP_ON[:, 0:PROXIMITY_RANGE] = 1
for _k in range(4):
    PROXIMITY_MAP_ON[10:40, _SIDE + _k * _SPACING : _SIDE + SHELF_WIDTH + _k * _SPACING] = 0

PROXIMITY_MAP_OFF[PROXIMITY_MAP_ON == 1] = 0
PROXIMITY_MAP_OFF[BASE_MAP == 1] = 0

# Sensor and robot parameters
RFID_MAX_RANGE = 8.0  # meters

RFID_SIGMA = 1.0  # meters, range noise used by the likelihood
# The zone reader: an infrared (IR) zone beacon in every named area of the
# warehouse broadcasts that area's id. IR does not pass through shelves or walls,
# so the robot hears the beacon of the zone it is in (near the edge of a zone it
# sometimes hears the next one). ZONE_MAP[i, j] is the zone of cell (i, j), -1 for
# shelf cells; ZONE_NAMES[k] is the name of zone k. RECEIVING (west of shelf A) and
# SHIPPING (east of shelf D) are cut into 10 m x 10 m bays: bands 1-5 from the
# south, W and E halves.

ZONE_NAMES = (
    tuple(f"RECEIVING {b}{side}" for b in range(1, 6) for side in "WE")
    + ("AISLE 1", "AISLE 2", "AISLE 3", "CROSS AISLE S", "CROSS AISLE N")
    + tuple(f"SHIPPING {b}{side}" for b in range(1, 6) for side in "WE")
)
NUM_ZONES = len(ZONE_NAMES)  # 25

ZONE_MAP = np.full((MAP_HEIGHT, MAP_WIDTH), -1, dtype=int)
_EAST = _SIDE + 3 * _SPACING + SHELF_WIDTH  # 80: the first column east of shelf D

for _b in range(5):  # RECEIVING (west of shelf A) and SHIPPING (east of shelf D): 10 m x 10 m bays
    for _h in range(2):  # bands 1-5 from the south; W and E halves
        ZONE_MAP[10 * _b : 10 * _b + 10, 10 * _h : 10 * _h + 10] = 2 * _b + _h
        ZONE_MAP[10 * _b : 10 * _b + 10, _EAST + 10 * _h : _EAST + 10 * _h + 10] = 15 + 2 * _b + _h

for _k in range(3):  # AISLE 1-3: between the shelf blocks
    ZONE_MAP[10:40, _SIDE + SHELF_WIDTH + _k * _SPACING : _SIDE + (_k + 1) * _SPACING] = 10 + _k

ZONE_MAP[:10, _SIDE:_EAST] = 13  # CROSS AISLE S: south of the shelves
ZONE_MAP[40:, _SIDE:_EAST] = 14  # CROSS AISLE N: north of the shelves

ZONE_P_CORRECT = 0.9  # probability that the reader names the zone the robot is in

ROBOT_RADIUS = 0.5  # meters (the book's circular robot)
WHEEL_RADIUS = 0.2  # meters
NUM_WHEELS = 3
MOTION_SIGMA = 0.2  # meters per step, the book's motion-model standard deviation

# Planning parameters
GAMMA = 0.975
GOAL_REWARD = 100.0
PROXIMITY_PENALTY = 50.0

# Actions move exactly one cell. The numbering matches the book's Q_value code
# (0: j-1, 1: j+1, 2: i-1, 3: i+1); we use compass names to avoid confusion.
WEST, EAST, SOUTH, NORTH = 0, 1, 2, 3
ACTION_NAMES = ("WEST", "EAST", "SOUTH", "NORTH")
ACTION_DELTAS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])  # (di, dj) per action
ACTION_VECTORS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])  # (dx, dy) meters


###############################################################################
#                       PROVIDED HELPERS                                      #
###############################################################################


def point_to_cell(point: ArrayLike) -> tuple[int, int]:
    """Return the grid cell containing a point.

    Parameters
    ----------
    point : array_like of shape (2,)
        Position ``(x, y)`` in meters.

    Returns
    -------
    cell : tuple of int
        Grid indices ``(row, col) = (round(y), round(x))``. The cell may
        lie outside the map; use `in_bounds` to check.
    """
    # convert ArrayLike to numpy array
    x, y = np.asarray(point, dtype=float)[:2]
    return int(np.round(y)), int(np.round(x))


def points_to_cells(
    points: ArrayLike,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.bool_]]:
    """Convert points to clipped grid indices and an in-bounds mask.

    Parameters
    ----------
    points : array_like of shape (N, 2) or (2,)
        Positions ``(x, y)`` in meters. A single point counts as ``N = 1``.

    Returns
    -------
    rows : ndarray of int, shape (N,)
        Row indices, clipped to the map bounds.
    cols : ndarray of int, shape (N,)
        Column indices, clipped to the map bounds.
    inside : ndarray of bool, shape (N,)
        Whether each point's rounded cell lies inside the map. Combine
        map lookups with this mask to exclude points outside the map.
    """
    # convert ArrayLike to numpy array
    points = np.asarray(points, dtype=float).reshape(-1, 2)

    cols = np.round(points[:, 0]).astype(int)
    rows = np.round(points[:, 1]).astype(int)
    inside = (rows >= 0) & (rows < MAP_HEIGHT) & (cols >= 0) & (cols < MAP_WIDTH)

    return np.clip(rows, 0, MAP_HEIGHT - 1), np.clip(cols, 0, MAP_WIDTH - 1), inside


def cell_to_point(
    cell: tuple[int, int] | NDArray[np.integer],
) -> np.ndarray:
    """Return the center of a grid cell in world coordinates.

    Parameters
    ----------
    cell : tuple of int or ndarray of int, shape (2,)
        Grid indices ``(row, col)``.

    Returns
    -------
    point : ndarray of float64, shape (2,)
        Cell center ``(x, y) = (col, row)`` in meters.
    """
    i, j = cell
    return np.array([float(j), float(i)])


def in_bounds(
    cell: tuple[int, int] | NDArray[np.integer],
    shape: tuple[int, int] = (MAP_HEIGHT, MAP_WIDTH),
) -> bool:
    """Check whether a grid cell lies inside the given map shape.

    Parameters
    ----------
    cell : tuple of int or ndarray of int, shape (2,)
        Grid indices ``(row, col)``.
    shape : tuple of int, optional
        Map dimensions ``(height, width)``. Defaults to
        ``(MAP_HEIGHT, MAP_WIDTH)``.

    Returns
    -------
    bool
        Whether both indices lie inside the grid.
    """
    i, j = cell
    return 0 <= i < shape[0] and 0 <= j < shape[1]


def next_cell(
    cell: tuple[int, int] | NDArray[np.integer],
    action: Literal[0, 1, 2, 3],
    shape: tuple[int, int] = (MAP_HEIGHT, MAP_WIDTH),
) -> tuple[int, int]:
    """Move one grid cell, staying in place if the move leaves the map.

    Parameters
    ----------
    cell : tuple of int or ndarray of int, shape (2,)
        Starting grid indices ``(row, col)``.
    action : {0, 1, 2, 3}
        Direction: 0 = WEST (col - 1), 1 = EAST (col + 1),
        2 = SOUTH (row - 1), or 3 = NORTH (row + 1).
    shape : tuple of int, optional
        Map dimensions ``(height, width)``. Defaults to
        ``(MAP_HEIGHT, MAP_WIDTH)``.

    Returns
    -------
    next_cell : tuple of int
        Destination ``(row, col)``, or the original cell if the move
        would leave the map. Shelf occupancy is not checked.
    """
    di, dj = ACTION_DELTAS[action]
    candidate = (int(cell[0] + di), int(cell[1] + dj))
    return candidate if in_bounds(candidate, shape) else (int(cell[0]), int(cell[1]))


###############################################################################
#                          SECTION 0: SETUP                                   #
###############################################################################


def student_info() -> dict[str, str]:
    """Return the student's name and Georgia Tech username.

    Returns
    -------
    info : dict of str to str
        Student information with keys ``name`` and ``gt_username``.
    """
    return {"name": "Joao Pedro Dudziak Fonseca", "gt_username": "jfonseca32"}


def robot_profile() -> dict[str, str]:
    """Personalize the robot shown in the Webots demo.

    Returns
    -------
    profile : dict of str to str
        Robot appearance with keys ``name`` and ``color``. The name is at
        most 16 plain ASCII characters. Supported colors are ``gold``,
        ``navy``, ``orange``, ``cyan``, ``magenta``, ``lime``, ``white``,
        and ``red``.
    """
    return {"name": "Brenda", "color": "magenta"}


###############################################################################
#               SECTION 1: STATE ESTIMATION                                   #
###############################################################################


def sample_motion_model(
    states: ArrayLike, u: ArrayLike, sigma: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sample next positions from the Gaussian motion model.

    Each position follows ``x_next = x + u + w``, where every coordinate
    of every state receives independent noise ``w ~ N(0, sigma**2)``.

    Parameters
    ----------
    states : array_like of shape (2,) or (N, 2)
        Current positions ``(x, y)`` in meters.
    u : array_like of shape (2,)
        Commanded displacement ``(dx, dy)`` in meters, shared by all states.
    sigma : float
        Nonnegative noise standard deviation in meters. With ``sigma = 0``,
        the result is exactly ``states + u``.
    rng : numpy.random.Generator
        Generator used for all random draws, allowing replay from a seed.

    Returns
    -------
    next_states : ndarray of float64
        New positions with the same shape as ``states``. The input is
        not modified in place.

    Raises
    ------
    ValueError
        If ``states`` has neither shape ``(2,)`` nor ``(N, 2)``, or if
        ``u`` does not have shape ``(2,)``.
    """
    # convert ArrayLike to numpy array
    x_k = np.asarray(states)
    u_k = np.asarray(u)

    if not (x_k.shape == (2,) or (x_k.ndim == 2 and x_k.shape[1] == 2)):
        raise ValueError(f"states must have shape (2,) or (N, 2), got {x_k.shape}.")
    if u_k.shape != (2,):
        raise ValueError(f"``u`` must have dimension (2,) but is currently {u_k.shape}.")

    noise = rng.normal(0.0, sigma, size=np.shape(states))  # sampling normal distribution
    next_states = x_k + u_k + noise  # x_{k+1} = x_k + u_k + w_k for all states
    return next_states


def proximity_likelihood(points: ArrayLike, z_on: bool) -> NDArray[np.float64]:
    """Compute likelihoods for one proximity sensor reading.

    The sensor reads ON near shelves and walls, as recorded in
    ``PROXIMITY_MAP_ON``. The corresponding OFF likelihoods are stored
    in ``PROXIMITY_MAP_OFF``. Shelf interiors have zero likelihood in both.

    Parameters
    ----------
    points : array_like of shape (N, 2) or (2,)
        Candidate positions ``(x, y)`` in meters. A single point counts
        as ``N = 1``.
    z_on : bool
        Observed sensor reading: True for ON, False for OFF.

    Returns
    -------
    likelihoods : ndarray of float64, shape (N,)
        One for positions consistent with the reading, zero otherwise.
        Points whose rounded cells lie outside the map receive zero.
    """
    rows, cols, inside = points_to_cells(points)
    lookup = PROXIMITY_MAP_ON if z_on else PROXIMITY_MAP_OFF
    return np.where(inside, lookup[rows, cols], 0.0).astype(float)


def rfid_likelihood(
    points: ArrayLike,
    measurement: tuple[int | np.integer | None, float],
    beacons: NDArray[np.float64] = BEACONS,
    max_range: float = RFID_MAX_RANGE,
    sigma: float = RFID_SIGMA,
) -> NDArray[np.float64]:
    """Compute likelihoods for one RFID range measurement.

    The reader reports the nearest beacon within ``max_range`` as
    ``(beacon_id, r)``, or ``(None, inf)`` if none is detected. For a
    reported beacon, let ``d`` be the distance to that beacon. The
    likelihood is ``exp(-(r - d)**2 / (2 * sigma**2))`` when
    ``d <= max_range``, and zero otherwise.

    For a no-detection reading, the likelihood is one only where every
    beacon is strictly farther than ``max_range``. These formulas apply
    at any position; map bounds and shelf occupancy are not checked.

    Parameters
    ----------
    points : array_like of shape (N, 2) or (2,)
        Candidate positions ``(x, y)`` in meters. A single point counts
        as ``N = 1``.
    measurement : tuple
        ``(beacon_id, r)``, where the ID is a Python or NumPy integer
        indexing ``beacons``, and ``r`` is the measured range in meters.
        A no-detection reading is ``(None, inf)``.
    beacons : ndarray of float64, shape (B, 2), optional
        Beacon positions ``(x, y)`` in meters. Defaults to ``BEACONS``.
    max_range : float, optional
        Detection range in meters. Defaults to ``RFID_MAX_RANGE``.
    sigma : float, optional
        Positive range noise standard deviation in meters. Defaults to
        ``RFID_SIGMA``.

    Returns
    -------
    likelihoods : ndarray of float64, shape (N,)
        Unnormalized measurement likelihood for each point, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``points`` has neither shape ``(2,)`` nor ``(N, 2)``, if a
        missing beacon ID is paired with a range other than positive
        infinity, or if a present beacon ID is paired with positive infinity.
    """
    beacon_id = measurement[0]
    beacon_range = measurement[1]

    # convert ArrayLike to numpy array
    points = np.asarray(points, dtype=float)

    if not (points.shape == (2,) or (points.ndim == 2 and points.shape[1] == 2)):
        raise ValueError(f"states must have shape (2,) or (N, 2), got {points.shape}.")
    if beacon_id is None and beacon_range != math.inf:
        raise ValueError(
            "measurement (beacon_id, r) yielded beacon_id None, but has finite range. "
            "If beacon_id is None, r should to be infinite."
        )
    if beacon_id is not None and beacon_range == math.inf:
        raise ValueError(
            "measurement (beacon_id, r) yielded beacon_id not None, has infinite range. "
            "If beacon_id is not None, r should be finite."
        )

    points = points.reshape(-1, 2)  # Reshape (2,) and (N, 2) to (N, 2)
    point_beacon_distances = np.linalg.norm(points[:, None, :] - beacons[None, :, :], axis=-1)

    # measurement found no beacons in range
    if beacon_id is None:
        # likelihood array has 0s for points in range; 1s for points out of range
        out_of_range = np.all(point_beacon_distances > max_range, axis=1)
        return out_of_range.astype(np.float64)  # cast boolean array to float

    # measurement found a beacon in range
    else:
        beacon_distances = point_beacon_distances[
            :, beacon_id
        ]  # distances of points to measured beacon

        # likelihood function exp(-(r - d)^2 / (2 * sigma^2))
        beacon_likelihoods = np.exp(-((beacon_range - beacon_distances) ** 2) / (2 * sigma**2))

        # mask to 0 if points are farther than max_range
        return np.where(beacon_distances <= max_range, beacon_likelihoods, 0.0)


def zone_likelihood(
    points: ArrayLike,
    z_zone: int | np.integer,
    zone_map: NDArray[np.integer] = ZONE_MAP,
    p_correct: float = ZONE_P_CORRECT,
    num_zones: int = NUM_ZONES,
) -> NDArray[np.float64]:
    """Compute likelihoods for one zone sensor reading.

    The reader names the correct zone with probability ``p_correct``.
    The remaining probability is spread evenly across the other
    ``num_zones - 1`` zones.

    Parameters
    ----------
    points : array_like of shape (N, 2) or (2,)
        Candidate positions ``(x, y)`` in meters. A single point counts
        as ``N = 1``.
    z_zone : int or numpy.integer
        Observed zone ID, indexing ``ZONE_NAMES``.
    zone_map : ndarray of int, shape (50, 100), optional
        Zone ID at each grid cell, with ``-1`` marking shelf cells.
        Defaults to ``ZONE_MAP``.
    p_correct : float, optional
        Probability of reporting the correct zone. Defaults to
        ``ZONE_P_CORRECT``.
    num_zones : int, optional
        Number of zones, greater than one. Defaults to ``NUM_ZONES``.

    Returns
    -------
    likelihoods : ndarray of float64, shape (N,)
        ``p_correct`` for points in the observed zone and
        ``(1 - p_correct) / (num_zones - 1)`` for points in other zones.
        Shelf cells and points outside the map receive zero.

    Raises
    ------
    ValueError
        If ``points`` has neither shape ``(2,)`` nor ``(N, 2)``.
    """
    # convert ArrayLike to numpy array
    points = np.asarray(points, dtype=float)
    if not (points.shape == (2,) or (points.ndim == 2 and points.shape[1] == 2)):
        raise ValueError(f"states must have shape (2,) or (N, 2), got {points.shape}.")

    rows, cols, inside = points_to_cells(points)
    zones_per_point = zone_map[rows, cols]

    # p_correct for cells in z_zone and (1 - p_correct) / (num_zones - 1) for cells not in z_zone
    likelihoods = np.where(zones_per_point == z_zone, p_correct, (1 - p_correct) / (num_zones - 1))

    # 0 to cells outside map (inside == 0) and to cells inside shelves (zones_per_point == -1)
    return np.where(inside & (zones_per_point != -1), likelihoods, 0)


def mcl_predict(
    particles: ArrayLike,
    weights: ArrayLike,
    u: ArrayLike,
    sigma: float,
    rng: np.random.Generator,
    num_samples: int | None = None,
) -> NDArray[np.float64]:
    """Resample weighted particles and predict their next positions.

    Draw particle indices with replacement in proportion to their weights,
    then apply `sample_motion_model` to the selected positions. Each copy
    of a selected particle receives independent motion noise.

    Parameters
    ----------
    particles : array_like of shape (N, 2)
        Current particle positions ``(x, y)`` in meters.
    weights : array_like of shape (N,)
        Nonnegative weights with a positive sum. They need not be normalized;
        zero-weight particles are never selected. The input is not modified.
    u : array_like of shape (2,)
        Commanded displacement ``(dx, dy)`` in meters.
    sigma : float
        Nonnegative motion noise standard deviation in meters.
    rng : numpy.random.Generator
        Generator used for both resampling and motion noise.
    num_samples : int or None, optional
        Number of particles to return. If None, use ``N``.

    Returns
    -------
    next_particles : ndarray of float64, shape (num_samples, 2)
        Predicted, unweighted particle positions ``(x, y)`` in meters.

    Raises
    ------
    ValueError
        If ``particles`` does not have shape ``(N, 2)``, or if the
        particles and weights have different lengths.

    Notes
    -----
    These particles form the cyan cloud in the simulation.
    """
    # convert ArrayLike to numpy array
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError(f"particles must have shape (N, 2), got {particles.shape}.")
    if particles.shape[0] != weights.shape[0]:
        raise ValueError(
            "particles and weights must have same length. Currently "
            f"particles: {particles.shape}; weights: {weights.shape}."
        )

    # resample previous posterior
    normalized_weights = weights / np.sum(weights)
    num_samples = len(particles) if num_samples is None else num_samples
    resampled_indices = rng.choice(len(particles), size=num_samples, p=normalized_weights)
    resampled = particles[resampled_indices]

    # Predict positions from the resampled posterior.
    return sample_motion_model(states=resampled, u=u, sigma=sigma, rng=rng)


def mcl_update(
    particles: ArrayLike,
    z_on: bool,
    rfid_measurement: tuple[int | np.integer | None, float],
    zone_measurement: int | np.integer | None = None,
) -> NDArray[np.float64]:
    """Computes one unnormalized sensor weight per particle.

    The sensors are independent given the state, so each weight is the
    product of the proximity and RFID likelihoods, and the zone likelihood
    when a zone reading is available. Previous weights are not used.

    Parameters
    ----------
    particles : array_like of shape (N, 2)
        Particle positions ``(x, y)`` in meters.
    z_on : bool
        Observed proximity reading: True for ON, False for OFF.
    rfid_measurement : tuple
        RFID reading ``(beacon_id, r)``. The ID is a Python or NumPy
        integer indexing ``BEACONS``, and ``r`` is the range in meters.
        A no-detection reading is ``(None, inf)``.
    zone_measurement : int or numpy.integer or None, optional
        Observed zone ID. If None, omit the zone likelihood factor.

    Returns
    -------
    weights : ndarray of float64, shape (N,)
        Products of the available sensor likelihoods, without normalization.
        The engine uses their raw magnitude to detect loss of localization.

    Raises
    ------
    ValueError
        If the particle shape or RFID measurement is rejected by
        `rfid_likelihood`.

    Notes
    -----
    These weights control particle size, the confidence bar, and the
    LOCKED/LOST status in the simulation.
    """
    # multiply proximity and rfid likelihoods from measurements
    weights = proximity_likelihood(points=particles, z_on=z_on) * rfid_likelihood(
        points=particles, measurement=rfid_measurement
    )

    # multiply zone likelihoods if measurement is present
    if zone_measurement is not None:
        weights *= zone_likelihood(points=particles, z_zone=zone_measurement)

    return weights


def estimate_position(
    particles: ArrayLike,
    weights: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the weighted mean and population covariance of particles.

    For normalized weights ``w_s = weights[s] / sum(weights)``, the mean
    is ``sum(w_s * x_s)`` and the covariance is
    ``sum(w_s * (x_s - mean) @ (x_s - mean).T)``, treating each position
    as a column vector. No sample covariance correction is applied.

    Parameters
    ----------
    particles : array_like of shape (N, 2)
        Particle positions ``(x, y)`` in meters. Supply a single particle
        with shape ``(1, 2)``.
    weights : array_like of shape (N,)
        Nonnegative weights with a positive sum. They need not be normalized;
        zero-weight particles do not contribute. The input is not modified.

    Returns
    -------
    mean : ndarray of float64, shape (2,)
        Weighted mean position ``(x, y)`` in meters.
    cov : ndarray of float64, shape (2, 2)
        Weighted population covariance in square meters. A single particle
        gives an all-zero covariance matrix.

    Raises
    ------
    ValueError
        If ``particles`` does not have shape ``(N, 2)``, or if the
        particles and weights have different lengths.

    Notes
    -----
    The simulation shows this estimate as a magenta ring with a
    2-sigma ellipse and position uncertainty telemetry.
    """
    # convert ArrayLike to numpy array
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError(f"particles must have shape (N, 2), got {particles.shape}.")
    if particles.shape[0] != weights.shape[0]:
        raise ValueError(
            "particles and weights must have same length. Currently "
            f"particles: {particles.shape}; weights: {weights.shape}."
        )

    # find weighted average
    average = np.average(particles, axis=0, weights=weights)

    # columns of particles.T are observations; NumPy centers them.
    # ddof=0 computes population covariance without a sample correction.
    covariance = np.cov(m=particles.T, aweights=weights, ddof=0)

    return (average, covariance)


###############################################################################
#           PART 2: PLANNING WITH VALUE ITERATION (Book Section 4.5)          #
###############################################################################


def q_value(
    v: ArrayLike,
    cell: tuple[int, int],
    action: Literal[0, 1, 2, 3],
    goal_mask: ArrayLike,
    obstacle_map: NDArray[np.float64] = BASE_MAP,
    proximity_map: NDArray[np.float64] = PROXIMITY_MAP_ON,
    gamma: float = GAMMA,
) -> float:
    """Q(x, a; V) for one grid cell and one action.

    Actions are deterministic. For ``cell = (i, j)``:

    1. If ``(i, j)`` is a goal cell (``goal_mask[i, j]`` is True), returns 0:
       the reward is only collected when FIRST arriving at the goal.
    2. If ``(i, j)`` is an obstacle (``obstacle_map[i, j]`` is nonzero),
       returns 0 (an invalid state).
    3. Otherwise the next cell is ``(r, c) = next_cell((i, j), action,
       V.shape)`` (moving off the grid leaves the robot where it is), and the
       reward starts at 0 and gets
         +100 (``GOAL_REWARD``) if ``(r, c)`` is a goal cell,
         -50 (``PROXIMITY_PENALTY``) if ``(i, j)`` is in proximity
         (``proximity_map[i, j]`` nonzero), and
         another -50 if ``(r, c)`` is in proximity.
       Returns ``reward + gamma * v[r, c]``.

    Paramters
    ---------
    v : array of shape ``(H, W)``
        The current value function. Its shape defines the grid.
    cell : tuple[int, int]
        ``(i, j)``, the (row, col) of the current cell.
    action : Literal[0, 1, 2, 3]
        0 = WEST, 1 = EAST, 2 = SOUTH, or 3 = NORTH.
    goal_mask : boolean array of shape ``(H, W)``,
        True at the goal cell(s); there may be more than one.
    obstacle_map : array of shape ``(H, W)``, default BASE_MAP
        Nonzero at obstacles.
    proximity_map : array of shape ``(H, W)``, default PROXIMITY_MAP_ON
        Nonzero where the robot is too close to something.
    gamma : float, default GAMMA
        The discount factor.

    Returns
    -------
    float
        The Q value.

    Notes
    -----
    In the demo: every planning decision (``value_iteration``, ``greedy_action``
    and ``vote_action``) is built from it.
    """
    # convert ArrayLike to numpy array
    v = np.asarray(v, dtype=float)
    goal_mask = np.asarray(goal_mask, dtype=bool)
    y1, x1 = cell  # row, col
    reward = 0

    if goal_mask[y1, x1] or obstacle_map[y1, x1] != 0:
        return reward

    y2, x2 = next_cell(cell=cell, action=action, shape=(v.shape[0], v.shape[1]))

    reward += GOAL_REWARD if goal_mask[y2, x2] else 0  # next cell is goal
    reward -= PROXIMITY_PENALTY if proximity_map[y1, x1] else 0  # current cell is close to wall
    reward -= PROXIMITY_PENALTY if proximity_map[y2, x2] else 0  # next cell is close to wall

    return reward + gamma * v[y2, x2]  # Bellman Equation


def value_iteration(
    goal_mask: ArrayLike,
    obstacle_map: NDArray[np.float64] = BASE_MAP,
    proximity_map: NDArray[np.float64] = PROXIMITY_MAP_ON,
    gamma: float = GAMMA,
    tol: float = 1e-5,
    max_iterations: int = 1000,
) -> NDArray[np.float64]:
    """Run value iteration on the grid until the value function converges.

    Starts from ``v = 0`` everywhere and repeats "sweeps". One sweep computes, for every cell,

        v_new[i, j] = max over the 4 actions a of q((i, j), a; V),

    where q is ``q_value``, so goal and obstacle cells always get 0.

    Sweeps are synchronous: every entry of ``v_new`` is computed from the previous ``v``,
    never from entries already updated in the current sweep.

    After each sweep computes ``error = max |v_new - v|`` and sets ``v = v_new``;
    stops as soon as ``error <= tol``, or after ``max_iterations`` sweeps, whichever comes first.

    Parameters
    ----------
    goal_mask : boolean array of shape ``(H, W)``,
        True at the goal cell(s); there may be more than one.
    obstacle_map : array of shape ``(H, W)``, default BASE_MAP
        Nonzero at obstacles.
    proximity_map : array of shape ``(H, W)``, default PROXIMITY_MAP_ON
        Nonzero where the robot is too close to something.
    gamma : float, default GAMMA
        The discount factor.
    tol : float, default 1e-5
        Tolerance to stop when the largest change in one sweep is less than or equal to it.
    max_iterations : int, default 1000
        The maximum number of sweeps.

    Returns
    -------
    Float array of shape ``(H, W)``
        V after the last sweep. For example, ``max_iterations=1`` returns the result of
        exactly one sweep from ``V = 0``.

    Notes
    -----
    In the demo: the value-function layer of the hologram (key M), the yellow
    planned path, and all re-planning around obstacles.
    """
    # convert ArrayLike to numpy array
    goal_mask = np.asarray(goal_mask, dtype=bool)
    obstacle_map = np.asarray(obstacle_map)
    proximity_map = np.asarray(proximity_map) != 0  # if array has values not just 0s and 1s
    zero_rewards = goal_mask | (obstacle_map != 0)  # goal and occupied cells have 0 reward

    # build v and index arrays
    height, width = goal_mask.shape
    v = np.zeros(shape=(height, width), dtype=float)  # start with all zeros
    rows, cols = np.indices(dimensions=(height, width))  # arrays containing row / col index

    # build state transition arrays
    dy = np.array(
        [
            [[0]],  # West
            [[0]],  # East
            [[-1]],  # South
            [[1]],  # North
        ]
    )
    dx = np.array(
        [
            [[-1]],  # West
            [[1]],  # East
            [[0]],  # South
            [[0]],  # North
        ]
    )  # shape (4, 1, 1)
    rows = np.broadcast_to(array=rows, shape=(4, height, width))  # shape to work with dx and dy
    cols = np.broadcast_to(array=cols, shape=(4, height, width))
    next_rows = np.clip(a=rows + dy, a_min=0, a_max=height - 1)  # clip destinations to inside map
    next_cols = np.clip(a=cols + dx, a_min=0, a_max=width - 1)

    # build immediate rewards array (one step reward for every cell and action)
    rewards = (
        GOAL_REWARD * goal_mask[next_rows, next_cols]  # next cell reward
        - PROXIMITY_PENALTY * proximity_map[None, :, :]  # current cell proximity penalty
        - PROXIMITY_PENALTY * proximity_map[next_rows, next_cols]  # next cell proximity penalty
    )

    while max_iterations > 0:
        max_iterations -= 1

        # q for all cells for all actions
        q = rewards + gamma * v[next_rows, next_cols]  # Bellman equation with broadcasting

        # best of the 4 actions at each cell
        v_new = np.max(q, axis=0)
        v_new[zero_rewards] = 0.0  # zero goal and occupied cells

        error = np.max(np.abs(v_new - v))
        v = v_new

        if error <= tol:
            break

    return v


def greedy_action(
    v: ArrayLike,
    cell: tuple[int, int],
    goal_mask: ArrayLike,
    obstacle_map: NDArray[np.float64] = BASE_MAP,
    proximity_map: NDArray[np.float64] = PROXIMITY_MAP_ON,
    gamma: float = GAMMA,
) -> Literal[0, 1, 2, 3]:
    """The best action at ``cell``: the one with the largest Q value.

    Computes ``q_value`` for the 4 actions and returns the index of the largest;
    ties go to the lowest action index.

    Paramters
    ---------
    v : array of shape ``(H, W)``
        The current value function. Its shape defines the grid.
    cell : tuple[int, int]
        ``(i, j)``, the (row, col) of the current cell.
    goal_mask : boolean array of shape ``(H, W)``,
        True at the goal cell(s); there may be more than one.
    obstacle_map : array of shape ``(H, W)``, default BASE_MAP
        Nonzero at obstacles.
    proximity_map : array of shape ``(H, W)``, default PROXIMITY_MAP_ON
        Nonzero where the robot is too close to something.
    gamma : float, default GAMMA
        The discount factor.

    Returns
    -------
    Literal[0, 1, 2, 3]
        The best action as a Python int in {0, 1, 2, 3}.

    Notes
    -----
    Used for the yellow path preview in Webots, and by ``vote_action``.
    """
    actions: tuple[Literal[0, 1, 2, 3], ...] = (0, 1, 2, 3)

    # rounded so tiny floating-point differences between implementations don't change a tie
    q = [
        round(
            float(
                q_value(
                    v=v,
                    cell=cell,
                    action=a,
                    goal_mask=goal_mask,
                    obstacle_map=obstacle_map,
                    proximity_map=proximity_map,
                    gamma=gamma,
                )
            ),
            6,
        )
        for a in actions
    ]

    return actions[np.argmax(q)]


def vote_action(
    v: ArrayLike,
    particles: ArrayLike,
    weights: ArrayLike,
    goal_mask: ArrayLike,
    obstacle_map: NDArray[np.float64] = BASE_MAP,
    proximity_map: NDArray[np.float64] = PROXIMITY_MAP_ON,
    gamma: float = GAMMA,
) -> Literal[0, 1, 2, 3]:
    """Choose one action for a whole particle belief by weighted voting.

    While the robot is unsure where it is, every particle votes for ``greedy_action``
    at its own cell, and each vote counts as much as the particle's weight.

    Particles off the map, in a goal or obstacle cell, or with weight 0 do not vote.

    Returns the action with largest total weight (ties: the lowest index; no valid particle: 0).

    Parameters
    ----------
    v : array of shape ``(H, W)``
        The current value function. Its shape defines the grid.
    particles : array_like of shape (N, 2)
        Particle positions ``(x, y)`` in meters. Supply a single particle
        with shape ``(1, 2)``.
    weights : array_like of shape (N,)
        Nonnegative weights with a positive sum. They need not be normalized;
        zero-weight particles do not contribute. The input is not modified.
    goal_mask : boolean array of shape ``(H, W)``,
        True at the goal cell(s); there may be more than one.
    obstacle_map : array of shape ``(H, W)``, default BASE_MAP
        Nonzero at obstacles.
    proximity_map : array of shape ``(H, W)``, default PROXIMITY_MAP_ON
        Nonzero where the robot is too close to something.
    gamma : float, default GAMMA
        The discount factor.

    Returns
    -------
    Literal[0, 1, 2, 3]
        The winning action as a Python int in {0, 1, 2, 3}.

    Notes
    -----
    In the Webots demo the robot picks every move this way.
    """
    # actions literal
    actions: tuple[Literal[0, 1, 2, 3], ...] = (0, 1, 2, 3)

    # convert ArrayLike and particles to arrays
    rows, cols, inside = points_to_cells(particles)
    weights = np.asarray(weights, dtype=float)
    goal = np.asarray(goal_mask, dtype=bool)
    valid = (
        inside & ~goal[rows, cols] & (np.asarray(obstacle_map)[rows, cols] == 0) & (weights > 0)
    )  # particles are only valid if inside map, outside obstacles and positive weight

    votes = np.zeros(4)
    tally = {}

    # combine weights of particles in the same cell
    for i, j, w in zip(rows[valid], cols[valid], weights[valid], strict=True):
        tally[(int(i), int(j))] = tally.get((int(i), int(j)), 0.0) + float(w)

    # each cell votes for its best action using its combined weight
    for cell, w in tally.items():
        votes[greedy_action(v, cell, goal, obstacle_map, proximity_map, gamma)] += w

    return actions[np.argmax(votes)]


###############################################################################
#          WEBOTS ROBOT                                                       #
###############################################################################


def omni_jacobian(
    num_wheels: int = NUM_WHEELS, wheel_radius: float = WHEEL_RADIUS
) -> NDArray[np.float64]:
    """For Webots robot. The Jacobian J that maps the velocity (vx, vy) to wheel speeds.

    Wheel ``i`` sits at the angle ``theta_i = 2 pi i / num_wheels`` and row
    ``i`` of J is ``[-sin(theta_i), cos(theta_i)] / r``, so ``omega = J @ v``.

    Parameters
    ----------
    num_wheels : int, default NUM_WHEELS
        Number of wheels in the robot.
    wheel_radius : float, default WHEEL_RADIUS
        Radius of the wheels. Same for all.

    Returns
    -------
    NDArray[np.float64]
        Float array of shape ``(num_wheels, 2)``.
    """
    thetas = 2 * np.pi * np.arange(num_wheels) / num_wheels  # get wheel angles relative to world
    return np.column_stack((-np.sin(thetas), np.cos(thetas))) / wheel_radius  # Jacobian for each


def estimate_motion_model(
    states: ArrayLike, controls: ArrayLike, next_states: ArrayLike
) -> tuple[np.ndarray, np.ndarray, float]:
    """For Webots robot. Estimate A, B, and sigma of x' = A x + B u + w from data.

    Least squares for ``[A B]`` and the unbiased noise estimate from the residuals.
    At boot, the robot gets a calibration log of 400 (state, control, next state) samples
    and uses sigma_hat as the motion noise of its particle filter (the boot screen
    shows the result).

    Parameters
    ----------
    states : array_like of shape (N, 2)
        Current positions ``(x, y)`` for N motion samples, in meters.
    controls : array_like of shape (N, 2)
        Control vectors ``(ux, uy)`` applied at each sample.
    next_states : array_like of shape (N, 2)
        Observed positions ``(x, y)`` after applying the corresponding controls,
        in meters. Row k corresponds to row k of ``states`` and ``controls``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(A_hat, B_hat, sigma_hat)``. Two ``(2, 2)`` arrays and a float.
    """
    X = np.asarray(states, dtype=float)
    U = np.asarray(controls, dtype=float)
    Z = np.asarray(next_states, dtype=float)
    XU = np.hstack((X, U))  # (N, 4): one row [x, y, ux, uy] per sample

    theta = np.linalg.solve(XU.T @ XU, XU.T @ Z)  # (4, 2) == [A B]^T
    A_hat, B_hat = theta[:2].T, theta[2:].T

    residuals = Z - XU @ theta
    n_samples = len(X)
    sigma_hat = float(np.sqrt(np.sum(residuals**2) / (2 * n_samples - 8)))

    return A_hat, B_hat, sigma_hat
