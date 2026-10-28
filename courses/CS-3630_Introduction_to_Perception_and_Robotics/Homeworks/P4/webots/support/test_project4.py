"""Public tests for CS 3630 Project 4: Autonomous Warehouse Challenge.

The easiest way to run them is the ``check(...)`` cell after each TODO in
``project4.ipynb``. From a terminal in the project folder (the one that
contains ``project4.py``):

    python -m pytest -q                      (all tests)
    python -m pytest -q -k rfid_likelihood   (only the tests of one function)

Every test is named after the function it checks. Until you implement a
function, its tests fail with ``NotImplementedError: TODO <n>``.

Each test checks one function. When a TODO builds on an earlier one, its
tests swap in a simple stand-in for the earlier function. For example, the
``mcl_predict`` tests use their own motion model, so a bug in
``sample_motion_model`` fails the ``sample_motion_model`` tests, not the
``mcl_predict`` tests.

Gradescope runs these tests plus hidden tests of the edge cases described in
the docstrings of ``project4.py``.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# project4.py lives two folders up (in the student release) or next to this file
# (on Gradescope); make both importable however pytest is launched.
for _folder in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[2]):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

p4 = importlib.import_module(os.environ.get("P4_MODULE", "project4"))

GOAL = (35, 50)  # the book's goal cell in Section 4.5.1
ROBOT_COLORS = {"gold", "navy", "orange", "cyan", "magenta", "lime", "white", "red"}


def _goal_mask(*cells, shape=(50, 100)):
    """Boolean grid that is True exactly at the given cells."""
    mask = np.zeros(shape, dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


def _manhattan_value(goal_cell, gamma=0.975, shape=(50, 100)):
    """The converged value function of an empty warehouse (no obstacles, no
    proximity): 100 * gamma**(d - 1) at Manhattan distance d >= 1 from the goal
    cell, and 0 at the goal itself."""
    rows, cols = np.indices(shape)
    distance = np.abs(rows - goal_cell[0]) + np.abs(cols - goal_cell[1])
    return np.where(distance == 0, 0.0, 100.0 * gamma ** (distance - 1.0))


###############################################################################
#                                    SETUP                                    #
###############################################################################


def test_student_info():
    info = p4.student_info()
    assert set(info) == {"name", "gt_username"}
    for value in info.values():
        text = value.strip().lower() if isinstance(value, str) else ""
        # a placeholder is empty, "TODO", or still contains "TODO:" (a name like "Todorov" is fine)
        assert text and text != "todo" and "todo:" not in text, (
            "replace both TODO strings in student_info()"
        )


def test_robot_profile_format():
    """Not graded: checks that the Webots HUD can show your robot's name and color."""
    profile = p4.robot_profile()
    name, color = profile["name"], profile["color"]
    assert isinstance(name, str) and name.strip() and len(name) <= 16, "name: 1 to 16 characters"
    assert all(32 <= ord(ch) < 127 for ch in name), "name: plain ASCII characters only"
    assert color in ROBOT_COLORS, f"color must be one of {sorted(ROBOT_COLORS)}"


###############################################################################
#                        PART 1: THE MOTION MODEL (4.2)                       #
###############################################################################


def test_sample_motion_model_statistics():
    states = np.tile([10.0, 20.0], (20000, 1))
    moved = p4.sample_motion_model(states, np.array([1.0, -2.0]), 0.5, np.random.default_rng(0))
    assert moved.shape == (20000, 2)
    np.testing.assert_allclose(
        moved.mean(axis=0),
        [11.0, 18.0],
        atol=0.02,
        err_msg="the mean of 20000 samples must be x + u; a random offset usually means "
        "that every state got the SAME noise vector",
    )
    np.testing.assert_allclose(
        moved.std(axis=0),
        [0.5, 0.5],
        rtol=0.05,
        err_msg="every coordinate of every state needs its own noise sample",
    )
    assert abs(np.corrcoef(moved.T)[0, 1]) < 0.05, "x and y need independent noise"


def test_sample_motion_model_uses_rng():
    states = np.array([[1.0, 2.0], [30.0, 40.0], [5.0, 5.0]])
    first = p4.sample_motion_model(states, [0.5, 0.0], 0.2, np.random.default_rng(7))
    again = p4.sample_motion_model(states, [0.5, 0.0], 0.2, np.random.default_rng(7))
    other = p4.sample_motion_model(states, [0.5, 0.0], 0.2, np.random.default_rng(8))
    assert first.shape == (3, 2)
    np.testing.assert_array_equal(
        first, again, err_msg="same seed, same samples: draw all noise from rng"
    )
    assert not np.array_equal(first, other)
    np.testing.assert_allclose(first, states + [0.5, 0.0], atol=1.5)


###############################################################################
#                         PART 1: SENSOR MODELS (4.3)                         #
###############################################################################


# The book's RFID simulator (Section 4.3.3), used below to build the expected likelihoods.
def _book_rfid_range(position, beacon, max_range=8):
    distance = np.linalg.norm(position - beacon)
    return float("inf") if distance > max_range else distance


def _book_rfid_measurement(position, max_range=8):
    ranges = [_book_rfid_range(position, beacon, max_range) for beacon in p4.BEACONS]
    nearest = min(ranges)
    return (np.argmin(ranges), nearest) if nearest <= max_range else (None, nearest)


def test_rfid_likelihood_book_annulus():
    # Section 4.3.2: at (20.5, 7.5) the robot measures a range of 4.03 m to beacon 0.
    measurement = _book_rfid_measurement(np.array([20.5, 7.5]))  # (np.int64(0), 4.03...)
    assert measurement[0] == 0 and measurement[1] == pytest.approx(4.03, abs=5e-3)
    likelihood = p4.rfid_likelihood(p4.MAP_COORDS.reshape(-1, 2), measurement)
    assert likelihood.shape == (5000,)
    # The book's Figure 5: exp(-(z - d)^2 / (2 sigma^2)) with d = inf beyond 8 m.
    distance = np.array(
        [[_book_rfid_range(xy, p4.BEACONS[0]) for xy in row] for row in p4.MAP_COORDS]
    )
    expected = np.exp(-1 / (2 * 1.0**2) * (measurement[1] - distance) ** 2)
    np.testing.assert_allclose(
        likelihood.reshape(50, 100),
        expected,
        atol=1e-12,
        err_msg="the annulus around beacon 0 (0 is a real beacon id: test `beacon_id is None`)",
    )
    assert p4.rfid_likelihood(np.array([[20.5, 7.5]]), measurement)[0] == pytest.approx(1.0)


def test_rfid_likelihood_out_of_range_matches_book_map():
    # The book's out_of_bound_map (Figure 6): 1 where no beacon is within 8 m, else 0.
    out_of_bound_map = np.array(
        [[_book_rfid_measurement(xy)[0] is None for xy in row] for row in p4.MAP_COORDS]
    )
    likelihood = p4.rfid_likelihood(p4.MAP_COORDS.reshape(-1, 2), (None, float("inf")))
    np.testing.assert_array_equal(
        likelihood.reshape(50, 100),
        out_of_bound_map.astype(float),
        err_msg="(None, inf): 1 only where EVERY beacon is farther than max_range",
    )


def test_rfid_likelihood_zero_beyond_max_range():
    beacon = p4.BEACONS[0]
    # 9 m from beacon 0: even a matching 9 m reading is impossible, the reader only hears beacons
    # within 8 m.
    assert p4.rfid_likelihood(np.array([beacon + [0.0, -9.0]]), (0, 9.0))[0] == 0.0
    # 7 m away: a 7 m reading is a perfect match, a 6 m reading is one sigma off.
    assert p4.rfid_likelihood(np.array([beacon + [0.0, -7.0]]), (0, 7.0))[0] == pytest.approx(1.0)
    assert p4.rfid_likelihood(np.array([beacon + [0.0, -7.0]]), (0, 6.0))[0] == pytest.approx(
        np.exp(-0.5)
    )


def test_zone_likelihood_reads_the_map():
    # (32.5, 25) is in AISLE 1 (zone 10), (5, 25) in RECEIVING 3W (zone 4), (25, 25) inside shelf A.
    points = np.array([[32.5, 25.0], [5.0, 25.0], [25.0, 25.0], [-3.0, 5.0]])
    likelihood = p4.zone_likelihood(points, 10)
    assert likelihood.shape == (4,)
    # the right zone: 0.9; another zone: 0.1 / 24; in a shelf or off the map: 0
    np.testing.assert_allclose(likelihood, [0.9, 0.1 / 24, 0.0, 0.0], rtol=1e-12)


def test_zone_likelihood_names_and_parameters():
    assert p4.ZONE_NAMES[p4.ZONE_MAP[25, 5]] == "RECEIVING 3W"
    assert p4.ZONE_NAMES[p4.ZONE_MAP[45, 50]] == "CROSS AISLE N"
    likelihood = p4.zone_likelihood(np.array([[50.0, 45.0], [50.0, 5.0]]), 14, p_correct=0.7)
    np.testing.assert_allclose(likelihood, [0.7, 0.3 / 24], rtol=1e-12)


###############################################################################
#                    PART 1: MONTE CARLO LOCALIZATION (4.4)                   #
###############################################################################


def _noise_free_motion(states, u, sigma, rng):
    """Stand-in for sample_motion_model without the noise: x' = x + u."""
    return np.asarray(states, dtype=float) + np.asarray(u, dtype=float)


def _jitter_motion(states, u, sigma, rng):
    """Stand-in for sample_motion_model: x' = x + u plus a uniform jitter in [-sigma, sigma) drawn
    from rng.
    """
    states = np.asarray(states, dtype=float)
    return states + np.asarray(u, dtype=float) + rng.uniform(-sigma, sigma, size=states.shape)


# The mcl_predict tests swap in the simple stand-ins above for your sample_motion_model, so they
# check only what mcl_predict itself does (resampling, then moving); TODO 1 has its own tests.


def test_mcl_predict_resamples_by_weight(monkeypatch):
    monkeypatch.setattr(p4, "sample_motion_model", _noise_free_motion)
    a, b = np.array([20.0, 5.0]), np.array([70.0, 45.0])
    particles = np.vstack([np.tile(a, (2000, 1)), np.tile(b, (2000, 1))])
    weights = np.concatenate(
        [np.full(2000, 1e-3), np.full(2000, 3e-3)]
    )  # unnormalized: b is 3x as likely
    u = np.array([3.0, -2.0])
    moved = p4.mcl_predict(particles, weights, u, 0.2, np.random.default_rng(0))
    assert moved.shape == (4000, 2)
    near_a = np.linalg.norm(moved - (a + u), axis=1) < 1.5
    near_b = np.linalg.norm(moved - (b + u), axis=1) < 1.5
    assert np.all(near_a | near_b), "every new particle is a chosen old particle, moved by u"
    assert near_b.mean() == pytest.approx(0.75, abs=0.04), "resample in proportion to the weights"


def test_mcl_predict_num_samples_and_rng(monkeypatch):
    monkeypatch.setattr(p4, "sample_motion_model", _noise_free_motion)
    setup = np.random.default_rng(1)
    particles = setup.uniform([0.0, 0.0], [99.0, 49.0], size=(300, 2))
    weights = setup.uniform(0.0, 1.0, size=300)
    small = p4.mcl_predict(
        particles, weights, [0.0, 1.0], 0.3, np.random.default_rng(5), num_samples=50
    )
    assert small.shape == (50, 2)
    again = p4.mcl_predict(
        particles, weights, [0.0, 1.0], 0.3, np.random.default_rng(5), num_samples=50
    )
    np.testing.assert_array_equal(
        small, again, err_msg="same seed, same particles: draw all randomness from rng"
    )
    assert p4.mcl_predict(particles, weights, [0.0, 1.0], 0.3, np.random.default_rng(5)).shape == (
        300,
        2,
    )


def test_mcl_predict_every_copy_gets_its_own_noise(monkeypatch):
    monkeypatch.setattr(p4, "sample_motion_model", _jitter_motion)
    particles = np.array([[10.0, 10.0], [60.0, 30.0]])
    # Only the first particle can be chosen, so the 500 new particles are all copies of it, moved by
    # u.
    moved = p4.mcl_predict(
        particles, np.array([2.0, 0.0]), [1.0, 0.0], 0.5, np.random.default_rng(3), num_samples=500
    )
    assert moved.shape == (500, 2)
    np.testing.assert_allclose(moved, np.tile([11.0, 10.0], (500, 1)), atol=0.5)
    distinct = len(np.unique(moved, axis=0))
    assert distinct == 500, (
        f"resample first, then move: every copy needs its own noise ({distinct} distinct of 500)"
    )


def test_mcl_update_multiplies_likelihoods():
    particles = np.random.default_rng(2).uniform([-2.0, -2.0], [101.0, 51.0], size=(500, 2))
    zone = 10  # AISLE 1
    for z_on in (True, False):
        for rfid in ((0, 4.0), (5, 6.5), (None, float("inf"))):
            expected = p4.proximity_likelihood(particles, z_on) * p4.rfid_likelihood(
                particles, rfid
            )
            weights = p4.mcl_update(particles, z_on, rfid)
            assert weights.shape == (500,)
            np.testing.assert_allclose(weights, expected, rtol=1e-12)
            with_zone = p4.mcl_update(particles, z_on, rfid, zone)
            np.testing.assert_allclose(
                with_zone, expected * p4.zone_likelihood(particles, zone), rtol=1e-12
            )


def test_mcl_update_keeps_raw_weights():
    # A zone reading that disagrees with every particle (SHIPPING 5E, zone 24) makes every weight
    # small, as right after a kidnapping. mcl_update must return that raw product, not normalized
    # weights.
    particles = np.array([[50.0, 25.0], [50.4, 25.0], [30.0, 45.0]])
    no_beacon = (None, float("inf"))
    weights = p4.mcl_update(particles, False, no_beacon, 24)
    expected = (
        p4.proximity_likelihood(particles, False)
        * p4.rfid_likelihood(particles, no_beacon)
        * p4.zone_likelihood(particles, 24)
    )
    assert np.sum(weights) < 0.1
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=0.0)


def test_estimate_position_weighted():
    particles = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    mean, cov = p4.estimate_position(particles, np.array([1.0, 1.0, 2.0]))  # unnormalized weights
    np.testing.assert_allclose(mean, [0.5, 2.0], rtol=1e-12)
    np.testing.assert_allclose(cov, [[0.75, -1.0], [-1.0, 4.0]], rtol=1e-12)


def test_estimate_position_equal_weights():
    particles = np.random.default_rng(4).normal([30.0, 12.0], [2.0, 0.5], size=(1000, 2))
    mean, cov = p4.estimate_position(particles, np.full(1000, 0.001))
    assert mean.shape == (2,) and cov.shape == (2, 2)
    np.testing.assert_allclose(mean, particles.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(
        cov, np.cov(particles.T, bias=True), rtol=1e-10, err_msg="population covariance (no N - 1)"
    )


###############################################################################
#                        PART 2: VALUE ITERATION (4.5)                        #
###############################################################################


def test_q_value_book_examples():
    goal = _goal_mask(GOAL)
    V = np.zeros((50, 100))
    for cell, action in [
        ((35, 49), p4.EAST),
        ((35, 51), p4.WEST),
        ((36, 50), p4.SOUTH),
        ((34, 50), p4.NORTH),
    ]:
        assert p4.q_value(V, cell, action, goal) == pytest.approx(100.0), (
            "arriving at the goal earns 100"
        )
    assert p4.q_value(V, (35, 49), p4.WEST, goal) == pytest.approx(0.0)
    V = np.full((50, 100), 10.0)
    assert p4.q_value(V, (35, 49), p4.WEST, goal) == pytest.approx(9.75)  # 0 + 0.975 * 10
    assert p4.q_value(V, (35, 49), p4.EAST, goal) == pytest.approx(109.75)  # 100 + 0.975 * 10
    assert p4.q_value(V, GOAL, p4.EAST, goal) == 0.0  # already at the goal
    assert p4.q_value(V, (20, 24), p4.EAST, goal) == 0.0  # inside shelf A


def test_q_value_proximity_penalties():
    goal = _goal_mask(GOAL)
    V = np.zeros((50, 100))
    # (8, 30) is in the 2 m band around shelf A; (8, 31) and (7, 30) are not; (8, 29) is.
    assert p4.q_value(V, (8, 30), p4.EAST, goal) == pytest.approx(-50.0)
    assert p4.q_value(V, (8, 30), p4.SOUTH, goal) == pytest.approx(-50.0)
    assert p4.q_value(V, (8, 30), p4.WEST, goal) == pytest.approx(-100.0)
    assert p4.q_value(V, (7, 30), p4.NORTH, goal) == pytest.approx(-50.0)  # moving into the band
    assert p4.q_value(V, (7, 30), p4.SOUTH, goal) == pytest.approx(0.0)
    # Driving into the wall leaves the robot in the corner cell (0, 0), inside the wall band.
    V[0, 0] = 4.0
    assert p4.q_value(V, (0, 0), p4.WEST, goal) == pytest.approx(-100.0 + 0.975 * 4.0)
    assert p4.q_value(V, (0, 0), p4.SOUTH, goal) == pytest.approx(-100.0 + 0.975 * 4.0)


def test_q_value_uses_the_next_cells_value():
    goal, V = _goal_mask(GOAL), _manhattan_value(GOAL)  # here V differs from cell to cell
    # Q = reward + gamma * V[next cell], in the open aisle south of the goal:
    assert p4.q_value(V, (30, 49), p4.NORTH, goal) == pytest.approx(0.975 * V[31, 49])
    assert p4.q_value(V, (30, 49), p4.SOUTH, goal) == pytest.approx(0.975 * V[29, 49])
    assert p4.q_value(V, (30, 49), p4.NORTH, goal, gamma=0.5) == pytest.approx(0.5 * V[31, 49])
    # From (8, 34), the cells EAST and NORTH are equally close to the goal, but EAST enters the
    # 2 m band around shelf B and costs 50, so NORTH is the better move.
    assert p4.q_value(V, (8, 34), p4.EAST, goal) == pytest.approx(-50.0 + 0.975 * V[8, 35])
    assert p4.q_value(V, (8, 34), p4.NORTH, goal) == pytest.approx(0.975 * V[9, 34])


@functools.cache
def _book_value_function():
    """value_iteration for the book's goal (35, 50), computed once and shared by the tests."""
    return p4.value_iteration(_goal_mask(GOAL))


def test_value_iteration_book_warehouse():
    V = _book_value_function()
    assert V.shape == (50, 100)
    assert V[GOAL] == 0.0
    assert np.all(V[p4.BASE_MAP == 1] == 0.0), "obstacle cells keep V = 0"
    for cell in [(35, 49), (35, 51), (34, 50), (36, 50)]:
        assert V[cell] == pytest.approx(100.0)  # one step from the goal
    assert V[35, 48] == pytest.approx(97.5)  # two steps: 0.975 * 100
    assert V[25, 50] == pytest.approx(100.0 * 0.975**9)  # ten steps straight up the aisle
    assert V.max() == pytest.approx(100.0)
    # Farther away, and near the walls and shelves, where the proximity penalties bite:
    expected = {
        (5, 5): 15.358243952130534,
        (45, 90): 28.921856650350446,
        (8, 30): -20.33655728169185,
        (0, 0): -326.9824090456218,
    }
    for cell, value in expected.items():
        assert V[cell] == pytest.approx(value, abs=1e-6), f"V{cell}"


def test_value_iteration_one_sweep():
    # One synchronous sweep from V = 0 gives every cell its best immediate reward.
    V1 = p4.value_iteration(_goal_mask(GOAL), max_iterations=1)
    assert V1[35, 49] == pytest.approx(100.0)
    for cell in [(35, 48), (36, 49), (36, 51), (37, 50), (5, 5)]:
        assert V1[cell] == pytest.approx(0.0), (
            f"V1{cell}: compute every sweep from the previous V only"
        )
    assert V1[0, 0] == pytest.approx(-100.0)  # in the wall band, and every move stays in the band
    assert V1[8, 30] == pytest.approx(-50.0)  # in the band, but moving east leaves it
    assert set(np.unique(V1)) <= {-100.0, -50.0, 0.0, 100.0}


def test_value_iteration_book_sweep_count():
    # The book converges "after 86 iterations, error = 0.0": the 86th sweep changes nothing,
    # so 85 sweeps already give the converged V, but 84 sweeps do not.
    V = _book_value_function()
    np.testing.assert_allclose(
        p4.value_iteration(_goal_mask(GOAL), max_iterations=85), V, atol=1e-9
    )
    assert not np.allclose(p4.value_iteration(_goal_mask(GOAL), max_iterations=84), V)
