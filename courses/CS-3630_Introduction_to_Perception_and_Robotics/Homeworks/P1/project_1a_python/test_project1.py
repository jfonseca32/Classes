"""Public tests. Set P1_MODULE=instructor.solution for the staff solution."""

import importlib
import os

import numpy as np
import pytest

p1 = importlib.import_module(os.environ.get("P1_MODULE", "project1"))


def test_setup():
    info = p1.student_info()
    assert set(info) == {"name", "gt_username"}
    assert all(
        isinstance(value, str) and value.strip() and "todo" not in value.lower()
        for value in info.values()
    )


def test_describe_point():
    assert p1.describe_point("A", 1.5, -2) == "A is at (1.5, -2)"


def test_temperature_label():
    assert [p1.temperature_label(x) for x in [5, 10, 24, 25]] == ["cold", "mild", "mild", "hot"]


def test_count_below():
    assert p1.count_below([0.2, 1.0, 0.9, 3.0], 1.0) == 2


def test_summarize_values():
    assert p1.summarize_values([1.0, 2.0, 6.0]) == pytest.approx((3.0, 1.0))


def test_update_position():
    record = {"name": "Ada", "position": [0, 0]}
    returned = p1.update_position(record, 3, -1)
    assert returned is record and record["position"] == [3, -1]


def test_squared_values():
    assert p1.squared_values([-2, 0, 3]) == [4, 0, 9]


def test_pair_items():
    assert p1.pair_items(["apple", "banana"], [1.2, 0.7]) == [(0, "apple", 1.2), (1, "banana", 0.7)]


def test_scale_vector():
    result = p1.scale_vector([1, 2, 3], 2.5)
    assert isinstance(result, np.ndarray) and result.dtype.kind == "f"
    np.testing.assert_allclose(result, [2.5, 5, 7.5])


def test_column_means():
    np.testing.assert_allclose(p1.column_means([[1, 2], [3, 6]]), [2, 4])


def test_vector_length():
    assert p1.vector_length([3, 4]) == pytest.approx(5)


def test_point_move():
    point = p1.Point(1, 2)
    assert point.move(3, -1) == (4, 1)
    assert (point.x, point.y) == (4, 1)


def test_direction_from_degrees():
    np.testing.assert_allclose(p1.direction_from_degrees(90), [0, 1], atol=1e-12)
