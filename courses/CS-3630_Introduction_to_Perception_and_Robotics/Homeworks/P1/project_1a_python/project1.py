"""CS 3630 Project 1: Python Fundamentals."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Section 0: setup and submission
def student_info() -> dict[str, str]:
    """Return your name and GT username in the provided dictionary."""
    return {"name": "Joao Pedro Dudziak Fonseca", "gt_username": "jfonseca32"}


# Section 1: variables, operators, and strings — 4 points total
def describe_point(name: str, x: float, y: float) -> str:
    """Build one sentence describing a labeled 2D point.

    Args:
        name: The point's label.
        x: Its x-coordinate.
        y: Its y-coordinate.

    Returns:
        A string formatted exactly as ``"<name> is at (<x>, <y>)"``.

    Example:
        ``describe_point("A", 1.5, -2)`` returns
        ``"A is at (1.5, -2)"``. Use an f-string rather than joining strings.
    """
    return f"{name} is at ({x}, {y})"


# Section 2: conditionals and loops
def temperature_label(celsius: float) -> str:
    """
    Choose a temperature label using an if/elif/else statement.

    Returns "cold" when celsius is below 10, "mild" when at
    least 10 but below 25, and "hot" otherwise.
    """
    if celsius < 10:
        return "cold"
    elif celsius >= 10 and celsius < 25:
        return "mild"
    else:
        return "hot"


def count_below(values: list[float], threshold: float) -> int:
    """
    Count how many values are strictly below a threshold.

    Starts a counter at zero, visits each item with a for loop, and
    increments counter when appropriate.

    A value equal to threshold is not below it.
    An empty list returns zero.
    """
    if not values:
        return 0

    # start counter and loop
    counter = 0
    for value in values:
        if value < threshold:
            counter += 1

    return counter


# Section 3: lists, tuples, and dictionaries
def summarize_values(values: list[float]) -> tuple[float, float]:
    """
    Summarize a nonempty list of numeric values.

    Returns:
        A two-item tuple (average, minimum).
        Computes the average using sum and len and the smallest value using min.

    Raises:
    ValueError for an empty list because its average and minimum are undefined.
    Returning two comma-separated values automatically makes a tuple in Python.
    """
    if not values:
        raise ValueError("List cannot be empty.")

    average = sum(values) / len(values)
    smallest = min(values)

    return (average, smallest)


def update_position(record: dict[str, object], x: float, y: float) -> dict[str, object]:
    """
    Update the position stored in a record dictionary.

    Set the "position" key to a new list [x, y].
    Intentionally mutates dictionary supplied by the caller.

    Returns
    Same dictionary object after updating it; all of its other keys preserved.
    """
    # Reassign values
    record["position"] = [x, y]
    return record


# Section 4: useful Python idioms
def squared_values(values: list[float]) -> list[float]:
    """
    Create a new list containing the square of each input value.

    Uses a list comprehension of the form [expression for item in values].
    Keeps the original order and does not modify values.
    An empty input produces an empty list.
    """
    if not values:
        return []

    squared = [value**2 for value in values]
    return squared


def pair_items(names: list[str], values: list[float]) -> list[tuple[int, str, float]]:
    """
    Combines parallel names and values with indices.

    Returns:
        Tuples (index, name, value) in a list.

    Notes:
        First uses zip to pair each name with its value, then
        enumerate those pairs to obtain indices beginning at zero.
        Like ordinary zip, stops when the shorter input list runs out.
    """
    # Zips names and values and enumerates together; strict=False like ordinary zip
    pairs = [(i, name, val) for i, (name, val) in enumerate(zip(names, values, strict=False))]
    return pairs


# Section 5: NumPy basics
def scale_vector(values: ArrayLike, factor: float) -> NDArray[np.floating]:
    """
    Converts values to a float NumPy array and scales every element.

    values may be a Python list or an array.
    Uses ``np.asarray`` with a float dtype, then multiplies the whole array by factor.
    NumPy multiplication is elementwise; unlike [1, 2] * 2, the result should be [2.0, 4.0].
    Does not modify the input.
    """
    # Convert first to float array, then multiply elementwise
    return np.asarray(values, dtype=float) * factor


def column_means(matrix: ArrayLike) -> NDArray[np.floating]:
    """
    Calculates one average for each column of a 2D collection.

    Converts matrix to a float array, then uses np.mean with the axis that
    keeps columns and combines rows.

    Examples:
        For ``[[1, 2], [3, 6]]``, returns the array ``[2.0, 4.0]``.
    """
    # create np array and column mean of it
    means = np.mean(np.asarray(matrix, dtype=float), axis=0)
    return means


def vector_length(vector: ArrayLike) -> float:
    """
    Return the Euclidean length of a vector as a Python float.

    Uses np.linalg.norm rather than writing a loop.

    For example, the length of [3, 4] is 5.
    Converts NumPy's scalar result with float(...) before returning it.
    """
    distance = np.linalg.norm(vector, ord=2)  # Euclidean L2 is ord=2
    return float(distance)  # cast for safety (from np.float64)


# Section 6: classes and imports
class Point:
    """A minimal mutable 2D point used to practice object syntax."""

    def __init__(self, x: float = 0, y: float = 0) -> None:
        self.x = x
        self.y = y

    def move(self, dx: float, dy: float) -> tuple[float, float]:
        """
        Move this object by (dx, dy), then return its updated position.

        Adds dx to the existing self.x and dy to self.y.

        Because attributes store object state, repeated calls must build on the prior
        position.

        Returns:
            (self.x, self.y) after the update.
        """
        self.x = self.x + dx  # update values
        self.y = self.y + dy

        return (self.x, self.y)


def direction_from_degrees(degrees: float) -> tuple[float, float]:
    """
    Convert an angle in degrees into a 2D unit direction.

    The functions math.cos and math.sin expect radians.

    First converts with math.radians(degrees), then returns (cos(theta), sin(theta)).
    Small floating-point errors are normal; 90 degrees is approximately (0, 1).
    """
    rad = math.radians(degrees)
    return (math.cos(rad), math.sin(rad))
