"""A small ASCII picture of the warehouse for the offline runner.

One character covers 2 m x 2 m; north is up. Legend::

    R robot (truth)   E estimate   @ both   . particles   G goal   * planned path
    # shelves         : proximity band      X obstacle    F forklift   b RFID beacon
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .records import StepResult
from .world import BASE_MAP, BEACONS, PROXIMITY_MAP_ON

CELL = 2  # meters per character
LEGEND = (
    "R robot  E estimate  @ both  . particles  G goal  * path\n"
    "# shelf  : proximity band  X obstacle  F forklift  b RFID beacon"
)


def _block(x: float, y: float):
    """Character (row from the top, column) that contains the point (x, y)."""
    col = int(np.clip(np.floor((x + 0.5) / CELL), 0, C.MAP_WIDTH // CELL - 1))
    row = int(np.clip(np.floor((y + 0.5) / CELL), 0, C.MAP_HEIGHT // CELL - 1))
    return C.MAP_HEIGHT // CELL - 1 - row, col


def render(result: StepResult) -> str:
    """Render one StepResult as a framed 50 x 25 character map plus a legend."""
    rows, cols = C.MAP_HEIGHT // CELL, C.MAP_WIDTH // CELL
    base = BASE_MAP.reshape(rows, CELL, cols, CELL).max(axis=(1, 3))
    prox = PROXIMITY_MAP_ON.reshape(rows, CELL, cols, CELL).max(axis=(1, 3))
    canvas: list[list[str]] = [[" "] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if base[rows - 1 - r, c]:
                canvas[r][c] = "#"
            elif prox[rows - 1 - r, c]:
                canvas[r][c] = ":"

    def put(x: float, y: float, ch: str) -> None:
        r, c = _block(x, y)
        canvas[r][c] = ch

    for bx, by in BEACONS:
        put(bx, by, "b")
    if result.belief is not None:
        particles = result.belief.particles
        for x, y in particles[:: max(1, len(particles) // 400)]:
            if C.FLOOR_X[0] <= x <= C.FLOOR_X[1] and C.FLOOR_Y[0] <= y <= C.FLOOR_Y[1]:
                r, c = _block(x, y)
                if canvas[r][c] in " :":
                    canvas[r][c] = "."
    for i, j in result.path:
        put(j, i, "*")
    for ob in result.obstacles:
        put(ob["x"], ob["y"], "X")
    for actor in result.actors:
        put(actor["x"], actor["y"], "F")
    if result.goal_cell is not None:
        put(result.goal_cell[1], result.goal_cell[0], "G")
    robot = _block(result.end[0], result.end[1])
    if result.belief is not None:
        estimate = _block(result.belief.mean[0], result.belief.mean[1])
        canvas[estimate[0]][estimate[1]] = "@" if estimate == robot else "E"
    if canvas[robot[0]][robot[1]] != "@":
        canvas[robot[0]][robot[1]] = "R"
    border = "+" + "-" * cols + "+"
    lines = [border] + ["|" + "".join(row) + "|" for row in canvas] + [border, LEGEND]
    return "\n".join(lines)
