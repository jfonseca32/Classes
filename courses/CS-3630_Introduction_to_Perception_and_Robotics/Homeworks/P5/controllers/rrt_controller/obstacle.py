"""
Implementation of the polygonal obstacles as well as of the walls
"""

import numpy as np
from matplotlib.path import Path


class Obstacle:
    """
    Class implementing simple 2D polygonal obstacles
    """

    def __init__(self, points):
        """
        Initializes the obstacle from a given list of vertices
        """
        self.points = np.array(points, dtype=float)

        # Calculate bounding box, center, and polygon from the provided points
        self.bounding_box = (
            self.points[:, 0].min(),
            self.points[:, 1].min(),
            self.points[:, 0].max(),
            self.points[:, 1].max(),
        )
        self.center = np.mean(self.points, axis=0)
        self.path = Path(self.points, closed=False)

    def colides(self, x, y):
        """
        Checks if the given point is in the obstacle or not.
        """
        return bool(self.path.contains_point((x, y)))

    def contains_points(self, points):
        """Vectorized version of `colides` for an (N, 2) array of points."""
        return self.path.contains_points(np.asarray(points, dtype=float).reshape(-1, 2))

    def distance_to_points(self, points):
        """
        Distance from each of the (N, 2) points to the polygon (0 inside the polygon).
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        best = np.full(len(pts), np.inf)
        verts = self.points
        for a, b in zip(verts, np.roll(verts, -1, axis=0), strict=False):
            ab = b - a
            denom = float(ab @ ab)
            if denom == 0.0:
                d = np.linalg.norm(pts - a, axis=1)
            else:
                t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
                d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
            best = np.minimum(best, d)
        best[self.contains_points(pts)] = 0.0
        return best

    def plot(self, ax=None, **kwargs):
        """
        Draws the polygon on screen.
        """
        import matplotlib.patches as pat
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        style = dict(color="black", fill=True)
        style.update(kwargs)
        ax.add_patch(pat.Polygon(self.points, **style))


# --- The Wall class is unchanged ---


class Wall:
    """
    Class implementing a wall with a moving hole in it
    """

    def __init__(self, width, bottom_y, thickness, moving=False):
        self.width = width
        self.bottom_y = bottom_y
        self.hole = width * np.random.rand()
        self.thickness = thickness
        self.speed = (np.random.rand() - 1 / 2) * 2 if moving else 0

    def colides(self, x, y, time=0):
        """
        Checks if the given point is in the obstacle or not.
        """

        if time == 0:
            return (x < self.hole - self.width * 0.05 or x > self.hole + self.width * 0.05) and (
                self.bottom_y <= y <= self.bottom_y + self.thickness
            )

        hole = (self.hole + self.speed * time) % (self.width)
        return (x < hole - self.width * 0.05 or x > hole + self.width * 0.05) and (
            self.bottom_y <= y <= self.bottom_y + self.thickness
        )

    def visible(self, view_top, view_bottom):
        """
        Checks if the wall is in the field of view
        """

        return self.bottom_y <= view_top and self.bottom_y + self.thickness >= view_bottom
