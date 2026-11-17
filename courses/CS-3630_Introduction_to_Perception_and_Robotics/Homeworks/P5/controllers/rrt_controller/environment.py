"""
The environment with static polygonal obstacles
"""

import numpy as np
from obstacle import Obstacle  # Make sure obstacle.py is in the same folder


class StaticEnvironment:
    """
    Class implementing a very simple bounded 2D world, containing polygonal
    obstacles.

    The robot is not a point: it is a disc of radius `inflation` (in planner
    units, robot radius + a safety margin). Instead of checking a disc against
    every obstacle, we plan in the *configuration space*: every obstacle (and the
    boundary of the room) is grown by `inflation`, and the robot center only has
    to stay outside of the grown obstacles. The grown obstacles are rasterized
    once into a boolean occupancy grid, so a collision check is a table lookup.
    """

    def __init__(self, dimensions, obstacle_vertices_list, inflation=0.0, resolution=0.5):
        """
        Initializes the environment

        Args:
            dimensions: (width, height) of the world in planner units; the world is [0, w] x [0, h]
            obstacle_vertices_list: list of polygons, each a list of (x, y) vertices in planner
            units
            inflation: how much every obstacle and the boundary are grown (planner units)
            resolution: size of one occupancy-grid cell (planner units)
        """
        self.dimensions = dimensions
        self.inflation = float(inflation)
        self.resolution = float(resolution)
        self.obstacles = [Obstacle(vertices) for vertices in obstacle_vertices_list]
        self._build_occupancy_grid()

    def _build_occupancy_grid(self):
        """Rasterizes the inflated obstacles and the inflated boundary."""
        w, h = self.dimensions
        self.grid_shape = (
            int(np.ceil(w / self.resolution)) + 1,
            int(np.ceil(h / self.resolution)) + 1,
        )
        xs = np.arange(self.grid_shape[0]) * self.resolution
        ys = np.arange(self.grid_shape[1]) * self.resolution
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel()])

        blocked = (
            (pts[:, 0] < self.inflation)
            | (pts[:, 0] > w - self.inflation)
            | (pts[:, 1] < self.inflation)
            | (pts[:, 1] > h - self.inflation)
        )
        for obstacle in self.obstacles:
            x0, y0, x1, y1 = obstacle.bounding_box
            m = self.inflation + self.resolution
            near = (
                (pts[:, 0] > x0 - m)
                & (pts[:, 0] < x1 + m)
                & (pts[:, 1] > y0 - m)
                & (pts[:, 1] < y1 + m)
            )
            if not near.any():
                continue
            idx = np.nonzero(near)[0]
            d = obstacle.distance_to_points(pts[idx])
            blocked[idx[d <= self.inflation]] = True
        self.occupancy = blocked.reshape(self.grid_shape)

    def is_free(self, x, y, time=0):
        """
        Returns False if a point is within an (inflated) obstacle or outside of
        the boundaries of the environment.
        """
        return bool(self.are_free(np.array([[x, y]]))[0])

    def are_free(self, points):
        """
        Vectorized collision check: one boolean per (x, y) row of `points`.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        free = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] <= self.dimensions[0])
            & (pts[:, 1] >= 0)
            & (pts[:, 1] <= self.dimensions[1])
        )
        i = np.clip(np.rint(pts[:, 0] / self.resolution).astype(int), 0, self.grid_shape[0] - 1)
        j = np.clip(np.rint(pts[:, 1] / self.resolution).astype(int), 0, self.grid_shape[1] - 1)
        free &= ~self.occupancy[i, j]
        return free

    def is_path_free(self, points):
        """True if every point of a sampled path is collision free."""
        if len(points) == 0:
            return True
        return bool(self.are_free(points).all())

    def random_free_space(self):
        """
        Returns a randomly selected point in the free space.
        """

        x = np.random.rand() * self.dimensions[0]
        y = np.random.rand() * self.dimensions[1]
        while not self.is_free(x, y):
            x = np.random.rand() * self.dimensions[0]
            y = np.random.rand() * self.dimensions[1]
        return x, y, np.random.rand() * np.pi * 2

    def plot(self, ax=None, show_inflation=True):
        """
        Draws the obstacles (black) and, optionally, the inflated free-space
        boundary (orange) on a matplotlib axis.
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        if show_inflation:
            ax.imshow(
                self.occupancy.T,
                origin="lower",
                cmap="Oranges",
                alpha=0.35,
                vmin=0,
                vmax=1.6,
                extent=(
                    -self.resolution / 2,
                    self.grid_shape[0] * self.resolution - self.resolution / 2,
                    -self.resolution / 2,
                    self.grid_shape[1] * self.resolution - self.resolution / 2,
                ),
            )
        for obstacle in self.obstacles:
            obstacle.plot(ax)
        ax.set_xlim(0, self.dimensions[0])
        ax.set_ylim(0, self.dimensions[1])
        ax.set_aspect("equal", adjustable="box")
        return ax
