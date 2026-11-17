"""
Construction of the Rapidly-exploring Random Tree (RRT) and of its
asymptotically optimal version, RRT*, plus the smoothing of the final path.

Both trees live in the planner's coordinate system (planner units, see
rrt_controller.py) and plan for the POSITION (x, y) of the robot only, as in
Section 5.5.6 of the textbook: the e-puck is round, so whether it collides does
not depend on its heading, and it can turn in place, so any heading can be
reached at any free point. Every edge of the tree is a straight segment.

The robot only drives forward. A straight-line path has corners: smooth_path
rounds the gentle ones with circular arcs (fillet_corner) and leaves the sharp
ones, where the robot stops and rotates in place (low_level_controller.py).
"""

import json
import math
from collections import deque

import numpy as np


def dist(pt_a, pt_b):
    """Euclidean distance between two (x, y) points"""
    return ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** 0.5


def densify_path(path, ds=0.05):
    dense_path = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]["x"], path[i]["y"]
        x1, y1 = path[i + 1]["x"], path[i + 1]["y"]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        steps = max(int(math.ceil(dist / ds)), 1)
        for s in range(steps):
            t = s / steps
            x = x0 + dx * t
            y = y0 + dy * t
            theta = math.atan2(dy, dx)
            dense_path.append({"x": x, "y": y, "theta": theta})
    # append last point
    dense_path.append(path[-1])
    return dense_path


def straight_points(a, b, point_separation=1.0):
    """(N, 2) points from a to b (both included), at most point_separation apart."""
    a, b = np.asarray(a[:2], dtype=float), np.asarray(b[:2], dtype=float)
    n = max(int(math.ceil(dist(a, b) / point_separation)), 1) + 1
    return np.linspace(a, b, n)


# Collision checks look up a grid with 0.5-unit cells: checking only the path's own
# points (0.5 apart) can miss a segment that clips the corner of an occupied cell.
COLLISION_CHECK_STEP = 0.125
DEFAULT_SHARP_ANGLE = math.radians(60)


def is_free_path(environment, points):
    """
    True if the polyline through `points` is collision free, checked every COLLISION_CHECK_STEP.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) > 1:
        pts = np.vstack(
            [
                straight_points(a, b, COLLISION_CHECK_STEP)[:-1]
                for a, b in zip(pts, pts[1:], strict=False)
            ]
            + [pts[-1:]]
        )
    if hasattr(environment, "is_path_free"):
        return environment.is_path_free(pts)
    return all(environment.is_free(p[0], p[1]) for p in pts)


class Node:
    """
    Node of the rapidly exploring random tree.
    """

    def __init__(self, position, time, cost, parent=None):
        self.position = position
        self.time = time
        self.cost = cost  # cost-to-come: length of the path from the root to this node
        self.parent = parent  # key of the parent node (None for the root)
        self.destination_list = []  # keys of the children
        self.path = []  # sampled straight segment from parent to this node


class Edge:
    """
    Edge of the rapidly exploring random tree.
    """

    def __init__(self, node_from, node_to, path, cost):
        self.node_from = node_from
        self.node_to = node_to
        self.path = deque(path)
        self.cost = cost


class RRT:
    """
    Class implementing a Rapidly Exploring Random Tree in two dimensions with
    straight-line edges: the algorithm of Section 5.5.5 of the textbook.
    """

    def __init__(self, environment, precision=(5, 5), step_size=20.0, point_separation=0.5):
        self.nodes = {}
        self.edges = {}
        self.environment = environment
        self.step_size = step_size
        self.point_separation = point_separation
        self.goal = (0, 0)
        self.root = (0, 0)
        self.precision = precision
        self.verbose = False

    def set_start(self, start):
        """
        Resets the graph, and sets the start node as root of the tree.
        A heading, if given (x, y, theta), is ignored: the tree is built for the position.
        """
        start = tuple(float(v) for v in start[:2])
        self.nodes = {}
        self.edges = {}
        self.nodes[start] = Node(start, 0, 0)
        self.root = start

    def sample(self, goal_rate):
        """The goal with probability goal_rate, otherwise a random free (x, y)."""
        if np.random.rand() < goal_rate:
            return self.goal
        return tuple(float(v) for v in self.environment.random_free_space()[:2])

    def nearest_node(self, sample):
        """Key of the tree node closest to the sample (linear time)."""
        return min(self.nodes, key=lambda key: dist(key, sample))

    def steer(self, from_key, sample):
        """
        Takes a step of at most `step_size` from from_key toward the sample
        (turn toward the target, then drive forward).
        """
        d = dist(from_key, sample)
        if d <= self.step_size:
            return (float(sample[0]), float(sample[1]))
        dx, dy = sample[0] - from_key[0], sample[1] - from_key[1]
        return (from_key[0] + dx / d * self.step_size, from_key[1] + dy / d * self.step_size)

    def edge_cost(self, node_from, node_to):
        """Cost of the edge node_from -> node_to: the length of the straight segment."""
        return dist(node_from, node_to)

    def edge_path(self, node_from, node_to):
        """Sampled (N, 2) points of the straight segment from node_from to node_to."""
        return straight_points(node_from, node_to, self.point_separation)

    def run(self, goal, nb_iteration=100, goal_rate=0.1, metric=None):
        """
        Executes the RRT algorithm: sample, find the nearest node, steer toward the
        sample, and add the new node if the segment is collision free. Stops as
        soon as the goal region is reached.
        """
        self.goal = tuple(float(v) for v in goal[:2])

        for _ in range(nb_iteration):
            sample = self.sample(goal_rate)
            nearest = self.nearest_node(sample)
            new_key = self.steer(nearest, sample)
            if new_key in self.nodes:
                continue
            path = self.edge_path(nearest, new_key)
            if not self.path_is_free(path):
                continue
            self._attach(nearest, new_key, path)

            if self.in_goal_region(new_key):
                # finish exactly on the goal when it can be reached in a straight line
                if new_key != self.goal and self.goal not in self.nodes:
                    to_goal = self.edge_path(new_key, self.goal)
                    if self.path_is_free(to_goal):
                        self._attach(new_key, self.goal, to_goal)
                if self.verbose:
                    print("Goal region reached!")
                return

    def _attach(self, parent_key, key, path):
        length = self.edge_cost(parent_key, key)
        path_list = [tuple(p) for p in path]
        self.nodes[key] = Node(
            key,
            self.nodes[parent_key].time + length,
            self.nodes[parent_key].cost + length,
            parent=parent_key,
        )
        self.nodes[key].path = path_list
        self.nodes[parent_key].destination_list.append(key)
        self.edges[parent_key, key] = Edge(parent_key, key, path_list, length)
        if self.verbose:
            print(f"Edge added: {parent_key} -> {key}")

    def path_is_free(self, path_points):
        """
        True if a sampled path is collision free (see is_free_path).
        """
        return is_free_path(self.environment, path_points)

    def goal_node(self):
        """
        Key of the node used as the end of the path: the goal itself if it is
        in the tree, otherwise the cheapest node inside the goal region.
        """
        if self.goal in self.nodes:
            return self.goal
        in_region = [key for key in self.nodes if self.in_goal_region(key)]
        if not in_region:
            return None
        return min(in_region, key=lambda key: self.nodes[key].cost)

    def path_cost(self):
        """Length of the current best path to the goal (inf if there is none)."""
        key = self.goal_node()
        return float("inf") if key is None else self.nodes[key].cost

    def extract_waypoints(self):
        """
        The tree nodes on the path from the root to the goal, as an (N, 2) array
        (the corners of the straight-line path). Empty if the goal was not reached.
        """
        key = self.goal_node()
        keys = []
        while key is not None:
            keys.append(key)
            key = self.nodes[key].parent
        return np.array(keys[::-1], dtype=float).reshape(-1, 2)

    def extract_path(self):
        """
        Traces the path from the goal back to the root.
        """
        goal_node_key = self.goal_node()

        if goal_node_key is None:
            if self.verbose:
                print("No node reached the goal region during extraction.")
            return []

        segments = []
        current_key = goal_node_key
        while current_key is not None:
            node = self.nodes[current_key]
            if node.path:
                segments.append(node.path)
            current_key = node.parent
        path = []
        for segment in reversed(segments):
            path.extend(segment)
        return path

    def export_path(self, filename="path.json"):
        """Reconstructs the full path and exports it."""
        final_path_points = self.extract_path()

        if not final_path_points:
            print("Path extraction failed, cannot export.")
            return

        traj_with_theta = []
        for i in range(len(final_path_points) - 1):
            x, y = final_path_points[i]
            x_next, y_next = final_path_points[i + 1]
            theta = math.atan2(y_next - y, x_next - x)
            traj_with_theta.append({"x": x, "y": y, "theta": theta})

        if traj_with_theta:
            last_point = final_path_points[-1]
            traj_with_theta.append(
                {"x": last_point[0], "y": last_point[1], "theta": traj_with_theta[-1]["theta"]}
            )

        if traj_with_theta:
            dense_traj = densify_path(traj_with_theta)
            with open(filename, "w") as f:
                json.dump(dense_traj, f, indent=2)
            print(f"Path exported to {filename}")

    def in_goal_region(self, sample):
        """Checks if a point is within the goal region."""
        return all(abs(self.goal[i] - sample[i]) <= self.precision[i] for i in range(2))

    def plot(
        self,
        ax=None,
        nodes=False,
        path=True,
        tree_color="tab:blue",
        path_color="tab:red",
        smoothed=None,
        smoothed_color="tab:green",
    ):
        """
        Displays the tree (and the current best path) using matplotlib.
        smoothed: optional (N, 2) path from smooth_path, drawn on top.
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        for edge in self.edges.values():
            if edge.path:
                arr = np.array(list(edge.path))
                ax.plot(arr[:, 0], arr[:, 1], "-", color=tree_color, alpha=0.35, lw=0.8)
        if nodes and self.nodes:
            arr = np.array(list(self.nodes.keys()))
            ax.scatter(arr[:, 0], arr[:, 1], s=4, color=tree_color)
        if path:
            pts = self.extract_path()
            if pts:
                arr = np.array(pts)
                ax.plot(arr[:, 0], arr[:, 1], "-", color=path_color, lw=2.2)
        if smoothed is not None and len(smoothed):
            arr = np.asarray(smoothed)
            ax.plot(arr[:, 0], arr[:, 1], "-", color=smoothed_color, lw=1.6)
        ax.plot(self.root[0], self.root[1], "o", color="tab:green", ms=8)
        ax.plot(self.goal[0], self.goal[1], "*", color="tab:red", ms=12)
        return ax


class RRTStar(RRT):
    """
    RRT* (Karaman & Frazzoli, 2011) with straight-line edges.

    Differences with the RRT above:
      1. A new node is not simply attached to the node it was sampled from: among
         all the tree nodes NEAR it, we pick the parent that gives the lowest
         cost-to-come (`choose_parent`).
      2. After inserting the new node, we check whether going THROUGH the new node
         would be cheaper for any of the near nodes, and if so we make the new node
         their parent (`rewire`).
      3. We do not stop when the goal is reached: we keep sampling for the whole
         iteration budget, and every rewire can only make the path to the goal
         shorter. As the number of samples grows, the path converges toward the
         optimal one.

    Notation (all lengths in planner units):
      self.nodes[key].cost        cost-to-come of a node (path length from the root)
      self.edge_cost(a, b)        length of the straight segment a -> b
      self.edge_path(a, b)        sampled points of that segment
      self.path_is_free(points)   collision check of sampled points
    """

    def __init__(
        self,
        environment,
        precision=(5, 5),
        step_size=20.0,
        point_separation=0.5,
        near_radius_gamma=120.0,
        near_radius_min=8.0,
        near_radius_max=22.0,
    ):
        super().__init__(environment, precision, step_size, point_separation)
        self.near_radius_gamma = near_radius_gamma
        self.near_radius_min = near_radius_min
        self.near_radius_max = near_radius_max
        self.cost_history = []  # best path cost after every iteration (inf until the goal is found)
        self._keys = []  # node keys, aligned with self._xy (fast nearest-neighbor queries)
        self._xy = np.zeros((0, 2))

    # ------------------------------------------------------------------
    # Helpers provided by the staff
    # ------------------------------------------------------------------
    def set_start(self, start):
        super().set_start(start)
        self.cost_history = []
        self._keys = [self.root]
        self._xy = np.array([self.root], dtype=float)

    def near_radius(self):
        """Shrinking RRT* neighborhood radius gamma * sqrt(log(n) / n), clamped to [min, max]."""
        n = len(self.nodes) + 1
        r = self.near_radius_gamma * math.sqrt(math.log(n) / n)
        return min(max(r, self.near_radius_min), self.near_radius_max)

    def near_nodes(self, sample, radius=None):
        """
        Keys of the tree nodes within `radius` of the sample, sorted from closest
        to farthest. Falls back to the 5 nearest nodes if the ball is empty, so
        the new sample always has candidate parents.
        """
        radius = self.near_radius() if radius is None else radius
        d = np.hypot(self._xy[:, 0] - sample[0], self._xy[:, 1] - sample[1])
        order = np.argsort(d)
        near = [self._keys[i] for i in order if d[i] <= radius]
        if not near:
            near = [self._keys[i] for i in order[:5]]
        return near

    def nearest_node(self, sample):
        d = np.hypot(self._xy[:, 0] - sample[0], self._xy[:, 1] - sample[1])
        return self._keys[int(np.argmin(d))]

    def steer(self, from_key, sample):
        """
        Moves a random sample toward the tree: if it is farther than `step_size`
        from its nearest node, it is pulled back along the line to that node.
        The goal is never moved.
        """
        if sample == self.goal:
            return sample
        return super().steer(from_key, sample)

    def add_node(self, key, parent_key, cost, path_points):
        """Inserts a new node in the tree as a child of parent_key."""
        path_list = [tuple(p) for p in path_points]
        self.nodes[key] = Node(key, cost, cost, parent=parent_key)
        self.nodes[key].path = path_list
        self.nodes[parent_key].destination_list.append(key)
        self.edges[parent_key, key] = Edge(
            parent_key, key, path_list, cost - self.nodes[parent_key].cost
        )
        self._keys.append(key)
        self._xy = np.vstack([self._xy, key[:2]])

    def change_parent(self, node_key, new_parent_key, path_points, edge_cost):
        """
        Re-attaches node_key below new_parent_key (used by rewire): removes the old
        edge, adds the new one, updates the node's cost-to-come and propagates the
        change of cost to all of its descendants.
        """
        node = self.nodes[node_key]
        old_parent = node.parent
        if old_parent is not None:
            self.nodes[old_parent].destination_list.remove(node_key)
            self.edges.pop((old_parent, node_key), None)
        path_list = [tuple(p) for p in path_points]
        node.parent = new_parent_key
        node.path = path_list
        self.nodes[new_parent_key].destination_list.append(node_key)
        self.edges[new_parent_key, node_key] = Edge(new_parent_key, node_key, path_list, edge_cost)
        new_cost = self.nodes[new_parent_key].cost + edge_cost
        self._propagate_cost(node_key, new_cost - node.cost)

    def _propagate_cost(self, node_key, delta):
        stack = [node_key]
        while stack:
            key = stack.pop()
            self.nodes[key].cost += delta
            self.nodes[key].time = self.nodes[key].cost
            stack.extend(self.nodes[key].destination_list)

    def choose_parent(
        self, new_key: tuple[float, float], near_keys: list[tuple[float, float]]
    ) -> tuple[tuple[float, float] | None, float, np.ndarray | None]:
        """Choose the lowest-cost collision-free parent for a new node.

        Parameters
        ----------
        new_key : tuple of float
            Position (x, y) of the new node, not yet inserted into the tree.
        near_keys : list of tuple of float
            Positions (x, y) of existing nodes to consider as parents.

        Returns
        -------
        parent_key : tuple of float or None
            Selected parent's position, or None if no valid parent exists.
        cost : float
            Total path length from the root through the selected parent
            to the new node. Infinite if no valid parent exists.
        path_points : numpy.ndarray or None
            Sampled segment from the parent to the new node, with shape
            (N, 2). None if no valid parent exists.

        Notes
        -----
        Candidate cost is the parent's cost-to-come plus the straight-line
        distance to the new node. Candidates are checked in ascending cost
        order, so the first collision-free connection has the lowest cost.
        """
        # Each entry is (total cost through this parent, parent position (x, y))
        ctc: list[tuple[float, tuple[float, float]]] = []

        for point in near_keys:
            # Path length from root to parent + distance from parent to new node
            cost = self.nodes[point].cost + self.edge_cost(point, new_key)
            ctc.append((cost, point))

        ctc.sort()  # sort by cost

        for cost, point in ctc:
            # Find cheapest free path
            path = self.edge_path(point, new_key)

            if self.path_is_free(path):
                return point, cost, path

        return None, float("inf"), None

    def rewire(self, new_key: tuple[float, float], near_keys: list[tuple[float, float]]) -> int:
        """Reconnect nearby nodes when routing through the new node is cheaper.

        Parameters
        ----------
        new_key : tuple of float
            Position (x, y) of the node just inserted into the tree.
        near_keys : list of tuple of float
            Positions (x, y) of nearby nodes to consider for rewiring.

        Returns
        -------
        int
            Number of nearby nodes assigned a new parent.

        Notes
        -----
        Only cheaper, collision-free connections are accepted.
        ``change_parent`` updates tree connections and propagates the
        cost change to descendants.
        """
        rewired = 0

        for point in near_keys:
            node = self.nodes[point]

            # Skip the new node itself, its parents, and the root
            if point == new_key or point == self.nodes[new_key].parent or node.parent is None:
                continue

            # Compare route to nearby node through the new node with the nearby node's current route
            edge_length = self.edge_cost(new_key, point)
            # total cost = root to new node cost + distance from new node to nearby node
            cost = self.nodes[new_key].cost + edge_length
            if cost >= node.cost:
                continue

            # Check collisions if new route is cheaper
            path = self.edge_path(new_key, point)
            if self.path_is_free(path):
                self.change_parent(point, new_key, path, edge_length)
                rewired += 1

        return rewired

    # ------------------------------------------------------------------
    # Main loop (provided)
    # ------------------------------------------------------------------
    def run(
        self, goal, nb_iteration=1500, goal_rate=0.1, metric=None, min_iterations_after_goal=None
    ):
        """
        Executes RRT*. Unlike RRT, it keeps improving the tree after the goal is
        reached. If min_iterations_after_goal is set, it stops that many
        iterations after the goal was first reached (or at nb_iteration).
        """
        self.goal = tuple(float(v) for v in goal[:2])
        found_at = None

        for it in range(nb_iteration):
            sample = self.sample(goal_rate)

            nearest = self.nearest_node(sample)
            new_key = self.steer(nearest, sample)
            if new_key in self.nodes or not self.environment.is_free(new_key[0], new_key[1]):
                self.cost_history.append(self.path_cost())
                continue

            near = self.near_nodes(new_key)
            parent, cost, path = self.choose_parent(new_key, near)
            if parent is not None:
                self.add_node(new_key, parent, cost, path)
                self.rewire(new_key, near)
                if self.verbose:
                    print(f"Edge added: {parent} -> {new_key}")

            best = self.path_cost()
            self.cost_history.append(best)
            if found_at is None and best < float("inf"):
                found_at = it
                if self.verbose:
                    print(f"Goal reached after {it + 1} iterations (cost {best:.1f})")
            if (
                found_at is not None
                and min_iterations_after_goal is not None
                and it - found_at >= min_iterations_after_goal
            ):
                break


# ----------------------------------------------------------------------
# Path smoothing
# ----------------------------------------------------------------------
def fillet_corner(
    p_prev: tuple[float, float] | np.ndarray,
    p: tuple[float, float] | np.ndarray,
    p_next: tuple[float, float] | np.ndarray,
    radius: float,
    point_separation: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Round a polyline corner with a tangent circular arc.

    Parameters
    ----------
    p_prev : tuple of float or numpy.ndarray
        Position (x, y) preceding the corner.
    p : tuple of float or numpy.ndarray
        Corner position (x, y), distinct from both neighboring points.
    p_next : tuple of float or numpy.ndarray
        Position (x, y) following the corner.
    radius : float
        Requested arc radius, in the same units as the positions.
    point_separation : float, optional
        Maximum spacing along the arc between consecutive sampled
        points. Must be positive. Default is 1.0.

    Returns
    -------
    points : numpy.ndarray
        Sampled arc of shape (N, 2), including both tangent endpoints.
        For a straight path, complete reversal, or nonpositive radius,
        contains only the original corner, with shape (1, 2).
    radius_used : float
        Actual arc radius, reduced if necessary to fit the adjacent
        segments. Zero when the original corner is returned.

    Notes
    -----
    Let u1 and u2 be the incoming and outgoing unit directions,
    and phi the magnitude of the turn angle. The tangent points are:

        T1 = p - t * u1
        T2 = p + t * u2
        t = radius * tan(phi / 2)

    The tangent distance t is limited to half the length of each
    adjacent segment to prevent neighboring fillets from overlapping.
    The radius is then recomputed as t / tan(phi / 2).

    This function constructs the arc without checking for collisions.
    """
    # convert points to numpy arrays of floats
    p_prev, p, p_next = (np.asarray(q[:2], dtype=float) for q in (p_prev, p, p_next))

    # Directions and lengths of the incoming and outgoing segments
    v1 = p - p_prev
    v2 = p_next - p
    length1 = float(np.linalg.norm(v1))
    length2 = float(np.linalg.norm(v2))
    u1 = v1 / length1  # unit vectors
    u2 = v2 / length2

    # Positive turn angles are left turns; negative are right
    cross = u1[0] * u2[1] - u1[1] * u2[0]
    turn_angle = math.atan2(cross, float(np.dot(u1, u2)))
    phi = abs(turn_angle)

    if phi < 1e-9 or math.pi - phi < 1e-9 or radius <= 0:
        # No usable circular fillet
        return (np.array([p]), 0)

    # Limit tangent distance so arc fits the 2 adjacent segments
    tan_half = math.tan(phi / 2)
    t = min(radius * tan_half, length1 / 2, length2 / 2)
    radius_used = t / tan_half

    # Points where arc touches the incoming / outgoing segments
    t1 = p - t * u1
    t2 = p + t * u2

    # Circle on the inside of the turn
    turn_sign = 1.0 if turn_angle > 0 else -1.0
    left_normal = np.array([-u1[1], u1[0]])
    center = t1 + turn_sign * radius_used * left_normal

    start_angle = math.atan2(t1[1] - center[1], t1[0] - center[0])  # around center circle

    # Sample enough intervals given the maximum arc spacing
    arc_length = radius_used * phi
    intervals = max(1, math.ceil(arc_length / point_separation))
    angles = np.linspace(start_angle, start_angle + turn_angle, intervals + 1)

    # Convert circle angles to positions in the world's coordinate
    points = center + radius_used * np.column_stack((np.cos(angles), np.sin(angles)))

    # Set tangent endpoints to avoid floating point errors
    points[0] = t1
    points[-1] = t2

    return points, radius_used


def smooth_path(
    waypoints,
    environment,
    radius=4.0,
    sharp_angle=DEFAULT_SHARP_ANGLE,
    point_separation=0.5,
    min_radius=1.5,
):
    """
    Turns the straight-line path through `waypoints` into a path the robot can
    follow driving forward: every corner that turns by less than sharp_angle is
    rounded with a collision-free fillet (the radius is halved until the arc is
    free, down to min_radius); sharper corners, and corners that cannot be
    rounded, are kept: the robot stops there and rotates in place.

    This post-processes the FINAL path once planning is done; it never touches
    the RRT* tree (that is RRTStar.rewire, which runs during planning).

    Args:
        waypoints: (N, 2) corners of the path, from the start to the goal
        environment: used to collision-check the arcs
    Returns:
        (M, 2) array of points, at most point_separation apart, from the first to
        the last waypoint.
    """
    pts = [np.asarray(w[:2], dtype=float) for w in waypoints]
    pts = [q for i, q in enumerate(pts) if i == 0 or np.linalg.norm(q - pts[i - 1]) > 1e-9]
    if len(pts) < 3:
        return (
            np.vstack(
                [
                    straight_points(a, b, point_separation)
                    for a, b in zip(pts, pts[1:], strict=False)
                ]
            )
            if len(pts) == 2
            else np.array(pts).reshape(-1, 2)
        )

    pieces = []  # each corner becomes a list of points (an arc, or just the corner)
    for i in range(1, len(pts) - 1):
        u1, u2 = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
        phi = abs(math.atan2(u1[0] * u2[1] - u1[1] * u2[0], float(np.dot(u1, u2))))
        corner = np.array([pts[i]])
        r = radius
        while phi <= sharp_angle and r >= min_radius:
            # arc points close enough together that each one turns the heading by at most ~0.25 rad
            arc, used = fillet_corner(
                pts[i - 1], pts[i], pts[i + 1], r, min(point_separation, 0.25 * r)
            )
            if used < min_radius:
                break
            if is_free_path(environment, arc):
                corner = arc
                break
            r /= 2.0
        pieces.append(corner)

    out = [pts[0]]
    for piece in pieces + [np.array([pts[-1]])]:
        out.extend(straight_points(out[-1], piece[0], point_separation)[1:])
        out.extend(piece[1:])
    out = np.array(out)
    keep = np.ones(len(out), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-9
    return out[keep]
