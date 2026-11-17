import os
import sys

import numpy as np

# add the project root and the rrt_controller folder to python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "controllers", "rrt_controller"))

from environment import StaticEnvironment  # noqa: E402 - requires the project path above
from rrt import RRTStar  # noqa: E402 - requires the project path above


def _straight(a, b, n=20):
    """Sampled straight segment, used to build a hand-made tree."""
    return np.linspace(np.array(a[:2], float), np.array(b[:2], float), n)


def test_choose_parent():
    """
    choose_parent must return the cheapest COLLISION-FREE parent.
        root R, candidate parents P1 (cheapest, but its segment to the new node
        crosses the wall), P2 (free, cost-to-come 75) and P3 (free, cost-to-come 65).
        Expected: P3 with cost 65.
    """
    wall = [(45, 0), (55, 0), (55, 30), (45, 30)]
    env = StaticEnvironment((100, 60), [wall])
    rrt = RRTStar(env)
    R = (10.0, 45.0)
    P1, P2, P3 = (35.0, 15.0), (40.0, 45.0), (20.0, 45.0)
    new = (75.0, 45.0)
    rrt.set_start(R)
    rrt.add_node(P1, R, 1.0, _straight(R, P1))
    rrt.add_node(P2, R, 40.0, _straight(R, P2))
    rrt.add_node(P3, R, 10.0, _straight(R, P3))

    parent, cost, path = rrt.choose_parent(new, [P1, P2, P3])
    try:
        assert parent == P3, f"expected parent {P3}, got {parent}"
        assert abs(cost - 65.0) < 1e-6, f"expected cost 65, got {cost}"
        assert path is not None and np.allclose(path[-1], new[:2]), "path must end at the new node"
        assert env.is_path_free(path), "returned path is not collision free"
        # with only the blocked candidate there is no valid parent
        parent, cost, path = rrt.choose_parent(new, [P1])
        assert parent is None and cost == float("inf"), (
            "expected (None, inf, None) when every candidate collides"
        )
        return True
    except AssertionError as e:
        print(e)
        return False


def test_rewire():
    """
    rewire must make the new node the parent of a near node when that is cheaper
    and collision free, and the cost of that node's descendants must follow.
        R -> Q (cost 200, an artificially bad path) -> Q2 (cost 220)
        R -> N (cost 25), N just inserted. Through N, Q costs 25 + 25 = 50.
    """
    env = StaticEnvironment((100, 60), [])
    rrt = RRTStar(env)
    R, Q, Q2, N = (10.0, 45.0), (60.0, 45.0), (80.0, 45.0), (35.0, 45.0)
    rrt.set_start(R)
    rrt.add_node(Q, R, 200.0, _straight(R, Q))
    rrt.add_node(Q2, Q, 220.0, _straight(Q, Q2))
    rrt.add_node(N, R, 25.0, _straight(R, N))
    try:
        n = rrt.rewire(N, [R, Q])
        assert n == 1, f"expected 1 rewired node, got {n}"
        assert rrt.nodes[Q].parent == N, "Q should now be a child of N"
        assert abs(rrt.nodes[Q].cost - 50.0) < 1e-6, (
            f"expected cost(Q) = 50, got {rrt.nodes[Q].cost}"
        )
        assert abs(rrt.nodes[Q2].cost - 70.0) < 1e-6, (
            f"expected cost(Q2) = 70, got {rrt.nodes[Q2].cost}"
        )
        assert Q not in rrt.nodes[R].destination_list, (
            "Q must be removed from its old parent's children"
        )

        # A wall between N and Q: no rewiring allowed.
        env2 = StaticEnvironment((100, 60), [[(45, 30), (50, 30), (50, 60), (45, 60)]])
        rrt2 = RRTStar(env2)
        rrt2.set_start(R)
        rrt2.add_node(Q, R, 200.0, _straight(R, Q))
        rrt2.add_node(N, R, 25.0, _straight(R, N))
        assert rrt2.rewire(N, [R, Q]) == 0 and rrt2.nodes[Q].parent == R, "rewired through a wall"

        # Not cheaper: no rewiring.
        rrt3 = RRTStar(env)
        rrt3.set_start(R)
        rrt3.add_node(Q, R, 40.0, _straight(R, Q))
        rrt3.add_node(N, R, 25.0, _straight(R, N))
        assert rrt3.rewire(N, [R, Q]) == 0 and rrt3.nodes[Q].parent == R, (
            "rewired although it is not cheaper"
        )
        return True
    except AssertionError as e:
        print(e)
        return False


def test_rrt_star_improves():
    """
    Full RRT* run around an obstacle: a path is found, it is collision free, the
    best cost never increases, and every node's cost equals its parent's cost plus
    the length of its edge.
    """
    np.random.seed(3630)
    env = StaticEnvironment((100, 60), [[(40, 0), (55, 0), (55, 38), (40, 38)]], inflation=2.0)
    rrt = RRTStar(env)
    rrt.set_start((10.0, 10.0))
    rrt.run(goal=(85.0, 10.0), nb_iteration=600, goal_rate=0.1)
    try:
        path = rrt.extract_path()
        assert len(path) > 0, "no path found"
        assert env.is_path_free(np.array(path)), "path collides"
        hist = [c for c in rrt.cost_history if c < float("inf")]
        assert all(b <= a + 1e-9 for a, b in zip(hist, hist[1:], strict=False)), (
            "best cost increased during the run"
        )
        for key, node in rrt.nodes.items():
            if node.parent is None:
                continue
            length = rrt.edge_cost(node.parent, key)
            assert abs(rrt.nodes[node.parent].cost + length - node.cost) < 1e-6, (
                f"inconsistent cost at {key}"
            )
        print(f"first solution {hist[0]:.1f}, final {hist[-1]:.1f} (planner units)")
        return True
    except AssertionError as e:
        print(e)
        return False


if __name__ == "__main__":
    print("Passed test_choose_parent:", test_choose_parent())
    print("Passed test_rewire:", test_rewire())
    print("Passed test_rrt_star_improves:", test_rrt_star_improves())
