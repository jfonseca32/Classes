"""
CS 3630 Project 3 (Fall 2026): local sanity checks.

These are lightweight versions of the autograder tests.  The notebook calls them as
    from project3_test import TestProject3, verify
    unit_test = TestProject3()
    print(verify(unit_test.test_get_prior, get_prior))
and you can also run the whole file from a terminal against your main.py:
    python -m pytest project3_test.py -q
(or against the solution, for staff: CS3630_P3_MODULE=main_sols python -m pytest project3_test.py
-q).

Passing these does not guarantee full credit; the autograder has more cases.
"""

import importlib
import inspect
import os
import unittest

import gtsam
import numpy as np
from gtbook.discrete import Variables

np.random.seed(3630)

# Reference constants (identical to main.py).  Tests pass these into your functions so a
# mistake in one TODO does not hide a correct implementation of another.
ROOMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
ACTIONS = [
    "2->1",
    "1->2",
    "2->3",
    "3->2",
    "2->4",
    "4->2",
    "2->5",
    "5->2",
    "2->6",
    "6->7",
    "7->6",
    "6->2",
    "2->9",
    "9->8",
    "8->9",
    "9->2",
]
LIGHT_LEVELS = ["dark", "medium", "light"]

PROB_SPEC = """
    100/0/0/0/0/0/0/0/0 3/95/2/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0 100/0/0/0/0/0/0/0/0
    93/4/3/0/0/0/0/0/0 0/100/0/0/0/0/0/0/0 3/6/88/3/0/0/0/0/0 0/100/0/0/0/0/0/0/0 0/6/3/87/4/0/0/0/0 0/100/0/0/0/0/0/0/0 0/5/0/2/92/1/0/0/0 0/100/0/0/0/0/0/0/0 0/4/0/0/1/93/2/0/0 0/100/0/0/0/0/0/0/0 0/100/0/0/0/0/0/0/0 0/100/0/0/0/0/0/0/0 0/6/2/0/0/0/0/4/88 0/100/0/0/0/0/0/0/0 0/100/0/0/0/0/0/0/0 0/100/0/0/0/0/0/0/0
    0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 2/90/5/3/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0 0/0/100/0/0/0/0/0/0
    0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/90/2/6/2/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0 0/0/0/100/0/0/0/0/0
    0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/94/0/1/4/1/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0 0/0/0/0/100/0/0/0/0
    0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/5/0/0/0/5/90/0/0 0/0/0/0/0/100/0/0/0 0/94/0/0/0/4/2/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0 0/0/0/0/0/100/0/0/0
    0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/96/4/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0 0/0/0/0/0/0/100/0/0
    0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/100/0 0/0/0/0/0/0/0/7/93 0/0/0/0/0/0/0/100/0
    0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/0/100 0/0/0/0/0/0/0/92/8 0/0/0/0/0/0/0/0/100 0/90/0/0/0/0/0/4/6
"""  # noqa: E501 - One transition-table row per room.

SENSOR_SPEC = """
    15/50/35 20/60/20 15/45/40 20/55/25 20/55/25 20/55/25 20/55/25 40/50/10 60/35/5
"""


def verify(function, *args) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
    - function: Python function object
    - *args: arguments to pass to the function
    Returns:
    - string
    """
    try:
        function(*args)
        return '\x1b[32m"Correct"\x1b[0m'
    except AssertionError as e:
        return '\x1b[31m"Wrong"\x1b[0m' + (f"  ({e})" if str(e) else "")
    except Exception as e:
        return '\x1b[31m"Wrong"\x1b[0m' + f"  ({type(e).__name__}: {e})"


def _student(name):
    """
    Resolve a student function/constant by name when a test is run by pytest (no argument passed).
    """
    module = importlib.import_module(os.environ.get("CS3630_P3_MODULE", "main"))
    return getattr(module, name)


class TestProject3(unittest.TestCase):
    """
    Sanity checks for the vacuum cleaning robot assignment.  Each test takes the function
    under test as an argument (the notebook passes it); when run by pytest the function is
    looked up in main.py instead.
    """

    # --- Section 3.1 -------------------------------------------------------------------------

    def test_create_action_series(self, create_action_series=None):
        create_action_series = create_action_series or _student("create_action_series")
        for N_test in range(3, 10):
            A_test = create_action_series("A", range(1, N_test))
            self.assertIsNotNone(A_test, "returned None")
            self.assertEqual(len(A_test), N_test - 1)
            self.assertEqual(
                A_test[1][1], len(ACTIONS), "cardinality must be the number of actions"
            )

    def test_create_state_series(self, create_state_series=None):
        create_state_series = create_state_series or _student("create_state_series")
        for N_test in range(3, 10):
            X_test = create_state_series("X", range(1, N_test + 1))
            self.assertIsNotNone(X_test, "returned None")
            self.assertEqual(len(X_test), N_test)
            self.assertEqual(X_test[1][1], len(ROOMS), "cardinality must be the number of rooms")

    # --- Section 3.2 -------------------------------------------------------------------------

    def test_create_motion_model(self, create_motion_model=None):
        create_motion_model = create_motion_model or _student("create_motion_model")
        V = Variables()
        X_1 = V.discrete_series("X", range(1, 6), ROOMS)
        A_1 = V.discrete_series("A", range(1, 5), ACTIONS)
        model = create_motion_model(X_1, A_1, PROB_SPEC)
        self.assertIsNotNone(model, "returned None")
        self.assertEqual(model.size(), 3, "P(X2 | X1, A1) involves exactly three variables")

    def test_create_prob_spec(self, prob_spec=None):
        """Checks the two blanks in row 1 (from the printer area) and the table shape."""
        prob_spec = prob_spec if prob_spec is not None else _student("prob_spec")
        rows = [line.split() for line in prob_spec.strip().splitlines() if line.strip()]
        self.assertEqual(len(rows), len(ROOMS), "one line per room")
        for r, row in enumerate(rows):
            self.assertEqual(len(row), len(ACTIONS), f"line {r + 1}: one entry per action")
            for a, tok in enumerate(row):
                self.assertEqual(
                    len(tok.split("/")), len(ROOMS), f"line {r + 1} entry {a + 1}: nine values"
                )
        self.assertEqual(
            rows[0][0], "100/0/0/0/0/0/0/0/0", "row 1, action 2->1 is impossible from room 1: stay"
        )
        self.assertEqual(
            rows[0][1], "3/95/2/0/0/0/0/0/0", "row 1, action 1->2: see the prose bullet for 1->2"
        )

    def test_create_bayes_net(self, create_bayes_net=None):
        create_bayes_net = create_bayes_net or _student("create_bayes_net")
        V = Variables()
        X_1 = V.discrete_series("X", range(1, 6), ROOMS)
        A_1 = V.discrete_series("A", range(1, 5), ACTIONS)
        motion_model = gtsam.DiscreteConditional(X_1[2], [X_1[1], A_1[1]], PROB_SPEC)
        reference = gtsam.DiscreteBayesNet()
        reference.add(motion_model)
        result = create_bayes_net(motion_model)
        self.assertIsNotNone(result, "returned None")
        self.assertTrue(reference.equals(result))

    # --- Section 3.3 -------------------------------------------------------------------------

    def test_create_sensor_series(self, create_sensor_series=None):
        create_sensor_series = create_sensor_series or _student("create_sensor_series")
        V = Variables()
        for N_test in range(3, 10):
            Z_test = create_sensor_series("?", range(1, N_test + 1))
            self.assertIsNotNone(Z_test, "returned None")
            self.assertEqual(len(Z_test), N_test)
            Z_ref = V.discrete_series("?", range(1, N_test + 1), LIGHT_LEVELS)
            for k in range(1, N_test + 1):
                self.assertEqual(tuple(Z_test[k]), tuple(Z_ref[k]))

    def test_get_prior(self, get_prior=None):
        get_prior = get_prior or _student("get_prior")
        self.assertEqual(get_prior(), "1/1/1/1/1/1/1/1/1")

    def test_get_kitchen_start_prior(self, get_kitchen_start_prior=None):
        get_kitchen_start_prior = get_kitchen_start_prior or _student("get_kitchen_start_prior")
        self.assertEqual(get_kitchen_start_prior(), "0/0/1/0/0/0/0/0/0")

    def test_create_dbn(self, create_dbn=None):
        create_dbn = create_dbn or _student("create_dbn")
        V = Variables()
        N_test = 5
        test_prior = "0/0/0/1/0/0/0/0/0"
        X_1 = V.discrete_series("X", range(1, N_test + 1), ROOMS)
        A_1 = V.discrete_series("A", range(1, N_test), ACTIONS)
        Z_1 = V.discrete_series("Z", range(1, N_test + 1), LIGHT_LEVELS)
        dbn = create_dbn(test_prior, X_1, A_1, Z_1, SENSOR_SPEC, PROB_SPEC, N_test)
        self.assertIsNotNone(dbn, "returned None")
        reference = gtsam.DiscreteBayesNet()
        for k in range(1, N_test + 1):
            reference.add(Z_1[k], [X_1[k]], SENSOR_SPEC)
        for k in reversed(range(1, N_test)):
            reference.add(X_1[k + 1], [X_1[k], A_1[k]], PROB_SPEC)
        reference.add(X_1[1], test_prior)
        self.assertTrue(
            dbn.equals(reference),
            (
                "conditionals must be added in the order: all Z[k]|X[k], then X[k+1]|X[k],A[k] for "
                "k = N-1 down to 1, then the prior on X[1]"
            ),
        )

    def test_create_simple_action_sequence(self, create_simple_action_sequence=None):
        create_simple_action_sequence = create_simple_action_sequence or _student(
            "create_simple_action_sequence"
        )
        A_1 = Variables().discrete_series("A", range(1, 4), ACTIONS)
        seq = create_simple_action_sequence(A_1, 3)
        self.assertIsInstance(seq, dict, "Action series must be dict.")
        self.assertEqual(list(seq.keys()), [A_1[1], A_1[2], A_1[3]], "keys must be A[1], A[2], ...")
        self.assertEqual(list(seq.values()), ["3->2", "3->2", "3->2"])

    def test_create_custom_action_sequence(self, create_custom_action_sequence=None):
        create_custom_action_sequence = create_custom_action_sequence or _student(
            "create_custom_action_sequence"
        )
        A_1 = Variables().discrete_series("A", range(1, 8), ACTIONS)
        self.assertIsInstance(
            create_custom_action_sequence(A_1, 2), dict, "Action series must be dict."
        )
        self.assertEqual(
            list(create_custom_action_sequence(A_1, 6).values()),
            ["3->2", "2->6", "6->7", "7->6", "6->2", "2->3"],
        )
        self.assertEqual(
            list(create_custom_action_sequence(A_1, 7).values()),
            ["3->2", "2->6", "6->7", "7->6", "6->2", "2->3", "3->2"],
        )

    def _reference_dbn(self, N_test, prior):
        V = Variables()
        X_1 = V.discrete_series("X", range(1, N_test + 1), ROOMS)
        A_1 = V.discrete_series("A", range(1, N_test), ACTIONS)
        Z_1 = V.discrete_series("Z", range(1, N_test + 1), LIGHT_LEVELS)
        dbn = gtsam.DiscreteBayesNet()
        for k in range(1, N_test + 1):
            dbn.add(Z_1[k], [X_1[k]], SENSOR_SPEC)
        for k in reversed(range(1, N_test)):
            dbn.add(X_1[k + 1], [X_1[k], A_1[k]], PROB_SPEC)
        dbn.add(X_1[1], prior)
        cycle = ["3->2", "2->6", "6->7", "7->6", "6->2", "2->3"]
        seq = {A_1[i + 1]: cycle[i % 6] for i in range(N_test - 1)}
        return dbn, X_1, A_1, Z_1, seq

    def test_ancestral_sampling(self, ancestral_sampling=None):
        ancestral_sampling = ancestral_sampling or _student("ancestral_sampling")
        N_test = 3
        dbn, X_1, A_1, Z_1, seq = self._reference_dbn(N_test, "0/0/1/0/0/0/0/0/0")
        sample = ancestral_sampling(dbn, seq)
        self.assertIsNotNone(sample, "returned None")
        self.assertEqual(len(sample), 3 * N_test - 1, "a sample assigns every X, Z and A variable")
        self.assertEqual(
            sample[X_1[1][0]], ROOMS.index("3"), "with the kitchen prior X1 is always the kitchen"
        )

    def test_simulate_multiple(self, simulate_multiple=None):
        simulate_multiple = simulate_multiple or _student("simulate_multiple")
        N_test = 8
        dbn, X_1, A_1, Z_1, seq = self._reference_dbn(N_test, "0/0/1/0/0/0/0/0/0")
        result = simulate_multiple(dbn, seq, 20, N_test)
        self.assertEqual(len(result), 20)
        for room in result:
            self.assertIn(room, ROOMS, "each entry must be a room string such as '2'")
        self.assertEqual(
            set(simulate_multiple(dbn, seq, 20, 1)),
            {"3"},
            "K=1 with the kitchen prior is always '3'",
        )

    def test_plot_state_histogram(self, plot_state_histogram=None):
        plot_state_histogram = plot_state_histogram or _student("plot_state_histogram")
        fig = plot_state_histogram(["2", "2", "3", "6"])
        self.assertIsNotNone(fig, "returned None")
        layout = fig.layout
        self.assertTrue(layout.xaxis.title.text, "x axis needs a title")
        self.assertTrue(layout.yaxis.title.text, "y axis needs a title")

    # --- Section 3.4 -------------------------------------------------------------------------

    def test_create_factor_graph(self, create_factor_graph=None):
        """Provided function; checks the fixed Fall 2026 version builds N likelihood factors."""
        create_factor_graph = create_factor_graph or _student("create_factor_graph")
        graph, X = create_factor_graph(4)
        self.assertEqual(graph.size(), 1 + 3 + 4, "prior + 3 transitions + 4 measurements")
        graph, X = create_factor_graph(
            4,
            measurements=["dark"] * 4,
            actions=["3->2", "2->6", "6->7"],
            prior="1/1/1/1/1/1/1/1/1",
        )
        self.assertEqual(graph.size(), 8)

    def test_naive_MPE(self, naive_MPE=None):
        naive_MPE = naive_MPE or _student("naive_MPE")
        create_factor_graph = _student("create_factor_graph")
        N = 4
        graph, X = create_factor_graph(N)
        value, trajectory = naive_MPE(graph, N, X)
        self.assertIsNotNone(trajectory, "returned None trajectory")
        self.assertGreater(value, 0)
        self.assertEqual(list(trajectory.items()), list(graph.optimize().items()))

    def test_GTSAM_MPE(self, GTSAM_MPE=None):
        GTSAM_MPE = GTSAM_MPE or _student("GTSAM_MPE")
        create_factor_graph = _student("create_factor_graph")
        graph, X = create_factor_graph(6)
        mpe = GTSAM_MPE(graph)
        self.assertIsNotNone(mpe, "returned None")
        self.assertEqual(list(mpe.items()), list(graph.optimize().items()))

    def test_plot_time_complexity(self, plot_time_complexity=None):
        plot_time_complexity = plot_time_complexity or _student("plot_time_complexity")
        source = inspect.getsource(plot_time_complexity)
        for needed in ["naive_MPE", "GTSAM_MPE", "create_factor_graph", "px.line"]:
            self.assertIn(needed, source, f"plot_time_complexity must use {needed}")

    # --- Section 3.5 -------------------------------------------------------------------------

    def test_create_markov_chain(self, create_markov_chain=None):
        create_markov_chain = create_markov_chain or _student("create_markov_chain")
        N = 8
        V = Variables()
        X_1 = V.discrete_series("X", range(1, N + 1), ROOMS)
        A_1 = V.discrete_series("A", range(1, N), ACTIONS)
        reference = gtsam.DiscreteBayesNet()
        for k in reversed(range(1, N)):
            reference.add(X_1[k + 1], [X_1[k], A_1[k]], PROB_SPEC)
        chain = create_markov_chain(PROB_SPEC, N)
        self.assertIsNotNone(chain, "returned None")
        self.assertTrue(chain.equals(reference), "add X[k+1] | X[k], A[k] for k = N-1 down to 1")

    def test_generate_reward_table(self, generate_reward_table=None):
        generate_reward_table = generate_reward_table or _student("generate_reward_table")
        reward_function = _student("reward_function")
        R = generate_reward_table(PROB_SPEC)
        self.assertEqual(R.shape, (len(ROOMS), len(ACTIONS), len(ROOMS)))
        expected = np.zeros_like(R)
        expected[:, :, ROOMS.index("3")] = reward_function("2", "2->3", "3")
        self.assertTrue(
            np.array_equal(R, expected),
            "R[x, a, y] = reward_function(ROOMS[x], ACTIONS[a], ROOMS[y])",
        )

    def test_control_tape_reward(self, control_tape_reward=None):
        control_tape_reward = control_tape_reward or _student("control_tape_reward")
        generate_reward_table = _student("generate_reward_table")
        create_markov_chain = _student("create_markov_chain")
        N = 8
        V = Variables()
        R = generate_reward_table(PROB_SPEC)
        X_1 = V.discrete_series("X", range(1, N + 1), ROOMS)
        A_1 = V.discrete_series("A", range(1, N), ACTIONS)
        chain = create_markov_chain(PROB_SPEC, N)
        # From room 1 the action 2->1 is impossible, so the robot stays in room 1 and earns nothing.
        actions = {A_1[k]: "2->1" for k in range(1, N)}
        self.assertEqual(control_tape_reward(chain, "1", actions, R, A_1, X_1), 0.0)

    def test_get_custom_policy(self, get_custom_policy=None):
        get_custom_policy = get_custom_policy or _student("get_custom_policy")
        policy = get_custom_policy()
        self.assertIsNotNone(policy, "returned None")
        self.assertEqual(len(policy), len(ROOMS), "one action index per room")
        for a in policy:
            self.assertIsInstance(
                a, (int, np.integer), "entries are action indices, e.g. ACTIONS.index('2->3')"
            )
            self.assertIn(a, range(len(ACTIONS)))

    def test_policy_iteration(self, policy_iteration=None):
        policy_iteration = policy_iteration or _student("policy_iteration")
        generate_reward_table = _student("generate_reward_table")
        get_transition_prob = _student("get_transition_prob")
        R = generate_reward_table(PROB_SPEC)
        T = get_transition_prob(PROB_SPEC)
        policy, values = policy_iteration(R, T)
        self.assertEqual(len(policy), len(ROOMS))
        self.assertEqual(len(values), len(ROOMS))
        self.assertTrue(np.isfinite(values).all())
        self.assertIn(
            ACTIONS[policy[ROOMS.index("2")]],
            {"2->1", "2->3", "2->4", "2->5", "2->6", "2->9"},
            "from the corridor the optimal action leaves the corridor",
        )
        self.assertAlmostEqual(
            values[ROOMS.index("3")],
            100.0,
            places=3,
            msg="staying in the kitchen earns 10 every step: 10 / (1 - 0.9) = 100",
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
