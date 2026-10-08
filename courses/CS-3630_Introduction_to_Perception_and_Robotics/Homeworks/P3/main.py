import time

import gtbook
import gtbook.display
import gtsam
import numpy as np
import pandas as pd
import plotly.express as px
from gtbook.discrete import DiscreteKey, Variables
from plotly.graph_objects import Figure

VARIABLES = Variables()  # container object for variables


def pretty(obj):
    """
    Pretty print a gtsam object using gtbook.display.pretty().
    """
    return gtbook.display.pretty(obj, VARIABLES)


def show(obj, **kwargs):
    """
    Show a gtsam object using gtbook.display.show().
    """
    figure = gtbook.display.show(obj, VARIABLES, **kwargs)
    try:
        figure.pipe(format="svg")
    except Exception:
        print(
            "show(): the Graphviz `dot` program was not found, so this diagram is skipped. It "
            "is optional (diagrams are not graded); install it with: conda install -c "
            "conda-forge graphviz"
        )
        return None
    return figure


###########
# Constants
###########
ROOMS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]  # using numbers instead of room names for conciseness, map above of room and number

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
]  # all possible actions robot can take (2 * total number of doorways)

LIGHT_LEVELS = [
    "dark",
    "medium",
    "light",
]  # possible light levels in each room


###############################################################################
# Modeling the State Space
###############################################################################


def create_action_series(action_variable_char: str, indices: list[int]) -> dict[int, DiscreteKey]:
    """
    Returns a discrete series of actions.

        Parameters:
            character (str): a single character assigned to the action state variable.
            indices (list of ints): a list of integer indices.

        Returns:
            A (dict): a dictionary with the keys being integer indices and the values
                      being gtsam.DiscreteKeys.
    """
    return VARIABLES.discrete_series(
        character=action_variable_char, indices=indices, domain=ACTIONS
    )


def create_state_series(state_variable_char: str, indices: list[int]) -> dict[int, DiscreteKey]:
    """
    Returns a discrete series of rooms.

        Parameters:
            character (str): a single character assigned to the room state variable.
            indices (list of ints): a list of integer indices.

        Returns:
            X (dict): a dictionary with the keys being integer indices and the values
                      being gtsam.DiscreteKeys.
    """
    return VARIABLES.discrete_series(character=state_variable_char, indices=indices, domain=ROOMS)


###############################################################################
# Probabilistic Outcomes
###############################################################################


def create_motion_model(
    states: dict[int, DiscreteKey], actions: dict[int, DiscreteKey], action_spec: str
) -> gtsam.DiscreteConditional:
    """
    Constructs the first-step transition / motion model P(X2 | X1, A1). `states` must contain
    state variables at indices 1 and 2, and `actions` must contain an action variable at index 1.
    Additional entries are ignored.

        Note:
            This is what the assignment asks for, but is brittle because it hard-codes the required
            indices. A reusable implementation would accept a timestep `t` and use `states[t + 1]`,
            `states[t]`, and `actions[t]`

        Parameters:
            states (dict): a discrete series of states (rooms).
            actions (dict): a discrete series of actions.
            action_spec (str): a table that contains all possible values for P(X2|X1, A1).

        Returns:
            motion_model (gtsam.DiscreteConditional): a DiscreteConditional that
            describes the state transition model P(X2|X1, A1).
    """
    return gtsam.DiscreteConditional(
        key=states[2],  # X2: next room
        parents=[states[1], actions[1]],  # X1: current room; A1: chosen action
        spec=action_spec,  # string probability spec
    )


# Probability table containing all possible transition matrices
prob_spec = """
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


def create_bayes_net(motion_model: gtsam.DiscreteConditional) -> gtsam.DiscreteBayesNet:
    """
    Returns a discrete bayes net of the motion model P(X2|X1, A1).

        Parameters:
            motion_model (gtsam.DiscreteConditional): a DiscreteConditional that describes
            the state transition model P(X2|X1, A1)

        Returns:
            bayes_net (gtsam.DiscreteBayesNet): a DiscreteBayesnet that describes the motion model
    """
    bayes_net = gtsam.DiscreteBayesNet()  # create object; keeps list of DiscreteConditional
    bayes_net.add(motion_model)  # add single DiscreteConditional to constructor
    return bayes_net


###############################################################################
# Section 3.3: Dynamic Bayes Nets & Sensor Models
###############################################################################


def create_sensor_series(sensor_variable_char: str, indices: list[int]) -> dict[int, DiscreteKey]:
    """
    Returns a discrete series of light levels.

        Parameters:
            sensor_variable_char (str): a single character assigned to the light
            levels state variable
            indices (list of ints): a list of integer indices

        Returns:
            Z (dict): a dictionary with the keys being integer indices and the values
                      being gtsam.DiscreteKeys
    """
    return VARIABLES.discrete_series(
        character=sensor_variable_char, indices=indices, domain=LIGHT_LEVELS
    )


sensor_spec = """
    15/50/35 20/60/20 15/45/40 20/55/25 20/55/25 20/55/25 20/55/25 40/50/10 60/35/5
"""


def get_prior() -> str:
    """
    Returns the priors to have uniform distribution over all the rooms.

        Returns:
            prior (str): the uniform distribution.
    """
    return "1/1/1/1/1/1/1/1/1"


def get_kitchen_start_prior() -> str:
    """
    Returns a prior that reflects the knowledge that the robot always starts in the kitchen.

        Returns:
            kitchen_start_prior (str): the prior distribution.
    """
    return "0/0/1/0/0/0/0/0/0"


def create_dbn(
    priors: str,
    X: dict[int, DiscreteKey],
    A: dict[int, DiscreteKey],
    Z: dict[int, DiscreteKey],
    sensor_spec: str,
    prob_spec: str,
    N: int,
) -> gtsam.DiscreteBayesNet:
    """
    Return a dynamic bayes net. Loops over X's/A's/Z's entries, creating
    a `gtsam.DiscreteConditional` for each transition and each observation
    and adds all to a `gtsam.DiscreteBayesNet` object.

        Note:
            GTSAM stores conditionals in reverse-topological order because
            DiscreteBayesNet.sample() traverses the collection in reverse to
            perform ancestral sampling, where parents must be sampled before their children.

            So, observation conditionals need to be stored first, transition conditionals
            stored from the latest to the earliest, and the root prior P(X1) stored last.
            During reverse traversal, GTSAM samples X1 first, then X2, X3,
            and so on, followed by the observations.

            The conditionals are connected through shared variable keys: X[k] is the child
            of one transition conditional and the parent of both the next transition and
            the corresponding observation conditional.

        Parameters:
            priors (str): the prior knowledge of the robot with regards to the room state
            X (dict): a discrete series of (room) states
            A (dict): a discrete series of actions
            Z (dict): a discrete series of sensor measurements (light levels)
            sensor_spec (str): A table that contains all possible values of P(Z1|X1)
            prob_spec (str): A table that contains all possible values of P(X2|X1, A1)
            N: number of states for the DBN

        Returns:
            bayes_net (gtsam.DiscreteBayesNet): the dynamic bayes net that represents
            the evolution of the (room) state over time
    """
    if len(X) != len(Z) or len(Z) != N or len(A) != N - 1:
        raise ValueError(
            f"Length of X must be same as Z and match N; length of A must be N-1. Currently: "
            f"X={len(X)}, Z={len(Z)}, A={len(A)}, N={N}"
        )
    bayes_net = gtsam.DiscreteBayesNet()

    # add observation models - N observations
    for index in range(1, N + 1):
        observation_model = gtsam.DiscreteConditional(
            key=Z[index],  # Zi: this rooms observation
            parents=[X[index]],  # Xi: current room; only depends on current room
            spec=sensor_spec,  # string sensor spec
        )
        bayes_net.add(observation_model)

    # add motion models - N-1 actions / transitions; reverse order addition
    for index in range(N - 1, 0, -1):
        motion_model = gtsam.DiscreteConditional(
            key=X[index + 1],  # Xi+1: next room
            parents=[X[index], A[index]],  # Xi: current room; Ai: chosen action
            spec=prob_spec,  # string probability spec
        )
        bayes_net.add(motion_model)

    # priors at first state; added last
    prior_distribution = gtsam.DiscreteConditional(key=X[1], parents=[], spec=priors)
    bayes_net.add(prior_distribution)
    return bayes_net


def create_simple_action_sequence(
    A: dict[int, DiscreteKey], num_actions: int
) -> dict[DiscreteKey, str]:
    """
    Return an action sequence such that all actions are 3->2 (kitchen to corridor).

        Parameters:
            A (dict): Actions dictionary
            num_actions (int): number of actions within the sequence

        Returns:
            action_seq (dict): a dictionary with the key being the action
            state variable (for instance, "A1", "A2", ...) and the value being
            the action from "3->2"
    """
    if num_actions < 1:
        raise ValueError("number of actions must be at least 1.")

    if len(A) < num_actions:
        raise ValueError(
            f"num_actions ({num_actions}) cannot exceed the available action variables ({len(A)})"
        )

    action_seq: dict[DiscreteKey, str] = {}
    for i in range(1, num_actions + 1):
        action_seq[A[i]] = "3->2"

    return action_seq


def create_custom_action_sequence(
    A: dict[int, DiscreteKey], num_actions: int
) -> dict[DiscreteKey, str]:
    """
    Return an action sequence for starting from the kitchen and visiting both
    of the small offices, and then returning to the kitchen.

    The action cycle should be as follows: 3->2, 2->6, . . . . and so on.
    It will be a cycle of 6. If the number of actions is not a multiple of 6,
    that is ok, the robot will simply stop afer the last action in whatever state it is in.

        Parameters:
            A: Actions dictionary
            num_actions (int): number of actions within the sequence

        Returns:
            action_seq (dict): a dictionary with the key being the action state
            variable (for instance, "A1", "A2", ...) and the value being the specific
            action from "L", "R", "U", "D"
    """
    if num_actions < 1:
        raise ValueError("number of actions needs to be greater than 0.")

    if len(A) < num_actions:
        raise ValueError(
            f"num_actions ({num_actions}) cannot exceed the available action variables ({len(A)})"
        )

    action_seq: dict[DiscreteKey, str] = {}
    actions: dict[int, str] = {1: "3->2", 2: "2->6", 3: "6->7", 4: "7->6", 5: "6->2", 6: "2->3"}

    for i in range(1, num_actions + 1):
        action_seq[A[i]] = actions[(i - 1) % 6 + 1]  # convert to 0-base and back

    return action_seq


def ancestral_sampling(
    dbn: gtsam.DiscreteBayesNet, action_sequence: dict[DiscreteKey, str]
) -> gtsam.DiscreteValues:
    """
    Returns an ancestral sampling of a dynamic bayes net.

        Parameters:
            dbn (gtsam.DiscreteBayesNet): the dynamic bayes net that represents the
            evolution of the (room) state over time.
            action_sequence (dict): a dictionary with the key being the action state
            variable (for instance, "A1", "A2", ...) and the value being the specific
            action from "L", "R", "U", "D".

        Returns:
            samples (gtsam.DiscreteValues): a DiscreteValues variable that contains
            key-value pairs with the key being all variables within the dynamic bayes
            net (for instance X1, X2, A1, A2, Z1, Z2) and the value being its
            corresponding sampled value.
    """
    actions = VARIABLES.assignment(action_sequence)  # convert to DiscreteValues
    return dbn.sample(actions)  # sample accepts DiscreteValues


def simulate_multiple(
    dbn: gtsam.DiscreteBayesNet, action_sequence: dict[DiscreteKey, str], N: int, K: int
) -> list[str]:
    """
    Runs ancestral sampling multiple times and returns the Kth state of each iteration.

        Parameters:
            dbn (gtsam.DiscreteBayesNet): the dynamic bayes net that represents
            the evolution of the (room) state over time.
            action_sequence (dict): a dictionary with the key being the action state
            variable (for instance, "A1", "A2", ...) and the value being the specific
            action from "3->2", "2->6", "6->7", "7->6", "6->2", "2->3".
            N (int): number of times to simulate the dbn.
            K (int): which value to return.

        Returns:
            resulting_states (list): a list of strings containing the final state
            of ancestral sampling.
    """
    resulting_states: list[str] = []

    # run ancestral sampling N times, appending kth state to result
    for _ in range(N):
        samples = ancestral_sampling(dbn=dbn, action_sequence=action_sequence)
        resulting_states.append(ROOMS[samples[gtsam.Symbol("X", K).key()]])

    return resulting_states


def plot_state_histogram(state_list: list[str]) -> Figure:
    """
    Plots the graph showing the number of times each state is visited in the list provided.
    Check out https://plotly.com/python/plotly-express/ for more info about using plotly.

        Parameters:
            state_list (list): the list of visited states

        Returns:
            figure (histogram): the histogram of visited states
    """
    x_title = "Room / State"
    df = pd.DataFrame({x_title: state_list})
    return px.histogram(
        df,
        x=x_title,
        title="Visited Rooms at Same Timestamp",
        color_discrete_sequence=["indianred"],
        text_auto=True,
    )


############
# Perception
############


def create_factor_graph(
    N: int,
    measurements: list[str] | None = None,
    actions: list[str] | None = None,
    prior: str | None = None,
) -> tuple[gtsam.DiscreteFactorGraph, dict[int, DiscreteKey]]:
    """
    Returns (graph, X): the HMM factor graph over N room states and its state series.

        Parameters:
            N (int): number of states X1..XN
            measurements (list of str, optional): N light levels from LIGHT_LEVELS, one per state.
                Default: LIGHT_LEVELS[k % 3] for k = 0..N-1 (dark, medium, light, dark, ...).
            actions (list of str, optional): N-1 actions from ACTIONS, one per transition.
                Default: the values of create_custom_action_sequence(A, N-1).
            prior (str, optional): 9-number spec for P(X1).
                Default: the robot starts in the kitchen.

        Returns:
            graph (gtsam.DiscreteFactorGraph), X (dict): the factor graph and the state series
    """
    X = create_state_series("X", range(1, N + 1))
    A = create_action_series("A", range(1, N))
    Z = create_sensor_series("Z", range(1, N + 1))

    if actions is None:
        action_assignment = VARIABLES.assignment(create_custom_action_sequence(A, N - 1))
    else:
        action_assignment = VARIABLES.assignment({A[k]: actions[k - 1] for k in range(1, N)})

    if measurements is None:
        measurements = [LIGHT_LEVELS[k % len(LIGHT_LEVELS)] for k in range(N)]

    if prior is None:
        prior = "0 0 1 0 0 0 0 0 0"

    # A DiscreteFactorGraph prior is a space-separated spec, while the priors
    # returned by get_prior()/get_kitchen_start_prior() are slash-separated
    # DiscreteConditional specs. Accept both.
    prior = prior.replace("/", " ")

    graph = gtsam.DiscreteFactorGraph()
    graph.add(X[1], prior)  # \phi(X_1) = P(X_1) prior

    for k in range(1, N):
        conditional = gtsam.DiscreteConditional(X[k + 1], [X[k], A[k]], prob_spec)
        conditional_a_k = conditional.choose(action_assignment)  # \phi(X,X+) = P(X+|X,A=a)
        graph.push_back(conditional_a_k)

    for k, measurement in enumerate(measurements):
        conditional = gtsam.DiscreteConditional(Z[k + 1], [X[k + 1]], sensor_spec)
        z_k = LIGHT_LEVELS.index(measurement)
        factor = conditional.likelihood(z_k)  # \phi(X) = P(Z=z|X)
        graph.push_back(factor)

    return graph, X


def generate_all_trajectories(
    n: int,
    X: dict[int, DiscreteKey],
    arr: list[str],
    i: int,
    trajectories_list: list[gtsam.DiscreteValues],
):
    """
    Recursively generates every possible state trajectory of length ``n``.

    Each trajectory assigns one room to every state variable ``X[1]`` through
    ``X[n]``. Completed trajectories are converted to
    ``gtsam.DiscreteValues`` and appended to ``trajectories_list``.

    Args:
        n (int): Number of states in each trajectory.
        X (dict[int, DiscreteKey]): State variables indexed from 1 to ``n``.
        arr (list[str]): Working list used to construct the current trajectory.
        i (int): Index in ``arr`` currently being assigned.
        trajectories_list (list[gtsam.DiscreteValues]): List populated with
            all generated trajectories.

    Returns:
        None: Results are added to ``trajectories_list`` in place.
    """
    if i == n:
        trajectories_list.append(VARIABLES.assignment({X[k + 1]: arr[k] for k in range(n)}))
        return

    for elem in range(9):
        arr[i] = ROOMS[elem]
        generate_all_trajectories(n, X, arr, i + 1, trajectories_list)


def naive_MPE(
    graph: gtsam.DiscreteFactorGraph, N: int, X: dict[int, DiscreteKey]
) -> tuple[float, gtsam.DiscreteValues]:
    """
    Implement the function to return the value for maximum probable explanation and
    its corresponding state trajectory.

    The naive implementation tries every possible trajectory and selects the one
    with the highest probability.

      Parameters:
          graph (gtsam.DiscreteFactorGraph): the factor graph used to find mpe.
          N (int): number of states.
          X (dict): states dictionary.

      Returns:
          mpe_value (float): the value for maximum probable explanation.
          mpe_trajectory (gtsam.DiscreteValues): the DiscreteValues variable that contains
          the key-value pair with the key being the (room) state variable and the value
          being its corresponding value.
    """
    mpe_value: float = 0
    mpe_trajectory = gtsam.DiscreteValues()
    trajectories: list[gtsam.DiscreteValues] = []

    # Get all possible trajectories given the inputs
    generate_all_trajectories(n=N, X=X, arr=[""] * N, i=0, trajectories_list=trajectories)

    for trajectory in trajectories:
        value = graph(trajectory)

        if value > mpe_value:
            mpe_value = value
            mpe_trajectory = trajectory

    return mpe_value, mpe_trajectory


def GTSAM_MPE(graph: gtsam.DiscreteFactorGraph) -> gtsam.DiscreteValues:
    """
    Implements the function to return the mpe trajectory using gtsam.

        Parameters:
            graph (gtsam.DiscreteFactorGraph): the factor graph used to find mpe.

        Returns:
            mpe_trajectory (gtsam.DiscreteValues): the DiscreteValues variable that
            contains the key-value pair with the key being the (room) state variable
            and the value being its corresponding value.
    """
    return graph.optimize()


def plot_time_complexity() -> Figure:
    """
    Plots the graph to compare the execution time between the naive mpe approach and
    the gtsam approach.

    Check out https://plotly.com/python/plotly-express/ for more info about using plotly.

    X-axis is the N (number states) and Y-axis should be execution time

        Returns:
            time_complexity (Figure): the figure containing the time complexity comparison
            between naive MPE and gtsam MPE for 3 to 7 number of states.
    """
    naive_times: list[float] = []
    gtsam_times: list[float] = []

    # run both functions over a sweep N 3-9 recording times
    for i in range(3, 8):
        graph, X = create_factor_graph(i)

        naive_start = time.perf_counter()
        naive_MPE(graph=graph, N=i, X=X)
        naive_end = time.perf_counter()

        gtsam_start = time.perf_counter()
        GTSAM_MPE(graph=graph)
        gtsam_end = time.perf_counter()

        naive_times.append(naive_end - naive_start)
        gtsam_times.append(gtsam_end - gtsam_start)

    # convert to data frame for plotting
    df = pd.DataFrame(
        {
            "Number of States (N)": range(3, 8),
            "Naive MPE": naive_times,
            "GTSAM MPE": gtsam_times,
        }
    )
    fig = px.line(
        df,
        x="Number of States (N)",
        y=["Naive MPE", "GTSAM MPE"],
        title="Time vs. N States: Naive vs. GTSAM",
        labels={
            "value": "Time (seconds)",
            "variable": "Function",
        },
        markers=True,
    )
    return fig


########################################
# Section 3.5: Markov Decision Processes
########################################

N = 8


def reward_function(state: str, action: str, next_state: str) -> float:
    """
    Reward that returns 10 upon entering the Kitchen.

        Parameters:
            state (str): The current state the robot is in e.g. "2",
            meaning Room 2 -> the Corridor.
            action (str): The action the robot will take e.g. "2->3",
            meaning going from Room 2 (Corridor) to Room 3 (Kitchen).
            next_state (str) -> The resulting state the robot will be
            after taking ``action`` e.g. "3" meaning Room 3 (Kitchen).

        Returns:
            reward (float): the reward achieved. 10.0 if ``next_state``
            is kitchen; 0.0 if not kitchen.
    """
    return 10.0 if next_state == "3" else 0.0


def perform_rollout(markov_chain, x1, actions, X):
    """
    Roll out states given actions as a dictionary.
    """
    dict = actions.copy()  # copy to not edit in-place
    dict[X[1]] = x1  # assign x1 state to dict entry
    given = VARIABLES.assignment(dict)  # convert dict to DiscreteValues
    return markov_chain.sample(given)  # rollout outcome to x2, x3, etc. return sampled values dict


def reward(R, rollout, A, X, k):
    """Return state, action, next_state triple for given rollout at time k."""
    state = rollout[X[k][0]]  # lookup current state at time k
    action = rollout[A[k][0]]
    next_state = rollout[X[k + 1][0]]
    return R[state, action, next_state]  # return reward for this combination


def rollout_reward(R, rollout, A, X, horizon, gamma=1.0):
    """Calculate reward for a given rollout"""
    # discount factor of gamma; k-1 powered as cummulative
    discounted_rewards = [gamma ** (k - 1) * reward(R, rollout, A, X, k) for k in range(1, horizon)]
    return sum(discounted_rewards)


def get_transition_prob(prob_spec: str) -> np.array:
    """
    Convert a probability matrix from str to array.

        Parameters:
            prob_spec (str): The probability matrix in string form e.g. "10/2/23/0/90".

        Returns:
            T (np.array): The array probability matrix of size (S, U, S), where
            S = len(ROOMS) and U = len(ACTIONS).
            P(next state | state, action) for all next states, states and actions.
    """
    S, U = len(ROOMS), len(ACTIONS)  # size of state space

    # P(next state | state, action) for all next state, state and action given prob_spec
    transition_matrix = gtsam.DiscreteConditional((2, S), [(0, S), (1, U)], prob_spec)

    T = np.empty((S, U, S), float)
    for x_a_y, prob in transition_matrix.enumerate():
        x, a, y = x_a_y[0], x_a_y[1], x_a_y[2]
        T[x, a, y] = prob  # populate transition probability matrix from DiscreteConditional
    return T


def create_markov_chain(prob_spec: str, N: int) -> gtsam.DiscreteBayesNet:
    """
    Implement the function such that it returns a Markov Chain P(X_{t + 1} | X_t, A_t)

        Parameters:
            prob_spec (str): The probability matrix in string form e.g. "10/2/23/0/90".
            N (int): number of time steps.

        Returns:
            markov_chain (gtsam.DiscreteBayesNet): DiscreteBayesNet that describes the Markov Chain.
    """
    X = VARIABLES.discrete_series("X", range(1, N + 1), ROOMS)  # X1, X2, X3, X4, etc.
    A = VARIABLES.discrete_series("A", range(1, N), ACTIONS)  # A1, A2, A3, A4, etc.
    markov_chain = gtsam.DiscreteBayesNet()

    for i in reversed(range(1, N)):
        transition_matrix = gtsam.DiscreteConditional(
            key=X[i + 1],
            parents=[X[i], A[i]],
            spec=prob_spec,
        )  # build transition matrix for every state and action
        markov_chain.add(transition_matrix)

    return markov_chain


def generate_reward_table(prob_spec: str) -> np.array:
    """
    Generates a reward table using the reward_function.

    Parameters:
            action_spec (str): A table that contains all possible values of P(X2|X1, A1)

    Returns:
            R (np.array): the rewards table containing rewards for every next state, state
            and action.
    """
    transition_matrix = gtsam.DiscreteConditional(
        key=(2, len(ROOMS)), parents=[(0, len(ROOMS)), (1, len(ACTIONS))], spec=prob_spec
    )  # build transition matrix for every next state, state and action

    R = np.empty((len(ROOMS), len(ACTIONS), len(ROOMS)), float)  # 3D array

    for x_a_y, _ in transition_matrix.enumerate():
        x, a, y = x_a_y[0], x_a_y[1], x_a_y[2]
        R[x, a, y] = reward_function(ROOMS[x], ACTIONS[a], ROOMS[y])

    return R


def control_tape_reward(
    markov_chain: gtsam.DiscreteBayesNet,
    x1: str,
    actions: dict[DiscreteKey, str],
    R: np.array,
    A,
    X,
) -> float:
    """
    Implement the function to calculate the reward given a sequence of actions.

    Parameters:
        markov_chain (gtsam.DiscreteBayesNet): the markov chain for all P(X_2 | X_1, A_1).
        x1 (str): the initial (room) state.
        actions (dict): the dictionary that represents a sequence of actions taken by the agent.
        R (np.array): array of expected reward given the current state.
        A (dict): actions dictionary. A1, A2, A3, A4, etc.
        X (dict): states dictionary. X1, X2, X3, X4, X5, etc.

    Returns:
        reward (float): the total reward of the sequence
    """
    rollout = perform_rollout(markov_chain=markov_chain, x1=x1, actions=actions, X=X)
    reward = rollout_reward(R=R, rollout=rollout, A=A, X=X, horizon=(len(actions) + 1))
    return reward


def get_custom_policy():
    """
    Returns a custom policy.
    """
    return [
        ACTIONS.index("1->2"),
        ACTIONS.index("2->3"),
        ACTIONS.index("3->2"),
        ACTIONS.index("4->2"),
        ACTIONS.index("5->2"),
        ACTIONS.index("6->7"),
        ACTIONS.index("7->6"),
        ACTIONS.index("8->9"),
        ACTIONS.index("9->2"),
    ]


def get_test_policy():
    # The test policy provides a preferred action from each room
    reasonable_policy = [ACTIONS.index("2->3")] * len(ROOMS)
    return reasonable_policy


def Q_value(R, T, value_function, x, a, gamma=0.9):
    """Calculate Q(x,a) from given value function"""
    return T[x, a] @ (R[x, a] + gamma * value_function)


def calculate_value_system(pi, R, T, gamma=0.9):
    """Calculate A, b matrix of linear system for value computation."""
    b = np.empty((len(ROOMS),), float)
    AA = np.empty((len(ROOMS), len(ROOMS)), float)
    for x, _room in enumerate(ROOMS):
        a = pi[x]  # action under policy
        b[x] = T[x, a] @ R[x, a]  # expected reward under policy pi
        AA[x] = -gamma * T[x, a]
        AA[x, x] += 1
    return AA, b


def calculate_value_function(pi, R, T, gamma=0.9):
    """Calculate value function for given policy"""
    AA, b = calculate_value_system(pi, R, T, gamma)
    return np.linalg.solve(AA, b)


def update_policy(R, T, value_function):
    """Update policy given a value function"""
    new_policy = [None for _ in range(len(ROOMS))]
    for x, _room in enumerate(ROOMS):
        Q_values = [Q_value(R, T, value_function, x, a) for a in range(len(ACTIONS))]
        new_policy[x] = np.argmax(Q_values)
    return new_policy


def policy_iteration(R, T, pi=None, max_iterations=100):
    """Do policy iteration, starting from policy `pi`."""
    if pi is None:
        num_states = T.shape[0]
        pi = np.zeros(num_states, dtype=int)

    for _ in range(max_iterations):
        value = calculate_value_function(pi=pi, R=R, T=T)  # evaluate policy
        new_pi = update_policy(R=R, T=T, value_function=value)  # update policy

        if np.array_equal(pi, new_pi):
            return pi, value  # converged

        pi = new_pi

    raise RuntimeError("No stable policy found after {max_iterations} iterations")
