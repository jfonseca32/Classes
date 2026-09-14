"""
This document depends on the implicit correct ordering of the items in every dictionary and list.
That is brittle and should definitely include checks and be more idiomatic in a real implementation.
"""

import math
from enum import Enum
from typing import Literal, TypeVar

import gtsam
import numpy as np
from gtbook.discrete import Variables

# Initialize the global ranom number generator for sampling
RNG: np.random.Generator = np.random.default_rng(seed=42)  # seed for reproducible results


### ENUMS ###
class Item(Enum):
    PRINTED_PRODUCT = 0
    STRUCTURAL_BRACING = 1
    PCB_BOARD = 2
    WING_COMPONENT = 3
    MOTOR = 4


class Thickness(Enum):
    THIN = 0
    MEDIUM = 1
    THICK = 2


class Rigidity(Enum):
    RIGID = 0
    FLEXIBLE = 1


class Material(Enum):
    PLASTIC = 0
    METAL = 1
    CERAMIC = 2


class Actions(Enum):
    PRINTED_BIN = 0
    STRUCTURAL_BIN = 1
    BOARD_BIN = 2
    WING_BIN = 3
    MOTOR_BIN = 4


# TypeVar for all possible enums we may want to sample from
KeyType = TypeVar("KeyType", Item, Thickness, Rigidity, Material)


### CONSTANTS ###
# All possible item classes
CLASSES = ["3D_Printed_Product", "Structural_Bracing", "PCB_Board", "Wing_Component", "Motor"]

PRIOR = {
    Item.PRINTED_PRODUCT: 0.3,
    Item.STRUCTURAL_BRACING: 0.2,
    Item.PCB_BOARD: 0.35,
    Item.WING_COMPONENT: 0.1,
    Item.MOTOR: 0.05,
}

# dict of dict, where first key is Action, and second key is item category
COST_DICT = {
    Actions.PRINTED_BIN: {
        Item.PRINTED_PRODUCT: 0,
        Item.STRUCTURAL_BRACING: 8,
        Item.PCB_BOARD: 10,
        Item.WING_COMPONENT: 20,
        Item.MOTOR: 18,
    },
    Actions.STRUCTURAL_BIN: {
        Item.PRINTED_PRODUCT: 2,
        Item.STRUCTURAL_BRACING: 0,
        Item.PCB_BOARD: 4,
        Item.WING_COMPONENT: 14,
        Item.MOTOR: 12,
    },
    Actions.BOARD_BIN: {
        Item.PRINTED_PRODUCT: 3,
        Item.STRUCTURAL_BRACING: 3,
        Item.PCB_BOARD: 0,
        Item.WING_COMPONENT: 6,
        Item.MOTOR: 6,
    },
    Actions.WING_BIN: {
        Item.PRINTED_PRODUCT: 14,
        Item.STRUCTURAL_BRACING: 12,
        Item.PCB_BOARD: 6,
        Item.WING_COMPONENT: 0,
        Item.MOTOR: 5,
    },
    Actions.MOTOR_BIN: {
        Item.PRINTED_PRODUCT: 16,
        Item.STRUCTURAL_BRACING: 14,
        Item.PCB_BOARD: 6,
        Item.WING_COMPONENT: 5,
        Item.MOTOR: 0,
    },
}


def get_item_prior_pmf() -> list[float]:
    """
    Prior probabilities PMF.

    Returns the probability mass function (PMF) of the prior probabilities of the item classes.

    Returns
    -------
    list[float]
        A list of the PMF.
    """
    item_prior_pmf: list[float] = []

    for prob in PRIOR.values():
        item_prior_pmf.append(prob)

    return item_prior_pmf


def get_cdf(probs: dict[KeyType, float]) -> dict[KeyType, float]:
    """
    Create a Cumulative Distribution Function for sampling.

    Adds each item's probability to the cumulative sum of previous
    items' probabilities.

    Parameters
    ----------
    probs : dict[Item | Thickness | Thickness | Rigidity | Material, float]
        A dictionary with each Item's probabilities to make the CDF from.

    Returns
    -------
    dict[Item, float]
        The generated CDF dictionary.

    Raises
    -------
    ValueError
        If the sum of probabilities is not close to 1.0.
    """
    cdf = {}
    cum_prob: float = 0  # keep track of current cummulative sum

    for item, prob in probs.items():
        cdf[item] = cum_prob + prob
        cum_prob += prob

    # isclose so floating point error is ignored
    if not math.isclose(cum_prob, 1.0):
        raise ValueError("Sum of probabilites is not 1.0.")

    return cdf


def sample_item() -> Literal[0, 1, 2, 3, 4]:
    """
    Returns a sample of an item class by sampling with the prior probabilities.

    Returns
    -------
    Literal[0, 1, 2, 3, 4]
        An int indicating the sampled item class.

    Raises
    ------
    ValueError
        If the generated number isn't within any of the bounds of the CDF.
    """
    num: float = RNG.random()  # 0<=num<=1
    cdf = get_cdf(PRIOR)

    for item, prob in cdf.items():
        if num <= prob:
            return item.value

    raise ValueError("Could not generate sample.")


def get_cost_table() -> np.ndarray:
    """
    Returns the cost table.

    Returns
    -------
    cost_table : np.ndarray
        numpy array where rows are Actions and columns are Item classes.
    """
    num_rows: int = len(COST_DICT)
    num_cols: int = len(next(iter(COST_DICT.values())))  # len of first inner dict; all same len
    cost_table: np.ndarray = np.zeroes((num_rows, num_cols))

    for row, items in enumerate(COST_DICT.values()):
        for col, cost in enumerate(items.values()):
            cost_table[row][col] = cost

    return cost_table


# 1. Weight - continuous-valued sensor
def get_pWT() -> np.array:
    """
    Makes P(Weight | Item Class) as means and standard deviations.

    Returns
    -------
    np.array
        A 5x2 numpy array of [mean, std_dev] for each class.
    """
    # Mean and Std. Dev. for:
    # 3D_Printed_Product, Structural_Bracing, PCB_Board, Wing_Component, Motor
    pWT: np.array = np.array([[150, 75], [800, 400], [50, 25], [250, 125], [600, 300]])
    return pWT


def sample_weight(item_class: Literal[0, 1, 2, 3, 4] | None = None) -> float:
    """
    Returns a sample of weight for a given item class.

    Having item_class be an int is very brittle. Should instead use Item
    and inside call Item.value.

    Parameters
    ----------
    item_class : Literal[0, 1, 2, 3, 4] | None
        integer indicating which item from Item enum. If None, an item is
        automatically sampled using sample_item().

    Returns
    -------
    float
        sampled weight in grams.
    """
    mean: int
    std_dev: int
    sample: float
    chosen_class: int = sample_item() if item_class is None else item_class

    mean, std_dev = get_pWT()[chosen_class]  # very brittle; assumes it is always in the same order
    sample = RNG.normal(mean, std_dev)  # np's normal distribution sampler

    return sample


# 2. Thickness - multi-valued sensor
def get_pTT() -> np.array:
    """
    Returns P(Thickness | Item Class) as a NumPy array.

    Returns
    -------
    np.array
        A 5x3 NumPy array where pTT[i, j] is the probability of thickness j given class i.
    """
    # P(Thin | Class), P(Medium | Class), P(Thick | Class)
    pTT: np.array = np.array(
        [
            [0.9, 0.1, 0.0],  # 3D Printed Product
            [0.0, 0.2, 0.8],  # Structural Bracing
            [0.3, 0.55, 0.15],  # PCB Board
            [0.0, 0.8, 0.2],  # Wing Component
            [0.4, 0.5, 0.1],  # Motor
        ]
    )

    return pTT


def sample_thickness(item_class: Literal[0, 1, 2, 3, 4] | None = None) -> Literal[0, 1, 2]:
    """
    Returns a sample of thickness for a given item class.

    Having item_class be an int is very brittle. Should instead use Item
    and inside call Item.value.

    Parameters
    ----------
    item_class : Literal[0, 1, 2, 3, 4] | None
        Integer indicating which item from Item enum. If None, an item is
        automatically sampled using sample_item().

    Returns
    -------
    Literal[0, 1, 2]
        sampled thickness (thin: 0, medium: 1, thick: 2).
    """
    num = RNG.random()  # 0<=num<=1
    chosen_class = sample_item() if item_class is None else item_class

    thin, medium, thick = get_pTT()[chosen_class]
    item_cdf = get_cdf({Thickness.THIN: thin, Thickness.MEDIUM: medium, Thickness.THICK: thick})

    for thickness, prob in item_cdf.items():
        if num <= prob:
            return thickness.value

    raise ValueError("Could not generate sample.")


# 3. Rigidity - binary sensor
def get_pRT() -> np.array:
    """
    Returns P(Rigidity | Item Class) as a NumPy array.

    Returns
    -------
    np.array
        A 5x2 NumPy array where pRT[i, j] is the probability of rigidity j given class i.
    """
    # P(Rigid | Class), P(Flexible | Class)
    pRT = np.array(
        [
            [0.2, 0.8],  # 3D Printed Product
            [0.9, 0.1],  # Structural Bracing
            [0.8, 0.2],  # PCB Board
            [0.4, 0.6],  # Wing Component
            [0.3, 0.7],  # Motor
        ]
    )

    return pRT


def sample_rigidity(item_class: Literal[0, 1, 2, 3, 4] | None = None) -> Literal[0, 1]:
    """
    Returns a sample of rigidity for a given item class.

    Having item_class be an int is very brittle. Should instead use Item
    and inside call Item.value.

    Parameters
    ----------
    item_class : Literal[0, 1, 2, 3, 4] | None
        integer indicating which item from Item enum. If None, an item is
        automatically sampled using sample_item().

    Returns
    -------
    Literal[0, 1]
        sampled thickness (rigid: 0, flexible: 1).
    """
    num = RNG.random()  # 0<=num<=1
    chosen_class = sample_item() if item_class is None else item_class

    rigid, flexible = get_pRT()[chosen_class]
    item_cdf = get_cdf({Rigidity.RIGID: rigid, Rigidity.FLEXIBLE: flexible})

    for rigidity, prob in item_cdf.items():
        if num <= prob:
            return rigidity.value

    raise ValueError("Could not generate sample.")


# 4. Surface Material - multi-valued sensor
def get_pMT() -> np.array:
    """
    Returns P(Material | Item Class) as a NumPy array.

    Returns
    -------
    np.array
        A 5x3 NumPy array where pMT[i, j] is the probability of material j given class i.
    """
    # P(Plastic | Class), P(Metal | Class), P(Ceramic | Class)
    pMT = np.array(
        [
            [0.95, 0.02, 0.03],  # 3D Printed Product
            [0.05, 0.85, 0.1],  # Structural Bracing
            [0.05, 0.35, 0.6],  # PCB Board
            [0.8, 0.15, 0.05],  # Wing Component
            [0.1, 0.85, 0.05],  # Motor
        ]
    )

    return pMT


def sample_material(item_class: Literal[0, 1, 2, 3, 4] | None = None) -> Literal[0, 1, 2]:
    """
    Returns a sample of rigidity for a given item class.

    Having item_class be an int is very brittle. Should instead use Item
    and inside call Item.value.

    Parameters
    ----------
    item_class : Literal[0, 1, 2, 3, 4] | None
        integer indicating which item from Item enum. If None, an item is
        automatically sampled using sample_item().

    Returns
    -------
    Literal[0, 1]
        sampled thickness (rigid: 0, flexible: 1).
    """
    num = RNG.random()  # 0<=num<=1
    chosen_class = sample_item() if item_class is None else item_class

    plastic, metal, ceramic = get_pMT()[chosen_class]
    item_cdf = get_cdf(
        {Material.PLASTIC: plastic, Material.METAL: metal, Material.CERAMIC: ceramic}
    )

    for material, prob in item_cdf.items():
        if num <= prob:
            return material.value

    raise ValueError("Could not generate sample.")


def Gaussian(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Calculate the probability density function (PDF) of a Gaussian distribution.

    Parameters
    ----------
    x : float
        The point at which to evaluate the probability density.
    mu : float, default 0.0
        The mean (center) of the Gaussian distribution.
    sigma : float, default 1.0
        The standard deviation (width) of the Gaussian distribution.

    Returns
    -------
    float
        The probability density value at `x`.
    """
    if sigma == 0:
        return 1.0 if x == mu else 0.0
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# Bayes with all sensors
def bayes_given_all_sensors(
    weight: float, thickness: Literal[0, 1, 2], rigidity: Literal[0, 1], material: Literal[0, 1, 2]
) -> list[float]:
    """
    Calculates the posteriors of all item classes by combining all four
    sensors and the priors.

    Parameters
    ----------
    weight : float
        The measured weight of the item.
    thickness : Literal[0, 1, 2]
        The measured thickness of the item.
    rigidity : Literal[0, 1]
        The measured rigidity of the item.
    material : Literal[0, 1, 2]
        The measured material of the item.

    Returns
    -------
    list[float]
        A list of posterior probabilities of each item class.
    """
    # Get distributions for each parameter
    weight_dist = get_pWT()
    thickness_dist = get_pTT()
    rigid_dist = get_pRT()
    material_dist = get_pMT()
    scores: list[float] = []

    # Iterate over Items, calculating their probabilities
    for item_class in Item:
        mean, std_dev = weight_dist[item_class.value]

        # P(I | W, T, R, M) numerator
        item_score = (
            PRIOR[item_class]
            * Gaussian(x=weight, mu=mean, sigma=std_dev)
            * thickness_dist[item_class.value][thickness]
            * rigid_dist[item_class.value][rigidity]
            * material_dist[item_class.value][material]
        )
        scores.append(item_score)

    total = sum(scores)
    posteriors = [score / total for score in scores]  # normalization step (denominator)

    return posteriors


def make_decision(posteriors: list[float]) -> Literal[0, 1, 2, 3, 4]:
    """
    Returns the decision made by the robot given the posteriors.

    Parameters
    ----------
    posteriors : list[float]
        A list of posteriors of each item class.

    Returns
    -------
    Literal[0, 1, 2, 3, 4]
        An int indicating the action (chosen class) taken by the robot.
    """
    item_class = np.argmax(posteriors)
    return item_class  # works only because Item and Action are ordered the same


# Create Variables object for GTSAM discrete inference
variables = Variables()
classes = CLASSES
Class = variables.discrete("Class", classes)
ThicknessVar = variables.discrete("Thickness", ["thin", "medium", "thick"])
RigidityVar = variables.binary("Rigidity")
MaterialVar = variables.discrete("Material", ["plastic", "metal", "ceramic"])


def get_item_prior_gtsam() -> gtsam.DiscreteDistribution:
    """
    Returns the prior probabilities of the item classes using GTSAM.

    Returns
    -------
    gtsam.DiscreteDistribution
        A DiscreteDistribution that summarizes the prior probabilities of all the items.
    """
    priors = [PRIOR[item] for item in Item]  # gtsam needs a list

    gtsam_priors = gtsam.DiscreteDistribution(
        Class,  # all items (1, 5)
        priors,
    )

    return gtsam_priors


# Prior probabilities PMF
def get_item_prior_pmf_gtsam() -> list[float]:
    """
    Returns the probability mass function (PMF) of the prior probabilities
    of the item classes.

    Returns:
    list
        A list of the PMF.
    """
    prior = get_item_prior_gtsam()
    return list(prior.pmf())  # gtsam has a function for this


def sample_item_gtsam() -> Literal[0, 1, 2, 3, 4]:
    """
    Returns a sample of an item class by sampling with the prior probabilities.

    Returns
    -------
    Literal[0, 1, 2, 3, 4]
        An int indicating the sampled item class.
    """
    prior = get_item_prior_gtsam()
    return prior.sample()  # gtsam has a function for this


# 2. Thickness - multi-valued sensor
def get_pTT_gtsam() -> gtsam.DiscreteConditional:
    """
    Returns P(Thickness | Item Class) as GTSAM Gaussian factors.

    Returns
    -------
    gtsam.DiscreteConditional
        A DiscreteConditional that indicates the conditional probability
        of thickness given the item class.
    """
    pTT = gtsam.DiscreteConditional(
        ThicknessVar,
        [Class],
        (
            "0.9/0.1/0.0 "  # 3D Printed Product
            "0.0/0.2/0.8 "  # Structural Bracing
            "0.3/0.55/0.15 "  # PCB Board
            "0.0/0.8/0.2 "  # Wing Component
            "0.4/0.5/0.1 "  # Motor
        ),
    )
    return pTT


def sample_thickness_gtsam(item_class: Literal[0, 1, 2, 3, 4] | None = None) -> Literal[0, 1, 2]:
    """
    Returns a sample of thickness using the conditional probability given
    the item class.

    Parameters
    ----------
    class : int
        An int indicating the item class.

    Returns
    -------
    int
        An int indicating the sampled thickness, the int-thickness
        mapping is given by the Thickness enum.
    """
    distribution = get_pTT_gtsam()
    sample = distribution.sample(item_class)
    return sample


# 3. Rigidity - binary sensor
def get_pRT_gtsam() -> gtsam.DiscreteConditional:
    """
    Returns P(Rigidity | Item Class) as GTSAM discrete factors.

    Returns
    -------
    gtsam.DiscreteConditional
        A DiscreteConditional that indicates the conditional
        probability of rigidity given the item class.
    """
    pRT = gtsam.DiscreteConditional(
        RigidityVar,
        [Class],
        (
            "0.2/0.8 "  # 3D Printed Product
            "0.9/0.1 "  # Structural Bracing
            "0.8/0.2 "  # PCB Board
            "0.4/0.6 "  # Wing Component
            "0.3/0.7 "  # Motor
        ),
    )
    return pRT


def sample_rigidity_gtsam(item_class: Literal[0, 1, 2, 3, 4] | None = None) -> Literal[0, 1, 2]:
    """
    Samples the rigidity given the item class using GTSAM.

    Returns
    -------
    int
        The sampled rigidity value.
    """
    distribution = get_pRT_gtsam()
    sample = distribution.sample(item_class)
    return sample


# 4. Surface Material - multi-valued sensor
def get_pMT_gtsam() -> gtsam.DiscreteConditional:
    """
    Returns P(Material | Item Class) as GTSAM discrete factors.

    Returns
    -------
    gtsam.DiscreteConditional
        A DiscreteConditional that indicates the conditional probability
        of material given the item class.
    """
    pMT = gtsam.DiscreteConditional(
        MaterialVar,
        [Class],
        (
            "0.95/0.02/0.03"  # 3D Printed Product
            "0.05/0.85/0.1"  # Structural Bracing
            "0.05/0.35/0.6"  # PCB Board
            "0.8/0.15/0.05"  # Wing Component
            "0.1/0.85/0.05"  # Motor
        ),
    )
    return pMT


def sample_material_gtsam(item_class: Literal[0, 1, 2, 3, 4] | None = None) -> Literal[0, 1, 2]:
    """
    Samples the material given the item class using GTSAM.

    Returns
    -------
    int
        The sampled material value.
    """
    distribution = get_pMT_gtsam()
    sample = distribution.sample(item_class)
    return sample


def posterior_no_sensors() -> list[float]:
    """
    Returns the posterior of all item class using only priors,
    aka no sensors.

    Returns
    -------
    list[float]
        A list of posterior of each item class.
    """
    return get_item_prior_pmf_gtsam()


def likelihood_given_weight(weight: float) -> list[float]:
    """
    Returns the likelihoods of all item classes using only the weight
    sensor (no priors).

    Parameters
    ----------
    weight : float
        A float indicating the weight of item.

    Returns
    -------
    list[float]
        A list of likelihoods of each item class.
    """
    likelihoods = []
    for mean, std_dev in get_pWT():
        likelihoods.append(Gaussian(weight, mean, std_dev))
    return likelihoods


def bayes_given_weight(weight: float) -> list[float]:
    """
    Returns the posteriors of all item classes by combining the weight
    sensor and the priors.

    Parameters
    ----------
    weight : float
        A float indicating the weight of the item.

    Returns
    -------
    list[float]
        A list of posterior probabilities of each item class.
    """
    prior = get_item_prior_gtsam()

    likelihoods = likelihood_given_weight(weight)
    likelikhood_factor = gtsam.DecisionTreeFactor(
        Class,
        likelihoods,
    )  # P(Weight | Class) for each class using likelihoods

    # likelihood of weight for each class * probability of that class existing
    # DiscreteDistribution is the normalization step
    posterior = gtsam.DiscreteDistribution(likelikhood_factor * prior)

    return list(posterior.pmf())


# Bayes with all sensors
def bayes_given_all_sensors_gtsam(
    weight: float, thickness: Literal[0, 1, 2], rigidity: Literal[0, 1], material: Literal[0, 1, 2]
) -> list[float]:
    """
    Returns the posteriors of all item classes by combining all four
    sensors and the priors.

    Parameters
    ----------
    weight : float
        A float indicating the weight of the item.
    thickness : int
        A int indicating the thickness of the item.
    rigidity : int
        A int indicating the rigidity of the item.
    material : int
        A int indicating the material of the item.

    Returns
    -------
    list[float]
        A list of posterior probabilities of each item class.
    """
    # weight_likelihood DecisionTreeFactor and rest DiscreteConditional for multiplying
    prior = get_item_prior_gtsam()
    weight_likelihood = gtsam.DecisionTreeFactor(
        Class,
        likelihood_given_weight(weight),
    )  # cannot use bayes_given_weight() as returns list[float]
    thickness_likelihood = get_pTT_gtsam().likelihood(thickness)
    rigidity_likelihood = get_pRT_gtsam().likelihood(rigidity)
    material_likelihood = get_pMT_gtsam().likelihood(material)

    unnormalized = (
        weight_likelihood  # DecisionTreeFactor comes first per gtsam
        * thickness_likelihood
        * rigidity_likelihood
        * material_likelihood
        * prior
    )
    posterior = gtsam.DiscreteDistribution(unnormalized)  # normalize

    return list(posterior.pmf())


def fit_log_normal(data: list[float]) -> tuple[float, float]:
    """
    Returns mu, sigma for a log-normal distribution.

    Parameters
    ----------
    data : list[float]
        A list of positive floats that represent the weight of an item.

    Returns
    -------
    tuple[float, float]
        The mu and sigma for a log-normal distribution.
    """
    log_data = np.log(data)  # log-normal is normal distribution on logged data

    mu = np.mean(log_data)
    sigma = np.std(log_data, ddof=1)  # assumes data is saample of population (not whole) (ddof=1)
    return (mu, sigma)


if __name__ == "__main__":
    from project2_test import TestProject2, verify

    np.random.seed(3630)
    unit_test = TestProject2()

    print("Testing item prior PMF: ", verify(unit_test.test_get_item_prior_pmf, get_item_prior_pmf))

    print("Testing item sampling: ", verify(unit_test.test_sample_item, sample_item))

    print("Testing weight sampling: ", verify(unit_test.test_sample_weight, sample_weight))

    print("Testing thickness sampling: ", verify(unit_test.test_sample_thickness, sample_thickness))

    print("Testing rigidity sampling: ", verify(unit_test.test_sample_rigidity, sample_rigidity))

    print("Testing material sampling: ", verify(unit_test.test_sample_material, sample_material))

    print(
        "Testing Bayesian inference: ",
        verify(unit_test.test_bayes_given_all_sensors, bayes_given_all_sensors),
    )

    print("Testing decision making: ", verify(unit_test.test_make_decision, make_decision))

    ### EXAMPLE RUN ###
    # Let's simulate a Structural Bracing and see what the robot does.
    # A structural bracing is typically: heavy, thick, rigid, and made of metal.
    print("\n--- SIMULATION: Testing with a typical Structural Bracing ---")
    simulated_weight = 820.0  # grams
    simulated_thickness = Thickness.THICK.value  # 2
    simulated_rigidity = Rigidity.RIGID.value  # 0
    simulated_material = Material.METAL.value  # 1

    # print(
    #     f"Sensor readings: Weight={simulated_weight}g, "
    #     f"Thickness={Thickness(simulated_thickness).name}, "
    #     f"Rigidity={Rigidity(simulated_rigidity).name}, "
    #     f"Material={Material(simulated_material).name}"
    # )

    # Calculate posterior probabilities based on these readings
    posteriors = bayes_given_all_sensors(
        simulated_weight, simulated_thickness, simulated_rigidity, simulated_material
    )

    print("\nPosterior Probabilities:")
    for i, p in enumerate(posteriors):
        print(f"  P({CLASSES[i]} | sensors) = {p:.4f}")

    # Make a decision
    decision_index = make_decision(posteriors)
    decision_class = CLASSES[decision_index]

    print(f"\nRobot's Decision: Sort as a '{decision_class}'")
    if decision_class == "Structural_Bracing":
        print("Correct decision! ✅")
    else:
        print("Incorrect decision. ❌")

    print("Testing your prior probabilities of the item classes: ")
    print(verify(unit_test.test_get_item_prior_pmf, get_item_prior_pmf_gtsam))

    print("Testing item sampling: ", verify(unit_test.test_sample_item, sample_item_gtsam))

    print(
        "Testing thickness sampling: ",
        verify(unit_test.test_sample_thickness, sample_thickness_gtsam),
    )

    print(
        "Testing your sample rigidity: ",
        verify(unit_test.test_sample_rigidity, sample_rigidity_gtsam),
    )

    print(
        "Testing material sampling: ", verify(unit_test.test_sample_material, sample_material_gtsam)
    )

    print("Testing your posterior with no sensors: ")
    print(verify(unit_test.test_posterior_no_sensor, posterior_no_sensors))

    print("Testing your posteriors with the weight sensor and priors: ")
    print(verify(unit_test.test_bayes_given_weight, bayes_given_weight))

    print(
        "Testing Bayesian inference: ",
        verify(unit_test.test_bayes_given_all_sensors, bayes_given_all_sensors_gtsam),
    )

    print(
        "Testing your log-normal distribution: ",
        verify(unit_test.test_fit_log_normal, fit_log_normal),
    )
