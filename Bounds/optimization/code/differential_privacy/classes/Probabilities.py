"""
A class to help study the probability distribution of the
outputs of a DiPA given an input.
"""
from fractions import Fraction

import numpy as np

from helpers.dipa_constructors import construct_svt_dipa
from classes.DiPAClass import DiPATraverser, DiPA

OutputPMF = dict[str, float]


def compute_PMF_stochastic(dipa: DiPA,
                           input_sequence: list[float],
                           epsilon: float,
                           n_trials: int = 100000):
    """

    :param dipa: The DiPA to compute the probability mass function for.
    :param input_sequence: The input sequence to feed into the DiPA.
    :param epsilon: The privacy parameter.
    :param n_trials: The number of trials to run.
    :return: A dictionary mapping output strings to their probabilities.
    """

    trav = DiPATraverser(dipa, epsilon)

    observed_outputs = {}

    for i in range(n_trials):
        trav.feed_sequence(input_sequence)
        output = trav.get_output_string()
        if output not in observed_outputs:
            observed_outputs[output] = 0
        observed_outputs[output] += 1
        trav.reset_state_variables()

        # Print progress
        if i % 100000 == 0:
            print(f"Progress: {i}/{n_trials}")

    return {k: observed_outputs[k] / n_trials for k in observed_outputs}


def pretty_print_pmf(pmf: dict[str, float]):
    """
    Pretty print a probability mass function.
    :param pmf: The probability mass function to pretty print.
    :return: None
    """

    # Sort the keys
    keys = list(pmf.keys())
    keys.sort()

    # Print the keys such that the probabilities line up

    # Find the longest key
    longest_key = max(len(key) for key in keys)

    # Print the keys and probabilities

    for key in keys:
        print(f"{key:>{longest_key}}: {pmf[key]}")


def compute_distance_asymmetric(pmf1: OutputPMF, pmf2: OutputPMF):
    """
    Compute the distance between two probability mass functions,
    given by the maximum ratio (pmf1/pmf2) of their probabilities of the same output.
    Note: this is asymmetric, because we only consider ratios of the form pmf1/pmf2.
    :param pmf1: The first probability mass function.
    :param pmf2: The second probability mass function.
    :return: The distance between the two probability mass functions.
    """
    # assert that they have the same keys
    assert set(pmf1.keys()) == set(pmf2.keys())

    # Compute the ratios
    ratios = [pmf1[key] / pmf2[key] for key in pmf1]

    # Return the maximum ratio
    return max(ratios)


def compute_distance(pmf1: OutputPMF, pmf2: OutputPMF):
    """
    Compute the distance between two probability mass functions,
    given by the maximum ratio (pmf1/pmf2) of their probabilities of the same output.
    :param pmf1: The first probability mass function.
    :param pmf2: The second probability mass function.
    :return: The distance between the two probability mass functions.
    """
    return max(compute_distance_asymmetric(pmf1, pmf2),
               compute_distance_asymmetric(pmf2, pmf1))


if __name__ == "__main__":
    dipa = construct_svt_dipa()

    epsilon = 0.01
    n_trials = 1000000

    input_sequence = np.arange(-10.0, 10.0, 1)
    pmf1 = compute_PMF_stochastic(dipa, input_sequence, epsilon, n_trials)

    # compute noise for each input between -1 and 1
    noise = np.random.random(len(input_sequence)) * 2 - 1
    input_sequence += noise
    pmf2 = compute_PMF_stochastic(dipa, input_sequence, epsilon, n_trials)

    pretty_print_pmf(pmf1)
    pretty_print_pmf(pmf2)

    print(compute_distance(pmf1, pmf2))
    print(np.exp(epsilon))