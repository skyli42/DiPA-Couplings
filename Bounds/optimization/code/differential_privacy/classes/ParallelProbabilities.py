"""
A class to help study the probability distribution of the
outputs of a DiPA given an input.
"""

import numpy as np

from classes.DiPAPresenter import DiPAPresenter
from helpers.constants import OUTPUT_ALPHABET
from helpers.dipa_constructors import *
from classes.DiPAClass import DiPATraverser, DiPA
import scipy.integrate

from helpers.laplace import laplace_pdf_without_x, integral_of_laplace
from helpers.string_functions import format_output_history, format_input_history

OutputPMF = dict[str, float]

import concurrent.futures


def compute_chunk(dipa: DiPA,
                  input_sequence: list[float],
                  epsilon: float,
                  start: int,
                  end: int):
    trav = DiPATraverser(dipa, epsilon)
    observed_outputs = {}

    for i in range(start, end):
        trav.feed_sequence(input_sequence)
        output = trav.get_output_string()
        if output not in observed_outputs:
            observed_outputs[output] = 0
        observed_outputs[output] += 1
        trav.reset_state_variables()

        # Print progress
        if i % 100000 == 0:
            print(f"Progress: {i}/{end}")

    return observed_outputs


def compute_PMF_stochastic_parallel(dipa: DiPA,
                                    input_sequence: list[float],
                                    epsilon: float,
                                    n_trials: int = 100000,
                                    n_workers: int = 4):
    # split n_trials into chunks for each worker
    chunk_size = n_trials // n_workers
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_workers)]
    if n_trials % n_workers:
        chunks[-1] = (chunks[-1][0], n_trials)

    total_observed_outputs = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_chunk = {executor.submit(compute_chunk, dipa, input_sequence, epsilon, chunk[0], chunk[1]): chunk for
                           chunk in chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                chunk_observed_outputs = future.result()
                # merge the results
                for key, value in chunk_observed_outputs.items():
                    if key not in total_observed_outputs:
                        total_observed_outputs[key] = 0
                    total_observed_outputs[key] += value
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    return {k: total_observed_outputs[k] / n_trials for k in total_observed_outputs}


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


def generate_output_sequences_stochastic(dipa: DiPA, epsilon: float, input_sequence: list[float],
                                         n_trials: int = 10000):
    output_sequences = set()
    trav = DiPATraverser(dipa, epsilon)
    for i in range(n_trials):
        trav.feed_sequence(input_sequence)
        output_sequences.add(tuple(trav.get_output_history()))
        trav.reset_state_variables()
    return output_sequences


def compute_numerical_probability(dipa: DiPA,
                                  epsilon: float,
                                  input_history: list[float],
                                  output_history: list[Outputs]):
    def pr(epsilon: float, x: float, rho_index: int):
        """
        Compute the probability of the state sequence with current
        value x and start index = rho_index.
        :param epsilon:
        :param x:
        :param rho_index:
        :return:
        """
        if rho_index == n - 1:  # last state in the sequence
            return 1.0

        cur_state = state_sequence[rho_index]
        cur_trans = trans_sequence[rho_index]

        v, w = -np.inf, np.inf
        l, u = v, w

        # k, k_err = scipy.integrate.quad(
        #     lambda z: laplace_pdf(z - cur_state.mu - input_history[rho_index],
        #                           cur_state.d * epsilon),
        #     v,
        #     w
        # )
        k = 1.0

        # k_prime, k_prime_err = scipy.integrate.quad(
        #     lambda z: laplace_pdf(z - cur_state.mu_prime - input_history[rho_index],
        #                           cur_state.d_prime * epsilon),
        #     v,
        #     w
        # )
        k_prime = 1.0

        nu = cur_state.mu + input_history[rho_index]

        if not cur_trans.assignment_trans:
            if cur_trans.guard == Guards.TRUE_CONDITION:  # Case c = true
                if cur_trans.output in OUTPUT_ALPHABET:
                    return pr(epsilon, x, rho_index + 1)
                elif cur_trans.output == Outputs.INSAMPLE_OUTPUT:
                    return k * pr(epsilon, x, rho_index + 1)
                else:
                    return k_prime * pr(epsilon, x, rho_index + 1)

            elif cur_trans.guard == Guards.INSAMPLE_GTE_CONDITION:  # case c = insample >= x
                if cur_trans.output == Outputs.INSAMPLE_PRIME_OUTPUT:
                    return k_prime * \
                        integral_of_laplace(nu, cur_state.d * epsilon, x, np.inf) * \
                        pr(epsilon, x, rho_index + 1)
                else:
                    l_prime = max(x, l)
                    return integral_of_laplace(nu, cur_state.d * epsilon, l_prime, u) * \
                        pr(epsilon, x, rho_index + 1)

            else:  # case c = insample < x
                if cur_trans.output == Outputs.INSAMPLE_PRIME_OUTPUT:
                    return k_prime * \
                        integral_of_laplace(nu, cur_state.d * epsilon, -np.inf, x) * \
                        pr(epsilon, x, rho_index + 1)
                else:
                    u_prime = min(x, u)
                    ret = integral_of_laplace(nu, cur_state.d * epsilon, l, u_prime) * \
                          pr(epsilon, x, rho_index + 1)
                    return ret
        else:  # assignment transition
            if cur_trans.guard == Guards.TRUE_CONDITION:  # case c = true
                if cur_trans.output == Outputs.INSAMPLE_PRIME_OUTPUT:
                    return k_prime * \
                        scipy.integrate.quad(
                            lambda z: laplace_pdf_without_x(z - nu,
                                                            cur_state.d * epsilon) *
                                      pr(epsilon, z, rho_index + 1),
                            -np.inf,
                            np.inf)[0]

                else:  # AT btw
                    ret, err = scipy.integrate.quad(
                        lambda z: laplace_pdf_without_x(z - nu,
                                                        cur_state.d * epsilon) *
                                  pr(epsilon, z, rho_index + 1),
                        l,
                        u,
                        limit=2000,
                        epsabs=1e-20, )
                    print("Integration error: ", err)
                    return ret
            elif cur_trans.guard == Guards.INSAMPLE_GTE_CONDITION:  # case c = insample >= x
                if cur_trans.output == Outputs.INSAMPLE_PRIME_OUTPUT:
                    return k_prime * \
                        scipy.integrate.quad(
                            lambda z: laplace_pdf_without_x(z - nu,
                                                            cur_state.d * epsilon) *
                                      pr(epsilon, z, rho_index + 1),
                            x,
                            np.inf)[0]
                else:
                    l_prime = max(x, l)
                    return scipy.integrate.quad(
                        lambda z: laplace_pdf_without_x(z - nu,
                                                        cur_state.d * epsilon) *
                                  pr(epsilon, z, rho_index + 1),
                        l_prime,
                        x)[0]
            else:  # case c = insample < x
                if cur_trans.output == Outputs.INSAMPLE_PRIME_OUTPUT:
                    return k_prime * \
                        scipy.integrate.quad(
                            lambda z: laplace_pdf_without_x(z - nu,
                                                            cur_state.d * epsilon) *
                                      pr(epsilon, z, rho_index + 1),
                            -np.inf,
                            x)[0]
                else:
                    u_prime = min(x, u)
                    return scipy.integrate.quad(
                        lambda z: laplace_pdf_without_x(z - nu,
                                                        cur_state.d * epsilon) *
                                  pr(epsilon, z, rho_index + 1),
                        x,
                        u_prime)[0]

    # Get the sequence of states corresponding to this output history
    state_sequence = [dipa.init_state, ]
    trans_sequence = []
    cur_state = dipa.init_state
    for output in output_history:
        trans = dipa.get_transition_for_output(cur_state, output)
        next_state = trans.get_dest_state()
        state_sequence.append(next_state)
        trans_sequence.append(trans)
        cur_state = next_state

    # for each state state_sequence[i], the transition out of it is stored in trans_sequence[i]

    n = len(state_sequence)

    # Compute the probability of this state sequence

    return pr(epsilon, 0, 0)


def compute_PMF_numerical(dipa: DiPA,
                          input_sequence: list[float],
                          epsilon: float
                          ):
    """
    Computes the PMF of the output sequence produced by the given DiPA.
    :param dipa:
    :param input_sequence:
    :param epsilon:
    :return:
    """

    # need some way of getting all possible output sequences. for now,
    # just do it stochastically

    output_sequences: set[tuple[Outputs]] = generate_output_sequences_stochastic(dipa, epsilon, input_sequence, 100000)

    pmf = {format_output_history(seq): 0 for seq in output_sequences}
    for seq in output_sequences:
        print("Computing probability for sequence: ", format_output_history(seq))
        pmf[format_output_history(seq)] = compute_numerical_probability(dipa, epsilon, input_sequence, seq)

    return pmf


def distance_on_adj_input(dipa, epsilon, input_sequence, noise, output_seq):
    p1 = compute_numerical_probability(dipa, epsilon, input_sequence, output_seq)
    input_sequence += noise
    p2 = compute_numerical_probability(dipa, epsilon, input_sequence, output_seq)
    return p1 / p2


def test_svt_limiting_behaviour():
    dipa = construct_svt_dipa()

    # presenter = DiPAPresenter(dipa)
    # presenter.visualize()

    epsilon = 0.1
    i = 100
    input_sequence = np.array([0] + [0] * i + [0])
    noise = np.array([1] + [-1] * i + [1])
    output_seq = [Outputs.EMPTY_OUTPUT] + [Outputs.BOT] * i + [Outputs.TOP]
    output_seq_str = format_output_history(output_seq)

    # p1 = compute_PMF_stochastic_parallel(dipa, input_sequence, epsilon, n_trials=3 * 10 ** 6)
    # p2 = compute_PMF_stochastic_parallel(dipa, input_sequence + noise, epsilon, n_trials=3 * 10 ** 6)
    #
    # print("Stochastically:")
    # print("p1: ", p1[output_seq_str])
    # print("p2: ", p2[output_seq_str])
    # print("ratio: ", p1[output_seq_str] / p2[output_seq_str])

    print("Numerically:")
    p1_num = compute_numerical_probability(dipa, epsilon, input_sequence, output_seq)
    p2_num = compute_numerical_probability(dipa, epsilon, input_sequence + noise, output_seq)

    print("p1: ", p1_num)
    print("p2: ", p2_num)
    print("ratio: ", p2_num / p1_num)

    print("Bound: ", np.exp(4 * epsilon))


def test_svt_3_limiting_behaviour():
    dipa = construct_svt_3_dipa()

    i = 10
    j = i * 2
    k = i * 3
    X_1 = np.array([0] + [0] * i + [0] + [0] * j + [0] + [0] * k + [0])
    X_2 = np.array([1] + [-1] * i + [1] + [-1] * j + [1] + [-1] * k + [1])

    epsilon = 1

    output_seq = [Outputs.EMPTY_OUTPUT] + [Outputs.BOT] * i + [Outputs.TOP] + [Outputs.BOT] * j + [Outputs.TOP] + [
        Outputs.BOT] * k + [Outputs.TOP]

    print(format_input_history(X_1))
    print(format_input_history(X_2))
    print(format_input_history(output_seq))

    p1 = compute_numerical_probability(dipa, epsilon, X_1, output_seq)
    print("p1: ", p1)

    p2 = compute_numerical_probability(dipa, epsilon, X_2, output_seq)
    print("p2: ", p2)

    print("ratio: ", max(p1 / p2, p2 / p1))

    print("Bound: ", np.exp(2.5 * epsilon))


def test_tree_dipa_1():
    dipa = construct_tree_dipa_1()

    # presenter = DiPAPresenter(dipa)
    # presenter.visualize()

    epsilon = 0.1

    y = 200
    z = y
    X_1 = np.array([0, y, z])
    X_2 = np.array([1, y - 1, z - 1])

    output_seq = [Outputs.EMPTY_OUTPUT, Outputs.BOT, Outputs.TOP]

    p1 = compute_numerical_probability(dipa, epsilon, X_1, output_seq)
    print("p1: ", p1)

    p2 = compute_numerical_probability(dipa, epsilon, X_2, output_seq)
    print("p2: ", p2)

    print("ratio: ", max(p1 / p2, p2 / p1))

    print("Bound: ", np.exp(epsilon * 3.0))
    print("Target:", np.exp(epsilon * 2.0))


def test_line_dipa():
    dipa = construct_line_dipa()

    # presenter = DiPAPresenter(dipa)
    # presenter.visualize()

    epsilon = 0.1

    X_1 = np.array([0, -1, -1, -1, -1, 1])
    X_2 = np.array([-1, 0, 0, 0, 0])

    output_seq = [Outputs.BOT] * 4 + [Outputs.TOP]

    p1 = compute_numerical_probability(dipa, epsilon, X_1, output_seq)
    print("p1: ", p1)

    p2 = compute_numerical_probability(dipa, epsilon, X_2, output_seq)
    print("p2: ", p2)

    r = max(p1 / p2, p2 / p1)
    print("ratio:       ", r)
    # print(np.log2(r)/epsilon)

    print("Bound (S^N): ", np.exp(epsilon * (1 / 2 * 5)))
    print("Bound (S^L): ", np.exp(epsilon * (1)))


def test_construct_simple_fork_dipa():
    dipa = construct_simple_fork_dipa()

    presenter = DiPAPresenter(dipa)
    presenter.visualize()

    epsilon = 0.1

    X_1 = np.array([0.0, 230.0])
    X_2 = X_1 + np.array([1, -1])
    print(f"X : {X_1}")
    print(f"X': {X_2}")
    output_seq = [Outputs.BOT, Outputs.BOT]
    print(f"o : {format_output_history(output_seq)}")

    p1 = compute_numerical_probability(dipa, epsilon, X_1, output_seq)
    print("p1: ", p1)

    p2 = compute_numerical_probability(dipa, epsilon, X_2, output_seq)
    print("p2: ", p2)

    print("ratio: ", max(p1 / p2, p2 / p1))

    print("Bound: ", np.exp(epsilon * (2.0)))


def test_complex_fork_dipa():
    """
    TODO: Incomplete
    :return:
    """
    dipa = construct_complex_fork_dipa()
    presenter = DiPAPresenter(dipa)
    presenter.visualize()

    epsilon = 0.1

    x = 290.0
    X_1 = np.array([0.0, x, -x])
    X_2 = X_1 + np.array([1, -1, 1])
    print(f"X : {X_1}")
    print(f"X': {X_2}")
    output_seq = [Outputs.BOT, Outputs.BOT]
    print(f"o : {format_output_history(output_seq)}")
    print(f"o : {format_output_history(output_seq)}")

    p1 = compute_numerical_probability(dipa, epsilon, X_1, output_seq)
    print("p1: ", p1)

    p2 = compute_numerical_probability(dipa, epsilon, X_2, output_seq)
    print("p2: ", p2)

    print("ratio: ", max(p1 / p2, p2 / p1))

    print("Bound: ", np.exp(epsilon * (3.0)))


if __name__ == "__main__":
    test_tree_dipa_1()
