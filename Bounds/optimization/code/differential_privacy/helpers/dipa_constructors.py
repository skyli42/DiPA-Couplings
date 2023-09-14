"""
File containing different constructions of DiPAs.
"""

from classes.DiPAClass import DiPA
from classes.PrimitiveClasses import State
from helpers.constants import Guards, Outputs

from fractions import Fraction


def construct_svt_dipa() -> DiPA:
    init_state = State('0', Fraction(1, 1), 0, 1, 1)
    dipa = DiPA(init_state)

    one = State('1', Fraction(1, 1), 0, 1, 1)
    dipa.add_state(one)

    two = State('2', Fraction(1, 1), 0, 1, 1)
    dipa.add_state(two)

    dipa.add_transition_from_states(init_state, one, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)
    #
    dipa.add_transition_from_states(one, one, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(one, two, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    return dipa


def construct_svt_3_dipa() -> DiPA:
    """
    Constructs a DiPA that models the above_threshold algorithm thrice,
    outputting at most three TOPs.

    The DiPA will have the following structure:

    0 -> 1 (self-loop) -> 2 (self-loop) -> 3 (self-loop) -> 4

    :return: The DiPA.
    """

    zero = State('0', Fraction(1, 2), 0, 1, 1)
    dipa = DiPA(zero)

    one = State('1', Fraction(1, 4), 0, 1, 1)
    dipa.add_state(one)

    two = State('2', Fraction(1, 4), 0, 1, 1)
    dipa.add_state(two)

    three = State('3', Fraction(1, 4), 0, 1, 1)
    dipa.add_state(three)

    four = State('4', Fraction(1, 4), 0, 1, 1)
    dipa.add_state(four)

    dipa.add_transition_from_states(zero, one, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(one, one, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(one, two, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(two, two, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(two, three, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(three, three, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(three, four, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    return dipa


def construct_svt_3_dipa_assignments() -> DiPA:
    """
    Constructs a DiPA that models the above_threshold algorithm thrice,
    outputting at most three TOPs. This DiPA assigns the new threshold
    to be the outputs that are TOP.

    The DiPA will have the following structure:

    0 -> 1 (self-loop) -> 2 (self-loop) -> 3 (self-loop) -> 4

    :return: The DiPA.
    """

    zero = State('0', Fraction(1, 2), 0, 1, 1)
    dipa = DiPA(zero)

    one = State('1', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(one)

    two = State('2', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(two)

    three = State('3', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(three)

    four = State('4', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(four)

    dipa.add_transition_from_states(zero, one, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(one, one, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(one, two, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, True)

    dipa.add_transition_from_states(two, two, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(two, three, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, True)

    dipa.add_transition_from_states(three, three, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(three, four, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, True)

    return dipa


def construct_numeric_sparse_dipa() -> DiPA:
    """
    Constructs a DiPA that models the numeric sparse algorithm.
    :return:

    0 -> 1 (self-loop) -> 2
    """

    zero = State('0', Fraction(4, 9), 0, 1, 1)
    one = State('1', Fraction(2, 9), 0, 1, 1)
    two = State('2', Fraction(2, 9), 0, 1, 1)

    dipa = DiPA(zero)

    dipa.add_state(one)
    dipa.add_state(two)

    dipa.add_transition_from_states(zero, one, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(one, one, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(one, two, Guards.INSAMPLE_GTE_CONDITION, Outputs.INSAMPLE_PRIME_OUTPUT, False)

    return dipa


def construct_sort_dipa() -> DiPA:
    """
    Constructs a DiPA that checks whether the sequence of real numbers given as input are sorted in descending order
    :return: The DiPA.
    """

    zero = State('0', Fraction(1, 2), 0, 1, 1)
    dipa = DiPA(zero)

    one = State('1', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(one)

    two = State('2', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(two)

    dipa.add_transition_from_states(zero, one, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(one, one, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, True)
    dipa.add_transition_from_states(one, two, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    return dipa


def construct_branching_dipa() -> DiPA:
    """
    Construct a DiPA that branches into two states that have assignment transitions. The
    path graph is as follows:

    0 (assignment)  -> 1    -> 2 (assignment)     -> 3
                            -> 4 (assignment)     -> 5
    :return:
    """

    # init_state = State('0', Fraction(1, 2), 0, 1, 1)
    init_state = State('0', Fraction(1, 2), 0, 1, 1)

    dipa = DiPA(init_state)

    # one = State('1', Fraction(1, 4), 1, 1, 1)
    one = State('1', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(one)

    # two = State('2', Fraction(1, 4), 1, 1, 1)
    two = State('2', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(two)

    six = State('6', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(six)

    seven = State('7', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(seven)

    three = State('3', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(three)

    four = State('4', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(four)

    five = State('5', Fraction(1, 4), 1, 1, 1)
    dipa.add_state(five)

    dipa.add_transition_from_states(init_state, one, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)

    dipa.add_transition_from_states(one, two, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)

    dipa.add_transition_from_states(two, three, Guards.INSAMPLE_LT_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(two, six, Guards.INSAMPLE_GTE_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(six, seven, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, False)
    dipa.add_transition_from_states(seven, two, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, False)

    dipa.add_transition_from_states(one, four, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(four, five, Guards.INSAMPLE_LT_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(four, four, Guards.INSAMPLE_GTE_CONDITION, Outputs.EMPTY_OUTPUT, True)

    return dipa


def construct_svt_star() -> DiPA:
    """
    Not differentially private.
    Constructs modeling an algorithm that processes a sequence
    of real numbers and implements a “noisy’ version” of the
    following process.
    As long as the input numbers are less than threshold T (= 0)
    it outputs ⊥. Once it sees the first number ≥ T ,
    it moves to the second phase. In the phase, it outputs ⊤ as long
    as the numbers are ≥ T. When it sees the first number < T, it
    outputs ⊥ and stops. Since insample′ is never output, parameters
    used in its sampling are not shown and not important.
    :return: The DiPA.
    """

    q0 = State('0', Fraction(1, 2), 0, 1, 1)
    q1 = State('1', Fraction(1, 4), 1, 1, 1)
    q2 = State('2', Fraction(1, 4), 1, 1, 1)
    q3 = State('3', Fraction(1, 4), 1, 1, 1)

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.BOT, True)

    dipa.add_transition_from_states(q1, q1, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(q2, q2, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
    dipa.add_transition_from_states(q2, q3, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)

    return dipa


def construct_privacy_violating_path_c() -> DiPA:
    """
    Constructs a DiPA that models the algorithm A_mod, a modification of numeric sparse
    that contains a privacy violating path of type (c).
    :return:
    """

    q0 = State('0', Fraction(4, 9), Fraction(0), Fraction(1), Fraction(1))
    q1 = State('1', Fraction(4, 9), Fraction(0), Fraction(1), Fraction(1))
    q2 = State('2', Fraction(4, 9), Fraction(0), Fraction(1), Fraction(1))

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)
    dipa.add_transition_from_states(q1, q1, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_GTE_CONDITION, Outputs.INSAMPLE_OUTPUT, False)

    return dipa


def construct_privacy_violating_path_b() -> DiPA:
    """
    Constructs a DiPA that models the second case of a privacy violating path.
    :return:
    """

    q0 = State('0', Fraction(4, 9), 0, 1, 1)
    q1 = State('1', Fraction(2, 9), 0, 1, 1)
    q2 = State('2', Fraction(2, 9), 0, 1, 1)
    q3 = State('3', Fraction(1, 9), 0, 1, 1)
    q4 = State('4', Fraction(1, 9), 0, 1, 1)
    garbage = State('garbage', 1, 0, 1, 1)

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)
    dipa.add_state(q4)
    dipa.add_state(garbage)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)

    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_LT_CONDITION, Outputs.INSAMPLE_OUTPUT, True)
    dipa.add_transition_from_states(q1, garbage, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(q2, q3, Guards.INSAMPLE_GTE_CONDITION, Outputs.BOT, True)
    dipa.add_transition_from_states(q2, garbage, Guards.INSAMPLE_LT_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(q3, q3, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
    dipa.add_transition_from_states(q3, q4, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)

    return dipa


def construct_almost_privacy_violating_path_b() -> DiPA:
    """
    This DiPA is differentially private.

    This differs from the previous DiPA in that the transition from q1 to q2 outputs
    INSAMPLE_PRIME_OUTPUT instead of INSAMPLE_OUTPUT.

    This means that the transition in the first segment is no longer faulty.

    Constructs a DiPA that models the second case of an (almost) privacy violating path.
    :return: The DiPA.
    """

    q0 = State('0', Fraction(4, 9), 0, 1, 1)
    q1 = State('1', Fraction(2, 9), 0, 1, 1)
    q2 = State('2', Fraction(2, 9), 0, 1, 1)
    q3 = State('3', Fraction(1, 9), 0, 1, 1)
    q4 = State('4', Fraction(1, 9), 0, 1, 1)
    garbage = State('garbage', 1, 0, 1, 1)

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)
    dipa.add_state(q4)
    dipa.add_state(garbage)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, True)

    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q1, garbage, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(q2, q3, Guards.INSAMPLE_GTE_CONDITION, Outputs.BOT, True)
    dipa.add_transition_from_states(q2, garbage, Guards.INSAMPLE_LT_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(q3, q3, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
    dipa.add_transition_from_states(q3, q4, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)

    return dipa


def construct_tree_dipa_1():
    q0 = State('0', Fraction(2, 1), 0, 1, 1)

    q1 = State('1', Fraction(1, 2), 0, 1, 1)

    q2 = State('2', Fraction(1, 2), 0, 1, 1)

    q3 = State('3', Fraction(1, 2), 0, 1, 1)

    q4 = State('4', Fraction(1, 2), 0, 1, 1)

    q5 = State('5', Fraction(1, 2), 0, 1, 1)

    q6 = State('6', Fraction(1, 2), 0, 1, 1)

    q7 = State('7', Fraction(1, 2), 0, 1, 1)

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)
    dipa.add_state(q4)
    dipa.add_state(q5)
    dipa.add_state(q6)
    dipa.add_state(q7)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.EMPTY_OUTPUT, assignment_trans=True)

    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q1, q3, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(q2, q4, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q2, q5, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    dipa.add_transition_from_states(q3, q6, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q3, q7, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    return dipa


def construct_line_dipa():
    q0 = State('0', Fraction(1, 2), 0, 1, 1)
    q1 = State('1', Fraction(1, 2), 0, 1, 1)
    q2 = State('2', Fraction(1, 2), 0, 1, 1)
    q3 = State('3', Fraction(1, 2), 0, 1, 1)
    q4 = State('4', Fraction(1, 2), 0, 1, 1)
    q5 = State('5', Fraction(1, 2), 0, 1, 1)
    garbage = State('garbage', 1, 0, 1, 1)

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)
    dipa.add_state(q4)
    dipa.add_state(q5)
    dipa.add_state(garbage)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.BOT, True)

    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q2, q3, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q3, q4, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q4, q5, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)

    dipa.add_transition_from_states(q1, garbage, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
    dipa.add_transition_from_states(q2, garbage, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
    dipa.add_transition_from_states(q3, garbage, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
    dipa.add_transition_from_states(q4, garbage, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    return dipa


def construct_simple_fork_dipa():
    q0 = State('0', 1, 0, 1, 1)
    q1 = State('1', 1, 0, 1, 1)
    q2 = State('2', 1, 0, 1, 1)
    q3 = State('3', 1, 0, 1, 1)

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.BOT, True)
    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q1, q3, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    return dipa


def construct_complex_fork_dipa():
    q0 = State('0', 1, 2, 1, 1)
    q1 = State('1', 1, -2, 1, 1)
    q2 = State('2', 1, 0, 1, 1)
    q3 = State('3', 1, 0, 1, 1)
    q4 = State('4', 1, 0, 1, 1)

    dipa = DiPA(q0)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)
    dipa.add_state(q4)

    dipa.add_transition_from_states(q0, q1, Guards.TRUE_CONDITION, Outputs.BOT, True)
    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q1, q3, Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
    dipa.add_transition_from_states(q2, q4, Guards.INSAMPLE_GTE_CONDITION, Outputs.BOT, False)
    dipa.add_transition_from_states(q2, q3, Guards.INSAMPLE_LT_CONDITION, Outputs.TOP, False)

    return dipa


def construct_inhomogeneous_line_dipa(n):
    """
    A line DiPA (actually not a dipa cause of violation of output determinism)
    with many different input inequalities.
    :return:
    """

    init_state = State('0', Fraction(1, 1), 0, 1, 1)

    dipa = DiPA(init_state)

    for i in range(1, n):
        state = State(str(i), Fraction(1, 1), 0, 1, 1)
        dipa.add_state(state)

    dipa.add_transition_from_states(init_state, dipa.states['1'], Guards.TRUE_CONDITION, Outputs.BOT, True)

    for i in range(1, n - 1):
        dipa.add_transition_from_states(
            dipa.states[str(i)],
            dipa.states[str(i + 1)],
            Guards.INSAMPLE_LT_CONDITION,
            Outputs.BOT,
            False
        )

    return dipa


def construct_snake_segment(i: int, j: int):
    """
    Construct a segment with i < transitions and j >= transitions.
    :return:
    """

    dipa = DiPA(State('init', Fraction(1, 1), 0, 1, 1))

    dipa.add_state(State('0', Fraction(1, 1), 0, 1, 1))

    dipa.add_transition('init', '0', Guards.TRUE_CONDITION, Outputs.BOT, True)

    for k in range(1, i + 1):
        dipa.add_state(State(str(k), Fraction(1, 1), 0, 1, 1))
        dipa.add_transition(str(k - 1), str(k), Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)

    for k in range(i + 1, i + j + 1):
        dipa.add_state(State(str(k), Fraction(1, 1), 0, 1, 1))
        dipa.add_transition(str(k - 1), str(k), Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

    return dipa


def construct_disclosing_cycle_insample() -> DiPA:
    """
    Construct a DiPA with a disclosing cycle that outputs insample.
    Definition 9 (from DiPA paper). A cycle C of a DiPA A is a disclosing
    cycle if there is an i, 0 ≤ i < |C| such that trans(C[i]) is an input
    transition that outputs either insample or insample′
    :return: A DiPA with a disclosing cycle.
    """

    init_state = State('init', Fraction(1, 1), 0, 1, 1)
    q1 = State('1', Fraction(1, 1), 0, 1, 1)
    q2 = State('2', Fraction(1, 1), 0, 1, 1)

    dipa = DiPA(init_state)
    dipa.add_state(q1)
    dipa.add_state(q2)

    dipa.add_transition_from_states(init_state, q1, Guards.TRUE_CONDITION, Outputs.BOT, True)
    dipa.add_transition_from_states(q1, q1, Guards.INSAMPLE_LT_CONDITION, Outputs.INSAMPLE_OUTPUT, False)
    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_GTE_CONDITION, Outputs.BOT, False)

    return dipa


def construct_disclosing_cycle_insample_prime() -> DiPA:
    """
    Construct a DiPA with a disclosing cycle that outputs insample prime.
    Definition 9 (from DiPA paper). A cycle C of a DiPA A is a disclosing
    cycle if there is an i, 0 ≤ i < |C| such that trans(C[i]) is an input
    transition that outputs either insample or insample′
    :return: A DiPA with a disclosing cycle.
    """

    init_state = State('init', Fraction(1, 1), 0, 1, 1)
    q1 = State('1', Fraction(1, 1), 0, 1, 1)
    q2 = State('2', Fraction(1, 1), 0, 1, 1)

    dipa = DiPA(init_state)
    dipa.add_state(q1)
    dipa.add_state(q2)

    dipa.add_transition_from_states(init_state, q1, Guards.TRUE_CONDITION, Outputs.BOT, True)
    dipa.add_transition_from_states(q1, q1, Guards.INSAMPLE_LT_CONDITION, Outputs.INSAMPLE_PRIME_OUTPUT, False)
    dipa.add_transition_from_states(q1, q2, Guards.INSAMPLE_GTE_CONDITION, Outputs.BOT, False)

    return dipa


# if __name__ == "__main__":
#
#     dipa = construct_snake_segment(4, 0)
#
#     presenter = DiPAPresenter(dipa)
#     presenter.visualize()