from classes.DiPAClass import DiPA
from classes.DiPAGraph import DiPAGraph
from classes.DiPAValidator import DiPAValidator
from classes.PrimitiveClasses import State
from helpers.constants import Guards, Outputs

from fractions import Fraction

from classes.DiPAPresenter import DiPAPresenter

import cvxpy as cp


def construct_tree_explosive_dipa():
    """
    Constructs a tree dipa branching out with height 3.
    :return: A DiPA object.
    """

    init_state = State("0")
    dipa = DiPA(init_state)

    # Add states
    for i in range(1, 8):
        dipa.add_state(State(str(i)))

    # Add transitions
    dipa.add_transition("0", "1", Guards.TRUE_CONDITION, Outputs.BOT, True)

    dipa.add_transition("1", "2", Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, True)
    dipa.add_transition("1", "3", Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, True)

    dipa.add_transition("2", "4", Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, True)
    dipa.add_transition("2", "5", Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, True)

    dipa.add_transition("3", "6", Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, True)
    dipa.add_transition("3", "7", Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, True)

    return dipa



def construct_diamond_dipa():
    """
    Constructs a DiPA with a diamond structure.
    :return: A DiPA object.
    """
    init_state = State("0")
    dipa = DiPA(init_state)

    n = 10
    # Add states
    for i in range(1, n):
        dipa.add_state(State(str(i)))
    for i in range(1, n - 1):
        dipa.add_state(State(str(i) + "_top"))
        dipa.add_state(State(str(i) + "_bot"))

    # Add transitions
    dipa.add_transition("0", "1", Guards.TRUE_CONDITION, Outputs.BOT, True)
    for i in range(1, n - 1):
        dipa.add_transition(str(i), str(i) + "_top", Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, True)
        dipa.add_transition(str(i), str(i) + "_bot", Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, True)
        dipa.add_transition(str(i) + "_top", str(i + 1), Guards.TRUE_CONDITION, Outputs.TOP, False)
        dipa.add_transition(str(i) + "_bot", str(i + 1), Guards.TRUE_CONDITION, Outputs.BOT, False)

    return dipa


def construct_problem(dipa: DiPA):
    """
    Constructs the optimization problem for the DiPA.
    :param dipa: The DiPA object.
    :return: A tuple (gamma, delta, constraints, objective) where gamma and delta are the optimization variables,
    """

    dipagraph = DiPAGraph(dipa)
    segments = dipagraph.find_all_segments()

    for seq in dipagraph.find_all_segment_sequences(segments):
        gamma = cp.Variable(len(segments))
        delta = cp.Variable(len(segments))

        constraints = []
        for i in range(len(seq) - 1):
            if dipagraph.get_first_transition(seq[i + 1]).guard == Guards.INSAMPLE_GTE_CONDITION:
                constraints.append(gamma[i] >= gamma[i + 1])
            elif dipagraph.get_first_transition(seq[i + 1]).guard == Guards.INSAMPLE_LT_CONDITION:
                constraints.append(gamma[i] <= gamma[i + 1])

        constraints.append(gamma >= -1)
        constraints.append(gamma <= 1)


dipa = construct_diamond_dipa()
# presenter = DiPAPresenter(dipa)
# presenter.visualize()

construct_problem(dipa)
