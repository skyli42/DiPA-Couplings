import io

from classes.DiPAClass import DiPA
from classes.PrimitiveClasses import Transition, State
from helpers.constants import Guards, Outputs
from helpers.dipa_constructors import construct_svt_dipa, construct_svt_3_dipa, construct_privacy_violating_path_c, \
    construct_disclosing_cycle_insample_prime, construct_tree_dipa_1
from classes.DiPAPresenter import DiPAPresenter
from classes.DiPAGraph import DiPAGraph
import cvxpy as cp

from PIL import Image

def find_assignment_transition(transitions: list[Transition]) -> int:
    """
    Find the index of the assignment transition in a list of transitions.
    """
    for i, transition in enumerate(transitions):
        if transition.is_assignment_transition():
            return i

    return -1


def find_coupling_bound(dipa, deltas) -> float:
    """
    Find the coupling bound for a DiPA given a list of deltas.
    deltas[0] is the delta for the assignment transition.
    deltas[1] is the delta for the < transitions
    deltas[2] is the delta for the >= transitions
    :returns: the coupling bound
    """

    constraints = []
    objectives = []
    gammas = []

    # Find all constraints
    graph = DiPAGraph(dipa)
    segments = graph.find_all_segments()

    for i, segment in enumerate(segments):

        print(f"Segment {i}: ", segment)

        cycle_transitions = graph.cycle_transitions_in_segment(segment)
        transitions: list[Transition] = graph.find_traversable_transitions_in_segment(segment)

        assignment_index = find_assignment_transition(transitions)
        transitions[0], transitions[assignment_index] = transitions[assignment_index], transitions[0]

        segment_gammas = []
        segment_constraints = []
        segment_objectives = []
        segment_deltas = []

        assert transitions[0].is_assignment_transition()

        for transition in transitions:
            trans_var = cp.Variable()
            segment_gammas.append(trans_var)

            segment_constraints.append(trans_var >= -1)
            segment_constraints.append(trans_var <= 1)

            insample_prime_gamma = None

            if transition.guard == Guards.INSAMPLE_LT_CONDITION:
                segment_deltas.append(deltas[1])  # Feed in 1 if the guard is <
                segment_constraints.append(segment_gammas[-1] <= segment_gammas[0])
            elif transition.guard == Guards.INSAMPLE_GTE_CONDITION:  # Feed in -1 if the guard is >=
                segment_deltas.append(deltas[2])
                segment_constraints.append(segment_gammas[-1] >= segment_gammas[0])
            else:
                segment_deltas.append(deltas[0])  # maybe in this case when the guard is true, we just don't couple? no clue

            if transition.output == Outputs.INSAMPLE_OUTPUT:
                # if insample is output, then the values of insample
                # must be coupled to be the same
                segment_constraints.append(segment_gammas[-1] == 0)
            if transition.output == Outputs.INSAMPLE_PRIME_OUTPUT:
                insample_prime_gamma = cp.Variable()
                segment_constraints.append(insample_prime_gamma >= -1)
                segment_constraints.append(insample_prime_gamma <= 1)
                segment_constraints.append(insample_prime_gamma == 0)


            if transition in cycle_transitions:
                segment_constraints.append(segment_gammas[-1] == segment_deltas[-1])

                if insample_prime_gamma is not None:
                    segment_constraints.append(insample_prime_gamma == segment_deltas[-1])

            segment_objectives.append(cp.abs(segment_deltas[-1] - segment_gammas[-1]))

        gammas.append(segment_gammas)
        constraints.extend(segment_constraints)
        objectives.extend(segment_objectives)

    # Solve the problem
    objective = cp.Minimize(cp.sum(objectives))
    prob = cp.Problem(objective, constraints=constraints)
    prob.solve()

    # print the gammas
    for segment_gammas in gammas:
        print([gamma.value for gamma in segment_gammas])

    return prob.value


def display_image(image_bytes):
    import matplotlib.pyplot as plt
    image = Image.open(io.BytesIO(image_bytes))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# dipa = construct_svt_3_dipa()
# presenter = DiPAPresenter(dipa)
# image = Image.open(io.BytesIO(presenter.visualize()))
# image.save("optimisation_example.png")
#
# deltas = [-1, 1, -1]
# coupling_bound = find_coupling_bound(dipa, deltas)
# print(coupling_bound) # exists and is tight


# dipa = construct_privacy_violating_path_c()
# presenter = DiPAPresenter(dipa)
# image = presenter.visualize()
# display_image(image)
# deltas = [-1, 1, -1]
# coupling_bound = find_coupling_bound(dipa, deltas)
# print(coupling_bound)   # is infinity

# dipa = construct_disclosing_cycle_insample_prime()
# presenter = DiPAPresenter(dipa)
# image = presenter.visualize()
# display_image(image)
# deltas = [1, 1, -1]
# coupling_bound = find_coupling_bound(dipa, deltas)
# print(coupling_bound)   # is infinity


# dipa = construct_tree_dipa_1()
# presenter = DiPAPresenter(dipa)
# image = presenter.visualize()
# display_image(image)
# coupling_bound = find_coupling_bound(dipa, [1, 1, -1])
# print(coupling_bound)   # 6? TODO: check this


# Construct a branching DiPA
init_state = State("init")
s1 = State("s1")
s2 = State("s2")
s3 = State("s3")
dipa = DiPA(init_state)
dipa.add_state(s1)
dipa.add_state(s2)
dipa.add_state(s3)
dipa.add_transition('init', 's1', Guards.TRUE_CONDITION, Outputs.BOT, True)
dipa.add_transition('s1', 's2', Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
dipa.add_transition('s1', 's3', Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)
dipa.add_transition('s2', 's3', Guards.INSAMPLE_LT_CONDITION, Outputs.BOT, False)
# dipa.add_transition('s2', 's3', Guards.INSAMPLE_GTE_CONDITION, Outputs.TOP, False)

presenter = DiPAPresenter(dipa)
image = presenter.visualize()
display_image(image)

coupling_bound = find_coupling_bound(dipa, [-1, 1, -1])
print(coupling_bound)   # 6? TODO: check this