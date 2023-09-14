import numpy as np

from classes.DiPAClass import DiPATraverser
from helpers.dipa_constructors import construct_svt_dipa


def test_dipa_svt():
    dipa = construct_svt_dipa()

    # running the algorithm
    epsilon = 0.1
    T = 0
    queries = np.arange(0, 10000)
    print(queries)

    traverser = DiPATraverser(dipa, epsilon)

    c = 100

    j = 0
    for _ in range(c):
        traverser.input(T)  # input the threshold
        j = traverser.feed_sequence(queries)  # input the sequence of queries

        queries = queries[j:]

        traverser.reset_state_variables()  # reset the traverser state

    print(traverser.get_output_string())