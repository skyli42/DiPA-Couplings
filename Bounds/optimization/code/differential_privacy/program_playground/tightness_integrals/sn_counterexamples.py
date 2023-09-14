from classes.DiPAClass import DiPA
from helpers.constants import Outputs
from helpers.dipa_constructors import construct_snake_segment
from classes.DiPAPresenter import DiPAPresenter
from classes.ParallelProbabilities import compute_numerical_probability

# def test_output_probabilities(dipa: DiPA, X_1: list[float], X_2: list[float], output: list[Outputs]):


# i, j = 5, 1 # Number of < and >= transitions
# dipa = construct_snake_segment(i, j)
#
# # presenter = DiPAPresenter(dipa)
# # presenter.visualize()
#
# z = 1.0
# X_1 = [0.0] + [z] * i + [z] * j
# X_2 = [1.0] + [z-1] * i + [z-1] * j
# output = [Outputs.BOT] * i + [Outputs.TOP] * j
#
# print("X_1: " + str(X_1))
# print("X_2: " + str(X_2))

