from classes.DiPAPresenter import DiPAPresenter
from helpers.dipa_constructors import *

# construct and validate the DiPA.
dipa = construct_branching_dipa()
presenter = DiPAPresenter(dipa)

# print the segments.
presenter.print_segments()

# print the constraints.
presenter.print_constraints()

# visualize the DiPA.
presenter.visualize()