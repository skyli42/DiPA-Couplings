"""
An attempt to use the method of shifts to provide a proof 
that the OR counterexample is private.

There are five states and six transitions.
The states are 0, 1, 2, 3, 4.
The transitions are: 
0: 0->1, true guard, assign to x
1: 1->2, true guard, assign to y
2: 2->2, in < x or in >= y, no assignment
3: 2->3, in >= x and in < y, no assignment
4: 3->3, in >= x or in < y, no assignment
5: 3->4, in < x and in >= y, no assignment

The initial state is 0.
"""

from pysmt.shortcuts import Symbol, And, Or, is_sat, get_model, Equals
from pysmt.typing import INT
import itertools


for deltas in itertools.product([-1, 1], repeat=6):
    print(deltas)
    # For each transition, create gamma_x, gamma_y, and gamma_z

    variables = []

    for i in range(6): # 6 transitions
        variables.append([])
        for t in range(2):
            variables[i].append(Symbol("gamma_{}_{}".format(t, i), INT))

    # For each transition, create the constraints

    constraints = []

    # Inequality constraints

    for i in range(6):
        for t in range(2):
            constraints.append(variables[i][t] >= -1)
            constraints.append(variables[i][t] <= 1)

    # transition 2 has constraint gamma_2_0 <= gamma_0_0 or gamma_2_1 >= gamma_1_0

    constraints.append(Or(variables[2][0] <= variables[0][0], variables[2][1] >= variables[1][0]))

    # transition 3 has constraint gamma_3_0 >= gamma_0_0 and gamma_3_1 <= gamma_1_0

    constraints.append(And(variables[3][0] >= variables[0][0], variables[3][1] <= variables[1][0]))

    # transition 4 has constraint gamma_4_0 >= gamma_0_0 or gamma_4_1 <= gamma_1_0

    constraints.append(Or(variables[4][0] >= variables[0][0], variables[4][1] <= variables[1][0]))

    # transition 5 has constraint gamma_5_0 <= gamma_0_0 and gamma_5_1 >= gamma_1_0

    constraints.append(And(variables[5][0] <= variables[0][0], variables[5][1] >= variables[1][0]))

    # Cost constraints

    # Transition 2 and 4 are cycles, so they should have no cost. 
    # The x-cost of transition 2 is gamma_2_0 - delta_2
    # The y-cost of transition 2 is gamma_2_1 - delta_2
    constraints.append(variables[2][0] <= deltas[2])
    constraints.append(variables[2][0] >= deltas[2])
    constraints.append(variables[2][1] <= deltas[2])
    constraints.append(variables[2][1] >= deltas[2])

    # The x-cost of transition 4 is gamma_4_0 - delta_4
    # The y-cost of transition 4 is gamma_4_1 - delta_4
    constraints.append(variables[4][0] <= deltas[4])
    constraints.append(variables[4][0] >= deltas[4])
    constraints.append(variables[4][1] <= deltas[4])
    constraints.append(variables[4][1] >= deltas[4])

    # Solve


    if is_sat(And(constraints)):
        print("SAT")
        model = get_model(And(constraints))
        for i in range(6):
            if model[variables[i][0]] != model[variables[i][1]]:
                print(model)
                break
    else:
        print("UNSAT")
        break 