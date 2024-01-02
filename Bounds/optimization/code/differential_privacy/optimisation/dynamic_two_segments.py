import cvxpy as cp
import numpy as np 
import itertools
import random


# l_trans = [random.randint(0, 2) for _ in range(2)]
# g_trans = [random.randint(0, 2) for _ in range(2)]

l_trans = [3, 0]
g_trans = [0, 0]

worst = 0
for deltas in itertools.product([-1, 1], repeat=2):
    
    print(f"Deltas: {deltas}")

    gammas = [cp.Variable() for _ in range(2)]

    constraints = [
        -1 <= gammas[i] for i in range(2)
    ] + [
        gammas[i] <= 1 for i in range(2)
    ] + [
        # gammas[0] <= gammas[1]
    ]

    objective = cp.Minimize(
        cp.abs(gammas[0] - deltas[0]) + (1 - gammas[0]) * l_trans[0] + (1 + gammas[0]) * g_trans[0] +\
        cp.abs(gammas[1] - deltas[1]) + (1 - gammas[1]) * l_trans[1] + (1 + gammas[1]) * g_trans[1]
    )

    prob = cp.Problem(objective, constraints=constraints)

    prob.solve()

    print("\tGamma arrays:")
    # Round the gamma arrays to 2 decimal places and print
    print(f"\t{[np.round(gamma.value, 2) for gamma in gammas]}")

    print(f"\t{prob.value}")
    worst = max(worst, prob.value)

print(f"L: {l_trans}")
print(f"G: {g_trans}")
print(f"Worst: {worst}")
print(f"Sky's cost: {2 + sum(l_trans) + sum(g_trans)}")