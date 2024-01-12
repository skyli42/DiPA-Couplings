import cvxpy as cp
import itertools
import numpy as np 

gamma = cp.Variable(3)
max_val = 0

for delta in itertools.product([-1, 1], repeat=3):
    print(delta)

    delta = np.array(delta)
    
    constraints = [gamma >= -1,
                gamma <= 1,
                gamma[0] <= gamma[1],
                gamma[2] <= gamma[0],
                gamma[1] >= -1,
                gamma[1] <= -1
    ]
    objective = cp.Minimize(2 * cp.abs(gamma[0] - delta[0]) + 2 * cp.abs(gamma[1] - delta[1]) + cp.abs(gamma[2] - delta[2]))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    print(gamma.value)
    print(prob.value)

    if prob.value > max_val:
        max_val = prob.value

print(f"Approx value: {max_val}")