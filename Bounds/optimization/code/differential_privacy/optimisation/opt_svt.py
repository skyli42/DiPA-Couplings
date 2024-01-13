import cvxpy as cp
import itertools
import numpy as np


def compute_opt(looping_branch, d):
    n = len(looping_branch)
    gamma = cp.Variable(n)
    
    max_sol = 0
    max_delta = [0] * n
    min_gamma = [0] * n

    for delta in itertools.product([-1, 1], repeat=n):

        constraints = [gamma >= -1,
                    gamma <= 1]
        
        at = [0] * n # Compute the previous assignment transition for each transition
        for i in range(1, n):
            if looping_branch[i - 1][0]: # Previous transition is an assignment
                at[i] = i - 1
            else: # Take the at of the previous transition
                at[i] = at[i - 1]
        
        for i in range(1, n): # validity constraints
            if looping_branch[i][1] == "<": # leq guard
                constraints.append(gamma[i] <= gamma[at[i]])
            elif looping_branch[i][1] == ">=":
                constraints.append(gamma[i] >= gamma[at[i]])

        for i in range(1, n): # finite cost constraints
            if looping_branch[i][2]: # in cycle
                constraints.append(gamma[i] <= delta[i])
                constraints.append(gamma[i] >= delta[i])

        # break

        objective = cp.Minimize(sum(d[i] * cp.abs(gamma[i] - delta[i]) for i in range(n)))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)

        if prob.value > max_sol:
            max_sol = prob.value
            max_delta = delta
            min_gamma = [gamma[i].value for i in range(n)]

    return max_sol, max_delta, min_gamma


def compute_approx(looping_branch, d):
    n = len(looping_branch)
    gamma = cp.Variable(n)

    constraints = [gamma >= -1,
                gamma <= 1]
    
    at = [0] * n # Compute the previous assignment transition for each transition
    for i in range(1, n):
        if looping_branch[i - 1][0]: # Previous transition is an assignment
            at[i] = i - 1
        else: # Take the at of the previous transition
            at[i] = at[i - 1]
    
    for i in range(1, n): # validity constraints
        if looping_branch[i][1] == "<": # leq guard
            constraints.append(gamma[i] <= gamma[at[i]])
        elif looping_branch[i][1] == ">=":
            constraints.append(gamma[i] >= gamma[at[i]])

    for i in range(1, n): # finite cost constraints
        if looping_branch[i][2]: # in cycle

            chosen_delta = 1 if looping_branch[i][1] == "<" else -1
            if looping_branch[i][1] != "<" and looping_branch[i][1] != ">=":
                raise Exception("Invalid guard")

            constraints.append(gamma[i] <= chosen_delta)
            constraints.append(gamma[i] >= chosen_delta)

    non_cyclic = [i for i in range(n) if not looping_branch[i][2]]

    objective = cp.Minimize(sum(d[i] * (1 + cp.abs(gamma[i])) for i in non_cyclic))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    return prob.value, [gamma[i].value for i in range(n)]


# a transition is a tuple (True/False, "<"/">=/true", True/False), representing assignment and guard and whether it's in a cycle
# looping_branch = [    
#     (True, "true", False),
#     (False, "<", True),
#     (False, "<", True),
#     (False, ">=", False),
#     (True, "<=", False),
#     (False, ">=", True),
#     (False, "<", False),
#     (True, ">=", False),
#     (False, "<", True),
#     (False, ">=", False),
#     (True, "<=", False),
#     (False, ">=", True),
#     (False, "<", False),
# ]
looping_branch = [
    (True, "true", False),
    (False, "<", False),
    (False, ">=", False),
    (True, "<=", False),
    (False, ">=", False),
    (False, ">=", False),
    (True, "<=", False),
    (False, ">=", False),
    (False, ">=", False),
]
n = len(looping_branch)
d = np.random.rand(n)


opt, d_opt, g_opt = compute_opt(looping_branch, d)
print(f"Optimal cost: {opt}")
print(f"Delta array: {d_opt}")
print(f"g_opt: {g_opt}\n")

non_cyclic = [i for i in range(n) if not looping_branch[i][2]]
linear_term = sum(d[i] for i in non_cyclic)
print(f"Opt + linear term: {opt + linear_term}")

assignments = [i for i in range(n) if looping_branch[i][0]]
assignment_term = sum(d[i] for i in assignments)
print(f"Opt + assignment term: {opt + assignment_term}\n")

approx, g_approx = compute_approx(looping_branch, d)
print(f"Approximate cost: {approx}")

print(f"g_approx: {g_approx}\n")