import cvxpy as cp

def print_full_opt(weights, deltas):
    gammas = [cp.Variable() for _ in range(len(weights))]

    constraints = [
        gammas[i] <= 1 for i in range(3)
    ] + [
        gammas[i] >= -1 for i in range(3)
    ] + [
        gammas[1] <= gammas[0],
        gammas[2] >= gammas[0],
    ]

    objective = cp.Minimize(
        sum(weights[i] * cp.abs(gammas[i] - deltas[i]) for i in range(3))
    )

    prob = cp.Problem(objective, constraints=constraints)
    prob.solve()

    print([gamma.value for gamma in gammas])
    print("Costs per transition are:")
    print(f"Assignment: {weights[0]} * {round(abs(gammas[0].value - deltas[0]), 2)}")
    print(f"L         : {weights[1]} * {round(abs(gammas[1].value - deltas[1]), 2)}")
    print(f"G         : {weights[2]} * {round(abs(gammas[2].value - deltas[2]), 2)}")
    print(prob.value)


def print_reduced_opt(weights, delta_0):
    gamma_0 = cp.Variable()

    constraints = [
        gamma_0 <= 1,
        gamma_0 >= -1,
    ]

    objective = cp.Minimize(
        weights[0] * cp.abs(gamma_0 - delta_0) +
        weights[1] * cp.abs(1 - gamma_0) +
        weights[2] * cp.abs(1 + gamma_0)
    )

    prob = cp.Problem(objective, constraints=constraints)
    prob.solve()

    # Print the results for the reduced problem
    print(f"\nSolution to the reduced problem: {prob.value}")
    print(f"Gamma_0: {gamma_0.value}")


weights = [5, 1, 9]
deltas = [1, 1, -1]

print_full_opt(weights, deltas)

# Better way to formulate the problem:

print_reduced_opt(weights, deltas[0])