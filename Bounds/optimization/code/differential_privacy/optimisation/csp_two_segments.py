import cvxpy as cp

weights_1 = [1, 1, 5000]
weights_2 = [1, 3, 1]
gammas_1 = [cp.Variable() for _ in range(3)]
gammas_2 = [cp.Variable() for _ in range(3)]
deltas_1 = [1, 1, -1]
deltas_2 = [-1, 1, -1]

constraints = [
    gammas_1[i] <= 1 for i in range(3)
] + [
    gammas_1[i] >= -1 for i in range(3)
] + [
    gammas_1[1] <= gammas_1[0],
    gammas_1[2] >= gammas_1[0],
] + [
    gammas_2[i] <= 1 for i in range(3)
] + [
    gammas_2[i] >= -1 for i in range(3)
] + [
    gammas_2[1] <= gammas_2[0],
    gammas_2[2] >= gammas_2[0],
] + [
    # gammas_1[0] >= gammas_2[0],
]

objective = cp.Minimize(
    sum(weights_1[i] * cp.abs(gammas_1[i] - deltas_1[i]) for i in range(3)) +
    sum(weights_2[i] * cp.abs(gammas_2[i] - deltas_2[i]) for i in range(3))
)

prob = cp.Problem(objective, constraints=constraints)
prob.solve()

print("Gamma arrays:")
print([gamma.value for gamma in gammas_1])
print([gamma.value for gamma in gammas_2])

print("Cost arrays:")
print([weight * round(abs(gamma.value - delta), 2) for gamma, delta, weight in zip(gammas_1, deltas_1, weights_1)])
print([weight * round(abs(gamma.value - delta), 2) for gamma, delta, weight in zip(gammas_2, deltas_2, weights_2)])

print(prob.value)