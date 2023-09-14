import cvxpy as cp
import numpy as np
import itertools
from matplotlib import pyplot as plt

# Global variables
leq = lambda x, y: x <= y
geq = lambda x, y: x >= y


def name_of_inequality(inequality):
    if inequality == leq:
        return "<"
    elif inequality == geq:
        return ">"
    else:
        raise ValueError("Unknown inequality")


def print_gamma_class(gammas, inequalities):
    n = len(gammas)
    for i in range(n):
        if gammas[i] < 0:
            print(f"\033[91m{gammas[i]}\033[0m", end=" ")  # Print negative values in red
        else:
            print(f"\033[94m{gammas[i]}\033[0m", end=" ")  # Print positive values in blue

        if i < n - 1:
            print(f"{name_of_inequality(inequalities[i])} ", end="")
    print()


def evaluate(constants, gammas, deltas):
    return sum((constants[i] + deltas[i]) * gammas[i] for i in range(len(gammas)))


def satisfies_constraints(gammas, inequalities):
    for i in range(1, len(gammas)):
        if not inequalities[i - 1](gammas[i - 1], gammas[i]):
            return False
    return True


def determine_best_gamma_class(constants, inequalities, deltas):
    n = len(constants)
    gammas = [cp.Variable() for _ in range(n)]
    constraints = [
                      gammas[i] <= 1 for i in range(n)
                  ] + [
                      gammas[i] >= -1 for i in range(n)
                  ] + [
                      gammas[i - 1] <= gammas[i] if inequalities[i - 1] == leq else gammas[i - 1] >= gammas[i]
                      for i in range(1, n)
                  ]
    objective = cp.Minimize(
        sum((constants[i] + deltas[i]) * gammas[i] for i in range(n))
    )

    prob = cp.Problem(objective, constraints=constraints)
    prob.solve(solver='ECOS')

    gamma_values = [gamma.value for gamma in gammas]
    new_gamma_values = [0 for _ in range(n)]
    for i in range(n):
        if not (np.isclose(gamma_values[i], 1) or np.isclose(gamma_values[i], -1)):
            new_gamma_values[i] = -1
        else:
            new_gamma_values[i] = int(np.round(gamma_values[i]))

        if new_gamma_values[i] == 0:
            pass

    assert satisfies_constraints(new_gamma_values, inequalities)
    assert np.isclose(evaluate(constants, new_gamma_values, deltas), prob.value)

    return tuple(new_gamma_values)


def determine_worst_deltas_gammas(n, constants, inequalities, interesting_classes=[], step_size=1):
    gamma_classes = dict()

    progress = 0
    worst_deltas = None
    worst_gamma_class = None

    for deltas in itertools.product(np.arange(-1, 1, step_size), repeat=n):

        gamma_class = determine_best_gamma_class(constants, inequalities, deltas)

        if gamma_class not in gamma_classes:
            gamma_classes[gamma_class] = [deltas]
        else:
            gamma_classes[gamma_class].append(deltas)

        val = evaluate(constants, gamma_class, deltas)

        if worst_deltas is None or val >= worst_deltas[1]:
            worst_deltas = (deltas, val)
            worst_gamma_class = gamma_class

        progress += 1
        # if progress % 100 == 0:
        #     print(f"Progress: {progress} out of {2 ** n}. Worst deltas: {worst_deltas[0]} with cost {worst_deltas[1]}")

    # Check if -1*n or 1*n have maximizers
    for gamma_class in interesting_classes:
        if gamma_class in gamma_classes:
            for deltas in gamma_classes[gamma_class]:
                val = evaluate(constants, gamma_class, deltas)
                if np.isclose(val, worst_deltas[1]):
                    worst_deltas = (deltas, val)
                    worst_gamma_class = gamma_class

    # print(f"\nNumber of gamma classes: {len(gamma_classes)}")
    # for gamma_class, deltas in gamma_classes.items():
    #     print(f"Gamma class {gamma_class} has {len(deltas)} elements: {deltas}")

    # print(f"\nConstants: {constants}")
    # print(f"Inequalities: {[name_of_inequality(inequality) for inequality in inequalities]}")
    # print(f"Worst deltas: {worst_deltas[0]} with cost {worst_deltas[1]}")
    # print(f"Gamma class for worst deltas:")
    #
    # print_gamma_class(worst_gamma_class, inequalities)

    return worst_deltas[0], worst_gamma_class


def compute_best_bound(n, l: np.array, g: np.array, inequalities):

    constants = g - l
    additivity = np.sum(g + l + 1)

    deltas, gammas = determine_worst_deltas_gammas(n, constants, inequalities, step_size=1)

    print("Solution:")
    print(deltas, gammas)

    return evaluate(constants, gammas, deltas) + additivity



def sample_constants(n):
    return np.random.randint(-n, n, n)


def enumerate_strange_examples():
    n = 8
    num_examples = 10
    inequalities = generate_inequalities(n)

    interesting_classes = [
                              (-1,) * k + (1,) * (n - k) for k in range(0, n)
                          ] + [
                              (1,) * k + (-1,) * (n - k) for k in range(0, n)
                          ]

    for i in range(num_examples):
        # keep repeating until worst_gamma_class is non-monotone
        while True:
            constants = sample_constants(n)
            deltas, gammas = determine_worst_deltas_gammas(n, constants, inequalities, interesting_classes,
                                                           step_size=0.5)

            if gammas not in interesting_classes:
                break

        print()
        print(f"Constants: {constants}")
        print(f"Worst deltas: {deltas}")
        print(f"Worst gammas:")
        print_gamma_class(gammas, inequalities)


def generate_inequalities(n):
    return [leq if i % 2 == 0 else geq for i in range(n - 1)]


def enumerate_examples(n: int):
    inequalities = [leq if i % 2 == 0 else geq for i in range(n - 1)]

    while True:
        constants = sample_constants(n)
        deltas, gammas = determine_worst_deltas_gammas(n, constants, inequalities, step_size=0.5)
        val1 = evaluate(constants, gammas, deltas)

        deltas_2, gammas_2 = determine_worst_deltas_gammas(n, constants, inequalities, step_size=1)
        val2 = evaluate(constants, gammas_2, deltas_2)

        print(f"Constants: {constants}")
        print(f"Worst deltas: {deltas}")
        print(f"Worst gammas:")
        print_gamma_class(gammas, inequalities)
        print(f"Value: {val1} vs {val2}")


# n = 4
# constants = [0, -8, 15, -12, 11, 17]
# inequalities = [leq, geq, leq, geq, leq]
#
# interesting_classes = [
#                               (-1,) * k + (1,) * (n - k) for k in range(0, n)
#                           ] + [
#                               (1,) * k + (-1,) * (n - k) for k in range(0, n)
#                           ]
#
# deltas, gammas = determine_worst_deltas_gammas(n, constants, inequalities, interesting_classes)
#
# print()
# print(f"Constants: {constants}")
# print(f"Worst deltas: {deltas}")
# print(f"Worst gammas:")
# print_gamma_class(gammas, inequalities)

#
# n = 5
# constants = [-3, 1, 0, -5, 4]
# inequalities = generate_inequalities(n)
# for deltas in itertools.product([-1, 1], repeat=n):
#     gamma_class = determine_best_gamma_class(constants, inequalities, deltas)
#     if gamma_class == (1, 1, 1, 1, -1):
#         print(deltas)


def flipped_problem(n, inequalities, l, g):
    gammas = cp.Variable(n)
    constraints = [gammas >= -1, gammas <= 1] + [
        gammas[i - 1] <= gammas[i] if inequalities[i - 1] == leq else gammas[i - 1] >= gammas[i]
        for i in range(1, n)
    ]

    g = np.array(g)
    l = np.array(l)

    a = np.ones(n)
    b = g - l
    c = g + l + 1

    objective = cp.Minimize(
        sum([cp.abs(gammas[i]) * a[i] for i in range(n)]) +
        sum([gammas[i] * b[i] for i in range(n)]) +
        sum([c[i] for i in range(n)])
    )

    problem = cp.Problem(objective, constraints)

    problem.solve()

    return gammas.value, problem.value


# n = 1
# data = []
#
# for i in range(1):
#     l = np.array([2])
#     g = np.array([1])
#     # print(l)
#     # print(g)
#
#     inequalities = generate_inequalities(n)
#     gammas, val_1 = flipped_problem(n, inequalities, l, g)
#     print(val_1)
#
#     print(gammas)
#
#     val_2 = compute_best_bound(n, l, g, inequalities)
#     print(val_2)
#     # print(val_1)
#     # print(val_2)
#     data.append(np.round(val_1 - val_2, 5))
#
#     if val_1 != val_2:
#         print(l)
#         print(g)
#         break
#
#     # if i + 1 % 10 == 0:
#     #     print(f"Done {i + 1} iterations")
#     #     print(f"Min: {np.mean(data)}")
#     #     print(f"Max: {np.max(data)}")
#
# plt.hist(data)
# plt.show()


n = 3
l = np.array([1, 5, 7])
g = np.array([2, 2, 2])
inequalities = generate_inequalities(n)

constants = g - l
additivity = sum(g + l + 1)

deltas, gammas = determine_worst_deltas_gammas(n, constants, inequalities, step_size=1)

print(deltas)
print(gammas)

print(evaluate(constants, gammas, deltas) + additivity)