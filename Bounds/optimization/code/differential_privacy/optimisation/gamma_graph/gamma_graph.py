import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def solve_problem_from_graph(G, weights):
    gammas = {node: cp.Variable() for node in G.nodes}
    constraints = [
                      gammas[node] <= 1 for node in G.nodes
                  ] + [
                      gammas[node] >= -1 for node in G.nodes
                  ] + [
                      gammas[i] <= gammas[j] for i, j in G.edges
                  ]
    objective = cp.Minimize(
        sum(weights[node] * gammas[node] for node in G.nodes)
    )

    prob = cp.Problem(objective, constraints=constraints)
    prob.solve()

    print("Gamma array:")
    for node in G.nodes:
        print(f"\t{node}: {gammas[node].value}")

    print("Cost array:")
    for node in G.nodes:
        print(f"\t{node}: {weights[node] * np.round(gammas[node].value, 2)}")
    print("Total cost: ", prob.value)

    return {node: weights[node] * gammas[node].value for node in G.nodes}


def ex1():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6])
    weights = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 5000,
        5: 1,
        6: 1,
    }
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (4, 6),
    ])

    nx.draw(G, with_labels=True)
    plt.show()
    solve_problem_from_graph(G, weights)


def ex2():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    weights = [-1, 1, -2, 1]
    G.add_edges_from([
        (0, 1),
        (2, 1),
        (1, 3)
    ])
    segment_costs = solve_problem_from_graph(G, weights)
    # nx.draw(G, with_labels=True)
    # plt.show()

    sequences = [
        [0, 1, 2],
        [0, 1, 3]
    ]

    for sequence in sequences:
        print("\nSequence: ", sequence)
        print("Global cost of sequence: ", sum(segment_costs[node] for node in sequence))
        F = G.subgraph(sequence)
        solve_problem_from_graph(F, weights)


ex2()