from pysmt.shortcuts import Symbol, And, GE, LT, Int, Equals, Implies, Not, is_sat, get_model
from pysmt.typing import INT

S = [Symbol(f"S[{i}]", INT) for i in range(3)]
T = [Symbol(f"T[{i}]", INT) for i in range(3)]

S_L = Int(0)
S_N = Int(1)
S_G = Int(2)

# 0: represents the S^L strategy
# 1: represents the S^N strategy
# 2: represents the S^G strategy

domains = And([And(GE(s, S_L), LT(s, S_G)) for s in S])

"""
S[0]: represents the segment 0 -> 1
S[1]: represents the segment 0 -> 1 -> 2
S[2]: represents the segment 1 -> 1 

Adjacency of segments:
S[0] -> S[1]
S[1] -> S[1]

==============================CONSTRAINTS==============================
Constraints for valid couplings:
-----------------------------------------------------------------------
Constraint   0 : (S[0] = S^G) => (S[2] = S^G)
Constraint   1 : (S[0] = S^N) => (S[2] != S^L)
Constraint   2 : (S[2] = S^G) => (S[2] = S^G)
Constraint   3 : (S[2] = S^N) => (S[2] != S^L)
-----------------------------------------------------------------------
Constraints for finite cost:
-----------------------------------------------------------------------
Constraint   4 : S[2] = S^L
-----------------------------------------------------------------------
Constraints for bounded segments per path:
-----------------------------------------------------------------------
Constraint   5 : T[0] < T[1]
Constraint   6 : T[1] < T[1]
"""

# We now model the above constraints as a formula.

valid_coupling_constraints = And(
    Implies(Equals(S[0], S_G), Equals(S[2], S_G)),  # Constraint 0
    Implies(Equals(S[0], S_N), Not(Equals(S[2], S_L))),  # Constraint 1
    Implies(Equals(S[2], S_G), Equals(S[2], S_G)),  # Constraint 2
    Implies(Equals(S[2], S_N), Not(Equals(S[2], S_L))),  # Constraint 3
)

finite_cost_constraints = Equals(S[2], S_L)  # Constraint 4

bounded_segments_per_path_constraints = And(
    LT(T[0], T[1]),  # Constraint 5
    LT(T[1], T[1]),  # Constraint 6
)

formula = And(domains, valid_coupling_constraints, finite_cost_constraints, bounded_segments_per_path_constraints)

print(formula)
print(is_sat(formula))

model = get_model(formula)
print(model)
