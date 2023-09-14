from pysmt.shortcuts import Symbol, And, Not, is_sat

varA = Symbol("A")
varB = Symbol("B")

f = And(varA, Not(varB))

print(f)

print(is_sat(f))  # should be satisfiable

g = f.substitute({varB: varA})

print(g)

print(is_sat(g))  # should not be satisfiable
