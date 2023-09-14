from pysmt.shortcuts import Symbol, Int, And, Equals, is_sat
from pysmt.typing import INT
S = Symbol(f"S1", INT)
T = Symbol(f"S1", INT)

print(S == T)

f = Equals(S, Int(0))

g = Equals(T, Int(0))

print(f)

print(is_sat(And(f, g)))