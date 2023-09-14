import sympy as sp

x = sp.Symbol('x')
epsilon = sp.Symbol('epsilon')
l = sp.Symbol('l')

# y = sp.exp(-epsilon * x)
y = sp.Symbol('y')

sp.init_printing(use_unicode=True, wrap_line=False)

print(sp.integrate(
    epsilon * y / 2 * (1 - y / 2)**l
    , y))