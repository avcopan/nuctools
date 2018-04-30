import nuctools
from nuctools.muller import f, g, h

p1 = (-0.5, +1.4)
p2 = (+0.6, +0.0)
p3 = (-0.1, +0.5)

# x1 = nuctools.steepest_descent(f=f, x0=p1, g=g, s0=1e-3)
# x2 = nuctools.steepest_descent(f=f, x0=p2, g=g, s0=1e-3)
# x3 = nuctools.steepest_descent(f=f, x0=p3, g=g, s0=1e-3)
x1 = nuctools.newton_raphson(f=f, x0=p1, g=g, h=h)
x2 = nuctools.newton_raphson(f=f, x0=p2, g=g, h=h)
x3 = nuctools.newton_raphson(f=f, x0=p3, g=g, h=h)

print(x1)
print(x2)
print(x3)
