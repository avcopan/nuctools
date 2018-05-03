from nuctools.opt import stationary
from nuctools.surfaces.mullerbrown import f, g, h

# minima
p1 = (-0.5, +1.4)
p2 = (+0.6, +0.0)
p3 = (-0.1, +0.5)
# transition states
p4 = (-0.8, 0.6)
p5 = (+0.2, 0.3)

# x1 = nuctools.steepest_descent(f=f, x0=p1, g=g, s0=1e-3)
# x2 = nuctools.steepest_descent(f=f, x0=p2, g=g, s0=1e-3)
# x3 = nuctools.steepest_descent(f=f, x0=p3, g=g, s0=1e-3)
x1 = stationary.newton_raphson(f=f, x0=p1, g=g, h=h)
print(x1)
x2 = stationary.newton_raphson(f=f, x0=p2, g=g, h=h)
print(x2)
x3 = stationary.newton_raphson(f=f, x0=p3, g=g, h=h)
print(x3)
x4 = stationary.newton_raphson(f=f, x0=p4, g=g, h=h)
print(x4)
x5 = stationary.newton_raphson(f=f, x0=p5, g=g, h=h)
print(x5)
