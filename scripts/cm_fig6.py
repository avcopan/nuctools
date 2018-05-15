import numpy
from nuctools.opt import saddle
from nuctools.surf.cerjmill import f, g, h

import matplotlib.pyplot as pyplot

p = (0.1, 0.2)

print(f((1, 0)))
print(g((1, 0)))
print(h((1, 0)))

x, traj = saddle.rational_function_optimization(
    f=f, x0=p, g=g, h=h, smax=1., order=1)
print(x)

X = numpy.linspace(-0.1, 2.1)
Y = numpy.linspace(-1, 1)
Z = f(numpy.array(numpy.meshgrid(X, Y)))
print(Z)

pyplot.contour(X, Y, Z, 20)
pyplot.plot(*zip(*traj))
pyplot.scatter(*zip(*traj))
pyplot.show()
