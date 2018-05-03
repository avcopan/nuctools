import numpy
from nuctools.opt import saddle
from nuctools.surfaces.mullerbrown import f, g, h

# minima
p1 = numpy.array([-0.5, +1.4])
p2 = numpy.array([-0.1, +0.5])
p3 = numpy.array([+0.6, +0.0])
# transition states
t12 = numpy.array([-0.8, 0.6])
t23 = numpy.array([+0.2, 0.3])

x1, _ = saddle.rational_function_optimization(
        f=f, x0=p1, g=g, h=h, smax=0.1, order=0)
print(x1)
x2, _ = saddle.rational_function_optimization(
        f=f, x0=p2, g=g, h=h, smax=0.1, order=0)
print(x2)
x3, _ = saddle.rational_function_optimization(
        f=f, x0=p3, g=g, h=h, smax=0.1, order=0)
print(x3)

# guesses
s = 0.2
p12 = (1.-s)*p1 + s*t12
p21 = (1.-s)*p2 + s*t12
p23 = (1.-s)*p2 + s*t23
p32 = (1.-s)*p3 + s*t23

x12, traj12 = saddle.rational_function_optimization(f=f, x0=p12, g=g, h=h, smax=0.1, order=1)
print(x12)
x21, traj21 = saddle.rational_function_optimization(f=f, x0=p21, g=g, h=h, smax=0.1, order=1)
print(x21)
x23, traj23 = saddle.rational_function_optimization(f=f, x0=p23, g=g, h=h, smax=0.1, order=1)
print(x23)
x32, traj32 = saddle.rational_function_optimization(f=f, x0=p32, g=g, h=h, smax=0.1, order=1)
print(x32)


import matplotlib.pyplot as pyplot

X = numpy.linspace(-1.25, 0.75)
Y = numpy.linspace(-0.25, 2.0)
GRD = numpy.array(numpy.meshgrid(X, Y))
print(GRD.shape)
Z = f(GRD)
print(Z)

pyplot.contour(X, Y, Z, 100)
pyplot.plot(*zip(*traj12))
pyplot.plot(*zip(*traj21))
pyplot.plot(*zip(*traj23))
pyplot.plot(*zip(*traj32))
pyplot.scatter(*zip(x1, x12, x2, x23, x3))
pyplot.show()
