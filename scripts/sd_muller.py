import numpy
import nuctools
import scipy.misc
from itertools import starmap

from numpy.testing import assert_almost_equal


def central_difference(f, x, step=0.01, nder=1, npts=None):
    if npts is None:
        npts = nder + 1 + nder % 2
    if numpy.ndim(step) == 0:
        step = float(step) * numpy.ones_like(x)

    weights = scipy.misc.central_diff_weights(Np=npts, ndiv=nder)

    def derivative(index):
        dx = numpy.zeros_like(x)
        dx[index] = step[index]
        grid = [numpy.array(x) + (k - npts//2) * dx for k in range(npts)]
        vals = map(f, grid)
        return (sum(numpy.multiply(w, v) for w, v in zip(weights, vals))
                / (step[index] ** nder))

    der = tuple(map(derivative, numpy.ndindex(numpy.shape(x))))
    shape = numpy.shape(x) + numpy.shape(f(x))
    return numpy.reshape(der, shape)


def exponential_quadratic(a, b, c):

    def _f(x):
        dx = x - b
        cdx = numpy.dot(c, dx)
        return a * numpy.exp(numpy.dot(dx, cdx))

    return _f


def exponential_quadratic_grad(a, b, c):

    def _f(x):
        dx = x - b
        cdx = numpy.dot(c, dx)
        return 2 * a * numpy.exp(numpy.dot(dx, cdx)) * cdx

    return _f


def exponential_quadratic_hess(a, b, c):

    def _f(x):
        dx = x - b
        cdx = numpy.dot(c, dx)
        return (2 * a * numpy.exp(numpy.dot(dx, cdx)) * c +
                4 * a * numpy.exp(numpy.dot(dx, cdx)) * numpy.outer(cdx, cdx))

    return _f


def muller_function(x):
    a = numpy.array([-200, -100, -170, 15])
    b = numpy.array([[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]])
    c = numpy.array([[[-1, 0], [0, -10]], [[-1, 0], [0, -10]],
                     [[-6.5, 5.5], [5.5, -6.5]], [[0.7, 0.3], [0.3, 0.7]]])
    fs = starmap(exponential_quadratic, zip(a, b, c))
    return sum(f(x) for f in fs)


def muller_function_grad(x):
    a = numpy.array([-200, -100, -170, 15])
    b = numpy.array([[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]])
    c = numpy.array([[[-1, 0], [0, -10]], [[-1, 0], [0, -10]],
                     [[-6.5, 5.5], [5.5, -6.5]], [[0.7, 0.3], [0.3, 0.7]]])
    fs = starmap(exponential_quadratic_grad, zip(a, b, c))
    return sum(f(x) for f in fs)


def muller_function_hess(x):
    a = numpy.array([-200, -100, -170, 15])
    b = numpy.array([[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]])
    c = numpy.array([[[-1, 0], [0, -10]], [[-1, 0], [0, -10]],
                     [[-6.5, 5.5], [5.5, -6.5]], [[0.7, 0.3], [0.3, 0.7]]])
    fs = starmap(exponential_quadratic_hess, zip(a, b, c))
    return sum(f(x) for f in fs)


# check minima:
m1 = muller_function((-0.558, +1.442))
m2 = muller_function((+0.623, +0.028))
m3 = muller_function((-0.050, +0.467))

assert_almost_equal(m1, -146.69948920058778, decimal=13)
assert_almost_equal(m2, -108.16665005353303, decimal=13)
assert_almost_equal(m3, -80.767749248757720, decimal=13)

print(m1)
print(m2)
print(m3)

g1 = muller_function_grad((-0.558, +1.442))
g2 = muller_function_grad((+0.623, +0.028))
g3 = muller_function_grad((-0.050, +0.467))

print(g1)
print(g2)
print(g3)

dm1 = central_difference(muller_function, (-0.558, +1.442), step=1e-5, npts=9)
dm2 = central_difference(muller_function, (+0.623, +0.028), step=1e-5, npts=9)
dm3 = central_difference(muller_function, (-0.050, +0.467), step=1e-5, npts=9)

print(dm1)
print(dm2)
print(dm3)

h1 = muller_function_hess((-0.558, +1.442))
h2 = muller_function_hess((+0.623, +0.028))
h3 = muller_function_hess((-0.050, +0.467))

print(h1)
print(h2)
print(h3)

d2m1 = central_difference(muller_function, (-0.558, +1.442), step=1e-5, npts=9,
                          nder=2)
d2m2 = central_difference(muller_function, (+0.623, +0.028), step=1e-5, npts=9,
                          nder=2)
d2m3 = central_difference(muller_function, (-0.050, +0.467), step=1e-5, npts=9,
                          nder=2)

print(d2m1)
print(d2m2)
print(d2m3)

e1 = nuctools.muller.e((-0.558, +1.442))
e2 = nuctools.muller.e((+0.623, +0.028))
e3 = nuctools.muller.e((-0.050, +0.467))
g1 = nuctools.muller.g((-0.558, +1.442))
g2 = nuctools.muller.g((+0.623, +0.028))
g3 = nuctools.muller.g((-0.050, +0.467))
h1 = nuctools.muller.h((-0.558, +1.442))
h2 = nuctools.muller.h((+0.623, +0.028))
h3 = nuctools.muller.h((-0.050, +0.467))
numpy.set_printoptions(precision=14)
print(repr(e1))
print(repr(e2))
print(repr(e3))
print(repr(g1))
print(repr(g2))
print(repr(g3))
print(repr(h1))
print(repr(h2))
print(repr(h3))
