import numpy
from itertools import starmap


A = numpy.array([-200, -100, -170, 15])
B = numpy.array([[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]])
C = numpy.array([[[-1, 0], [0, -10]], [[-1, 0], [0, -10]],
                 [[-6.5, 5.5], [5.5, -6.5]], [[0.7, 0.3], [0.3, 0.7]]])


def f(x):
    eqfs = starmap(_exponential_quadratic_function, zip(A, B, C))
    return sum(eqf(x) for eqf in eqfs)


def g(x):
    eqgs = starmap(_exponential_quadratic_gradient, zip(A, B, C))
    return sum(eqg(x) for eqg in eqgs)


def h(x):
    eqhs = starmap(_exponential_quadratic_hessian, zip(A, B, C))
    return sum(eqh(x) for eqh in eqhs)


def _exponential_quadratic_function(a, b, c):

    def _f(x):
        dx = x - b
        cdx = numpy.dot(c, dx)
        return a * numpy.exp(numpy.dot(dx, cdx))

    return _f


def _exponential_quadratic_gradient(a, b, c):

    def _f(x):
        dx = x - b
        cdx = numpy.dot(c, dx)
        return 2 * a * numpy.exp(numpy.dot(dx, cdx)) * cdx

    return _f


def _exponential_quadratic_hessian(a, b, c):

    def _f(x):
        dx = x - b
        cdx = numpy.dot(c, dx)
        return (2 * a * numpy.exp(numpy.dot(dx, cdx)) * c +
                4 * a * numpy.exp(numpy.dot(dx, cdx)) * numpy.outer(cdx, cdx))

    return _f
