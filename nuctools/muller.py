import numpy
from itertools import starmap


A = numpy.array([-200, -100, -170, 15])
B = numpy.array([[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]])
C = numpy.array([[[-1, 0], [0, -10]], [[-1, 0], [0, -10]],
                 [[-6.5, 5.5], [5.5, -6.5]], [[0.7, 0.3], [0.3, 0.7]]])


def e(x):
    fs = starmap(_exponential_quadratic_function, zip(A, B, C))
    return sum(f(x) for f in fs)


def g(x):
    fs = starmap(_exponential_quadratic_gradient, zip(A, B, C))
    return sum(f(x) for f in fs)


def h(x):
    fs = starmap(_exponential_quadratic_hessian, zip(A, B, C))
    return sum(f(x) for f in fs)


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
