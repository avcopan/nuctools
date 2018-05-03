import numpy
from typing import Iterable
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
        dx = x - _cast(b, 0, numpy.ndim(x))
        cdx = numpy.tensordot(c, dx, axes=(1, 0))
        return a * numpy.exp(numpy.sum(dx * cdx, axis=0))

    return _f


def _exponential_quadratic_gradient(a, b, c):

    def _f(x):
        dx = x - _cast(b, 0, numpy.ndim(x))
        cdx = numpy.tensordot(c, dx, axes=(1, 0))
        return 2 * a * numpy.exp(numpy.sum(dx * cdx, axis=0)) * cdx

    return _f


def _exponential_quadratic_hessian(a, b, c):

    def _f(x):
        dx = x - _cast(b, 0, numpy.ndim(x))
        cdx = numpy.tensordot(c, dx, axes=(1, 0))
        return (2 * a * numpy.exp(numpy.sum(dx * cdx, axis=0)) *
                (_cast(c, (0, 1), numpy.ndim(x)+1)
                 + 2 * _insert(cdx, 1) * _insert(cdx, 0)))

    return _f


def _cast(a, ax, ndim=None):
    ax = tuple(ax) if isinstance(ax, Iterable) else (ax,)
    assert numpy.ndim(a) == len(ax)
    ndim = max(ax) + 1 if ndim is None else ndim
    ix = (slice(None) if i in ax else None for i in range(ndim))
    at = numpy.transpose(a, numpy.argsort(ax))
    return at[tuple(ix)]


def _insert(a, ax):
    ax = tuple(ax) if isinstance(ax, Iterable) else (ax,)
    ndim = numpy.ndim(a) + 1
    ix = (None if i in ax else slice(None) for i in range(ndim))
    return a[tuple(ix)]



if __name__ == '__main__':
    import matplotlib.pyplot as pyplot

    # minima
    x1 = [-0.55822363, 1.44172584]
    x2 = [-0.05001082, 0.4666941]
    x3 = [0.6234994, 0.02803776]
    # saddle points
    x12 = [-0.82200156, 0.6243128]
    x23 = [0.21248658, 0.29298833]


    X = numpy.linspace(-1.25, 0.75)
    Y = numpy.linspace(-0.25, 2.0)
    Z = f(numpy.array(numpy.meshgrid(X, Y)))

    pyplot.contour(X, Y, Z, 100)
    pyplot.scatter(*zip(x1, x12, x2, x23, x3))
    pyplot.show()
