import numpy


def linear_synchronous_transit(x0, x1, x=None):
    t = numpy.subtract(x1, x0)
    return t / numpy.linalg.norm(t)


def quadratic_synchronous_transit(x0, x1, x):
    dx0 = numpy.subtract(x0, x)
    dx1 = numpy.subtract(x1, x)
    ndx0 = numpy.linalg.norm(dx0)
    ndx1 = numpy.linalg.norm(dx1)
    t = dx1 / ndx1 ** 2 - dx0 / ndx0 ** 2
    return t / numpy.linalg.norm(t)
