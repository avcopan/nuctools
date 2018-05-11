import numpy


def bfgs(x, g, x0, g0, h0):
    dx = numpy.subtract(x, x0)
    dg = numpy.subtract(g, g0)

    if numpy.linalg.norm(dx) < numpy.finfo(float).eps:
        return h0
    else:
        h0dx = numpy.dot(h0, dx)
        h = (h0 + numpy.outer(dg, dg) / numpy.dot(dg, dx) -
             numpy.outer(h0dx, h0dx) / numpy.dot(dx, h0dx))
        return h


# Hratchian JCTC (2005) p. 61
def psb(x, g, x0, g0, h0):
    dx = numpy.subtract(x, x0)
    dg = numpy.subtract(g, g0)

    if numpy.linalg.norm(dx) < numpy.finfo(float).eps:
        return h0
    else:
        h0dx = numpy.dot(h0, dx)
        ddg = dg - h0dx
        h = (h0 + numpy.outer(ddg, dx) / numpy.dot(dx, dx) +
             numpy.outer(dx, ddg) / numpy.dot(dx, dx) -
             numpy.dot(dx, ddg) * numpy.outer(dx, dx) / numpy.dot(dx, dx) ** 2)
        return h


# Hratchian JCTC (2005) p. 61
def ms(x, g, x0, g0, h0):
    dx = numpy.subtract(x, x0)
    dg = numpy.subtract(g, g0)

    if numpy.linalg.norm(dx) < numpy.finfo(float).eps:
        return h0
    else:
        h0dx = numpy.dot(h0, dx)
        ddg = dg - h0dx
        h = h0 + numpy.outer(ddg, ddg) / numpy.dot(ddg, dx)
        return h


# Hratchian JCTC (2005) p. 61
def bofill(x, g, x0, g0, h0):
    dx = numpy.subtract(x, x0)
    dg = numpy.subtract(g, g0)

    if numpy.linalg.norm(dx) < numpy.finfo(float).eps:
        return h0
    else:
        h0dx = numpy.dot(h0, dx)
        ddg = dg - h0dx
        h_psb = psb(x=x, g=g, x0=x0, g0=g0, h0=h0)
        h_ms = ms(x=x, g=g, x0=x0, g0=g0, h0=h0)
        phi = numpy.dot(dx, ddg) ** 2 / numpy.dot(dx, dx) / numpy.dot(ddg, ddg)
        return phi * h_ms + (1. - phi) * h_psb
