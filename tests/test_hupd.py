import numpy
import pytest

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

import nuctools


@given(arrays('float64', shape=(2,), elements=floats(-1., 1.5)),
       arrays('float64', shape=(2,), elements=floats(-0.3, 0.3)))
def test__quasi_newton_condition(x0, dx):
    g0 = nuctools.surf.mullbrow.g(x0)
    h0 = nuctools.surf.mullbrow.h(x0)

    x = numpy.add(x0, dx)
    g = nuctools.surf.mullbrow.g(x)

    h_bfgs = nuctools.hupd.bfgs(x, g, x0, g0, h0)

    dg = numpy.subtract(g, g0)

    assert numpy.allclose(numpy.dot(h_bfgs, dx), dg, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    x0 = numpy.array([4., 4.])
    dx = numpy.array([2.22044605e-15, 2.22044605e-15])
    g0 = nuctools.surf.mullbrow.g(x0)
    h0 = nuctools.surf.mullbrow.h(x0)

    x = numpy.add(x0, dx)
    g = nuctools.surf.mullbrow.g(x)

    h_bfgs = nuctools.hupd.bfgs(x, g, x0, g0, h0)

    dg = numpy.subtract(g, g0)

    print(dx)
    print(numpy.dot(h_bfgs, dx))
    print(dg)

    assert numpy.allclose(numpy.dot(h_bfgs, dx), dg, rtol=1e-10, atol=1e-10)
