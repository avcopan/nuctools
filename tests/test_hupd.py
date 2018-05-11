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
    h_psb = nuctools.hupd.psb(x, g, x0, g0, h0)
    h_ms = nuctools.hupd.ms(x, g, x0, g0, h0)
    h_bofill = nuctools.hupd.bofill(x, g, x0, g0, h0)

    dg = numpy.subtract(g, g0)

    assert numpy.allclose(numpy.dot(h_bfgs, dx), dg, rtol=1e-10, atol=1e-10)
    assert numpy.allclose(numpy.dot(h_psb, dx), dg, rtol=1e-10, atol=1e-10)
    assert numpy.allclose(numpy.dot(h_ms, dx), dg, rtol=1e-10, atol=1e-10)
    assert numpy.allclose(numpy.dot(h_bofill, dx), dg, rtol=1e-10, atol=1e-10)
