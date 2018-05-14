import nuctools

import numpy
import pytest


def lin(x):
    f = nuctools.surf.mullbrow.f(x)
    g = nuctools.surf.mullbrow.g(x)
    return f, g


def test__steepest_descent():
    x, info = nuctools.algo.steepest_descent(
        lin, (-0.5, 1.4), 1e-5, 0.3, 1e-6, 50, True)
    assert numpy.allclose(x, [-0.55822363, 1.44172584])
    assert info['niter'] <= 12
    x, info = nuctools.algo.steepest_descent(
        lin, (+0.6, 0.0), 1e-5, 0.3, 1e-6, 50, True)
    assert numpy.allclose(x, [0.62349941, 0.02803776])
    assert info['niter'] <= 11
    x, info = nuctools.algo.steepest_descent(
        lin, (-0.1, 0.5), 1e-5, 0.3, 1e-6, 50, True)
    assert numpy.allclose(x, [-0.05001082,  0.4666941])
    assert info['niter'] <= 13

    with pytest.warns(UserWarning):
        nuctools.algo.steepest_descent(lin, (0., 0.), 1., 1., maxiter=1)
