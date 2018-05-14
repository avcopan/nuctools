import nuctools

import numpy
import pytest


def lin(x):
    f = nuctools.surf.mullbrow.f(x)
    g = nuctools.surf.mullbrow.g(x)
    return f, g


def quad(x, x0=None, g0=None, h0=None):
    f = nuctools.surf.mullbrow.f(x)
    g = nuctools.surf.mullbrow.g(x)
    h = nuctools.surf.mullbrow.h(x)
    return f, g, h


def quad_bfgs(x, x0=None, g0=None, h0=None):
    f = nuctools.surf.mullbrow.f(x)
    g = nuctools.surf.mullbrow.g(x)
    if any(thing is None for thing in (x0, g0, h0)):
        h = nuctools.surf.mullbrow.h(x)
    else:
        h = nuctools.hupd.bfgs(x, g, x0, g0, h0)
    return f, g, h


def quad_bofill(x, x0=None, g0=None, h0=None):
    f = nuctools.surf.mullbrow.f(x)
    g = nuctools.surf.mullbrow.g(x)
    if any(thing is None for thing in (x0, g0, h0)):
        h = nuctools.surf.mullbrow.h(x)
    else:
        h = nuctools.hupd.bofill(x, g, x0, g0, h0)
    return f, g, h


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


def test__newton_raphson():
    print('minima:')

    x0 = (-0.5, 1.4)
    x, info = nuctools.algo.newton_raphson(
        quad, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [-0.55822363, 1.44172584])
    assert info['niter'] <= 6

    x, info = nuctools.algo.newton_raphson(
        quad_bfgs, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [-0.55822363, 1.44172584])
    assert info['niter'] <= 7

    x0 = (+0.6, 0.0)
    x, info = nuctools.algo.newton_raphson(
        quad, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [0.62349941, 0.02803776])
    assert info['niter'] <= 5

    x, info = nuctools.algo.newton_raphson(
        quad_bfgs, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [0.62349941, 0.02803776])
    assert info['niter'] <= 7

    x0 = (-0.1, 0.5)
    x, info = nuctools.algo.newton_raphson(
        quad, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [-0.05001082, 0.46669411])
    assert info['niter'] <= 5

    x, info = nuctools.algo.newton_raphson(
        quad_bfgs, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [-0.05001082, 0.46669411])
    assert info['niter'] <= 7

    print('transition states:')

    x0 = (-0.8, 0.6)
    x, info = nuctools.algo.newton_raphson(
        quad, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [-0.82200156, 0.6243128])
    assert info['niter'] <= 5

    x, info = nuctools.algo.newton_raphson(
        quad_bofill, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [-0.82200156, 0.6243128])
    assert info['niter'] <= 8

    x0 = (+0.2, 0.3)
    x, info = nuctools.algo.newton_raphson(
        quad, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [0.21248658, 0.29298833])
    assert info['niter'] <= 5

    x, info = nuctools.algo.newton_raphson(
        quad_bofill, x0, 0.3, 1e-6, 50, True)
    print(x)
    assert numpy.allclose(x, [0.21248658, 0.29298833])
    assert info['niter'] <= 6

    with pytest.warns(UserWarning):
        nuctools.algo.newton_raphson(quad_bfgs, (0., 0.), 1., 1., maxiter=1)


if __name__ == '__main__':
    test__newton_raphson()
