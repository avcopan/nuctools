import nuctools

from numpy.testing import assert_almost_equal

P = (0.5, 0.25)

F = 0.2137814335323605
G = [0.5475943005970816, 0.15264990211607438]
H = [[-0.1825314335323605, -0.2920502936517768],
     [-0.2920502936517768, +0.6105996084642975]]


def test__f():
    f = nuctools.surf.cerjmill.f(P)
    assert_almost_equal(f, F, decimal=13)


def test__g():
    g = nuctools.surf.cerjmill.g(P)
    assert_almost_equal(g, G, decimal=13)


def test__h():
    h = nuctools.surf.cerjmill.h(P)
    assert_almost_equal(h, H, decimal=13)
