import nuctools

from numpy.testing import assert_almost_equal

P = (0.5, 0.25)

F = 0.1772751468258884
G = [0.4380754404776652, 0.05529980423214878]
H = [[-0.1460251468258884, -0.5841005873035536], [-0.5841005873035536, 1.]]


def test__f():
    f = nuctools.surfaces.cerjanmiller.f(P)
    assert_almost_equal(f, F, decimal=13)


def test__g():
    g = nuctools.surfaces.cerjanmiller.g(P)
    assert_almost_equal(g, G, decimal=13)


def test__h():
    h = nuctools.surfaces.cerjanmiller.h(P)
    assert_almost_equal(h, H, decimal=13)
