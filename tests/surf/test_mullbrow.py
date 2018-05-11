import nuctools

from numpy.testing import assert_almost_equal

P1 = (-0.558, +1.442)
P2 = (+0.623, +0.028)
P3 = (-0.050, +0.467)

F1 = -146.69948920058778
F2 = -108.16665005353303
F3 = -80.76774924875772

G1 = [-1.87353828849268e-04,  2.04493894713679e-01]
G2 = [-0.28214371532097, -0.1904339081248]
G3 = [0.04894392796121, 0.44871341799095]

H1 = [[2241.281310559507, -1828.889350339865],
      [-1828.889350339865,  2237.838155011883]]
H2 = [[552.8729185542339,  154.84146807354927],
      [154.84146807354927, 2994.2371962907396]]
H3 = [[239.68935029003293,  151.39774283167137],
      [151.39774283167137, 1462.4043557569194]]


def test__f():
    f1 = nuctools.surf.mullbrow.f(P1)
    f2 = nuctools.surf.mullbrow.f(P2)
    f3 = nuctools.surf.mullbrow.f(P3)
    assert_almost_equal(f1, F1, decimal=13)
    assert_almost_equal(f2, F2, decimal=13)
    assert_almost_equal(f3, F3, decimal=13)


def test__g():
    g1 = nuctools.surf.mullbrow.g(P1)
    g2 = nuctools.surf.mullbrow.g(P2)
    g3 = nuctools.surf.mullbrow.g(P3)
    assert_almost_equal(g1, G1, decimal=13)
    assert_almost_equal(g2, G2, decimal=13)
    assert_almost_equal(g3, G3, decimal=13)


def test__h():
    h1 = nuctools.surf.mullbrow.h(P1)
    h2 = nuctools.surf.mullbrow.h(P2)
    h3 = nuctools.surf.mullbrow.h(P3)
    assert_almost_equal(h1, H1, decimal=13)
    assert_almost_equal(h2, H2, decimal=13)
    assert_almost_equal(h3, H3, decimal=13)
