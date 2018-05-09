import numpy


# do-nothing template for quadratic information at a point
def quad(x, x0=None, f0=None, g0=None, h0=None):
    f = None
    g = None
    h = None
    return f, g, h


def bfgs(x, g, x0=None, g0=None, h0=None):
    dx = numpy.subtract(x, x0)
    dg = numpy.subtract(g, g0)
    h0dx = numpy.dot(h0, dx)
    h = (h0 - numpy.outer(h0dx, h0dx) / numpy.dot(dx, h0dx)
         + numpy.outer(dg, dg) / numpy.dot(dg, dx))
    return h


if __name__ == '__main__':
    import itertools
    from nuctools.surfaces.mullerbrown import f, g, h

    h0 = [[1000., 0.], [0., 1000.]]
    x0 = [-0.558, +1.442]
    g0 = g(x0)

    dx = [0., 0.001]
    x1 = numpy.add(x0, dx)
    g1 = g(x1)
    h1_approx = bfgs(x=x1, g=g1, x0=x0, g0=g0, h0=h0)
    g0 = g1
    x0 = x1
    h0 = h1_approx

    print('compare:')
    print(h1_approx)
    print(h(x1))

    dx = [0.001, 0.]
    x1 = numpy.add(x0, dx)
    g1 = g(x1)
    h1_approx = bfgs(x=x1, g=g1, x0=x0, g0=g0, h0=h0)
    g0 = g1
    x0 = x1
    h0 = h1_approx

    print('compare:')
    print(h1_approx)
    print(h(x1))

    dx = [0., -0.001]
    x1 = numpy.add(x0, dx)
    g1 = g(x1)
    h1_approx = bfgs(x=x1, g=g1, x0=x0, g0=g0, h0=h0)
    g0 = g1
    x0 = x1
    h0 = h1_approx

    print('compare:')
    print(h1_approx)
    print(h(x1))

    dx = [-0.001, 0.]
    x1 = numpy.add(x0, dx)
    g1 = g(x1)
    h1_approx = bfgs(x=x1, g=g1, x0=x0, g0=g0, h0=h0)
    g0 = g1
    x0 = x1
    h0 = h1_approx

    print('compare:')
    print(h1_approx)
    print(h(x1))
