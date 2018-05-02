import numpy


def f(z):
    x, y = z
    return (1. - y) * x ** 2 * numpy.exp(-x ** 2) + 1 / 2. * y ** 2


def g(z):
    gx = dfdx(z)
    gy = dfdy(z)
    return numpy.array([gx, gy])


def h(z):
    hxx = d2fdx2(z)
    hxy = d2fdxdy(z)
    hyy = d2fdy2(z)
    return numpy.array([[hxx, hxy], [hxy, hyy]])


def dfdx(z):
    x, y = z
    return 2 * (1. - y) * x * (1. - x ** 2) * numpy.exp(-x ** 2)


def dfdy(z):
    x, y = z
    return y - x ** 2 * numpy.exp(-x ** 2)


def d2fdx2(z):
    x, y = z
    return (2 * (1. - y) * (1. - 5. * x ** 2 + 2. * x ** 4) *
            numpy.exp(-x ** 2))


def d2fdxdy(z):
    x, y = z
    return -2 * x * (1. - x ** 2) * numpy.exp(-x ** 2)


def d2fdy2(z):
    x, y = z
    return 1.


if __name__ == '__main__':
    import scipy.misc
    from functools import partial

    print('vals')
    print(f((.5, .25)))
    print(g((.5, .25)))
    print(h((.5, .25)))

    def f_(x, y):
        return f((x, y))

    print('numerical gradient x:')
    print(scipy.misc.derivative(partial(f_, y=.25), .5, dx=1e-6))

    print('numerical gradient y:')
    print(scipy.misc.derivative(partial(f_, .5), .25, dx=1e-6))

    def dfdx_(x, y):
        return scipy.misc.derivative(partial(f_, y=y), x, dx=1e-4, order=15)

    print('numerical hessian xx:')
    print(scipy.misc.derivative(partial(f_, y=.25), .5, dx=1e-4, n=2, order=15))

    print('numerical hessian xy:')
    print(scipy.misc.derivative(partial(dfdx_, .5), .25, dx=1e-4, order=15))

    print('numerical hessian yy:')
    print(scipy.misc.derivative(partial(f_, .5), .25, dx=1e-4, n=2, order=15))
