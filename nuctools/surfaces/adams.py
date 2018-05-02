import numpy


def f(z):
    x, y = z
    r = numpy.linalg.norm(z, axis=0)
    return (2 * x ** 2 * (4. - x) + y ** 2 * (4 + y) -
            x * y * (6 - 17 * numpy.exp(-r ** 2 / 4.)))


if __name__ == '__main__':
    import scipy.misc
    from functools import partial

    p1 = (+0.0, +0.0)
    p2 = (-0.2, -2.3)
    p3 = (+2.4, +0.4)
    p4 = (+3.8, -4.4)
    print(f(p1))
    print(f(p2))
    print(f(p3))
    print(f(p4))

    print('vals:')
    p = (0.5, 0.25)
    print(f(p))

    def f_(x, y):
        return f((x, y))

    print('numerical gradient x:')
    print(scipy.misc.derivative(partial(f_, y=.25), .5, dx=1e-3, order=15))

    print('numerical gradient y:')
    print(scipy.misc.derivative(partial(f_, .5), .25, dx=1e-3, order=15))

    def dfdx_(x, y):
        return scipy.misc.derivative(partial(f_, y=y), x, dx=1e-3, order=15)

    print('numerical hessian xx:')
    print(scipy.misc.derivative(partial(f_, y=.25), .5, dx=1e-3, n=2, order=15))

    print('numerical hessian xy:')
    print(scipy.misc.derivative(partial(dfdx_, .5), .25, dx=1e-3, order=15))

    print('numerical hessian yy:')
    print(scipy.misc.derivative(partial(f_, .5), .25, dx=1e-3, n=2, order=15))
