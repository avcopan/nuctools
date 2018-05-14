import numpy
import warnings


def steepest_descent(lin, x0, s0, smax, gtol=1e-5, maxiter=50, print_info=True):
    '''
    :param lin: a function returning the point value and the gradient at x
    :type lin: function
    :param x0: the starting point
    :type x0: numpy.ndarray
    :param s0: the initial step size
    :type s0: float
    :param smax: the maximum step size
    :type smax: float
    :param gtol: convergence threshold for the gradient, by maximum element
    :type gtol: float
    :param maxiter: the maximum number of iterations
    :type maxiter: int
    '''
    f0, g0 = lin(x0)
    x = x0 - s0 * g0

    for iteration in range(1, maxiter + 1):
        f, g = lin(x)

        gmax = numpy.amax(numpy.abs(g))
        converged = gmax < gtol

        info = {'niter': iteration, 'f(x)': f, 'gmax': gmax,
                'conv_status': converged}

        if print_info:
            print(info)

        if converged:
            break

        dx = x - x0
        dg = g - g0
        eps = numpy.finfo(float).eps
        s = numpy.vdot(dx, dg) / (numpy.vdot(dg, dg) + eps)

        s = min(s, smax)

        if print_info:
            print('step size: {:f}'.format(s))

        dx = -s * g

        x0 = x
        g0 = g
        x = x0 + dx

    if not converged:
        warnings.warn("Did not converge!")

    return x, info


def newton_raphson(quad, x0, smax, gtol=1e-5, maxiter=50, print_info=True):
    '''
    :param quad: returns the value, gradient, and hessian as a function of the
        point x and the previous point information (x0, g0, h0)
    :type quad: function
    :param x0: the starting point
    :type x0: numpy.ndarray
    :param smax: the maximum step size
    :type smax: float
    :param gtol: convergence threshold for the gradient, by maximum element
    :type gtol: float
    :param maxiter: the maximum number of iterations
    :type maxiter: int
    '''
    x = x0
    x0 = None
    g0 = None
    h0 = None

    for iteration in range(1, maxiter + 1):
        f, g, h = quad(x, x0=x0, g0=g0, h0=h0)

        gmax = numpy.amax(numpy.abs(g))
        converged = gmax < gtol

        info = {'niter': iteration, 'f(x)': f, 'gmax': gmax,
                'conv_status': converged}

        if print_info:
            print(info)

        if converged:
            break

        dx = -numpy.dot(numpy.linalg.pinv(h), g)
        dx = truncate_length(dx, smax)

        x0 = x
        g0 = g
        h0 = h

        x = x0 + dx

    if not converged:
        warnings.warn("Did not converge!")

    return x, info


def truncate_length(v, max_norm):
    norm = numpy.linalg.norm(v)
    return (v if norm < max_norm else max_norm * v / norm)
