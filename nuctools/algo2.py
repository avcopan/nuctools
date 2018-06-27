""" optimization algorithms
"""
import warnings
import numpy


def enforce_max_step_size(dx, smax):
    """ enforces maximum step size
    """
    s = numpy.linalg.norm(dx)
    return dx if s < smax else dx * smax / s


def optimize_gradient_descent(lin, x0, smax=0.3, gtol=1e-5, maxiter=50,
                              print_info=True):
    """ optimizes by gradient descent

    :param lin: a function returning the point value and the gradient at x
    :type lin: function
    :param x0: the starting point
    :type x0: numpy.ndarray
    :param smax: the maximum step size
    :type smax: float
    :param gtol: convergence threshold for the gradient, by maximum element
    :type gtol: float
    :param maxiter: the maximum number of iterations
    :type maxiter: int
    """
    x0 = numpy.array(x0)
    ln = lin(x0)
    f0, g0 = ln[:2]
    s = smax

    converged = False

    traj = [(x0, ln)]

    for iteration in range(maxiter):
        dx = - s * g0
        dx = enforce_max_step_size(dx, smax)
        x = x0 + dx

        ln = lin(x)
        f, g = ln[:2]
        traj.append([(x, ln)])

        gmax = numpy.amax(numpy.abs(g))
        converged = gmax < gtol

        if print_info:
            print('iteration {:d} gmax={:.1e} f(x)={:15.10e}'
                  .format(iteration, gmax, f))

        if converged:
            break
        else:
            dg = g - g0
            eps = numpy.finfo(float).eps
            s = numpy.vdot(dx, dg) / (numpy.vdot(dg, dg) + eps)
            x0 = x
            g0 = g

    if not converged:
        warnings.warn("Did not converge!")

    return x, traj


def optimize_quasi_newton(lin, x0, hup, smax=0.3, gtol=1e-5,
                          maxiter=50, print_info=True):
    """ optimizes by gradient descent

    :param lin: a function returning the point value and the gradient at x
    :type lin: function
    :param x0: the starting point
    :type x0: numpy.ndarray
    :param hup: a function returning the updated hessian
    :type hup: function
    :param smax: the maximum step size
    :type smax: float
    :param gtol: convergence threshold for the gradient, by maximum element
    :type gtol: float
    :param maxiter: the maximum number of iterations
    :type maxiter: int
    """
    dim = numpy.size(x0)

    x0 = numpy.array(x0)
    ln = lin(x0)
    f0, g0 = ln[:2]
    h0 = numpy.linalg.norm(g0) / smax * numpy.eye(dim)

    converged = False

    traj = [(x0, ln)]

    for iteration in range(maxiter):
        dx = - numpy.dot(numpy.linalg.pinv(h0), g0)
        dx = enforce_max_step_size(dx, smax)
        x = x0 + dx

        ln = lin(x)
        f, g = ln[:2]
        traj.append([(x, ln)])

        gmax = numpy.amax(numpy.abs(g))
        converged = gmax < gtol

        if print_info:
            print('iteration {:d} gmax={:.1e} f(x)={:15.10e}'
                  .format(iteration, gmax, f))

        if converged:
            break
        else:
            h = hup(x=x, g=g, x0=x0, g0=g0, h0=h0)
            x0 = x
            g0 = g
            h0 = h

    if not converged:
        warnings.warn("Did not converge!")

    return x, traj
