import numpy
import warnings


def steepest_descent(f, x0, g, s0, gtol=1e-5, maxiter=50):
    x = x0
    s = s0
    gx0 = g(x0)
    converged = False
    for iteration in range(maxiter):
        x = numpy.array(x0) - s * numpy.array(gx0)
        fx = f(x)
        gx = g(x)
        s = (numpy.vdot(x - x0, gx - gx0) /
             (numpy.linalg.norm(gx - gx0) ** 2 + numpy.finfo(float).eps))
        x0 = numpy.array(x)
        gx0 = numpy.array(gx)

        gmax = numpy.amax(numpy.abs(gx))
        converged = gmax < gtol

        info = {'niter': iteration + 1, 'f(x)': fx, 'gmax': gmax, 's': s,
                'conv_status': converged}
        print(info)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge!")

    return x
