import numpy
import scipy.linalg
import warnings


def newton_raphson(f, x0, g, h, gtol=1e-5, maxiter=50):
    x = x0
    converged = False
    for iteration in range(maxiter):
        fx = f(x)
        gx = g(x)
        hx = h(x)
        x += - numpy.dot(scipy.linalg.pinv(hx), gx)

        gmax = numpy.amax(numpy.abs(gx))
        converged = gmax < gtol

        info = {'niter': iteration + 1, 'f(x)': fx, 'gmax': gmax,
                'conv_status': converged}
        print(info)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge!")

    return x
