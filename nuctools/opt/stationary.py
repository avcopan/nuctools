import numpy
import scipy.linalg
import warnings


def newton_raphson(f, x0, g, h, gtol=1e-5, maxiter=50):
    converged = False
    for iteration in range(maxiter):
        f0 = f(x0)
        g0 = g(x0)
        h0 = h(x0)
        x = x0 - numpy.dot(scipy.linalg.pinv(h0), g0)
        x0 = x

        gmax = numpy.amax(numpy.abs(g0))
        converged = gmax < gtol

        info = {'niter': iteration + 1, 'f(x)': f0, 'gmax': gmax,
                'conv_status': converged}
        print(info)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge!")

    return x
