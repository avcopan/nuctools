import numpy
import scipy.linalg
import warnings


def cerjan_miller(f, x0, g, h, order=0, gtol=1e-5, maxiter=50):
    converged = False
    for iteration in range(maxiter):
        f0 = f(x0)
        g0 = g(x0)
        h0 = h(x0)

        l, v = scipy.linalg.eigh(h0)

        gmax = numpy.amax(numpy.abs(g0))
        lorder = int(numpy.sum(l < gtol))

        lg0 = numpy.dot(g0, v)
        ld0 = 1./2 * (numpy.abs(l) + numpy.sqrt(l**2 + 4. * lg0**2))
        ld0[:order] *= -1.
        ldx = - lg0 / ld0
        dx = numpy.dot(v, ldx)

        x = x0 + dx
        x0 = x

        converged = gmax < gtol and order == lorder

        info = {'niter': iteration + 1, 'f(x)': f0, 'gmax': gmax,
                'lorder': lorder, 'conv_status': converged}
        print(info)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge!")

    return x
