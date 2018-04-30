import numpy
import scipy.linalg
import warnings


def cerjan_miller(f, x0, g, h, order=0, gtol=1e-5, maxiter=50):
    x = x0
    dim = len(x)
    order = int(order)
    converged = False
    for iteration in range(maxiter):
        fx = f(x)
        gx = g(x)
        hx = h(x)

        gmax = numpy.amax(numpy.abs(gx))

        lm, v_lm = scipy.linalg.eigh(hx)

        gx_lm = numpy.dot(gx, v_lm)
        sc_lm = numpy.array([-1.] * order + [+1.] * (dim - order))

        pc_lm = 1./2 * sc_lm * (numpy.abs(lm) +
                                numpy.sqrt(lm**2 + 4 * gx_lm**2))

        dx_lm = - gx_lm / pc_lm

        dx = numpy.dot(v_lm, dx_lm)

        x += dx

        nreac = int(numpy.sum(lm < 10 * gtol))

        converged = gmax < gtol and nreac == order

        info = {'niter': iteration + 1, 'f(x)': fx, 'gmax': gmax, 'nreac':
                nreac, 'conv_status': converged}
        print(info)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge!")

    return x
