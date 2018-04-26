import numpy
import pyscf
import pyscf.grad
from functools import partial
import warnings


def _method(basis, labels, coords):
    mol = pyscf.gto.Mole(atom=zip(labels, coords), unit="bohr", basis=basis)
    mol.build()
    method = pyscf.scf.RHF(mol)
    return method


def energy(basis, labels, coords):
    method = _method(basis, labels, coords)
    method.run()
    return method.e_tot


def gradient(basis, labels, coords):
    method = _method(basis, labels, coords)
    method.run()
    grad_method = pyscf.grad.RHF(method)
    return grad_method.kernel()


def steepest_descent(f, x0, g, s0=0.2, gtol=1e-5, maxiter=50):
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


# mol = pyscf.gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
# mf = pyscf.scf.RHF(mol).run()
# pyscf.grad.RHF(mf).kernel()

basis = 'cc-pvdz'
labels = ('n', 'n')

e_ = partial(energy, basis, labels)
g_ = partial(gradient, basis, labels)
r0 = ((0., 0., -1.), (0., 0., 1.))

x = steepest_descent(f=e_, x0=r0, g=g_, gtol=1e-6)

print(x)
print(numpy.linalg.norm(x[1] - x[0]))
