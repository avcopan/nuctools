import numpy
import pyscf
import pyscf.grad
from functools import partial


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


def steepest_descent(f, x0, fp, s0=0.2):
    x = x0
    s = s0
    fpx0 = fp(x0)
    for _ in range(10):
        print(s)
        x = numpy.array(x0) - s * numpy.array(fpx0)
        fx = f(x)
        fpx = fp(x)
        s = numpy.vdot(x - x0, fpx-fpx0) / numpy.linalg.norm(fpx0) ** 2
        x0 = numpy.array(x)
        fpx0 = numpy.array(fpx)
        print(fx)
    return x


# mol = pyscf.gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
# mf = pyscf.scf.RHF(mol).run()
# pyscf.grad.RHF(mf).kernel()

basis = 'cc-pvdz'
labels = ('n', 'n')

e_ = partial(energy, basis, labels)
g_ = partial(gradient, basis, labels)
r0 = ((0., 0., -1.), (0., 0., 1.))

x = steepest_descent(f=e_, x0=r0, fp=g_)

# cm = sum(x) / len(x)
# x -= cm

print(x)
print(numpy.linalg.norm(x[1] - x[0]))
