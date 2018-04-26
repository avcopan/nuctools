import numpy
import pyscf
import pyscf.grad
from functools import partial

import nuctools


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


basis = 'cc-pvdz'
labels = ('n', 'n')

e_ = partial(energy, basis, labels)
g_ = partial(gradient, basis, labels)
r0 = ((0., 0., -1.), (0., 0., 1.))

x = nuctools.steepest_descent(f=e_, x0=r0, g=g_, gtol=1e-6)

print(x)
print(numpy.linalg.norm(x[1] - x[0]))
