""" A library of PySCF electronic structure methods.
"""
import numpy
import pyscf
import pyscf.grad


def _rhf_method(basis, labels, coords):
    """ the RHF electronic structure method
    """
    mcoords = numpy.reshape(coords, (-1, 3))
    mol = pyscf.gto.Mole(atom=zip(labels, mcoords),
                         unit="bohr",
                         basis=basis)
    mol.build()
    method = pyscf.scf.RHF(mol)
    return method


def rhf_energy_point(basis, labels, coords):
    """ RHF energy
    """
    method = _rhf_method(basis, labels, coords)
    method.run()
    e = method.e_tot
    return e


def rhf_energy_line(basis, labels, coords):
    """ RHF gradient and energy
    """
    method = _rhf_method(basis, labels, coords)
    method.run()
    grad_method = pyscf.grad.RHF(method)
    e = method.e_tot
    g = numpy.ravel(grad_method.kernel())
    return e, g
