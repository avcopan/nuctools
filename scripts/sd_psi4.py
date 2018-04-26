import numpy
import psi4
from functools import partial
from toolz.functoolz import compose
import warnings

psi4.set_options({
    'e_convergence': 1e-12,
    'd_convergence': 1e-10,
    'guess': 'gwh',
    'reference': 'rhf'})


def _molecule(geometry):
    mol = psi4.core.Molecule.create_molecule_from_string(
            "units bohr\n" +
            '\n'.join("{:2s} {:f} {:f} {:f}".format(s, *xyz)
                      for s, xyz in geometry))
    mol.update_geometry()
    return mol


def _wavefunction(basis, geometry):
    mol = _molecule(geometry)
    bs = psi4.core.BasisSet.build(mol, '', basis)
    wfn = psi4.core.Wavefunction.build(mol, bs)
    sf, _ = psi4.driver.dft_funcs.build_superfunctional('hf', False)
    hf = psi4.core.RHF(wfn, sf)
    hf.compute_energy()
    return hf


def energy(basis, geometry):
    hf = _wavefunction(basis, geometry)
    return hf.energy()


def gradient(basis, geometry):
    hf = _wavefunction(basis, geometry)
    dr = psi4.core.Deriv(hf)
    dr.compute()
    return numpy.array(hf.gradient())


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


basis = 'cc-pvdz'
geometry = [('n', (0., 0., -1.)),
            ('n', (0., 0., +1.))]

r0 = ((0., 0., -1.),
      (0., 0., +1.))

geom = partial(zip, ('n', 'n'))

e_ = compose(partial(energy, basis), geom)
g_ = compose(partial(gradient, basis), geom)

x = steepest_descent(f=e_, x0=r0, g=g_, gtol=1e-6)

print(x)
print(numpy.linalg.norm(x[1] - x[0]))
