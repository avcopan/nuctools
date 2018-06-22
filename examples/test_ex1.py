""" steepest descent optimization of N2 with PySCF
"""
import numpy
from functools import partial
import nuctools
import eslibs.pyscf_ as eslib


def test__optimize_gradient_descent():
    lin = partial(eslib.rhf_energy_line, 'sto-3g', ('o', 'h', 'h'))
    x0 = numpy.ravel([(0., 0., 0.), (-1., 1., 0.), (1., 1., 0.)])
    x, traj = nuctools.algo2.optimize_gradient_descent(
        lin=lin, x0=x0, smax=0.3, gtol=1e-8)
    print(x)
    print(len(traj))
