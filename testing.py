# imports and functions
### IMPORTS ###
from triqs.gf import *
from triqs.lattice import BravaisLattice, BrillouinZone, TightBinding
from triqs.lattice.tight_binding import TBLattice, dos
from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice import *
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
from triqs.plot.mpl_interface import oplot, plt
import triqs.utility.mpi as mpi
from functools import lru_cache
import numpy as np
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from scipy import integrate
from h5 import HDFArchive
import time
### IMPORTS ###
from math import exp
def onefermion(tau, eps0, beta):
    return -exp((beta*(eps0<0) - tau.value) * eps0) / (1. + exp(-beta * abs(eps0)))

# parameters
beta = 5.0
eps0 = 1.2

# original greens function
it_dlr_mesh = MeshDLRImTime(beta=beta, statistic='Fermion', w_max=10.0, eps=1e-14)
g_it_dlr = Gf(mesh=it_dlr_mesh, target_shape=())
for tau in it_dlr_mesh:
    g_it_dlr[tau] = onefermion(tau, eps0, beta)

# fourier transform via normal
g_dlr = make_gf_dlr(g_it_dlr)
g_it = make_gf_imtime(g_dlr, 128)
g_iw = make_gf_from_fourier(g_it)

# fourier transform via dlr
g_iw_dlr = make_gf_dlr_imfreq(g_dlr)
g_iw_from_dlr = make_gf_imfreq(g_iw_dlr, 128)

oplot(g_it_dlr)
oplot(g_it)

plt.figure()
oplot(g_iw, '-')
oplot(g_iw_from_dlr, '--')