### commonly used functions ###
### IMPORTS ###
from triqs.gf import *

from triqs.lattice import BravaisLattice, BrillouinZone, TightBinding
from triqs.lattice.tight_binding import TBLattice, dos
from triqs_tprf.lattice import *
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk

from triqs.plot.mpl_interface import oplot, plt

import numpy as np
import time
from scipy.optimize import brentq

import triqs.utility.mpi as mpi
from h5 import HDFArchive
### IMPORTS ###

### SUMS OVER GREEN'S FUNCTIONS ###
def k_sum(g_wk):
    """
    Calculates the k-sum over the passed Green's function.
    
    Requires
    --------
    g_wk must be defined on a MeshProduct(..., MeshBrZone).
    
    Parameters
    ----------
    triqs.gf g_wk       : TRIQS Green's function

    Returns
    -------
    triqs.gf g_w        : TRIQS Green's function
                        : mesh = g_wk.mesh.components[0]
                        : target_shape = g_wk.target_shape
    """

    # extract mesh
    mesh = g_wk.mesh

    # check input
    if not isinstance(mesh, MeshProduct):
        raise TypeError('g_wk.mesh must be of type \'MeshProduct\'!')
    
    # check whether second component is a kmesh
    if not isinstance(mesh.components[1], MeshBrZone):
        raise TypeError('g_wk.mesh.components[1] must be of type \'MeshBrZone\'!')
    
    # extract meshes
    w_mesh, k_mesh = mesh.components

    # initialize resulting Green's function
    g_w = Gf(mesh=w_mesh, target_shape=g_wk.target_shape)

    # perform sum over k
    g_w.data[:] = np.sum(g_wk.data, axis=1) / len(k_mesh)

    # return result
    return g_w

def k_iw_sum(g_wk):
    """
    Calculates the (real part of the) k- and Matsubara sum over the passed Green's function.
    
    Requires
    --------
    g_wk must be defined on a MeshProduct(..., MeshBrZone).
    
    Parameters
    ----------
    triqs.gf g_wk       : TRIQS Green's function

    Returns
    -------
    double              : k- and Matsubara sum over passed Green's function
    """

    # calculate k-sum (will check whether mesh is correct)
    g_w = k_sum(g_wk)

    # extract mesh
    mesh = g_w.mesh

    # get correct sign from statistic
    if mesh.statistic == 'Boson':
        sign = -1   # need to account for minus sign in definition of density
    elif mesh.statistic == 'Fermion':
        sign = +1

    # check first component
    if isinstance(mesh, MeshDLRImFreq) or isinstance(mesh, MeshDLRImTime):
        # get dlr representation
        g_dlr = make_gf_dlr(g_w)

        # get density
        result = sign*density(g_dlr).real

    elif isinstance(mesh, MeshImFreq):
        # get density
        result = sign*density(g_w).real

    elif isinstance(mesh, MeshImTime):
        # get density
        result = -sign*g_w(0).real

    else:
        raise TypeError('g_wk.mesh.components[0] must be of type \'Mesh(DLR)ImTime/Freq\'!')
    
    # return result
    return result



### FUNCTIONS FOR CHEMICAL POTENTIAL ###
def Gf_density(mu, eps_k, Sigma_wk=None):
    """
    Calculates the density as a function of mu.

    Parameters
    ----------
    mu          : chemical potential
    eps_k       : dispersion relation (saved as a Green's function on a MeshBrZone)
    Sigma_wk    : Self-Energy (Green's function with MeshProduct(Mesh(DLR)ImFreq, MeshBrZone)) OR
                : Mesh(DLR)ImFreq; if so, calculate non-interacting density
    Returns
    -------
    double      : density of the Green's function
    """

    # check type of Sigma_wk
    if isinstance(Sigma_wk, Gf):
        # input is a Green's function!
        Sigma_wk_is_Greens_Func = True

        # extract mesh
        mesh = Sigma_wk.mesh

        # check input
        if not isinstance(mesh, MeshProduct):
            raise TypeError('Sigma_wk.mesh must be of type \'MeshProduct\'!')
        
        # extract meshes
        w_mesh, k_mesh = mesh.components

        if not isinstance(w_mesh, MeshDLRImFreq) or isinstance(w_mesh, MeshImFreq):
            raise TypeError('Sigma_wk.mesh.components[0] must be of type \'Mesh(DLR)ImFreq\'!')
        
        if not isinstance(k_mesh, MeshBrZone):
            raise TypeError('Sigma_wk.mesh.components[1] must be of type \'MeshBrZone\'!')
        
        if not Sigma_wk.target_shape == (1,1):
            raise TypeError('Sigma_wk.target_shape must be (1,1)!')
    
    elif isinstance(Sigma_wk, MeshDLRImFreq) or isinstance(Sigma_wk, MeshImFreq):
        # input is a mesh
        Sigma_wk_is_Greens_Func = False

        # get mesh
        w_mesh = Sigma_wk

    else:
        raise TypeError('Input Sigma_wk must be of type \'Gf\' or of type \'Mesh(DLR)ImFreq!')
    

    if Sigma_wk_is_Greens_Func == True:
        # Dyson equation
        g0_dlr_wk_inv = inverse(lattice_dyson_g0_wk(mu=mu, e_k=eps_k, mesh=w_mesh))
        G2 = inverse(g0_dlr_wk_inv - Sigma_wk)
    elif Sigma_wk_is_Greens_Func == False:
        # Only non-interacting Green's function
        G2 = lattice_dyson_g0_wk(mu=mu, e_k=eps_k, mesh=w_mesh)
    
    # return density
    return k_iw_sum(G2)

def calc_mu(dens, eps_k, Sigma_wk):
    """
    Calculates the chemical potential for a desired density

    Parameters
    ----------
    dens        : desired density
    eps_k       : dispersion relation (saved as a Green's function on a MeshBrZone)
    Sigma_wk    : Self-Energy (Green's function with MeshProduct(Mesh(DLR)ImFreq, MeshBrZone)) OR
                : Mesh(DLR)ImFreq; if so, calculate non-interacting density
    Returns
    -------
    double      : chemical potential leading to dens.
    """

    # find the mu that leads to the correct density
    mu_min = np.min(eps_k.data.real)
    mu_max = np.max(eps_k.data.real)
    mu_result = brentq(lambda m: Gf_density(mu=m, eps_k=eps_k, Sigma_wk=Sigma_wk) - dens/2, mu_min, mu_max)

    # return result
    return mu_result

def calc_G_from_Sigma(n, eps_k, Sigma_wk):
    """
    Calculates the Green's function for given density, dispersion, as well as the given self-energy.

    Requires
    --------
    Sigma_dlr_wk must be given on a MeshDLRImFreq.

    Parameters
    ----------
    n           : density
    eps_k       : dispersion relation
    Sigma_wk    : self-energy

    Returns
    -------
    G_wk        : Green's function with given self-energy and correct chemical potential
    mu          : chemical potential
    """

    # calculate mu
    mu = calc_mu(dens=n, eps_k=eps_k, Sigma_wk=Sigma_wk)

    # calculate G
    g0_wk_inv = inverse(lattice_dyson_g0_wk(mu=mu, e_k=eps_k, mesh=Sigma_wk.mesh.components[0]))
    G = inverse(g0_wk_inv - Sigma_wk)

    # return results
    return G, mu



### ADD AND SUBTRACT LOCAL GREEN'S FUNCTIONS ###
def add_k_ind_gf(g_wk, g_w):
    """
    Adds a k-independent Green's function to a k-dependent Green's function.

    Requires
    --------
    Both inputs must have the same first mesh.

    Parameters
    ----------
    triqs.gf g_wk   : k-dependent Green's function
    triqs.gf g_w    : k-independent Green's function

    Returns
    -------
    triqs.gf        : k-dependent Green's function
    """

    # check input
    if not isinstance(g_wk.mesh, MeshProduct):
        raise TypeError('g_wk.mesh must be of type \'MeshProduct\'!')
    if not isinstance(g_wk.mesh.components[1], MeshBrZone):
        raise TypeError('g_wk.mesh.components[1] must be of type \'MeshBrZone\'!')
    if not g_wk.target_shape == g_w.target_shape:
        raise AssertionError('Input Green\'s functions must have the same target_shape!')
    if not g_wk.mesh.components[0] == g_w.mesh:
        raise AssertionError('Input Green\'s functions must have the same imfreq_mesh!')

    # initialize result
    result = g_wk.copy()

    # assign values (use numpy boradcasting!)
    result.data[:] = g_wk.data[:] + g_w.data[:,None]

    # return result
    return result

def get_nonlocal_gf(g_wk):
    """
    Calculates and subtracts the local part from g_wk.

    Requires
    --------
    g_wk must be given on a MeshProduct(...,MeshBrZone)
    
    Parameters
    ----------
    triqs.gf g_wk   : k-dependent Green's function
    
    Returns
    -------
    triqs.gf        : k-dependent Green's function
    """
    
    # get local Gf
    g_loc_w = k_sum(g_wk)

    # ititialize result
    result = add_k_ind_gf(g_wk=g_wk, g_w=-g_loc_w)

    # return result
    return result



### APPROXIMATIONS ###
def solve_Hubbard_RPA(chi0_wk, U):
    """
    Calculates the RPA-(spin-)susceptibility of the Hubbard-Model.
    chi_RPA_wk = chi0_wk / (1 - U/2*chi0_wk)

    Requires
    --------
    calc_noninteracting_gf must have been called.

    Parameters
    ----------
    triqs.gf chi0_wk    : non-interacting susceptibility
    double U            : RPA vertex for charge/spin susceptibility

    Returns
    -------
    triqs.gf            : RPA-spin-susceptibility (U > 0) or RPA-charge-susceptibility (U < 0)
    """

    # extract mesh
    mesh = chi0_wk.mesh

    # check input
    if not isinstance(mesh, MeshProduct):
        raise TypeError('chi0_wk.mesh must be of type \'MeshProduct\'!')
    
    # extract meshes
    w_mesh, k_mesh = mesh.components

    if not isinstance(w_mesh, MeshDLRImFreq) or isinstance(w_mesh, MeshImFreq):
        raise TypeError('chi0_wk.mesh.components[0] must be of type \'Mesh(DLR)ImFreq\'!')
    
    if not isinstance(k_mesh, MeshBrZone):
        raise TypeError('chi0_wk.mesh.components[1] must be of type \'MeshBrZone\'!')
    
    # initialize RPA susceptibility
    chi_dlr_wk = chi0_wk.copy()

    # fill with data
    chi_dlr_wk.data[:] = chi0_wk.data[:]/(1 - U/2*chi0_wk.data[:])

    # return results
    return chi_dlr_wk



### FOURIER TRANSFORMS ###
def fourier_wk_to_tr(g_wk):
    """
    Fourier-transforms a Green's function from wk to tr representation.

    Requires
    --------
    g_wk must be defined on a MeshProduct(Mesh(DLR)ImFreq, MeshBrZone)

    Parameters
    ----------
    triqs.gf g_wk       : TRIQS Green's function

    Returns
    -------
    triqs.gf g_tr       : TRIQS Green's function on a MeshProduct(Mesh(DLR)ImTime, MeshCycLat)
    """

    # extract mesh
    mesh = g_wk.mesh

    # check input
    if not isinstance(mesh, MeshProduct):
        raise TypeError('g_wk.mesh must be of type \'MeshProduct\'!')
    
    # extract meshes
    w_mesh, k_mesh = mesh.components

    if not isinstance(w_mesh, MeshDLRImFreq) or isinstance(w_mesh, MeshImFreq):
        raise TypeError('g_wk.mesh.components[0] must be of type \'Mesh(DLR)ImFreq\'!')
    
    if not isinstance(k_mesh, MeshBrZone):
        raise TypeError('g_wk.mesh.components[1] must be of type \'MeshBrZone\'!')
    
    # fourier transform
    g_wr = fourier_wk_to_wr(g_wk)
    g_tr = fourier_wr_to_tr(g_wr)

    # return result
    return g_tr

def fourier_wk_to_mtr(g_wk):
    """
    Fourier-transforms a Green's function from wk to (-t,-r) representation.

    Requires
    --------
    g_wk must be defined on a MeshProduct(Mesh(DLR)ImFreq, MeshBrZone)

    Parameters
    ----------
    triqs.gf g_wk       : TRIQS Green's function

    Returns
    -------
    triqs.gf g_mtr      : TRIQS Green's function on a MeshProduct(Mesh(DLR)ImTime, MeshCycLat)
    """

    # extract mesh
    mesh = g_wk.mesh

    # check input
    if not isinstance(mesh, MeshProduct):
        raise TypeError('g_wk.mesh must be of type \'MeshProduct\'!')
    
    # extract meshes
    w_mesh, k_mesh = mesh.components

    if not isinstance(w_mesh, MeshDLRImFreq) or isinstance(w_mesh, MeshImFreq):
        raise TypeError('g_wk.mesh.components[0] must be of type \'Mesh(DLR)ImFreq\'!')
    
    if not isinstance(k_mesh, MeshBrZone):
        raise TypeError('g_wk.mesh.components[1] must be of type \'MeshBrZone\'!')
    
    # fourier transform
    g_wk_conj = g_wk.conjugate()
    g_wr_conj = fourier_wk_to_wr(g_wk_conj)
    g_tr_conj = fourier_wr_to_tr(g_wr_conj)
    g_mtr = g_tr_conj.conjugate()

    # return result
    return g_mtr

def fourier_tr_to_wk(g_tr):
    """
    Fourier-transforms a Green's function from wk to tr representation.

    Requires
    --------
    g_tr must be defined on a MeshProduct(Mesh(DLR)ImTime, MeshCycLat)

    Parameters
    ----------
    triqs.gf g_tr       : TRIQS Green's function

    Returns
    -------
    triqs.gf g_wk       : TRIQS Green's function on a MeshProduct(Mesh(DLR)ImFreq, MeshBrZone)
    """

    # extract mesh
    mesh = g_tr.mesh

    # check input
    if not isinstance(mesh, MeshProduct):
        raise TypeError('g_tr.mesh must be of type \'MeshProduct\'!')
    
    # extract meshes
    w_mesh, k_mesh = mesh.components

    if not isinstance(w_mesh, MeshDLRImTime) or isinstance(w_mesh, MeshImTime):
        raise TypeError('g_tr.mesh.components[0] must be of type \'Mesh(DLR)ImTime\'!')
    
    if not isinstance(k_mesh, MeshCycLat):
        raise TypeError('g_tr.mesh.components[1] must be of type \'MeshCycLat\'!')
    
    # fourier transform
    g_wr = fourier_tr_to_wr(g_tr)
    g_wk = fourier_wr_to_wk(g_wr)

    # return result
    return g_wk
