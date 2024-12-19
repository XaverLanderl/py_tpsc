from triqs.gf import *
from triqs.lattice.tight_binding import TBLattice
from triqs_tprf.lattice import lattice_dyson_g0_wk, fourier_wk_to_wr
from triqs.plot.mpl_interface import oplot, plt
import numpy as np
from scipy.optimize import brentq

def get_chi0_iw(G0_iw):

    # get G0_tau and G0_mtau
    G0_dlr = make_gf_dlr(G0_iw)
    G0_conj_dlr = make_gf_dlr(conjugate(G0_iw))
    G0_tau = make_gf_dlr_imtime(G0_dlr)
    G0_mtau = conjugate(make_gf_dlr_imtime(G0_conj_dlr))

    # get new mesh
    mesh_iw = G0_tau.mesh
    mesh_tau = MeshDLRImTime(beta=mesh_iw.beta, statistic='Boson', w_max=mesh_iw.w_max, eps=mesh_iw.eps)

    # get chi0_tau
    chi0_tau = Gf(mesh=mesh_tau, target_shape=G0_tau.target_shape)

    # write data to chi0_tau
    chi0_tau.data[:] = -2 * G0_tau.data * G0_mtau.data

    # get chi0_iwU
    chi0_dlr = make_gf_dlr(chi0_tau)
    chi0_iw = make_gf_dlr_imfreq(chi0_dlr)

    # return results
    return chi0_iw

def get_chiRPA_iw(chi0_iw, U):

    # initialize result
    chiRPA_iw = chi0_iw * inverse(1 - U/2 * chi0_iw)

    # return result
    return chiRPA_iw

def iw_sum(g):

    # get result
    if g.mesh.statistic == 'Boson':
        result = -density(g).real
    if g.mesh.statistic == 'Fermion':
        result = density(g).real
    
    # return result
    return result

def k_sum(g_Xk):

    # get meshes
    X_mesh, k_mesh = g_Xk.mesh.components

    # initialize result
    g_X = Gf(mesh=X_mesh, target_shape=g_Xk.target_shape)

    # perform sum
    g_X.data[:] = np.sum(g_Xk.data, axis=1) / len(k_mesh)

    # return result
    return g_X

def get_vertices(chi0_iw, n, U):

    # get spin vertex from sum rule
    Usp_root = lambda usp: (iw_sum(get_chiRPA_iw(chi0_iw, usp)) - (n - usp/U*n**2/2))
    Usp = brentq(Usp_root, 0.0, 2.0/np.amax(chi0_iw.data).real-1e-7)

    # get charge vertex from sum rule
    Uch_root = lambda uch: (iw_sum(get_chiRPA_iw(chi0_iw, -uch)) - (n + Usp/U*n**2/2 - n**2))
    Uch = brentq(Uch_root, 0.0, 1000.0)

    # return results
    return Usp, Uch

def get_Sigma_iw(G0_iw, chi0_iw, Usp, Uch, U):

    # get spin susceptibility
    chisp_iw = chi0_iw * inverse(1 - Usp/2 * chi0_iw)

    # get charge susceptibility
    chich_iw = chi0_iw * inverse(1 + Uch/2 * chi0_iw)

    # get effective potential
    V_iw = U/8 * (3*Usp*chisp_iw + Uch*chich_iw)

    # get V(-tau)
    V_dlr = make_gf_dlr(conjugate(V_iw))
    V_mtau = conjugate(make_gf_dlr_imtime(V_dlr))

    # get G_tau
    G0_dlr = make_gf_dlr(G0_iw)
    G0_tau = make_gf_dlr_imtime(G0_dlr)

    # get Sigma_tau
    Sigma_tau = G0_tau.copy()
    Sigma_tau.data[:] = V_mtau.data * G0_tau.data

    # get Sigma_iw
    Sigma_dlr = make_gf_dlr(Sigma_tau)
    Sigma_iw = make_gf_dlr_imfreq(Sigma_dlr)

    # return result
    return Sigma_iw

def get_G_iw(G0_iw, Sigma_iw, n):

    # get chemical potential
    n_root = lambda mu: iw_sum(inverse(inverse(G0_iw) + mu - Sigma_iw)) - n/2
    Mu = brentq(n_root, -10, 10)

    # get G_iw
    G_iw = inverse(inverse(G0_iw) + Mu - Sigma_iw)

    # return result
    return G_iw

def get_G_iw_from_G0_iw(G0_iw, n, U):

    # get chi0_iw
    chi0_iw = get_chi0_iw(G0_iw)

    # get vertices
    Usp, Uch = get_vertices(chi0_iw, n, U)

    # get Sigma_iw
    Sigma_iw = get_Sigma_iw(G0_iw, chi0_iw, Usp, Uch, U)

    # get G_iw
    G_iw = get_G_iw(G0_iw, Sigma_iw, n)

    # return result
    return Usp, Sigma_iw, G_iw

def get_G_loc_iw(eps_k, Sigma_iw, n):

    # define Green's function
    G_wk = Gf(mesh=MeshProduct(Sigma_iw.mesh, eps_k.mesh), target_shape=())

    iw = np.array([w.imag*1j for w in Sigma_iw.mesh])[:,None]
    
    # density as function of mu
    def n_root(m):
        # fill Gf
        G_wk.data[:] = 1 / (iw + m - eps_k.data[None,:,0,0] - Sigma_iw.data[:,None])
        return iw_sum(k_sum(G_wk)) - n/2

    # calculate new mu
    mu_new = brentq(n_root, -10, 10)
    
    # calculate new Green's function
    G_wk.data[:] = 1 / (iw + mu_new - eps_k.data[None,:,0,0] - Sigma_iw.data[:,None])

    # perform k-sum to get the local Green's function
    G_loc = k_sum(G_wk)

    # return result
    return G_loc


### test the method
# parameters
beta = 10
t = 1.0
U = 2.0
n = 0.875

basis = [(1,0,0), (0,1,0)]
hoppings = {
    (+1,0) : [[-t]],
    (-1,0) : [[-t]],
    (0,+1) : [[-t]],
    (0,-1) : [[-t]]
}
n_k = 128

# get dispersion
L = TBLattice(basis, hoppings)
k_mesh = L.get_kmesh(n_k=n_k)
eps_k = L.fourier(k_mesh)

# get non-interacting G
iw_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=12.0, eps=1e-14)
G0_iw = Gf(mesh=iw_mesh, target_shape=())
G0_iw << inverse(iOmega_n)

for iteration in range(20):

    # get interacting G from "Impurity Solver"
    Usp, Sigma_iw, G_iw = get_G_iw_from_G0_iw(G0_iw, n, U)

    # get local lattice Green's function
    G_loc = get_G_loc_iw(eps_k, Sigma_iw, n)

    if iteration >= 10:
        oplot(Sigma_iw)

    # update G0_iw
    G0_iw << inverse(inverse(G_loc) + Sigma_iw)

plt.figure()
Sigma_dlr = make_gf_dlr(Sigma_iw)
Sigma_iw_full = make_gf_imfreq(Sigma_dlr, n_iw=1025)
oplot(Sigma_iw_full, x_window=(0,40))