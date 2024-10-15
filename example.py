from TPSC_TRIQS_library import *
from triqs.gf.gf_factories import fit_gf_dlr

beta = 5
n = 0.875
U = 2
t = 1

n_k = 128

eps_k = dispersion_relation(basis=[(1,0,0),(0,1,0)], t=t, n_k=n_k).eps_k

model = tpsc_solver(n=n, U=U, beta=beta, eps_k=eps_k, plot=False)

model.run()

Sigma_wk = Gf(mesh=model.Sigma_wk.mesh, target_shape=())
Sigma_wk.data[:] = model.Sigma_wk.data[:,0,0]
Sigma_w = model.k_sum(model.Sigma_wk)
Sigma_dlr = fit_gf_dlr(Sigma_w, w_max=10.0, eps=1e-14)

iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=Sigma_wk.mesh.components[0].n_iw)

def dens(mu):
    g0_wk_inv = inverse(lattice_dyson_g0_wk(mu=mu, e_k=eps_k, mesh=iw_mesh))
    G2 = inverse(g0_wk_inv - Sigma_dlr_wk)
    return density(model.k_sum(G2)).real - n/2

mu2 = brentq(dens,-2,2)
print(mu2)
print(model.mu2)

density_dlr = []
density_iw = []
beta_list = np.linspace(1,20,30)
for beta in beta_list:
    dlr_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=10, eps=1e-14)
    g_dlr = Gf(mesh=dlr_mesh, target_shape=(1,1))
    g_dlr << 5
    g_coeff = make_gf_dlr(g_dlr)
    g_iw = make_gf_imfreq(g_coeff, 1025)
    density_dlr.append(density(g_coeff)[0][0].real)
    density_iw.append(density(g_iw)[0][0].real)
plt.plot(beta_list, density_iw)
plt.plot(beta_list, density_dlr, '--')
plt.xlabel('$\\beta$')