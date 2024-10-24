# imports
from TPSC_TRIQS_library import *
from triqs.sumk import SumkDiscreteFromLattice

# define model
disp = dispersion_relation()
eps_k = disp.eps_k
model = tpsc_solver(eps_k=eps_k, plot=False)

# set up k-sum
L = disp.H_r
SK = SumkDiscreteFromLattice(lattice=L, n_points=8, method='Riemann')

# run model
model.run()

# get local Sigma
Sigma_dlr_w = model.k_sum(model.Sigma2_dlr_wk)
Sigma_dlr = make_gf_dlr(Sigma_dlr_w)
Sigma_w = make_gf_imfreq(Sigma_dlr, n_iw=1025)

# plot
plt.figure()
oplot(Sigma_w, x_window=(0,40))
oplot(Sigma_dlr_w, x_window=(0,40))

### get Green's function
# direct method
G_wk = Gf(mesh=MeshProduct(Sigma_w.mesh, disp.eps_k.mesh), target_shape=(1,1))
w_mesh, k_mesh = G_wk.mesh.components
iw = np.array([w.imag*1j for w in w_mesh])[:,None,None,None]


G_wk.data[:] = 1 / (iw - eps_k.data[None,:] - Sigma_w.data[:,None])

G_w = model.k_sum(G_wk)


# indirect method
G_w_2 = SK(Sigma=BlockGf(name_list=['up'], block_list=[Sigma_w]), mu=0)

g0_wk=lattice_dyson_g0_wk(mu=0, e_k=eps_k, mesh=w_mesh)
g0_w=model.k_sum(g0_wk)

G_wk_full = Gf(mesh=model.Sigma2_dlr_wk.mesh, target_shape=(1,1))
w_dlr_mesh, k_mesh = G_wk_full.mesh.components
iw = np.array([w.imag*1j for w in w_dlr_mesh])[:,None,None,None]

G_wk_full.data[:] = 1 / (iw - eps_k.data[None,:] - model.Sigma2_dlr_wk.data)
G_w_full = model.k_sum(G_wk_full)

plt.figure()
oplot(g0_w, x_window=(0,40), name='non-interacting')
oplot(G_w, '--', x_window=(0,40), name='interacting')
oplot(G_w_full, x_window=(0,40), name='full Sigma')