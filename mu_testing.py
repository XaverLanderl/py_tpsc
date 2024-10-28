from TPSC_TRIQS_library import *

model = tpsc_solver(n=0.7, plot=False)
model.run()

Sigma_dlr_wk = model.Sigma2_dlr_wk

# does not calculate correct mu!
G_dlr_wk = inverse(inverse(lattice_dyson_g0_wk(mu=0, e_k=model.eps_k, mesh=Sigma_dlr_wk.mesh.components[0])) - Sigma_dlr_wk)
G_dlr_w = model.k_sum(G_dlr_wk)
def func(m):
    result = inverse(inverse(G_dlr_w) + m)
    return float(density(result).real) - model.n/2
mu_corr = brentq(func, -4,4)
G_dlr_w_corr = inverse(inverse(G_dlr_w) + mu_corr)

# calculates correct mu!
G_dlr_wk_2 = model.calc_G_from_Sigma(Sigma_dlr_wk=Sigma_dlr_wk)[0]
G_dlr_w_2 = model.k_sum(G_dlr_wk_2)

print()
print(density(G_dlr_w))
print(density(G_dlr_w_corr))
print(density(G_dlr_w_2))

oplot(G_dlr_w_2)
oplot(G_dlr_w_corr)
plt.title(np.linalg.norm(G_dlr_w_2.data - G_dlr_w_corr.data));