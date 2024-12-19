from AIM_TPSC import *

t = 1
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

beta_list = np.array([5,10,15,20])
U_list = np.array([1,2,3,4,5])
n = 0.875

doccs = np.zeros(shape=(beta_list.size, U_list.size))

for ind_b, beta in enumerate(beta_list):
    for ind_U, U in enumerate(U_list):

        print(ind_b + 1)
        print(ind_U + 1)
        print()

        # get non-interacting G
        iw_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=12.0, eps=1e-14)
        G0_iw = Gf(mesh=iw_mesh, target_shape=())
        G0_iw << inverse(iOmega_n)

        for iteration in range(10):

            # get interacting G from "Impurity Solver"
            Usp, Sigma_iw, G_iw = get_G_iw_from_G0_iw(G0_iw, n, U)

            # get local lattice Green's function
            G_loc = get_G_loc_iw(eps_k, Sigma_iw, n)

            if iteration >= 10:
                oplot(Sigma_iw)

            # update G0_iw
            G0_iw << inverse(inverse(G_loc) + Sigma_iw)

        doccs[ind_b, ind_U] = Usp/U*n**2/4

for ind_U, U in enumerate(U_list):
    plt.plot(beta_list, doccs[:,ind_U], 'p--', label='U = ' + str(U))
plt.xlabel('$\\beta$')
plt.title('$\\langle n_{\\uparrow}n_{\\downarrow}\\rangle(\\beta, U)$');
plt.legend(bbox_to_anchor=(1,1))