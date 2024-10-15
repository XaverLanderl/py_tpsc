from TPSC_TRIQS_library import *

def TPSC_GG(num_iter, n, U, beta, eps_k):

    # initialize results
    mus = []
    Usps = []
    Uchs = []
    diffnorms = []

    # TPSC solver (g0 from scratch)
    model = tpsc_solver(n=n, U=U, beta=beta, eps_k=eps_k, verbose=False, plot=False)

    # Calculate
    model.run()
    mus.append(model.mu2)
    Usps.append(model.Usp)
    Uchs.append(model.Uch)
    Sigma = model.Sigma_wk

    for k in range(num_iter):

        # initialize model with G2 as new g0
        model = tpsc_solver(n=n, U=U, beta=beta, eps_k=eps_k, g0_bubble=model.G2, verbose=False, plot=False)
        model.run()

        # write parameters to files
        mus.append(model.mu2)
        Usps.append(model.Usp)
        Uchs.append(model.Uch)
        diffnorms.append(np.linalg.norm(Sigma.data - model.Sigma_wk.data))
        Sigma = model.Sigma_wk

    # return results
    return mus, Usps, Uchs, diffnorms

Ulist = np.linspace(0.5,3.0,5)
blist = np.linspace(1.0,30.0,5)
mu = np.zeros(shape=(blist.size, Ulist.size))
Usp = np.zeros(shape=(blist.size, Ulist.size))
Uch = np.zeros(shape=(blist.size, Ulist.size))
diffnorm = np.zeros(shape=(blist.size, Ulist.size))

# Dispersion
eps_k = dispersion_relation(basis=[(1,0,0),(0,1,0)], t=1.0, n_k=128).eps_k


for ind_b, beta in enumerate(blist):
    for ind_U, U in enumerate(Ulist):

        prln = str(ind_b+1)+"/"+str(blist.size)+", "+str(ind_U+1)+"/"+str(Ulist.size)
        print(prln)

        mus, Usps, Uchs, diffnorms = TPSC_GG(num_iter=10, n=1.0, U=U, beta=beta, eps_k=eps_k)

        mu[ind_b, ind_U] = np.std(mus[5:])
        Usp[ind_b, ind_U] = np.std(Usp[5:])
        Uch[ind_b, ind_U] = np.std(Uch[5:])
        diffnorm[ind_b, ind_U] = np.mean(diffnorm[5:])

with HDFArchive('TPSC_GG.h5', 'w') as A:
    A['mu'] = mu
    A['Usp'] = Usp
    A['Uch'] = Uch
    A['diffnorm'] = diffnorm