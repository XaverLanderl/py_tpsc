from TPSC_TRIQS_library import *

# TPSC + GG
def TPSC_GG(num_iter, n, U, beta, eps_k):

    # initialize results
    mus = []
    Usps = []
    Uchs = []

    # TPSC solver (g0 from scratch)
    model = tpsc_solver(n=n, U=U, beta=beta, eps_k=eps_k, verbose=False, plot=False)

    # Calculate
    model.run()
    mus.append(model.mu2)
    Usps.append(model.Usp)
    Uchs.append(model.Uch)

    for k in range(num_iter):

        # initialize model with G2 as new g0
        model = tpsc_solver(n=n, U=U, beta=beta, eps_k=eps_k, g0_bubble=model.G2, verbose=False, plot=False)
        model.run()

        # write parameters to files
        mus.append(model.mu2)
        Usps.append(model.Usp)
        Uchs.append(model.Uch)

    # return results
    return Usps, Uchs, mus



# Parameters
beta = 1
U = 1
n = 1

# Dispersion
eps_k = dispersion_relation(basis=[(1,0,0),(0,1,0)], t=1.0, n_k=128).eps_k

Usps, Uchs, mus = TPSC_GG(num_iter=15, n=n, U=U, beta=beta, eps_k=eps_k)

plt.figure()
plt.plot(Usps)
plt.title("U_sp")
plt.xlabel("iteration")
plt.figure()
plt.plot(Uchs)
plt.title("U_ch")
plt.xlabel("iteration")
plt.figure()
plt.plot(mus)
plt.title("mu")
plt.xlabel("iteration")