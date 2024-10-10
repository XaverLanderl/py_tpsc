from TPSC_TRIQS_library import *

beta = 30
n = 1
U = 5
t = 1

n_k = 128

eps_k = dispersion_relation(basis=[(1,0,0),(0,1,0)], t=t, n_k=n_k).eps_k

model = tpsc_solver(n=n, U=U, beta=beta, eps_k=eps_k)

model.run()