from TPSC_TRIQS_library import *

# Parameters
n = 1.0
t = 1.0
U = 1.0
beta = 2.5

n_k = 128
basis=[(1,0,0),(0,1,0)]

# Dispersion
disp = dispersion_relation(basis=basis, t=t, n_k=n_k)

# TPSC solver
model = tpsc_solver(n=n, U=U, eps_k=disp.eps_k, beta=beta)

# Calculate
model.run()