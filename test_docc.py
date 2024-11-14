from TPSC_TRIQS_library import *

# construct and run solver
solver = tpsc_solver(plot=False)
solver.run()

# get docc from Sigma and G
solver.Sigma2_dlr_wk.data[:] += solver.U*solver.n/2

F1_dlr_wk = solver.Sigma2_dlr_wk * solver.g0_dlr_wk
F1 = solver.k_iw_sum(F1_dlr_wk)
F2_dlr_wk = solver.Sigma2_dlr_wk * solver.G2
F2 = solver.k_iw_sum(F2_dlr_wk)

# print results
print()
print(np.abs(F1/solver.U - solver.docc))
print(np.abs(F2/solver.U - solver.docc))
print(np.abs(F1/solver.U - F2/solver.U))