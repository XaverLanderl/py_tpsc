from TPSC_TRIQS_library_temp import tpsc_solver as solver_temp
from TPSC_TRIQS_library import tpsc_solver as solver
import numpy as np

# run solvers
s1 = solver(verbose=False, plot=False)
s1.run()

s2 = solver_temp(verbose=False, plot=False)
s2.run()

# compare values
print('differences of scalar values:')
print(abs(s1.docc - s2.docc))
print(abs(s1.Usp - s2.Usp))
print(abs(s1.Uch - s2.Uch))
print(abs(s1.mu1 - s2.mu1))
print(abs(s1.mu2 - s2.mu2))
print(abs(s1.mu2_phys - s2.mu2_phys))

print()
print('diffnorms of Gfs:')
print(np.linalg.norm((s1.g0_dlr_wk-s2.g0_dlr_wk).data))
print(np.linalg.norm((s1.chi0_dlr_wk-s2.chi0_dlr_wk).data))
print(np.linalg.norm((s1.g2_dlr_wk-s2.g2_dlr_wk).data))
print(np.linalg.norm((s1.Sigma2_dlr_wk-s2.Sigma2_dlr_wk).data))
print(np.linalg.norm((s1.chi1_sp_dlr_wk-s2.chi1_sp_dlr_wk).data))
print(np.linalg.norm((s1.chi1_ch_dlr_wk-s2.chi1_ch_dlr_wk).data))

# check self-cons
s1.verbose = True
s2.verbose = True
s1.check_for_self_consistency()
s2.check_for_self_consistency()