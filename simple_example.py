# imports
from TPSC_TRIQS_library import *

# construct solver
solver = tpsc_solver(plot=False)

# run solver
solver.run()

# get better chi
solver.imtime_bubble_chi2_wk()
oplot(solver.k_sum(solver.chi2_dlr_wk))