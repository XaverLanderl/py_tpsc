# imports
from TPSC_TRIQS_library import *

# define model
model = tpsc_solver(plot=False)

# run model
model.run()
Sigma = model.Sigma2_dlr_wk

for k in range(100):
    Sigma_nonloc = model.get_nonlocal_gf(Sigma)
    Sigma_loc = model.k_sum(Sigma)

    Sigma = model.add_local_gf(Sigma_nonloc, Sigma_loc)


