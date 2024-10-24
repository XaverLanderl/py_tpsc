from TPSC_TRIQS_library import *

# run model
model = tpsc_solver()
model.run()
S_nonloc = model.get_nonlocal_gf(model.Sigma2_dlr_wk)






m = model.Sigma2_dlr_wk
mloc = model.k_sum(m)
mnonloc = m.copy()
mnonloc.data[:] = m.data[:] - mloc.data[:,None]