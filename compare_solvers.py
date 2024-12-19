from TPSC_TRIQS_library import tpsc_solver as old_solver
from TPSC_TRIQS_library_temp import tpsc_solver as new_solver, k_sum
from h5 import HDFArchive
from triqs.plot.mpl_interface import oplot, plt
import numpy as np

# parameters
beta_list = np.linspace(1,20,2)
U_list = np.linspace(0.5,5,2)
n_list = np.linspace(0.5,1,2)

# initialize results
Usp_diff = np.zeros(shape=(beta_list.size, U_list.size, n_list.size))
Uch_diff = np.zeros(shape=(beta_list.size, U_list.size, n_list.size))
docc_diff = np.zeros(shape=(beta_list.size, U_list.size, n_list.size))
mu_diff = np.zeros(shape=(beta_list.size, U_list.size, n_list.size))
Sigma_diff = np.zeros(shape=(beta_list.size, U_list.size, n_list.size))

num_its = beta_list.size*U_list.size*n_list.size
counter = 1
# go over all parameters
for ind_b, beta in enumerate(beta_list):
    for ind_U, U in enumerate(U_list):
        for ind_n, n in enumerate(n_list):

            # progress
            print('Iteration ' + str(counter) + '/' + str(num_its))
            counter += 1

            # initialize solvers
            s_old = old_solver(beta=beta, U=U, n=n, verbose=False, plot=False)
            s_new = new_solver(beta=beta, U=U, n=n, verbose=False, plot=False)

            # run solvers
            s_old.run()
            s_new.run_TPSC()

            plt.figure()
            oplot(k_sum(s_old.g2_dlr_wk), name='old', x_window=(0,40))
            oplot(k_sum(s_new.g2_wk), name='new', x_window=(0,40))
            plt.title('Gf')
            plt.figure()
            oplot(k_sum(s_old.Sigma2_dlr_wk), name='old', x_window=(0,40))
            oplot(k_sum(s_new.Sigma2_wk), name='new', x_window=(0,40))
            plt.title('self-energy')

            # get data
            Usp_diff[ind_b, ind_U, ind_n] = s_old.Usp - s_new.Usp
            Uch_diff[ind_b, ind_U, ind_n] = s_old.Uch - s_new.Uch
            docc_diff[ind_b, ind_U, ind_n] = s_old.docc - s_new.docc
            mu_diff[ind_b, ind_U, ind_n] = s_old.mu2 - s_new.mu2
            Sigma_diff[ind_b, ind_U, ind_n] = np.linalg.norm(s_old.Sigma2_dlr_wk.data - s_new.Sigma2_wk.data)

with HDFArchive('CompareSolvers_Testing2.h5', 'w') as A:
    A['Usp'] = Usp_diff
    A['Uch'] = Uch_diff
    A['docc'] = docc_diff
    A['mu'] = mu_diff
    A['Sigma'] = Sigma_diff

print('-----------------------------------------------------------------')
print(np.max(np.abs(Usp_diff)))
print(np.max(np.abs(Uch_diff)))
print(np.max(np.abs(docc_diff)))
print(np.max(np.abs(mu_diff)))
print(np.max(np.abs(Sigma_diff)))
print('-----------------------------------------------------------------')