# Import libraries
from TPSC_TRIQS_library import *

# Set parameter (lists)
beta_list = 1/np.linspace(0.05, 1, 10)
U_list = np.array([1,2,3,4,5])
n = 1.2

# get filename
file_name = 'betaUdependence_n' + str(n) + '.h5'

# initialize results
Usp = np.zeros(shape=(beta_list.size, U_list.size))
Uch = np.zeros(shape=(beta_list.size, U_list.size))
docc = np.zeros(shape=(beta_list.size, U_list.size))
mu = np.zeros(shape=(beta_list.size, U_list.size))

# create model
model = tpsc_solver(n=n, verbose=False, plot=False)

print('----- Start! -----')

# Calculate beta- and U-dependence
num_iter = beta_list.size * U_list.size
counter = 1
start = time.time()
for ind_b, beta in enumerate(beta_list):

    # update beta
    model.beta = beta

    for ind_U, U in enumerate(U_list):
        
        # update U
        model.U = U

        # print out progress
        print('beta = ' + str(beta) + ', U = ' + str(U))
        print('calculation ' + str(counter) + '/' + str(num_iter))
        counter += 1

        # run model
        model.run()

        # save data to arrays
        Usp[ind_b, ind_U] = model.Usp
        Uch[ind_b, ind_U] = model.Uch
        docc[ind_b, ind_U] = model.docc
        mu[ind_b, ind_U] = model.mu2_phys

end = time.time()

print('----- Done! -----')
print('Runtime = ' + str((end-start)/60) + 'min')
print('----- Saving! -----')

# Save data to archive
with HDFArchive(file_name, 'w') as A:
    A['beta'] = beta_list
    A['U'] = U_list
    A['Usp'] = Usp
    A['Uch'] = Uch
    A['docc'] = docc
    A['mu'] = mu

print('----- Saved! -----')
print('----- All Done! -----')