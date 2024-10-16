# Import libraries
from TPSC_TRIQS_library import *

# Set parameter (lists)
beta_list = np.linspace(1,20,25)
U_list = np.linspace(0.5,5,10)
Usp = np.zeros(shape=(beta_list.size, U_list.size))
Uch = np.zeros(shape=(beta_list.size, U_list.size))

# Calculate beta- and U-dependence
for ind_b, beta in enumerate(beta_list):
    for ind_U, U in enumerate(U_list):

        print('beta = ' + str(beta) + ', U = ' + str(U))

        model = tpsc_solver(U=U, beta=beta, verbose=False, plot=False)
        model.calc_first_level_approx()
        Usp[ind_b, ind_U] = model.Usp
        Uch[ind_b, ind_U] = model.Uch

# Save data
with HDFArchive('betaUdependence.h5', 'w') as A:
    A['beta'] = beta_list
    A['U'] = U_list
    A['Usp'] = Usp
    A['Uch'] = Uch

# Read data
with HDFArchive('betaUdependence.h5', 'r') as A:
    beta_list = A['beta']
    U_list = A['U']
    Usp = A['Usp']
    Uch = A['Uch']

# Plot Usp
plt.figure()
for ind_U, U in enumerate(U_list):
    plt.plot(1/beta_list, Usp[:,ind_U], 'p-', label=('U = ' + str(U)))
plt.title('$U_{sp}$')
plt.xlabel('T')
plt.ylabel('[t]')
plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left');

# plot Uch
plt.figure()
for ind_U, U in enumerate(U_list):
    plt.plot(1/beta_list, Uch[:,ind_U], 'p-', label=('U = ' + str(U)))
plt.title('$U_{ch}$')
plt.xlabel('T')
plt.ylabel('[t]')
plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left');