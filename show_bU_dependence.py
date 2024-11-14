# Import libraries
from TPSC_TRIQS_library import *

# which density?
n = 1.0

# get filename
file_name = 'betaUdependence_n' + str(n) + '.h5'

# Read data
with HDFArchive(file_name, 'r') as A:
    beta_list   = A['beta']
    U_list      = A['U']
    Usp         = A['Usp']
    Uch         = A['Uch']
    docc        = A['docc']
    mu          = A['mu']

# Plot Usp
plt.figure()
for ind_U, U in enumerate(U_list):
    plt.plot(1/beta_list, Usp[:,ind_U], 'p-', label=('U = ' + str(U)))
plt.title('$U_{sp}$' + ', n = ' + str(n))
plt.xlabel('T')
plt.ylabel('[t]')
plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left');

# plot Uch
plt.figure()
for ind_U, U in enumerate(U_list):
    plt.plot(1/beta_list, Uch[:,ind_U], 'p-', label=('U = ' + str(U)))
plt.title('$U_{ch}$' + ', n = ' + str(n))
plt.xlabel('T')
plt.ylabel('[t]')
plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left');

plt.figure()
for ind_U, U in enumerate(U_list):
    plt.plot(1/beta_list, docc[:,ind_U], 'p-', label=('U = ' + str(U)))
plt.title('$\\langle n_{\\uparrow}n_{\\downarrow}\\rangle$' + ', n = ' + str(n))
plt.xlabel('T')
plt.ylabel('')
plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left');

plt.figure()
for ind_U, U in enumerate(U_list):
    plt.plot(1/beta_list, mu[:,ind_U], 'p-', label=('U = ' + str(U)))
plt.title('$\\mu$' + ', n = ' + str(n))
plt.xlabel('T')
plt.ylabel('')
plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left');