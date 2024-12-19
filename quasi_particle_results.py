### imports
from TPSC_TRIQS_library import *
from triqs_maxent import *
from triqs_maxent.sigma_continuator import *
from scipy.interpolate import CubicSpline, UnivariateSpline, lagrange
from scipy.integrate import simpson

# parameters
U_list = [1.0,2.0,3.0,4.0,5.0]
beta_list = [round(b,3) for b in np.linspace(1,30,20)]
n_list = [0.875, 1.0]


for U in U_list:
    for beta in beta_list:
        for n in n_list:

            file_name = 'U' + str(U) + '_beta' + str(beta) + '_n' + str(n) + '_fit0.h5'

            with HDFArchive(file_name, 'r') as A:
                result = A['result']
                Sigma_w = A['Sigma_w']
                Zs = A['Zs']
                tau_invs = A['tau_invs']

            plt.figure()
            result.analyzer_results['LineFitAnalyzer'].plot_linefit()
            plt.title(file_name)

            plt.figure()
            plt.plot(result.omega, result.analyzer_results['LineFitAnalyzer']['A_out'])
            plt.title(file_name)

            plt.figure()
            oplot(Sigma_w)
            plt.title(file_name)

            plt.figure()
            plt.plot(Zs, 'p', label='Z')
            plt.plot(tau_invs, '*', label='$\\tau^{-1}$')
            plt.title(file_name)
            plt.legend(bbox_to_anchor=(1,1))