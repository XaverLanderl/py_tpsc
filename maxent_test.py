from TPSC_TRIQS_library import *
from triqs_maxent import *

def add_gaussian_noise(G_tau, percent):
    """
    Adds gaussian noise to the passed Green's function.
    The standard deviation is percent/100 * max(G_tau).

    Parameters
    ----------
    G_tau       :   Green's function object
    percent     :   maximum relative error in percent

    Returns
    -------
    G_tau_noisy :   Green's function object
    """

    # extract max value of Green's function (real and imag separately)
    max_val = np.max(np.abs(G_tau.data.real))
    # get standard deviation
    std_dev = percent/100*max_val

    # Generate Gaussian noise for real part
    real_noise = np.random.normal(0.0, std_dev, G_tau.data.shape)

    # get noisy G_tau
    G_tau_noisy = G_tau.copy()
    G_tau_noisy.data[:] += real_noise

    # return result
    return G_tau_noisy, std_dev

# run model
solver = tpsc_solver(U=4, plot=False)
solver.run()

# Get G_tau
G_dlr = make_gf_dlr(solver.k_sum(solver.g2_dlr_wk))
G_tau = make_gf_imtime(G_dlr, n_tau=2001)
#G_tau = G_tauk(all,(np.pi,np.pi,0))

G_tau_noisy, std_dev = add_gaussian_noise(G_tau, 2)

plt.figure()
oplot(G_tau_noisy, '--')
oplot(G_tau)

tm = TauMaxEnt(cost_function='bryan', probability='normal')
tm.omega = HyperbolicOmegaMesh(omega_min=-20, omega_max=20, n_points=100)
alpha_max = 5e7 / 10001
alpha_min = 1 / 10001
tm.alpha_mesh = LogAlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=20)

tm.set_G_tau(G_tau_noisy)
tm.set_error(5*std_dev)
result = tm.run()
plt.figure()
plt.plot(result.omega, result.analyzer_results['BryanAnalyzer']['A_out'])
plt.figure()
result.plot_A(alpha_index=result.analyzer_results['LineFitAnalyzer']['alpha_index'])

with HDFArchive('maxent_test_DOS_3.h5', 'w') as B:
    B['G_tau_'] = result.data
plt.figure()
result.analyzer_results['LineFitAnalyzer'].plot_linefit()


if False:
    with HDFArchive('maxent_test_DOS.h5', 'r') as B:
        result = B['result']
    plt.plot(result.omega, result.analyzer_results['BryanAnalyzer']['A_out'])
    result.plot_A(alpha_index=result.analyzer_results['LineFitAnalyzer']['alpha_index'])

    for ind, a in enumerate(result.alpha):
        plt.figure()
        result.plot_A(alpha_index=ind)
        plt.title('$\\alpha = $' + str(a))