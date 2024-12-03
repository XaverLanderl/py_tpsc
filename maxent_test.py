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
solver = tpsc_solver(plot=False)
solver.run()

# Get G_tau
G_dlr = make_gf_dlr(solver.g2_dlr_wk)
G_tauk = make_gf_imtime(G_dlr, n_tau=2001)
G_tau = G_tauk(all,(0,0,0))

G_tau_noisy, std_dev = add_gaussian_noise(G_tau, 5)

plt.figure()
oplot(G_tau_noisy, '--')
oplot(G_tau)

tm = TauMaxEnt(cost_function='bryan', probability='normal')
tm.omega = HyperbolicOmegaMesh(omega_min=-20, omega_max=20, n_points=100)
alpha_max = 5e4 / 10001
alpha_min = 1e-1 / 10001
tm.alpha_mesh = LogAlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=10)

tm.set_G_tau(G_tau_noisy)
tm.set_error(5*std_dev)
result = tm.run()
plt.figure()
result.analyzer_results['LineFitAnalyzer'].plot_linefit()
alpha_index = result.analyzer_results['LineFitAnalyzer']['alpha_index']
plt.figure()
result.plot_A(alpha_index = alpha_index)

with HDFArchive('maxent_test_k0.h5', 'w') as B:
    B['G_tau_'] = result.data