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
    max_val = np.max(np.abs(G_tau.data.real)) + 1j*np.max(np.abs(G_tau.data.imag))
    # get standard deviation
    std_dev = percent/100*max_val

    # Generate Gaussian noise for real and imaginary parts
    real_noise = np.random.normal(0.0, std_dev, G_tau.data.shape)
    imag_noise = np.random.normal(0.0, std_dev, G_tau.data.shape)

    # Combine real and imaginary parts to form complex noise
    complex_noise = real_noise + 1j * imag_noise
    
    # get noisy G_tau
    G_tau_noisy = G_tau.copy()
    G_tau_noisy.data[:] += complex_noise

    # return result
    return G_tau_noisy

# run model
model = tpsc_solver(U=5, plot=False)
model.run()

# Get G_tau
G_dlr_k = make_gf_dlr(model.G2)
G_tk = make_gf_imtime(G_dlr_k, n_tau=2001)
G_tau = G_tk(all,(0,0,0))

G_tau_noisy = add_gaussian_noise(G_tau, 2)

plt.figure()
oplot(G_tau_noisy, '--')
oplot(G_tau)


tm = TauMaxEnt(cost_function='bryan', probability='normal')
tm.set_G_tau(G_tau_noisy)
tm.set_error(10*1e-5)
result = tm.run()
result.plot_A()