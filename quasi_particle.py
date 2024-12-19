### imports
from TPSC_TRIQS_library import *
from triqs_maxent import *
from triqs_maxent.sigma_continuator import *
from scipy.interpolate import CubicSpline, UnivariateSpline, lagrange
from scipy.integrate import simpson

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
    max_val = np.max(np.abs(G_tau.data))
    # get standard deviation
    std_dev = percent/100*max_val

    # Generate Gaussian noise for real part
    real_noise = np.random.normal(0.0, std_dev, G_tau.data.shape)
    imag_noise = np.random.normal(0.0, std_dev, G_tau.data.shape)

    # get noisy G_tau
    G_tau_noisy = G_tau.copy()
    G_tau_noisy.data[:] += real_noise + 1j*imag_noise

    # return result
    return G_tau_noisy, std_dev


### fit parameters
n_max = 5

# parameters
U_list = [1.0,2.0,3.0,4.0,5.0]
beta_list = [round(b,3) for b in np.linspace(1,30,20)]
n_list = [0.875, 1.0]


for U in U_list:
    for beta in beta_list:
        for n in n_list:

            file_name = 'U' + str(U) + '_beta' + str(beta) + '_n' + str(n) + '_fit0.h5'


            ### run the solver and get Sigma_iw
            solver = tpsc_solver(n=n, U=U, beta=beta, plot=False)
            solver.run()

            Sigma_dlr_k = make_gf_dlr(solver.Sigma2_dlr_wk)


            ### do maxent

            # get sigma
            Sigma_iwk = make_gf_imfreq(Sigma_dlr_k, n_iw=1025)
            Sigma_iw0 = Sigma_iwk(all, (0,0,0))

            # add noise
            Sigma_iw0_noisy, std_dev = add_gaussian_noise(Sigma_iw0, 2)

            # initialize continuator
            isc = InversionSigmaContinuator(Sigma_iw0_noisy)
            tm = TauMaxEnt()

            # set parameters
            tm.set_G_iw(isc.Gaux_iw)
            tm.set_error(5*std_dev)
            tm.omega = HyperbolicOmegaMesh(omega_min=-25, omega_max=25, n_points=100)
            tm.alpha_mesh = LogAlphaMesh(alpha_min=1e-4, alpha_max=1e2, n_points=20)

            # run continuation
            print()
            result = tm.run()

            # get real-freq Gaux
            Aaux_w = result.analyzer_results['LineFitAnalyzer']['A_out']
            isc.set_Gaux_w_from_Aaux_w(Aaux_w, result.omega)

            # get Sigma on real axis
            Sigma_w = isc.S_w

            # plot Aaux_w
            plt.figure()
            plt.plot(result.omega, result.analyzer_results['LineFitAnalyzer']['A_out'])
            plt.title('Integral = ' + str(simpson(Aaux_w, result.omega)))

            # plot Sigma
            plt.figure()
            oplot(Sigma_w)

            # save result
            with HDFArchive(file_name, 'w') as A:
                A['result'] = result.data
                A['Sigma_w'] = Sigma_w


            ### gets Z and tau_inv
            dw = 1
            Sigma_r_der = 100

            # get derivative
            while True:

                # get next iteration
                der_next = (Sigma_w(dw).real[0,0] - Sigma_w(-dw).real[0,0])/(2*dw)
                
                # compare to previous iteration
                if abs(der_next - Sigma_r_der) < 1e-8:
                    break

                # set up next iteration
                Sigma_r_der = der_next
                dw /= 10

            Z = 1 / (1 - Sigma_r_der)
            tau_inv = -Z * Sigma_w(1e-20).imag[0,0]

            print()
            print('Quantities from positive Spline Fit:')
            print('Z = ' + str(Z))
            print('tau_inv = ' + str(tau_inv))



            ### do fits
            Sigma_wk = make_gf_imfreq(Sigma_dlr_k, n_iw=n_max)
            Sigma_w0 = Sigma_wk(all,(0,0,0))

            ### get data for interpolation
            pos_iw_n = np.array([(2*n+1)*np.pi/solver.beta for n in range(n_max)])
            pos_Im_Sigma = np.squeeze(np.array([float(Sigma_w0(n).imag) for n in range(n_max)]))
            neg_iw_n = np.array([(2*n+1)*np.pi/solver.beta for n in range(-n_max,0)])
            neg_Im_Sigma = np.squeeze(np.array([float(Sigma_w0(n).imag) for n in range(-n_max,0)]))


            ### fits
            pos_fit = CubicSpline(pos_iw_n, pos_Im_Sigma)
            neg_fit = CubicSpline(neg_iw_n, neg_Im_Sigma)

            pos_fit_2 = lagrange(pos_iw_n, pos_Im_Sigma)
            neg_fit_2 = lagrange(neg_iw_n, neg_Im_Sigma)


            ### plot
            pos_w = np.linspace(0, (2*n_max-1)*np.pi/solver.beta, 100)
            neg_w = np.linspace(-(2*n_max-1)*np.pi/solver.beta, 0, 100)
            plt.figure()
            plt.plot(np.concatenate((neg_iw_n, pos_iw_n)),
                    np.concatenate((neg_Im_Sigma, pos_Im_Sigma)), 'p', label='Data')
            plt.plot(np.concatenate((neg_w, pos_w)), 
                    np.concatenate((neg_fit(neg_w), pos_fit(pos_w))), '-', label='Spline')
            plt.plot(np.concatenate((neg_w, pos_w)), 
                    np.concatenate((neg_fit_2(neg_w), pos_fit_2(pos_w))), '--', label='Lagrange')
            plt.title('$\\Sigma(i\\omega \\rightarrow 0)$')
            plt.legend(bbox_to_anchor=(1,1))


            # print fit parameters
            print()
            print('Spline, positive fit: Sigma(iw_n->0) = ' + str(round(float(pos_fit(0)),5)))
            print('derivative at w=0: ' + str(round(float(pos_fit.derivative()(0)),5)))
            print('Spline, negative fit: Sigma(iw_n->0) = ' + str(round(float(neg_fit(0)),5)))
            print('derivative at w=0: ' + str(round(float(neg_fit.derivative()(0)),5)))
            print()
            print('Lagrange, positive fit: Sigma(iw_n->0) = ' + str(round(float(pos_fit_2(0)),5)))
            print('derivative at w=0: ' + str(round(float(np.poly1d.deriv(pos_fit_2)(0)),5)))
            print('Lagrange, negative fit: Sigma(iw_n->0) = ' + str(round(float(neg_fit_2(0)),5)))
            print('derivative at w=0: ' + str(round(float(np.poly1d.deriv(neg_fit_2)(0)),5)))

            # set up lists --> first entry is MaxEnt result
            Z_fits = [Z]
            tau_inv_fits = [tau_inv]

            # get quasi-partice properties
            Z_curr = 1 / (1 - float(pos_fit.derivative()(0)))
            Z_fits.append(Z_curr)
            tau_inv_fits.append(-Z_curr * pos_fit(0))

            Z_curr = 1 / (1 - float(neg_fit.derivative()(0)))
            Z_fits.append(Z_curr)
            tau_inv_fits.append(Z_curr * neg_fit(0))

            Z_curr = 1 / (1 - float(np.poly1d.deriv(pos_fit_2)(0)))
            Z_fits.append(Z_curr)
            tau_inv_fits.append(-Z_curr * pos_fit_2(0))

            Z_curr = 1 / (1 - float(np.poly1d.deriv(neg_fit_2)(0)))
            Z_fits.append(Z_curr)
            tau_inv_fits.append(Z_curr * neg_fit_2(0))

            # save to archive
            with HDFArchive(file_name, 'a') as A:
                A['Zs'] = Z_fits
                A['tau_invs'] = tau_inv_fits