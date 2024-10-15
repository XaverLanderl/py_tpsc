from TPSC_TRIQS_library import *

def subtract_local_gf(g_iw_k, g_local):
    """
    Subtracts a local (k-independent) Green's function from a k-dependent Green's function.

    Parameters
    ----------
    self    : self
    g_iw_k  : k-dependent Green's function; iw must be on first axis.
    g_local : k-independent Green's function to be subtracted
            : must both have the same Matsubara-mesh

    Returns
    -------
    result  : TRIQS Green's function object
    """

    # extract data of k-dependent gf
    g_iw_k_data = np.squeeze(g_iw_k.data)

    # extract data of local gf; reshape for broadcasting reasons
    g_local_data = np.squeeze(g_local.data).reshape(-1,1)

    # initialize result
    result = Gf(mesh = g_iw_k.mesh, target_shape=(1,1))
    
    # feed values
    result.data[:,:,0,0] = g_iw_k_data - g_local_data

    # return result
    return result

model = tpsc_solver(n=0.875, plot=False)
model.run()



# get self-energy
Sigma_dlr_wk = model.Sigma2_dlr_wk
n_iw = 1025

### compare difference of order (dlr <-> imfreq conversion & k_sum)
if False:
    # sum first, then imfreq
    Sigma_dlr_w = model.k_sum(Sigma_dlr_wk)
    Sigma_dlr = make_gf_dlr(Sigma_dlr_w)
    Sigma_w_from_dlr = make_gf_imfreq(Sigma_dlr, n_iw)

    # imfreq first, then sum
    Sigma_dlr_k = make_gf_dlr(Sigma_dlr_wk)
    Sigma_wk_from_dlr = make_gf_imfreq(Sigma_dlr_k, n_iw)
    Sigma_w = model.k_sum(Sigma_wk_from_dlr)

    # compare
    oplot(Sigma_w_from_dlr, '-', label='sum first')
    oplot(Sigma_w, '--', label='imfreq first')
    plt.title('diffnorm = ' + str(np.linalg.norm(Sigma_w_from_dlr.data - Sigma_w.data)));

### compare computational effort (n_iw)
if False:
    n_iw_s_list = 2**np.arange(6, 12)
    runtimes = []
    for n_iw_s in n_iw_s_list:
        start = time.time()
        temp_w_from_dlr = make_gf_imfreq(Sigma_dlr, n_iw_s)
        end = time.time()
        runtimes.append(end-start)

    plt.figure()
    plt.plot(n_iw_s_list, runtimes)
    plt.xlabel('$n_{i\\omega}$')
    plt.ylabel('s')
    plt.title('dlr -> imfreq runtimes');

### compare difference of order (dlr <-> imfreq conversion & subtraction of local part)
if False:
    # with dlr, converersion at the end
    Sigma_dlr_local = model.k_sum(Sigma_dlr_wk)
    Sigma_dlr_nonlocal = subtract_local_gf(Sigma_dlr_wk, Sigma_dlr_local)
    Sigma_coeff_nonlocal = make_gf_dlr(Sigma_dlr_nonlocal)
    Sigma_nonlocal_from_dlr = make_gf_imfreq(Sigma_coeff_nonlocal, n_iw)

    # conversersion first
    Sigma_dlr_k = make_gf_dlr(Sigma_dlr_wk)
    Sigma_wk_from_dlr = make_gf_imfreq(Sigma_dlr_k, n_iw)
    Sigma_local_from_dlr = model.k_sum(Sigma_wk_from_dlr)
    Sigma_nonlocal = subtract_local_gf(Sigma_wk_from_dlr, Sigma_local_from_dlr)

    # plot
    oplot(model.k_sum(Sigma_nonlocal_from_dlr), '-', label='conversersion at end')
    oplot(model.k_sum(Sigma_nonlocal), '--', label='conversion at beginning')
    plt.title('diffnorm of k-dependent nonlocals = ' + str(np.linalg.norm(Sigma_nonlocal_from_dlr.data - Sigma_nonlocal.data)))

### compare local Green's functions
if True:

    # information on mu
    print('mu2 = ' + str(model.mu2))
    print('mu_phys = ' + str(model.mu2 + model.U*model.n/2))
