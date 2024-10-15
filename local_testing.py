from TPSC_TRIQS_library import *

model = tpsc_solver(plot=False)
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
    Sigma_dlr_nonlocal = model.subtract_local_gf(Sigma_dlr_wk, Sigma_dlr_local)
    Sigma_coeff_nonlocal = make_gf_dlr(Sigma_dlr_nonlocal)
    Sigma_nonlocal_from_dlr = make_gf_imfreq(Sigma_coeff_nonlocal, n_iw)

    # conversersion first
    Sigma_dlr_k = make_gf_dlr(Sigma_dlr_wk)
    Sigma_wk_from_dlr = make_gf_imfreq(Sigma_dlr_k, n_iw)
    Sigma_local_from_dlr = model.k_sum(Sigma_wk_from_dlr)
    Sigma_nonlocal = model.subtract_local_gf(Sigma_wk_from_dlr, Sigma_local_from_dlr)

    # plot
    oplot(model.k_sum(Sigma_nonlocal_from_dlr), '-', label='conversersion at end')
    oplot(model.k_sum(Sigma_nonlocal), '--', label='conversion at beginning')
    plt.title('diffnorm of k-dependent nonlocals = ' + str(np.linalg.norm(Sigma_nonlocal_from_dlr.data - Sigma_nonlocal.data)))

### compare local Green's functions
if False:

    # information on mu
    print('mu2 = ' + str(model.mu2))
    print('mu_phys = ' + str(model.mu2_phys))

### with Hartree-term
if False:
    Sigma_dlr_k = make_gf_dlr(Sigma_dlr_wk)
    Sigma_wk = make_gf_imfreq(Sigma_dlr_k, n_iw)
    Sigma_Hartree = Sigma_wk.copy()
    Sigma_Hartree.zero()
    Sigma_Hartree.data[:] = model.U * model.n/2
    Sigma_full_wk = Sigma_wk.copy()
    Sigma_full_wk.zero()
    Sigma_full_wk = Sigma_Hartree + Sigma_wk

    oplot(model.k_sum(Sigma_full_wk))

### non-local
Sigma_dlr_nonloc_wk = model.get_nonlocal_gf(Sigma_dlr_wk)
oplot(model.k_sum(Sigma_dlr_nonloc_wk))
plt.title(np.linalg.norm(model.k_sum(Sigma_dlr_nonloc_wk).data))