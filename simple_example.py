from TPSC_TRIQS_library import *

model = tpsc_solver(plot=False)
model.run()
print()

Sigma_dlr_wk = model.Sigma2_dlr_wk
Sigma_dlr_k = make_gf_dlr(Sigma_dlr_wk)
Sigma_wk = make_gf_imfreq(Sigma_dlr_k, 1025)

if False:
    Sigma_dlr_wk = model.Sigma2_dlr_wk
    Sigma_dlr_k = make_gf_dlr(Sigma_dlr_wk)
    Sigma_wk = make_gf_imfreq(Sigma_dlr_k, 128)
    model.calc_G_from_Sigma(Sigma_wk)
    oplot(model.k_sum(Sigma_wk), x_window=(0,30))
    Sigma_dlr_wk = model.Sigma2_dlr_wk
    Sigma_dlr_k = make_gf_dlr(Sigma_dlr_wk)
    Sigma_wk = make_gf_imfreq(Sigma_dlr_k, 1025)
    model.calc_G_from_Sigma(Sigma_wk)
    oplot(model.k_sum(Sigma_wk), '--', x_window=(0,30))

if False:
    N = 100
    start = time.time()
    for k in range(N):
        x = model.k_sum(Sigma_wk)
    end1 = time.time()
    for k in range(N):
        x = model.k_sum(Sigma_dlr_wk)
    end2 = time.time()

    print('k-sums runtimes:')
    print('runtime no DLR = ' + str((end1 - start)/N) + 's')
    print('runtime with DLR = ' + str((end2 - end1)/N) + 's')
    print()


    Sigma_dlr_w = model.k_sum(Sigma_dlr_wk)
    Sigma_w = model.k_sum(Sigma_wk)

    start = time.time()
    for k in range(N):
        x = density(Sigma_w)
    end1 = time.time()
    for k in range(N):
        x = density(Sigma_dlr_w)
    end2 = time.time()

    print('iw-sums runtimes:')
    print('runtime no DLR = ' + str((end1 - start)/N) + 's')
    print('runtime with DLR = ' + str((end2 - end1)/N) + 's')
    print()

# compare runtimes with different n_iw
N = 100
runtimes=[]
n_iw_list = [128, 256, 512, 1024, 2048]
for n_iw in n_iw_list:
    Sigma_wk_temp = make_gf_imfreq(Sigma_dlr_k, n_iw)
    start = time.time()
    for l in range(N):
        x = model.k_sum(Sigma_wk_temp)
    end = time.time()
    runtimes.append((end-start)/N)
plt.plot(n_iw_list, runtimes)