from TPSC_TRIQS_library import *

# parameters
num_iter = 10
n = 1.0
U = 5.0
beta = 15.0
eps_k = dispersion_relation(basis=[(1,0,0),(0,1,0)], t=1.0, n_k=128).eps_k

# figures
fig1, ax1 = plt.subplots(1,1)
fig2, ax2 = plt.subplots(1,1)
fig3, ax3 = plt.subplots(1,1)
w_max = 40

# initialize results
mus = []
Usps = []
Uchs = []

# TPSC solver (g0 from scratch)
model = tpsc_solver(n=n, U=U, beta=beta, eps_k=eps_k, verbose=False, plot=False)

for k in range(num_iter):

    try:
        # print progress
        print('Iteration = ' + str(k+1))
        
        # initialize model with G2 as new g0
        if k != 0:
            model.g0_bubble = model.G2
        model.run()

        # write parameters to files
        mus.append(model.mu2)
        Usps.append(model.Usp)
        Uchs.append(model.Uch)

        G_w = model.k_sum(model.G2)
        S_w = model.k_sum(model.Sigma2_dlr_wk)
        iw = np.array([iw.imag for iw in G_w.mesh])

        # plot
        ax1.plot(iw, np.squeeze(G_w.data.real), 'p', label='iteration '+str(k+1))
        ax2.plot(iw, np.squeeze(G_w.data.imag), 'p', label='iteration '+str(k+1))
        ax3.plot(iw, np.squeeze(S_w.data.imag), 'p', label='iteration '+str(k+1))

    except:
        print('Something went wrong during iteration %i!'%k)
        print('')
        plt.figure()
        U = np.linspace(0,10,30)
        f = np.vectorize(model.Usp_root)
        plt.plot(U,f(U))
        plt.title('Usp_root')
        plt.figure()
        g = np.vectorize(model.Uch_root)
        plt.plot(U,g(U))
        plt.title('Uch_root')
        break

# fancy graphics
ax1.set_title('Real part')
ax1.set_xlabel('$i\\omega_{n}$')
ax1.set_xlim([0,w_max])    
ax1.legend()
ax2.set_title('Imaginary part')
ax2.set_xlabel('$i\\omega_{n}$')
ax2.set_xlim([0,w_max]) 
ax2.legend()
ax3.set_title('Imaginary part of the self-energy')
ax3.set_xlabel('$i\\omega_{n}$')
ax3.set_xlim([0,w_max]) 
ax3.legend()