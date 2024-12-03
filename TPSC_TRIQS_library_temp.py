### IMPORTS ###
from Gf_Utils import *
### IMPORTS ###


### TPSC SOLVER CLASS ###
class tpsc_solver:
    """
    Class to solve TPSC.
    
    Parameters
    ----------
    Model Parameters:
    double n        : total electron density
        default     : n = 1.0 (half-filling)
    double U        : Hubbard U
        default     : U = 2.0
    double beta     : inverse temperature
        default     : beta = 2.5
    Gf eps_k        : Dispersion relation (TRIQS GF object, mesh=BrZoneMesh)
        default     : 2D square lattice nearest neighbour only TB model on a 128x128 k_mesh
    double docc     : double occupation <n_up*n_down>
        default     : docc = None
                    : if not None, will ignore TPSC-Ansatz and use passed docc to calculate the sum rules
    Gf g0_bubble    : non-interacting Green's function, with which the bubble is formed
        default     : g0_bubble = None
                    : if None, will calculate g0 from scratch
 
    Calculation Parameters:
    float w_max     : maximum frequency for Discrete Lehmann Representation (DLR)
        default     : w_max = 10.0
    float eps       : accuracy of DLR
        default     : eps = 1e-14
    double Usp_tol  : accuracy of solution of Usp
        default     : Usp_tol = 1e-12
    double Uch_tol  : accuracy of solution of Uch
        default     : Uch_tol = None, sets Uch_tol = Usp_tol
    
    Boolean Parameters:
    bool verbose    : if True, running the method will print out detailed information
        default     : verbose = True
    bool plot       : if True, plots local spectral function
        default     : plot = True
    """
    
    ### CLASS CONSTRUCTOR ###
    def __init__(self, n=1., U=2., beta=2.5, eps_k=None, docc=None, w_max=10.0, eps=1e-14, Usp_tol=1e-12, Uch_tol=None, verbose=True, plot=True):
        """
        Initialize a tpsc_solver object.
        """
        
        ### SET PARAMETERS
        # set model pareameters
        self.n = n
        self.U = U
        self.beta = beta
        
        # get dispersion relation
        if eps_k == None:
            # run with default values from dispersion_relation class
            disp = dispersion_relation()
            self.eps_k = disp.eps_k
        else:
            self.eps_k = eps_k

        # check if TPSC-Ansatz or docc are to be used in the sum rules
        if docc == None:
            self.use_tpsc_ansatz = True
        else:
            self.use_tpsc_ansatz = False
            self.docc = docc

        # set calculation parameters
        self.w_max = w_max
        self.eps = eps
        self.Usp_tol = Usp_tol
        if Uch_tol == None:
            self.Uch_tol = Usp_tol
        else:
            self.Uch_tol = Uch_tol
            
        # set other parameters
        self.verbose = verbose
        self.plot = plot
    ### CLASS CONSTRUCTOR ###        
    

    ### METHODS TO RUN THE CALCULATION ###
    def run(self):
        """
        Runs the TPSC-Calculation on the specified model.
        """

        # calculate first and second levels of approximation
        t1 = time.time()
        self.calc_first_level_approx()
        t2 = time.time()
        self.calc_second_level_approximation()
        t3 = time.time()

        # plot Self-energy
        if self.plot == True:
            self.plot_spectral_function(n_iw=1025)
            #self.plot_Sigma_zero_frequency_lagrange()
        t4 = time.time()

        # do self-consistency check
        self.check_for_self_consistency()

        # End of calculation
        self.vprint()
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint(f'Runtime of first level approximation = {(t2 - t1):.2f}s.')
        self.vprint(f'Runtime of second level approximation = {(t3 - t2):.2f}s.')
        if self.plot == True:
            self.vprint(f'Runtime of plots = {(t4 - t3):.2f}s.')
        self.vprint("DONE!")

    def calc_first_level_approx(self, Uch_max=1000.):
        """
        Runs the first-level approximation of the TPSC-Calculation on the specified model.
        If docc is specified in the model, it will be used to evaluate the sum rules.
        If docc is not specified in the model, the TPSC-Ansatz will be used to evaluate the sum rules.

        Parameters
        ----------
        self            : self
        double Uch_max  : upper bound for Uch
            default     : Uchmax = 1000.
        
        Returns
        -------
        Sets self.Usp, self.Uch and self.docc, if not included in model parameters

        """
        
        # begin calculation
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint("Calculate first-level approximation:")
        if self.use_tpsc_ansatz == True:
            self.vprint()
            self.vprint("   Double Occupation was not specified, will use the TPSC-Ansatz to evaluate the sum rules.")
        else:
            self.vprint()
            self.vprint("   Double Occupation was specified, will use it to evaluate the sum rules and ignore the TPSC-Ansatz.")
             
        # start with non-interacting calculation
        self.vprint()
        self.vprint("   Calculating non-interacting susceptibility...")
        self.calc_noninteracting_gf()
        
        # calculate vertices
        self.vprint()
        self.vprint("   Calculating Usp...")
        self.Usp = self.calc_Usp(chi0_dlr_wk=self.chi0_dlr_wk)
        
        self.vprint()
        self.vprint("   Calculating Uch...")
        self.Uch = self.calc_Uch(chi0_dlr_wk=self.chi0_dlr_wk, Usp=self.Usp, Uch_max=Uch_max)

        # calculate susceptibilities
        self.vprint()
        self.vprint("   Calculate TPSC-spin- and charge-susceptibilities...")
        self.chi1_sp_dlr_wk = solve_Hubbard_RPA(chi0_wk=self.chi0_dlr_wk, U=self.Usp)
        self.chi1_ch_dlr_wk = solve_Hubbard_RPA(chi0_wk=self.chi0_dlr_wk, U=-self.Uch)
        
        # calculate double occupation
        if self.use_tpsc_ansatz == True:
            self.vprint()
            self.vprint("   Calculate double occupation...")
            self.docc = self.calc_docc(Usp=self.Usp)
        
        # print out results
        self.vprint()
        self.vprint("Summary first level approximation:")
        self.vprint("Usp = " + str(self.Usp) + ", Uch = " + str(self.Uch))
        self.vprint("Double Occupation <n_up*n_down> = " + str(self.docc))
    
    def calc_second_level_approximation(self):
        """
        Runs the second-level approximation of the TPSC-Calculation on the specified model.

        Requires
        --------
        First-level approximation must have been run.

        Parameters
        ----------
        self    : self

        Returns
        -------
        Sets    : self.Sigma_dlr_wk,
                  self.mu2 (WITHOUT Hartree term),
                  self.mu2_phys (WITH Hartree term),
                  self.G2
        """

        # start calculation
        self.vprint()
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint("Calculate second-level approximation:")

        # calculate self-energy
        self.vprint()
        self.vprint("   Calculate self-energy...")
        self.calc_Sigma()
        
        # calculate G2
        self.vprint()
        self.vprint("   Calculate Green's function...")
        self.g2_dlr_wk, self.mu2 = self.calc_G_from_Sigma(Sigma_dlr_wk=self.Sigma2_dlr_wk)
        self.mu2_phys = self.mu2 + self.U*self.n/2          # add Hartree term
        self.vprint()
        self.vprint("Summary second level approximation:")
        self.vprint("mu^(2) = " + str(self.mu2_phys))

    def check_for_self_consistency(self):
        """
        Checks the evaluated model for self-consistency.
        
        Requires
        --------
        Must have calculated chi_sp and chi_ch.
        
        Parameters
        ----------
        self    : self

        Returns
        -------
        None

        """
        
        # check that the sum rule is fulfilled
        self.vprint()
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint('Doing self-consistency check of first-level approximation...')
        self.check_sum_rule = k_iw_sum(self.chi1_sp_dlr_wk + self.chi1_ch_dlr_wk) - (2*self.n - self.n**2)
        self.vprint(f'The sum rule is fulfilled with an accuracy of {float(abs(self.check_sum_rule))}.')

        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint('Doing self-consistency check of second-level approximation...')
        # add Hartree term to self-energy
        Sigma_dlr_wk = self.Sigma2_dlr_wk.copy()
        Sigma_dlr_wk.data[:] += self.U*self.n/2

        # get products Sigma*G
        F1_dlr_wk = Sigma_dlr_wk * self.g0_dlr_wk
        F2_dlr_wk = Sigma_dlr_wk * self.g2_dlr_wk

        # get traces
        trace_F1 = k_iw_sum(F1_dlr_wk) / self.U
        trace_F2 = k_iw_sum(F2_dlr_wk) / self.U

        # get the relative difference
        rel_diff = float(np.abs((trace_F1 - trace_F2) / trace_F1) * 100)

        # check consistency
        self.vprint('Sum_k {Sigma^(2)(k) * G^(1)(k)} - <n_up * n_down> = ' + str(float(np.abs(trace_F1 - self.docc))))
        self.vprint(f'Relative difference of traces = {rel_diff:.2f}%')
    ### METHODS TO RUN THE CALCULATION ###


    ### HELPER FUNCTIONS ###
    def vprint(self, *args):
        if self.verbose == True:
            if len(args) > 0:
                mpi.report(args[0])
            else:
                mpi.report('')
            
    def change_target_shape(self, g_dlr_wk):
        """
        Changes target_shape from (1,1,1,1) to (1,1) if necessary, does nothing otherwise.

        Parameters
        ----------
        self        : self
        g_dlr_wk    : TRIQS Green's function object

        Returns
        -------
        TRIQS Green's function object with changed target_shape
        """ 

        # extract mesh and target_shape
        mesh = g_dlr_wk.mesh
        target_shape = g_dlr_wk.target_shape
        
        # change target_shape and transfer data
        if target_shape == (1,1,1,1):
            result = Gf(mesh=mesh, target_shape=(1,1))
            result.data[:,:,:,:] = g_dlr_wk.data[:,:,:,:,0,0]
        else:
            result = g_dlr_wk
            
        # return new gf-object
        return result
    ### HELPER FUNCTIONS ###


    ### FIRST LEVEL OF APPROXIMATION ###
    def calc_noninteracting_gf(self):
        """
        Calculates the non-interacting quantities.
        
        Parameters
        ----------
        self    : self
        
        Returns
        -------
        Sets    :   if g0 is calculated from scratch:
                        self.iw_dlr_mesh, imaginary times DLR mesh
                        self.mu1,         non-interacting chemical potential
                        self.g0_dlr_wk,   non-interacting Green's function with correct chemical potential
                    always:
                    self.chi0_dlr_wk, target_shape=(1,1)
        """

        # calculate imaginary time mesh
        self.iw_dlr_mesh = MeshDLRImFreq(beta=self.beta, statistic='Fermion', w_max=self.w_max, eps=self.eps)

        # get mu from n (need for G0 and bubble)
        self.mu1 = calc_mu(dens=self.n, eps_k=self.eps_k, Sigma_wk=self.iw_dlr_mesh)
        
        # calculate non-interacting Green's function of the model
        self.g0_dlr_wk = lattice_dyson_g0_wk(mu=self.mu1, e_k=self.eps_k, mesh=self.iw_dlr_mesh)

        # calculate the non-interacting susceptibility of the model
        chi0_dlr_wk = 2*imtime_bubble_chi0_wk(self.g0_dlr_wk, nw=2, verbose=False)

        # change target_shape (we need (1,1))
        self.chi0_dlr_wk = self.change_target_shape(g_dlr_wk=chi0_dlr_wk)

    def imtime_bubble_chi2_wk(self):
        """
        Calculates chi2(r,tau) = - G2(r,tau)*G0(-r,-tau) - G2(-r,-tau)*G0(r,tau) in wk-space.

        Requires
        --------
        self.g0_dlr_wk and self.g2_dlr_wk must have been calculated

        Parameters
        ----------
        self                :   self

        Returns
        -------
        self.chi2_dlr_wk    :   second-level approximation of bubble in TPSC
        """

        # Fourier transform Gs to real space
        G2_dlr_tr = fourier_wk_to_tr(self.g2_dlr_wk)
        G2_dlr_mtr = fourier_wk_to_mtr(self.g2_dlr_wk)
        G0_dlr_tr = fourier_wk_to_tr(self.g0_dlr_wk)
        G0_dlr_mtr = fourier_wk_to_mtr(self.g0_dlr_wk)

        # initialize chi2_dlr_tr
        chi2_dlr_tr = fourier_wk_to_tr(self.chi0_dlr_wk.copy())

        # calculate chi2
        chi2_dlr_tr.data[:] = -G2_dlr_tr.data*G0_dlr_mtr.data - G2_dlr_mtr.data*G0_dlr_tr.data
        self.chi2_dlr_wk = fourier_tr_to_wk(chi2_dlr_tr)

    def Usp_root(self, chi0_dlr_wk, Usp):
        """
        Function whose root is the self-consistent value for Usp.
        
        Parameters
        ----------
        self        : self
        chi0_dlr_wk : non-interacting susceptibility
        double Usp  : given value of spin vertex
        
        Returns
        -------
        double      : spin sum over chi_RPA(Usp) - sum rule (Usp)
        """
        
        # calculate sum over RPA-like susceptibility
        chi_sum = k_iw_sum(solve_Hubbard_RPA(chi0_wk=chi0_dlr_wk, U=Usp))
        
        # calculate spin sum rule
        if self.use_tpsc_ansatz == True:
            sum_rule = self.n - Usp/self.U*self.n*self.n/2
        else:
            sum_rule = self.n - 2*self.docc

        # return difference
        return chi_sum - sum_rule
    
    def Uch_root(self, chi0_dlr_wk, Usp, Uch):
        """
        Function whose root is the self-consistent value for Uch.
        
        Parameters
        ----------
        self        : self
        chi0_dlr_wk : non-interacting susceptibility
        double Usp  : (previously determined) spin vertex
        double Uch  : given value of charge vertex
        
        Returns
        -------
        double      : charge sum over chi_RPA(Uch) - sum rule (Uch)
        """
        
        # calculate sum over RPA-like susceptibility
        chi_sum = k_iw_sum(solve_Hubbard_RPA(chi0_wk=chi0_dlr_wk, U=-Uch))
        
        # calculate charge sum rule
        if self.use_tpsc_ansatz == True:
            sum_rule = self.n + Usp/self.U*self.n*self.n/2 - self.n*self.n
        else:
            sum_rule = self.n + 2*self.docc - self.n*self.n
        
        # return the difference
        return chi_sum - sum_rule
    
    def calc_Usp(self, chi0_dlr_wk):
        """
        Calculates Usp self-consistently to obey spin sum rule.
        
        Parameters
        ----------
        self        : self
        chi0_dlr_wk : non-interacting susceptibility

        Returns
        -------
        double      : Usp

        """

        # set maximum value of Usp (where chi diverges)
        Usp_max = 2.0/np.amax(chi0_dlr_wk.data).real - 1e-7 # the 1e-7 is chosen for numerical stability.
        
        # calculate Usp self-consistently
        Usp = brentq(lambda x: self.Usp_root(chi0_dlr_wk=chi0_dlr_wk, Usp=x), 0.0, Usp_max, xtol=self.Usp_tol)

        # return result
        return Usp
        
    def calc_Uch(self, chi0_dlr_wk, Usp, Uch_max=1000.):
        """
        Calculates Uch self-consistently to obey charge sum rule.
        
        Parameters
        ----------
        self        : self
        chi0_dlr_wk : non-interacting susceptibility
        Usp         : spin-vertex required for sum rule
        Uch_max     : maximum search value for Uch (default = 1000.)

        Returns
        -------
        double  : Uch

        """
        
        # calculate Usp self-consistently
        Uch = brentq(lambda x: self.Uch_root(chi0_dlr_wk=chi0_dlr_wk, Usp=Usp, Uch=x), 0.0, Uch_max, xtol=self.Uch_tol)
        
        # return result
        return Uch
    
    def calc_docc(self, Usp):
        """
        Calculates the double occupation.
        
        Parameters
        ----------
        self    : self
        Usp     : spin vertex
        
        Returns
        -------
        double  : docc
        """
        
        # calculate and return result
        return Usp/self.U*self.n*self.n/4
    ### FIRST LEVEL OF APPROXIMATION ###


    ### SECOND LEVEL OF APPROXIMATION ###
    def calc_Sigma(self):
        """
        Calculates the second-level approximation of the self-energy.

        Requires
        --------
        Susceptibilities must have been calculated

        Parameters
        ----------
        self    : self

        Returns
        -------
        Sets self.Sigma (WITHOUT the Hartree-term!)
        """

        # define effective potential
        V_dlr_wk = self.U/8*(3*self.Usp*self.chi1_sp_dlr_wk + self.Uch*self.chi1_ch_dlr_wk)

        # get V(-t,-r)
        V_dlr_mtr = fourier_wk_to_mtr(V_dlr_wk)

        # get G(t,r)
        g0_dlr_tr = fourier_wk_to_tr(self.g0_dlr_wk)

        # multiply V(-t,-r) * G0(t,r) = Sigma(t,r)
        Sigma2_dlr_tr = g0_dlr_tr.copy()    # the 2 means second level of approximation, must be fermionic
        Sigma2_dlr_tr.data[:] = V_dlr_mtr.data * g0_dlr_tr.data
        
        # transform Sigma(t,r) to Sigma(w,k)
        self.Sigma2_dlr_wk = fourier_tr_to_wk(Sigma2_dlr_tr)

    def calc_G_from_Sigma(self, Sigma_dlr_wk):
        """
        Calculates the Green's function given a self-energy.

        Requires
        --------
        Sigma_dlr_wk must be given on a MeshDLRImFreq.

        Parameters
        ----------
        self            : self
        Sigma_dlr_wk    : self-energy

        Returns
        -------
        G_dlr_wk        : Green's function with given self-energy and correct chemical potential
        mu2             : chemical potential
        """
        
        # get mesh
        mesh = Sigma_dlr_wk.mesh.components[0]

        # calculate mu
        mu = calc_mu(dens=self.n, eps_k=self.eps_k, Sigma_wk=Sigma_dlr_wk)

        # calculate G
        g0_dlr_wk_inv = inverse(lattice_dyson_g0_wk(mu=mu, e_k=self.eps_k, mesh=mesh))
        G = inverse(g0_dlr_wk_inv - Sigma_dlr_wk)

        # return results
        return G, mu
    ### SECOND LEVEL OF APPROXIMATION ###


    ### FUNCTIONS FOR ANALYTIC CONTINUATION AND FITTING ###
    def evaluate_G_real_freq(self, G_Matsubara, window=(-10, 10), n_w=1000):
        """
        Analytically continues a Matsubara Green's function G to the real axis via TRIQS' Pade approximation.

        Requires
        --------
        G.mesh must be MeshImFreq (not DLR!).

        Parameters
        ----------
        self        : self
        G           : Green's function
        window      : frequency range of real-frequency Green's function
        n_w         : number of frequencies in MeshReFreq

        Returns
        -------
        Real-frequency Gf of same target shape as input.
        """

        # get mesh
        w_mesh = MeshReFreq(window=window, n_w=n_w)

        # Initialize ReFreq Gf
        G_omega = Gf(mesh=w_mesh, target_shape=G_Matsubara.target_shape)

        # perform pade fit
        G_omega.set_from_pade(G_Matsubara)

        # return resulg
        return G_omega

    def evaluate_non_int_spectral_function(self, n_iw=128, window=(-10, 10), n_w=1000):
        """
        Calculates the local non-interacting spectral function of the model.

        Requires
        --------
        First and second level approximation must have been calculated.

        Parameters
        ----------
        self        : self
        n_iw        : number of Matsubara-Frequencies considered for the Pade-fit
        window      : frequency range of real-frequency Green's function
        n_w         : number of frequencies in MeshReFreq

        Returns
        -------
        Sets self.A0_loc (Gf object).
        """

        # get local Green's function
        G0_loc_dlr_w = k_sum(self.g0_dlr_wk)

        # transform to ImFreq mesh
        G0_loc_dlr = make_gf_dlr(G0_loc_dlr_w)
        G0_loc_w = make_gf_imfreq(G0_loc_dlr, n_iw=n_iw)

        # analytically continue
        G0_loc_rw = self.evaluate_G_real_freq(G0_loc_w, window=window, n_w=n_w)
    
        # return local spectral function
        self.A0_loc = -1.0/np.pi * G0_loc_rw.imag

    def evaluate_spectral_function(self, n_iw=128, window=(-10, 10), n_w=1000):
        """
        Calculates the local spectral function of the model.

        Requires
        --------
        First and second level approximation must have been calculated.

        Parameters
        ----------
        self        : self
        n_iw        : number of Matsubara-Frequencies considered for the Pade-fit
        window      : frequency range of real-frequency Green's function
        n_w         : number of frequencies in MeshReFreq

        Returns
        -------
        Sets self.A_loc (Gf object).
        """

        # get local Green's function
        G2_loc_dlr_w = k_sum(self.g2_dlr_wk)

        # transform to ImFreq mesh
        G2_loc_dlr = make_gf_dlr(G2_loc_dlr_w)
        G2_loc_w = make_gf_imfreq(G2_loc_dlr, n_iw=n_iw)

        # enforce particle hole symmetry
        if self.n == 1.0:
            G2_loc_w.data.real = 0.0

        # analytically continue
        G2_loc_rw = self.evaluate_G_real_freq(G2_loc_w, window=window, n_w=n_w)
    
        # return local spectral function
        self.A_loc = -1.0/np.pi * G2_loc_rw.imag
    ### FUNCTIONS FOR ANALYTIC CONTINUATION AND FITTING ###

    ###### UNDER CONSTRUCTION ######
    ### METHODS FOR EXTRACTING G(k) AT OMEGA = 0 ###
    def evaluate_G_imfreq_range(self, G, n_max=30):
        """
        Evaluates an imaginary frequency Green's function on a given range of Matsubara frequencies.

        Parameters
        ----------
        self    : self
        G       : TRIQS Green's function object (G.mesh must be MeshImFreq!)
        n_max   : maximum Matsubara frequency index

        Returns
        -------
        np.arrays of Matsubara freqiencies and of Green's function evaluated at those frequencies.
        """
        
        if isinstance(G.mesh, MeshImFreq):
            
            # get range of indices
            indices = np.arange(-n_max, n_max + 1)

            # get frequencies iw_n and data G(iw_n)
            wn_data = np.squeeze(np.array([G.mesh.to_value(i).imag for i in indices]))
            G_data = np.squeeze(np.array([G(i) for i in indices]))
        
        elif isinstance(G.mesh, MeshDLRImFreq):
            
            # extract complete data
            wn_data = np.squeeze(np.array([wn.imag for wn in G.mesh]))
            G_data = np.squeeze(G.data)

        # return results
        return wn_data, G_data

    def langrage_fit_zero_freq(self, xs, ys):
        """
        Evaluates the Lagrange polynomial passing through the points xs=[x1, x2, ..., xn+1], ys=[y1, y2, ..., yn+1] at x = 0.0.

        Parameters
        ----------
        self    : self
        xs      : np.array containing x-data
        ys      : np.array containing y-data

        Returns
        -------
        Value of the Lagrange polynomial at x = 0.0.
        """

        val = 0
        for i in range(ys.size):
            prod_temp=1
            for j in range(ys.size):
                if j != i:
                    prod_temp *= -xs[j] / (xs[i] - xs[j])
            val += prod_temp * ys[i]
        return val

    def evaluate_G_k_zero_freq_lagrange(self, G, n_k=64, n_max=30):
        """
        Evaluates an imaginary frequency and k-dependent Green's function at omega = 0 via a Lagrange polynomial as a function of k.

        Parameters
        ----------
        self    : self
        G       : TRIQS Green's function object (G.mesh must be MeshImFreq!)
        n_k     : number of k-points per dimension
        n_max   : maximum Matsubara frequency index; determines how much data there is for the fit

        Returns
        -------
        np.arrays containing kx, ky, and fitted data of G 
        """

        # define k-grid
        kx = np.linspace(0.0, 2*np.pi, n_k)
        ky = np.linspace(0.0, 2*np.pi, n_k)

        # define result array
        G_0_k = np.zeros(shape=(kx.size, ky.size), dtype='complex')

        # go over all k-points and extract fit-data
        for indx, x in enumerate(kx):
            for indy, y in enumerate(ky):

                # get and evaluate G at this k-point
                G_k = G(all,(x,y,0.0))

                # get fit data for the Lagrange polynomial
                wn_data, G_data = self.evaluate_G_imfreq_range(G_k, n_max=n_max)

                # evaluate and save value at zero frequency
                G_0_k[indx, indy] = self.langrage_fit_zero_freq(wn_data, G_data)
        
        # return results
        return kx, ky, G_0_k
    
    def evaluate_G_k_zero_freq_TRIQS_pade(self, G, n_k=64, window=(-10, 10), n_w=1000):
        """
        Evaluates an imaginary frequency and k-dependent Green's function at omega = 0 via a Pade fit as a function of k.

        Parameters
        ----------
        self        : self
        G           : TRIQS Green's function object (G.mesh must be MeshImFreq!)
        n_k         : number of k-points per dimension
        window      : frequency range of real-frequency Green's function
        n_w         : number of frequencies in MeshReFreq

        Returns
        -------
        np.arrays containing kx, ky, and fitted data of G 
        """

        # define k-grid
        kx = np.linspace(0.0, 2*np.pi, n_k)
        ky = np.linspace(0.0, 2*np.pi, n_k)

        # define result array
        G_0_k = np.zeros(shape=(kx.size, ky.size), dtype='complex')

        # go over all k-points and extract fit-data
        for indx, x in enumerate(kx):
            for indy, y in enumerate(ky):

                # get and evaluate G at this k-point
                G_k = G(all,(x,y,0.0))

                G_0_k[indx, indy] = self.evaluate_G_real_freq(G_k, window=window, n_w=n_w)(0.0)

        # return results
        return kx, ky, G_0_k
    ### METHODS FOR EXTRACTING G(k) AT OMEGA = 0 ###


    ### FUNCTIONS FOR PLOTTING ###
    def plot_Sigma_wk(self, iwn=0, n_k=64):
        """
        Plots the k-dependence of the self-energy at omega = iwn

        Requires
        --------
        Self-Energy must have been calculated.

        Parameters
        ----------
        self        : self
        iwn         : Matsubara frequency index (!) at which to evaluate
        n_k         : number of k-points per dimension
        """

        # Get Sigma at iwn
        Sigma_dlr_k = make_gf_dlr(self.Sigma2_dlr_wk)
        Sigma_wk = make_gf_imfreq(Sigma_dlr_k, n_iw=iwn+10)

        Sigma_k = Sigma_wk(iwn, all)

         # define k-grid
        kx = np.linspace(0.0, 2*np.pi, n_k)
        ky = np.linspace(0.0, 2*np.pi, n_k)

        # define result array
        S_iwn_k = np.zeros(shape=(n_k, n_k), dtype='complex')

        # go over all k-points
        for indx, x in enumerate(kx):
            for indy, y in enumerate(ky):
                S_iwn_k[indx, indy] = Sigma_k((x,y,0))

        # plot real part
        plt.figure()
        plt.contourf(kx, ky, S_iwn_k.real, levels=25, cmap='magma')
        plt.xlabel(r"$k_x$")
        plt.ylabel(r"$k_y$")
        plt.colorbar()
        plt.title("real part of self energy at $\\omega_{n} = $" + str((2*iwn+1)*np.pi/self.beta))

        # plot imaginary part
        plt.figure()
        plt.contourf(kx, ky, S_iwn_k.imag, levels=25, cmap='magma')
        plt.xlabel(r"$k_x$")
        plt.ylabel(r"$k_y$")
        plt.colorbar()
        plt.title("imaginary part of self energy at $\\omega_{n} = $" + str((2*iwn+1)*np.pi/self.beta))
    
    def plot_Sigma_zero_frequency_pade(self, n_k=64, window=(-10, 10), n_w=1000):
        """
        Plots the k-dependence the second-level approximation of the self-energy at omega = 0.0 via a Pade fit.

        Requires
        --------
        Self-Energy must have been calculated.

        Parameters
        ----------
        self        : self
        n_k         : number of k-points per dimension
        window      : frequency range of real-frequency Green's function
        n_w         : number of frequencies in MeshReFreq

        Returns
        -------
        Plots self.Sigma at omega = 0.0.
        """

        # interpolate k-dependence
        kx, ky, Sigma_0_k = self.evaluate_G_k_zero_freq_TRIQS_pade(self.Sigma2_wk, n_k=n_k, window=window, n_w=n_w)

        # plot real part
        plt.figure()
        plt.contourf(kx, ky, Sigma_0_k.real, levels=25, cmap='magma')
        plt.xlabel(r"$k_x$")
        plt.ylabel(r"$k_y$")
        plt.colorbar()
        plt.title(r"real part, extrapolation of self energy at $\omega_n \to 0$")

        # plot imaginary part
        plt.figure()
        plt.contourf(kx, ky, Sigma_0_k.imag, levels=25, cmap='magma')
        plt.xlabel(r"$k_x$")
        plt.ylabel(r"$k_y$")
        plt.colorbar()
        plt.title(r"imaginary part, extrapolation of self energy at $\omega_n \to 0$")
    
    def plot_spectral_function(self, n_iw=128, n_w=1000, w_range=(-10,10)):
        """
        Plots the local spectral function of the model.

        Requires
        --------
        First and second level approximation must have been calculated.

        Parameters
        ----------
        self    : self
        n_iw    : number of Matsubara Frequencies
        n_w     : number of real frequencies on which to plot
        w_range : frequency range on which to plot

        Returns
        -------
        Plot of local spectral function evaluated on w_range.
        """

        # get local spectral function
        self.evaluate_spectral_function(n_iw=n_iw, n_w=n_w)
        
        # plot
        plt.figure()
        oplot(self.A_loc, x_window=w_range)
        plt.xlabel('$\\omega$');
        plt.ylabel('$A_{loc}(\\omega)$');
        plt.title('Local Spectral Function');
        plt.gca().get_legend().remove();
    ### FUNCTIONS FOR PLOTTING ###
### TPSC SOLVER CLASS ###



### DISPERSION RELATION CLASS ###
class dispersion_relation:
    """
    Class to calculate dispersion relations
    
    Parameters
    ----------
    Model Parameters:
    list basis      : primitive lattice vectors, defines the lattice
        default     : basis=[(1,0,0), (0,1,0)] (2D cubic lattice)
    float t         : nearest-neighbour hopping
        default     : t = 1.0
    float tp        : next-nearest neighbour hopping
        default     : tp = 0.0
    int n_k         : number of k-points per direction
        default     : n_k = 128
    str scheme      : dispersion scheme (square, triangular ...)
        default     : scheme = 'square'
    
    Attributes
    ----------
    self.eps_k : Green's Function object defined on a k_mesh containing the dispersion relation
    
    """
    
    def __init__(self, basis=[(1,0,0), (0,1,0)], t=1.0, tp=0.0, n_k=128, scheme='square'):
        """
        Initialize a dispersion_relation object.
        """
        
        # get attributes
        self.basis = basis
        self.dim = len(basis)
        self.t = t
        self.tp = tp
        self.n_k = n_k
        self.scheme = scheme

        # get the dispersion relation.
        self.get_eps_k()

        # get the dos
        self.get_dos()
        
    def get_eps_k(self):
        """
        Calculates the dispersion relation.

        Parameters
        ----------
        self    : self

        Returns
        -------
        Sets self.eps_k.
        """

        # get the hopping matrix
        hoppings = {
            (+1,0) : [[-self.t]],
            (-1,0) : [[-self.t]],
            (0,+1) : [[-self.t]],
            (0,-1) : [[-self.t]]
        }

        if self.scheme == 'square':
            hoppings.update({
                (+1,+1) : [[-self.tp]],
                (+1,-1) : [[-self.tp]],
                (-1,+1) : [[-self.tp]],
                (-1,-1) : [[-self.tp]]
            })
        elif self.scheme == 'triangular':
            hoppings.update({
                (1,-1) : [-self.t]
            })

        # initialize a tight binding model (real space)
        self.H_r = TBLattice(units=self.basis, hoppings=hoppings)
    
        # get dispersion relation
        k_mesh = self.H_r.get_kmesh(n_k=self.n_k)
        self.eps_k = self.H_r.fourier(k_mesh)
    
    def get_dos(self):
        """
        Calculates the density of states.

        Requires
        --------
        get_eps_k must have been called.

        Parameters
        ----------
        self    : self

        Returns
        -------
        Sets self.dos.
        """

        # get dos
        self.dos = dos(self.H_r.tb, n_kpts=int(15*self.n_k), n_eps=1000, name='density of states')[0]
    
    def plot_dispersion(self):
        """
        Plots the dispersion relation.

        Parameters
        ----------
        self    : self

        Returns
        -------
        Plots epsilon(k).
        """

        # get k-points
        k_points = np.array(list(self.eps_k.mesh.values()))
        k_points = k_points.reshape(self.n_k, self.n_k, 3)
        kx = k_points[...,0]
        ky = k_points[...,1]

        # get data points
        e_k = self.eps_k.data.reshape(self.n_k, self.n_k)

        # plot 
        plt.figure
        plt.contourf(kx, ky, e_k, levels=25, cmap='magma')
        plt.xlabel(r"$k_x$")
        plt.ylabel(r"$k_y$")
        plt.colorbar()
        plt.title(r"$\epsilon(\mathbf{k})$")

    def plot_dos(self):
        """
        Plots the dispersion relation.

        Parameters
        ----------
        self    : self

        Returns
        -------
        Plots dos(E)
        """

        # extract energies and density
        eps = self.dos.eps
        rho = self.dos.rho

        # plot
        plt.figure()
        plt.plot(eps, rho, '-')
        plt.xlabel('$\\epsilon$')
        plt.ylabel('$D(\\epsilon)$')
        plt.title('Non-Interacting Density of States')
### DISPERSION RLEATION CLASS ###