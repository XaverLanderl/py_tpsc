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
    def __init__(self, n=1., U=2., beta=2.5, eps_k=None, docc=None, w_max=10.0, eps=1e-14, Usp_tol=1e-12, Uch_tol=None, verbose=True):
        """
        Initialize a tpsc_solver object.
        """
        
        ### SET PARAMETERS
        # set model pareameters
        self.n = n
        self.U = U
        self.beta = beta

        if eps_k == None:   # default
            units = [(1,0,0), (0,1,0)]
            hoppings = {(+1,0) : [[-1]],
                        (-1,0) : [[-1]],
                        (0,+1) : [[-1]],
                        (0,-1) : [[-1]]}
            TBL = TBLattice(units=units, hoppings=hoppings)
            self.eps_k = TBL.fourier(TBL.get_kmesh(n_k=256))
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
    ### CLASS CONSTRUCTOR ###        
    

    ### METHODS TO RUN THE CALCULATION ###
    def run_TPSC(self):
        """
        Runs the TPSC-Calculation on the specified model.

        Parameters
        ----------
        self    : self
        """

        ### start with non-interacting calculation
        self.vprint()
        self.vprint("Calculating non-interacting quantities...")
        self.calc_noninteracting_gf()

        ### calculate first level of approximation
        t1 = time.time()
        result_first_level = self.calc_first_level_approximation(chi0_wk=self.chi0_wk)
        self.Usp, self.Uch, self.docc, self.chi1_sp_wk, self.chi1_ch_wk = result_first_level
        t2 = time.time()

        # print out results
        self.vprint()
        self.vprint("Summary first level approximation:")
        self.vprint("Usp = " + str(self.Usp) + ", Uch = " + str(self.Uch))
        self.vprint("Double Occupation <n_up*n_down> = " + str(self.docc))

        ### calculate second level of approximation
        t3 = time.time()
        result_second_level = self.calc_second_level_approximation(Usp=self.Usp, Uch=self.Uch,
                                                                   chi_sp_wk=self.chi1_sp_wk,
                                                                   chi_ch_wk=self.chi1_ch_wk)
        self.Sigma2_wk, self.g2_wk, self.mu2, self.mu2_phys = result_second_level
        t4 = time.time()

        # print out results
        self.vprint()
        self.vprint("Summary second level approximation:")
        self.vprint("mu^(2) = " + str(self.mu2_phys))

        ### do self-consistency check
        self.check_for_self_consistency(g2_wk=self.g2_wk, Sigma2_wk=self.Sigma2_wk,
                                        chi_sp_wk=self.chi1_sp_wk, chi_ch_wk=self.chi1_ch_wk)

        ### End of calculation
        self.vprint()
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint(f'Runtime of First-Level-Approximation = {(t2 - t1):.2f}s.')
        self.vprint(f'Runtime of Second-Level-Approximation = {(t4 - t3):.2f}s.')
        self.vprint("DONE!")

    def run_TPSC_plus(self, alpha=0.5, conv=1e-6):
        """
        Runs the TPSC+-Calculation on the specified model.

        Parameters
        ----------
        self    : self
        alpha   : mixing parameter
        conv    : convergence parameter
        """

        # run the TPSC-calculation
        self.vprint('TPSC+ Step 1: TPSC-Calculation:')
        self.run_TPSC()
        
        # initialize result
        g2_plus_wk = self.g2_wk

        # prepare the TPSC+-cycle
        self.vprint()
        self.vprint('TPSC+ Step 2: Self-Consistent Cycle')

        if self.verbose == True:
            turn_on_verbose = True
            self.verbose = False
        
        t1 = time.time()
        counter = 1

        while True:

            # get new "chi0_wk"
            if turn_on_verbose == True:
                print('Iteration ' + str(counter))
                print()
                print('   Get new "non-interacting" susceptibility:')
            chi2_wk = self.imtime_bubble_chi2_wk(g2_wk=g2_plus_wk)

            # do first and second level approximation with this chi
            if turn_on_verbose == True:
                print()
                print('   Do TPSC calculation')
            result_first_level = self.calc_first_level_approximation(chi0_wk=chi2_wk)
            Usp, Uch, docc, chi_sp_wk, chi_ch_wk = result_first_level
            result_second_level = self.calc_second_level_approximation(Usp=Usp, Uch=Uch, chi_sp_wk=chi_sp_wk, chi_ch_wk=chi_ch_wk)
            g2_wk = result_second_level[1]

            # check for convergence
            diff = np.linalg.norm(g2_wk.data - g2_plus_wk.data)
            print(diff)
            if diff < conv or counter >= 500:
                t2 = time.time()
                break

            # prepare for new iteration
            counter += 1
            g2_plus_wk = alpha*g2_wk + (1.0 - alpha)*g2_plus_wk # mixing

        # once convergence is reached, set parameters
        self.Usp_plus = Usp
        self.Uch_plus = Uch
        self.docc_plus = docc
        self.chi_sp_plus_wk = chi_sp_wk
        self.chi_ch_plus_wk = chi_ch_wk
        self.g2_plus_wk = g2_wk
        self.Sigma2_plus_wk = result_second_level[0]
        self.mu2_plus = result_second_level[2]
        self.mu2_phys_plus = result_second_level[3]

        # turn verbose back on
        if turn_on_verbose == True:
            self.verbose = True

        # do self-consistency check
        self.check_for_self_consistency(g2_wk=self.g2_plus_wk, Sigma2_wk=self.Sigma2_plus_wk,
                                        chi_sp_wk=self.chi_sp_plus_wk, chi_ch_wk=self.chi_ch_plus_wk)

        # end of calculation
        self.vprint()
        self.vprint('Convergence achieved after ' + str(t2-t1) + 's and ' + str(counter) + ' iterations!')
        self.vprint("DONE!")

    def calc_first_level_approximation(self, chi0_wk, Uch_max=1000.):
        """
        Runs the first-level approximation of the TPSC-Calculation on the specified model.
        If docc is specified in the model, it will be used to evaluate the sum rules.
        If docc is not specified in the model, the TPSC-Ansatz will be used to evaluate the sum rules.

        Parameters
        ----------
        self            : self
        gf chi0_wk      : "non"-interacting susceptibility, bubble
        double Uch_max  : upper bound for Uch
            default     : Uchmax = 1000.
        
        Returns
        -------
        Usp         : spin vertex
        Uch         : charge vertex
        docc        : double occupation
        chi_sp_wk   : spin susceptibility
        chi_ch_wk   : charge susceptibility
        """
        
        # begin calculation
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint("Calculate First-Level-Approximation:")
        if self.use_tpsc_ansatz == True:
            self.vprint()
            self.vprint("   Double Occupation was not specified, will use the TPSC-Ansatz to evaluate the sum rules.")
        else:
            self.vprint()
            self.vprint("   Double Occupation was specified, will use it to evaluate the sum rules and ignore the TPSC-Ansatz.")
             
        # calculate vertices
        self.vprint()
        self.vprint("   Calculating Usp...")
        Usp = self.calc_Usp(chi0_wk=chi0_wk)
        
        self.vprint()
        self.vprint("   Calculating Uch...")
        Uch = self.calc_Uch(chi0_wk=chi0_wk, Usp=Usp, Uch_max=Uch_max)

        # calculate susceptibilities
        self.vprint()
        self.vprint("   Calculate TPSC-spin- and charge-susceptibilities...")
        chi1_sp_wk = solve_Hubbard_RPA(chi0_wk=self.chi0_wk, U=Usp)
        chi1_ch_wk = solve_Hubbard_RPA(chi0_wk=self.chi0_wk, U=-Uch)
        
        # calculate double occupation
        if self.use_tpsc_ansatz == True:
            self.vprint()
            self.vprint("   Calculate double occupation...")
            docc = self.calc_docc(Usp=Usp)
        else:
            docc = self.docc

        # return results
        return Usp, Uch, docc, chi1_sp_wk, chi1_ch_wk
        
    def calc_second_level_approximation(self, Usp, Uch, chi_sp_wk, chi_ch_wk):
        """
        Runs the second-level approximation of the TPSC-Calculation on the specified model.

        Requires
        --------
        First-level approximation must have been run.

        Parameters
        ----------
        self        : self
        Usp         : spin vertex
        Uch         : charge vertex
        chi_sp_wk   : spin susceptibility
        chi_ch_wk   : charge susceptibility

        Returns
        -------
        Sigma2_wk   : TPSC Self-energy
        g2_wk       : TPSC Green's function
        mu2         : chemical potential in Gf with Hartree-Term
        mu2_phys    : chemical potential without Hartree-Term
        """

        # start calculation
        self.vprint()
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint("Calculate Second-Level-Approximation:")

        # calculate self-energy
        self.vprint()
        self.vprint("   Calculate self-energy...")
        Sigma2_wk = self.calc_Sigma(Usp=Usp, Uch=Uch, chi_sp_wk=chi_sp_wk, chi_ch_wk=chi_ch_wk)
        
        # calculate G2
        self.vprint()
        self.vprint("   Calculate Green's function...")
        g2_wk, mu2 = calc_G_from_Sigma(n=self.n, eps_k=self.eps_k, Sigma_wk=Sigma2_wk)
        mu2_phys = mu2 + self.U*self.n/2    # add Hartree term
        
        # return results
        return Sigma2_wk, g2_wk, mu2, mu2_phys
      
    def check_for_self_consistency(self, g2_wk, Sigma2_wk, chi_sp_wk, chi_ch_wk):
        """
        Checks the evaluated model for self-consistency.
        
        Requires
        --------
        Must have called first- and second-level and TPSC+ approximation.
        
        Parameters
        ----------
        self        : self
        g2_wk       : interacting Green's function
        Sigma2_wk   : self-energy
        chi_sp_wk   : spin susceptibility
        chi_ch_wk   : charge susceptibility

        Returns
        -------
        None

        """
        
        # check that the sum rule is fulfilled
        self.vprint()
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint('Doing self-consistency check of First-Level-Approximation...')
        check_sum_rule = k_iw_sum(chi_sp_wk + chi_ch_wk) - (2*self.n - self.n**2)
        self.vprint(f'The sum rule is fulfilled with an accuracy of {float(abs(check_sum_rule))}.')

        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint('Doing self-consistency check of Second-Level-Approximation...')
        
        # add Hartree term to self-energy
        Sigma2_wk_temp = Sigma2_wk.copy()
        Sigma2_wk_temp.data[:] += self.U*self.n/2     # Hartree-term

        # get products Sigma*G
        F1_wk = Sigma2_wk_temp * self.g0_wk
        F2_wk = Sigma2_wk_temp * g2_wk

        # get traces
        trace_F1 = k_iw_sum(F1_wk) / self.U
        trace_F2 = k_iw_sum(F2_wk) / self.U

        # get the relative difference
        rel_diff = float(np.abs((trace_F1 - trace_F2) / trace_F1) * 100)

        # check consistency
        self.vprint('Sum_k {Sigma^(2)(k) * G^(1)(k)} - U*<n_up * n_down> = ' + str(float(np.abs(trace_F1 - self.docc))))
        self.vprint('Sum_k {Sigma^(2)(k) * G^(2)(k)} - U*<n_up * n_down> = ' + str(float(np.abs(trace_F2 - self.docc))))
        self.vprint(f'Relative difference of traces = {rel_diff:.2f}%')
    ### METHODS TO RUN THE CALCULATION ###


    ### HELPER FUNCTIONS ###
    def vprint(self, *args):
        if self.verbose == True:
            if len(args) > 0:
                mpi.report(args[0])
            else:
                mpi.report('')
            
    def change_target_shape(self, g_wk):
        """
        Changes target_shape from (1,1,1,1) to (1,1) if necessary, does nothing otherwise.

        Parameters
        ----------
        self    : self
        g_wk    : TRIQS Green's function object

        Returns
        -------
        TRIQS Green's function object with changed target_shape
        """ 

        # extract mesh and target_shape
        mesh = g_wk.mesh
        target_shape = g_wk.target_shape
        
        # change target_shape and transfer data
        if target_shape == (1,1,1,1):
            result = Gf(mesh=mesh, target_shape=(1,1))
            result.data[:,:,:,:] = g_wk.data[:,:,:,:,0,0]
        else:
            result = g_wk
            
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
        Sets    :   self.mu1,     non-interacting chemical potential
                    self.g0_wk,   non-interacting Green's function with correct chemical potential
                    self.chi0_wk, target_shape=(1,1)
        """

        # calculate imaginary time mesh
        iw_mesh = MeshDLRImFreq(beta=self.beta, statistic='Fermion', w_max=self.w_max, eps=self.eps)

        # get mu from n (need for G0 and bubble)
        self.mu1 = calc_mu(dens=self.n, eps_k=self.eps_k, Sigma_wk=iw_mesh)
        
        # calculate non-interacting Green's function of the model
        self.g0_wk = lattice_dyson_g0_wk(mu=self.mu1, e_k=self.eps_k, mesh=iw_mesh)

        # calculate the non-interacting susceptibility of the model
        chi0_wk = 2*imtime_bubble_chi0_wk(self.g0_wk, nw=2, verbose=False)

        # change target_shape (we need (1,1))
        self.chi0_wk = self.change_target_shape(g_wk=chi0_wk)

    def imtime_bubble_chi2_wk(self, g2_wk):
        """
        Calculates chi2(r,tau) = - G2(r,tau)*G0(-r,-tau) - G2(-r,-tau)*G0(r,tau) in wk-space.

        Parameters
        ----------
        self    :   self
        g2_wk   :   interacting Green's function

        Returns
        -------
        chi2_wk :   second-level approximation of bubble in TPSC
        """

        # Fourier transform Gs to real space
        G2_tr = fourier_wk_to_tr(g2_wk)
        G2_mtr = fourier_wk_to_mtr(g2_wk)
        G0_tr = fourier_wk_to_tr(self.g0_wk)
        G0_mtr = fourier_wk_to_mtr(self.g0_wk)

        # initialize chi2_tr
        chi2_tr = fourier_wk_to_tr(self.chi0_wk.copy())

        # calculate chi2
        chi2_tr.data[:] = -G2_tr.data*G0_mtr.data - G2_mtr.data*G0_tr.data
        chi2_wk = fourier_tr_to_wk(chi2_tr)

        # return result
        return chi2_wk

    def Usp_root(self, chi0_wk, Usp):
        """
        Function whose root is the self-consistent value for Usp.
        
        Parameters
        ----------
        self        : self
        chi0_wk     : non-interacting susceptibility
        double Usp  : given value of spin vertex
        
        Returns
        -------
        double      : spin sum over chi_RPA(Usp) - sum rule (Usp)
        """
        
        # calculate sum over RPA-like susceptibility
        chi_sum = k_iw_sum(solve_Hubbard_RPA(chi0_wk=chi0_wk, U=Usp))
        
        # calculate spin sum rule
        if self.use_tpsc_ansatz == True:
            sum_rule = self.n - Usp/self.U*self.n*self.n/2
        else:
            sum_rule = self.n - 2*self.docc

        # return difference
        return chi_sum - sum_rule
    
    def Uch_root(self, chi0_wk, Usp, Uch):
        """
        Function whose root is the self-consistent value for Uch.
        
        Parameters
        ----------
        self        : self
        chi0_wk     : non-interacting susceptibility
        double Usp  : (previously determined) spin vertex
        double Uch  : given value of charge vertex
        
        Returns
        -------
        double      : charge sum over chi_RPA(Uch) - sum rule (Uch)
        """
        
        # calculate sum over RPA-like susceptibility
        chi_sum = k_iw_sum(solve_Hubbard_RPA(chi0_wk=chi0_wk, U=-Uch))
        
        # calculate charge sum rule
        if self.use_tpsc_ansatz == True:
            sum_rule = self.n + Usp/self.U*self.n*self.n/2 - self.n*self.n
        else:
            sum_rule = self.n + 2*self.docc - self.n*self.n
        
        # return the difference
        return chi_sum - sum_rule
    
    def calc_Usp(self, chi0_wk):
        """
        Calculates Usp self-consistently to obey spin sum rule.
        
        Parameters
        ----------
        self        : self
        chi0_wk     : non-interacting susceptibility

        Returns
        -------
        double      : Usp
        """

        # set maximum value of Usp (where chi diverges)
        Usp_max = 2.0/np.amax(chi0_wk.data).real - 1e-7 # the 1e-7 is chosen for numerical stability.
        
        # calculate Usp self-consistently
        Usp = brentq(lambda x: self.Usp_root(chi0_wk=chi0_wk, Usp=x), 0.0, Usp_max, xtol=self.Usp_tol)

        # return result
        return Usp
        
    def calc_Uch(self, chi0_wk, Usp, Uch_max=1000.):
        """
        Calculates Uch self-consistently to obey charge sum rule.
        
        Parameters
        ----------
        self        : self
        chi0_wk     : non-interacting susceptibility
        Usp         : spin-vertex required for sum rule
        Uch_max     : maximum search value for Uch (default = 1000.)

        Returns
        -------
        double  : Uch
        """
        
        # calculate Usp self-consistently
        Uch = brentq(lambda x: self.Uch_root(chi0_wk=chi0_wk, Usp=Usp, Uch=x), 0.0, Uch_max, xtol=self.Uch_tol)
        
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
    def calc_Sigma(self, Usp, Uch, chi_sp_wk, chi_ch_wk):
        """
        Calculates the second-level approximation of the self-energy.

        Requires
        --------
        Susceptibilities must have been calculated

        Parameters
        ----------
        self        : self
        Usp         : spin vertex
        Uch         : charge vertex
        chi_sp_wk   : spin susceptibility
        chi_ch_wk   : charge susceptibility

        Returns
        -------
        Sigma_wk (WITHOUT the Hartree-term!)
        """

        # define effective potential
        V_wk = self.U/8*(3*Usp*chi_sp_wk + Uch*chi_ch_wk)

        # get V(-t,-r)
        V_mtr = fourier_wk_to_mtr(V_wk)

        # get G(t,r)
        g0_tr = fourier_wk_to_tr(self.g0_wk)

        # multiply V(-t,-r) * G0(t,r) = Sigma(t,r)
        Sigma2_tr = g0_tr.copy()    # the 2 means second level of approximation, must be fermionic
        Sigma2_tr.data[:] = V_mtr.data * g0_tr.data
        
        # transform Sigma(t,r) to Sigma(w,k)
        Sigma2_wk = fourier_tr_to_wk(Sigma2_tr)

        # return result
        return Sigma2_wk
    ### SECOND LEVEL OF APPROXIMATION ###