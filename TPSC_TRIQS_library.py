### IMPORTS ###
from triqs.gf import *
from triqs.lattice import BravaisLattice, BrillouinZone, TightBinding
from triqs.lattice.tight_binding import TBLattice, dos
from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice import *
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
from triqs.plot.mpl_interface import oplot, plt
import triqs.utility.mpi as mpi
from functools import lru_cache
import numpy as np
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from scipy import integrate
from h5 import HDFArchive
import time
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
    def __init__(self, n=1., U=2., beta=2.5, eps_k=None, docc=None, g0_bubble=None, w_max=10.0, eps=1e-14, Usp_tol=1e-12, Uch_tol=None, verbose=True, plot=True):
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

        # check whether a g0 for the bubble has been passed
        if g0_bubble == None:
            self.g0_bubble = None
        else:
            self.g0_bubble = g0_bubble

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

        # check self consisteny
        self.check_for_self_consistency()
        t4 = time.time()

        # plot Self-energy
        if self.plot == True:
            self.plot_spectral_function()
            #self.plot_Sigma_zero_frequency_lagrange()
        t5 = time.time()

        # End of calculation
        self.vprint()
        self.vprint("------------------------------------------------------------------------------------------")
        self.vprint(f'Runtime of first level approximation = {(t2 - t1):.2f}s.')
        self.vprint(f'Runtime of second level approximation = {(t3 - t2):.2f}s.')
        self.vprint(f'Runtime of self-consistency check = {(t4 - t3):.2f}s.')
        if self.plot == True:
            self.vprint(f'Runtime of plots = {(t5 - t4):.2f}s.')
        self.vprint("DONE!")

    def calc_first_level_approx(self, Uchmax=1000.):
        """
        Runs the first-level approximation of the TPSC-Calculation on the specified model.
        If docc is specified in the model, it will be used to evaluate the sum rules.
        If docc is not specified in the model, the TPSC-Ansatz will be used to evaluate the sum rules.

        Parameters
        ----------
        self            : self
        double Uchmin   : lower bound for Uch
            default     : Uchmin = 0.
        double Uchmax   : upper bound for Uch
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
        self.calc_Usp()
        
        self.vprint()
        self.vprint("   Calculating Uch...")
        self.calc_Uch(Uchmax=Uchmax)

        # calculate susceptibilities
        self.vprint()
        self.vprint("   Calculate TPSC-spin- and charge-susceptibilities...")
        self.chi_sp = self.solve_rpa(self.Usp)
        self.chi_ch = self.solve_rpa(-self.Uch)
        
        # calculate double occupation
        if self.use_tpsc_ansatz == True:
            self.vprint()
            self.vprint("   Calculate double occupation...")
            self.calc_docc()
        
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
        self.G2, self.mu2 = self.calc_G_from_Sigma(Sigma_dlr_wk=self.Sigma2_dlr_wk)
        self.mu2_phys = self.mu2 + self.U*self.n/2 # add Hartree term
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
        self.vprint('Doing self-consistency check...')
        self.check_sum_rule = self.k_iw_sum(self.chi_sp + self.chi_ch) - (2*self.n - self.n**2)
        self.vprint(f'The sum rule is fulfilled with an accuracy of {abs(self.check_sum_rule)}.')
    ### METHODS TO RUN THE CALCULATION ###


    ### HELPER FUNCTIONS ###
    def vprint(self, *args):
        if self.verbose == True:
            if len(args) > 0:
                mpi.report(args[0])
            else:
                mpi.report('')
        
    def solve_rpa(self, U):
        """
        Calculates the rpa-susceptibility from the non-interacting susceptibility.

        Requires
        --------
        calc_noninteracting_gf must have been called.

        Parameters
        ----------
        self        : self
        double U    : RPA vertex for charge/spin susceptibility

        Returns
        -------
        Green's Function object containing the RPA-like charge (U < 0) or spin (U > 0) susceptibility for the given vertex
        Mesh is the same as the mesh of the non-interacting susceptibility.
        """

        # initialize RPA susceptibility
        chi_dlr_wk = self.chi0_dlr_wk.copy()

        # fill with data
        chi_dlr_wk.data[:] = self.chi0_dlr_wk.data[:]/(1 - U/2*self.chi0_dlr_wk.data[:])

        # return results
        return chi_dlr_wk

    def k_iw_sum(self, g_dlr_wk):
        """
        Calculates the k- and Matsubara sum over the passed Green's function.
        
        Requires
        --------
        g_dlr_wk must be defined on a MeshDLRImFreq-mesh.
        
        Parameters
        ----------
        self                : self
        triqs.gf g_dlr_wk   : TRIQS Green's function object

        Returns
        -------
        double              : k- and Matsubara sum over passed Green's function
                              target_shape=()
        """

        # extract meshes of passed Gf
        wmesh, kmesh = g_dlr_wk.mesh.components

        # define auxiliary quantity not dependent on k
        g_dlr_w = Gf(mesh=wmesh, target_shape=())

        # for each frequency, sum over k (second axis!) and normalize by number of k-points
        g_dlr_w.data[:] = np.sum(np.squeeze(g_dlr_wk.data), axis=1) / len(kmesh)
    
        # convert to dlr mesh for density call (DLRImFreq does not support density!)
        g_dlr = make_gf_dlr(g_dlr_w)
        
        # return result based on statistic
        if wmesh.statistic == 'Boson':
            return -g_dlr.density().real    # bosonic time order does not introduce the necessary -sign
        elif wmesh.statistic == 'Fermion':
            return g_dlr.density().real
        
    def k_sum(G_wkself, g_dlr_wk):
        """
        Calculates the k-sum over the passed Green's function.
        
        Requires
        --------
        g_dlr_wk must be defined on a MeshDLRImFreq-mesh.
        K-dependence must be along the second axis.
        
        Parameters
        ----------
        self                : self
        triqs.gf g_dlr_wk   : TRIQS Green's function object

        Returns
        -------
        triqs.gf            : DLRImFreq Green's function
                              target_shape=()
        """

        # extract meshes of passed Gf
        wmesh, kmesh = g_dlr_wk.mesh.components

        # define Green's function not dependent on k
        g_dlr_w = Gf(mesh=wmesh, target_shape=())

        # for each frequency, sum over k (second axis!) and normalize by number of k-points
        g_dlr_w.data[:] = np.sum(np.squeeze(g_dlr_wk.data), axis=1) / len(kmesh)

        # return the result
        return g_dlr_w
    
    def change_target_shape(self, g):
        """
        Changes target_shape between (1,1) and (1,1,1,1), as both are used by various functions.

        Requires
        --------
        g must be a Green's function.

        Parameters
        ----------
        self    : self
        g       : TRIQS Green's function object

        Returns
        -------
        TRIQS Green's function object with changed target_shape
        """ 

        # extract mesh and target_shape
        mesh = g.mesh
        target_shape = g.target_shape
        
        # change target_shape and transfer data
        if target_shape == (1,1,1,1):
            result = Gf(mesh=mesh, target_shape=(1,1))
            result.data[:,:,:,:] = g.data[:,:,:,:,0,0]
        elif target_shape == (1,1):
            result = Gf(mesh=mesh, target_shape=(1,1,1,1))
            result.data[:,:,:,:,:,:] = g.data[:,:,:,:,None,None]
            
        # return new gf-object
        return result

    def n_root(self, mu, Sigma_dlr_wk=None):
        """
        Calculates the density as a function of mu.

        Parameters
        ----------
        self            : self
        mu              : chemical potential
        Sigma_dlr_wk    : self-energy (must be MeshDLRImFreq)
                        : if None, calculates non-interacting density
        """

        if Sigma_dlr_wk is not None:
            # Dyson equation
            g0_dlr_wk_inv = inverse(lattice_dyson_g0_wk(mu=mu, e_k=self.eps_k, mesh=self.iw_dlr_mesh))
            G2 = inverse(g0_dlr_wk_inv - Sigma_dlr_wk)
        elif Sigma_dlr_wk == None:
            # Only non-interacting Green's function
            G2 = lattice_dyson_g0_wk(mu=mu, e_k=self.eps_k, mesh=self.iw_dlr_mesh)
        
        # return density
        return self.k_iw_sum(G2) - self.n/2     # n/2 because G2 is Green's function per spin.
    
    def calc_mu(self, Sigma_dlr_wk=None):
        """
        Calculates the chemical potential for the given self-energy.

        Parameters
        ----------
        self            : self
        Sigma_dlr_wk    : self-energy (must be MeshDLRImFreq)
                        : if None, Sigma is set to zero
        """

        # find the mu that leads to the correct density
        mu_min = np.min(self.eps_k.data.real)
        mu_max = np.max(self.eps_k.data.real)
        mu_result = brentq(lambda m: self.n_root(mu=m, Sigma_dlr_wk=Sigma_dlr_wk), mu_min, mu_max)

        # return the result
        return mu_result
    
    def add_local_gf(self, g_w_k, g_local_w):
        """
        Adds a local (k-independent) Green's function to a k-dependent Green's function.

        Parameters
        ----------
        self        : self
        g_iw_k      : k-dependent Green's function; iw must be on first axis.
        g_local_w   : k-independent Green's function to be subtracted
                    : must both have the same Matsubara-mesh

        Returns
        -------
        result      : TRIQS Green's function object of target_shape=(1,1)
        """

        # extract data of k-dependent gf
        g_w_k_data = np.squeeze(g_w_k.data)

        # extract data of local gf; reshape for broadcasting reasons
        g_local_w_data = np.squeeze(g_local_w.data).reshape(-1,1)

        # initialize result
        result = Gf(mesh = g_w_k.mesh, target_shape=(1,1))
        
        # feed values (utilize numpy broadcasting!)
        result.data[:,:,0,0] = g_w_k_data + g_local_w_data

        # return result
        return result

    def get_nonlocal_gf(self, g_wk):
        """
        Removes the local part of passed Green's function.

        Parameters
        ----------
        self        : self
        g_wk        : Green's function; iw must be on first axis

        Returns
        -------
        g_nonloc_wk : non-local part of passed Green's function, same target_shape
        """

        # get local part
        g_local_w = self.k_sum(g_wk)

        # subtract local part
        g_nonloc_wk = self.add_local_gf(g_wk, -1*g_local_w)

        # return result
        return g_nonloc_wk
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
                    self.chi0_dlr_wk, target_shape=(1,1,1,1)
        """

        # If no g0 for the bubble has been passed, calculate from scratch
        if self.g0_bubble == None:

            # calculate imaginary time mesh
            self.iw_dlr_mesh = MeshDLRImFreq(beta=self.beta, statistic='Fermion', w_max=self.w_max, eps=self.eps)

            # get mu from n (need for G0 and bubble)
            if self.n == 1:
                self.mu1 = 0.0  # half-filling
            else:
                self.mu1 = self.calc_mu(Sigma_dlr_wk=None)
            
            # calculate non-interacting Green's function of the model
            self.g0_dlr_wk = lattice_dyson_g0_wk(mu=self.mu1, e_k=self.eps_k, mesh=self.iw_dlr_mesh)
        
        # otherwise, use passed g0
        else:
            self.iw_dlr_mesh = self.g0_bubble.mesh.components[0]
            self.g0_dlr_wk = self.g0_bubble

        # calculate the non-interacting susceptibility of the model
        self.chi0_dlr_wk = 2*imtime_bubble_chi0_wk(self.g0_dlr_wk, nw=2, verbose=False)

    def Usp_root(self, Usp):
        """
        Function whose root is the self-consistent value for Usp.
        
        Requires
        --------
        calc_noninteracting_gf must have been called (requirement for solve_rpa).
        
        Parameters
        ----------
        self        : self
        double Usp  : given value of spin vertex
        
        Returns
        -------
        double      : spin sum over chi_RPA(Usp) - sum rule (Usp)
        """
        
        # calculate sum over RPA-like susceptibility
        chi_sum = self.k_iw_sum(self.solve_rpa(Usp))
        
        # calculate spin sum rule
        if self.use_tpsc_ansatz == True:
            sum_rule = self.n - Usp/self.U*self.n*self.n/2
        else:
            sum_rule = self.n - 2*self.docc

        # return difference
        return chi_sum - sum_rule
    
    def Uch_root(self, Uch):
        """
        Function whose root is the self-consistent value for Uch.
        
        Requires
        --------
        calc_noninteracting_gf must have been called (requirement for solve_rpa).
        calc_Usp must have been called (need self-consistent value for Usp, in charge sum rule).
        
        Parameters
        ----------
        self        : self
        double Usp  :  given value of spin vertex
        double Uch  : given value of charge vertex
        
        Returns
        -------
        double      : charge sum over chi_RPA(Uch) - sum rule (Uch)
        """
        
        # calculate sum over RPA-like susceptibility
        chi_sum = self.k_iw_sum(self.solve_rpa(-Uch))
        
        # calculate charge sum rule
        if self.use_tpsc_ansatz == True:
            sum_rule = self.n + self.Usp/self.U*self.n*self.n/2 - self.n*self.n
        else:
            sum_rule = self.n + 2*self.docc - self.n*self.n
        
        # return the difference
        return chi_sum - sum_rule
    
    def calc_Usp(self):
        """
        Calculates Usp self-consistently to obey spin sum rule.
        
        Requires
        --------
        calc_chi0 must have been called (requirement for Usp_root)
        
        Parameters
        ----------
        self    : self

        Returns
        -------
        double  : Usp

        """

        # set maximum value of Usp (where chi diverges)
        Uspmax = 2.0/np.amax(self.chi0_dlr_wk.data).real - 1e-7 # the 1e-7 is chosen for numerical stability.
        
        # calculate Usp self-consistently
        self.Usp = brentq(lambda x: self.Usp_root(x), 0.0, Uspmax, xtol=self.Usp_tol)
        
    def calc_Uch(self, Uchmax=1000.):
        """
        Calculates Uch self-consistently to obey charge sum rule.
        
        Requires
        --------
        calc_chi0 must have been called (requirement for Uch_root)
        calc_Usp must have been called (requirement for Uch_root)
        
        Parameters
        ----------
        self    : self

        Returns
        -------
        double  : Uch

        """
        
        # calculate Usp self-consistently
        self.Uch = brentq(lambda x: self.Uch_root(x), 0.0, Uchmax, xtol=self.Uch_tol)
        
    def calc_docc(self):
        """
        Calculates the double occupation.
        
        Requires
        --------
        calc_Usp must have been called
        
        Parameters
        ----------
        self    : self
        
        Returns
        -------
        Sets self.docc
        """
        
        self.docc = self.Usp/self.U*self.n*self.n/4
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
        V = self.U/8*(3*self.Usp*self.chi_sp + self.Uch*self.chi_ch)

        # get V(-t,-r)
        V = V.conjugate()
        V = chi_wr_from_chi_wk(V)
        V = chi_tr_from_chi_wr(V)
        V = V.conjugate()

        # change V to correct target shape
        V = self.change_target_shape(V)

        # get G(t,r)
        g0_dlr_wr = fourier_wk_to_wr(self.g0_dlr_wk)
        g0_dlr_tr = fourier_wr_to_tr(g0_dlr_wr)

        # multiply V(-t,-r) * G0(t,r) = Sigma(t,r)
        Sigma2_dlr_tr = g0_dlr_tr.copy()    # the 2 means second level of approximation
        Sigma2_dlr_tr.data[:] = V.data[:] * g0_dlr_tr.data[:]

        # transform Sigma(t,r) to Sigma(w,k)
        Sigma2_dlr_wr = fourier_tr_to_wr(Sigma2_dlr_tr)
        self.Sigma2_dlr_wk = fourier_wr_to_wk(Sigma2_dlr_wr)

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
        mu = self.calc_mu(Sigma_dlr_wk=Sigma_dlr_wk)

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
        G0_loc_dlr_w = self.k_sum(self.g0_dlr_wk)

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
        G2_loc_dlr_w = self.k_sum(self.G2)

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
    def plot_Sigma_zero_frequency_lagrange(self, n_k=64, n_max=10):
        """
        Plots the k-dependence the second-level approximation of the self-energy at omega = 0.0 via a Lagrange polynomial fit.

        Requires
        --------
        Self-Energy must have been calculated.

        Parameters
        ----------
        self        : self
        n_k         : number of k-points per dimension
        n_max       : maximum Matsubara frequency index; determines how much data there is for the fit

        Returns
        -------
        Plots self.Sigma at omega = 0.0.
        """

        # interpolate k-dependence
        kx, ky, Sigma_0_k = self.evaluate_G_k_zero_freq_lagrange(self.Sigma2_wk, n_k=n_k, n_max=n_max)

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
    
    def plot_spectral_function(self, w_range=(-10,10)):
        """
        Plots the local spectral function of the model.

        Requires
        --------
        First and second level approximation must have been calculated.

        Parameters
        ----------
        self    : self
        w_range : frequency range on which to plot

        Returns
        -------
        Plot of local spectral function evaluated on w_range.
        """

        # get local spectral function
        self.evaluate_spectral_function()
        
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