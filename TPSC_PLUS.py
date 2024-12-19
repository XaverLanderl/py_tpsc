# Imports
from TPSC_TRIQS_library_temp import *

# functions used
def do_self_consistent_loop(solver, g2_wk, alpha, conv):
    """
    Perform the self-consistent loop of the TPSC+ method.

    Parameters
    ----------
    solver  : tpsc_solver
    g2_wk   : initial guess of interacting G (better than getting it from TPSC)
    alpha   : mixing parameter
    conv    : convergence parameter

    Returns
    -------
    New g2_wk and Sigma2_wk.
    """

    # initialize result
    g2_plus_wk = g2_wk

    # prepare the TPSC+-cycle
    print('TPSC+ Step 2: solver-Consistent Cycle')

    # initialize self-consistent loop
    t1 = time.time()
    counter = 1

    # do self-consistent loop
    while True:

        # progress
        print('iteration = ' + str(counter))

        # get new "chi0_wk"
        print('    Get new "non-interacting" susceptibility:')
        chi2_wk = solver.imtime_bubble_chi2_wk(g2_wk=g2_plus_wk)

        # do first and second level approximation with this chi
        print('    Do TPSC calculation:')
        result_first_level = solver.calc_first_level_approximation(chi0_wk=chi2_wk)
        Usp, Uch, docc, chi_sp_wk, chi_ch_wk = result_first_level
        result_second_level = solver.calc_second_level_approximation(Usp=Usp, Uch=Uch, chi_sp_wk=chi_sp_wk, chi_ch_wk=chi_ch_wk)
        g2_wk = result_second_level[1]

        # check for convergence
        print('    Check for convergence:')
        diff = np.linalg.norm(g2_wk.data - g2_plus_wk.data)
        print(diff)
        if diff < conv:
            t2 = time.time()
            break

        # prepare for new iteration
        counter += 1
        g2_plus_wk = alpha*g2_wk + (1.0 - alpha)*g2_plus_wk # mixing

    # once convergence is reached, set parameters
    g2_plus_wk = g2_wk
    Sigma2_plus_wk = result_second_level[0]

    # end
    print('Convergence achieved after ' + str(t2-t1) + 's and ' + str(counter) + ' iterations!')

    # return results
    return Usp, Uch, g2_plus_wk, Sigma2_plus_wk

# Parameters
n = 1
U = 2

beta_list = np.linspace(1.0,20.0,20)

alpha = 1.0
conv = 1e-6

# get dispersion
units = [(1,0,0), (0,1,0)]
hoppings = {(+1,0) : [[-1]],
            (-1,0) : [[-1]],
            (0,+1) : [[-1]],
            (0,-1) : [[-1]]}
TBL = TBLattice(units=units, hoppings=hoppings)
eps_k = TBL.fourier(TBL.get_kmesh(n_k=128))

# initialize solver and do the initial TPSC calculation
solver = tpsc_solver(n=n, beta=beta_list[0], U=U, eps_k=eps_k, verbose=False)

# step 1: initial do TPSC
print('TPSC+ Step 1: TPSC-Calculation:')
solver.run_TPSC()
g2_wk_init = solver.g2_wk   # initial guess
Usp = solver.Usp
Uch = solver.Uch

for ind_b, beta in enumerate(beta_list):

    print()
    print('##### ----- ##### Iteration ' + str(ind_b+1) + '/' + str(beta_list.size) + ' ##### ----- #####')
    print('##### ----- ##### beta = ' + str(beta) + ' ##### ----- #####')
    print()

    # initialize solver for this temperature
    solver_temp = tpsc_solver(n=n, beta=beta, U=U, eps_k=eps_k, verbose=False)
    solver_temp.calc_noninteracting_gf()

    # get new initial guess of g2_wk_init
    chi_sp_wk = solve_Hubbard_RPA(solver_temp.chi0_wk, Usp)
    chi_ch_wk = solve_Hubbard_RPA(solver_temp.chi0_wk, -Uch)

    # get new self-energy and initial guess for Gf
    Sigma2_wk_init = solver_temp.calc_Sigma(Usp, Uch, chi_sp_wk, chi_ch_wk)
    g2_wk_init, mu2 = calc_G_from_Sigma(n=n, eps_k=eps_k, Sigma_wk=Sigma2_wk_init)

    # step 2: self-consistent TPSC+ loop
    Usp, Uch, g2_wk_init, Sigma2_plus_wk = do_self_consistent_loop(solver=solver_temp, g2_wk=g2_wk_init,
                                                                   alpha=alpha, conv=conv)

    # plot
    plt.figure()
    oplot(k_sum(Sigma2_plus_wk))
    plt.title('beta = ' + str(beta) + '$ \\mu = $' + str(mu2 + solver.U*solver.n/2))