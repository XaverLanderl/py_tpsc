from TPSC_TRIQS_library import *

# parameters
# parameters
beta = 5.0
U = 2.0
n = 1.0

# initialize solver and get "real" docc
solver = tpsc_solver(beta=beta, U=U, n=n, verbose=False, plot=False)

while True:
    try:
        solver.calc_first_level_approx()
        solver.beta += 20
    except:
        print('Failed at beta = ' + str(solver.beta))
        break