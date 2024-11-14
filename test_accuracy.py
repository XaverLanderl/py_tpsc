from TPSC_TRIQS_library import *
def generate_variants(X_0, relative_error_percent, num_vals):
    # Generate the relative error as a fraction (e.g., 0.01% = 0.0001)
    relative_error = relative_error_percent / 100
    
    # Calculate the noise value at each precision level
    noise_values = relative_error * X_0 * np.linspace(-1,1,num_vals)

    # Generate a list of values with added and subtracted noise
    X_variants = X_0 + noise_values

    return X_variants

# parameters
beta = 5.0
U = 2.0
n = 1.0

# initialize solver and get "real" docc
solver = tpsc_solver(beta=beta, U=U, n=n, verbose=False, plot=False)
solver.calc_first_level_approx()
real_docc = solver.docc

# prepare solver
solver.use_tpsc_ansatz = False

failed = []
worked = []
worked_sp = []
worked_ch = []

doccs = generate_variants(real_docc, 20, 15)

for ind, docc in enumerate(doccs):
    
    print(ind + 1)

    solver.docc = docc
    
    try:
        solver.calc_first_level_approx()
        worked.append(docc)
        worked_sp.append(solver.Usp)
        worked_ch.append(solver.Uch)

    except:
        failed.append(docc)

message = 'Calculation failed for docc = '
for ind, failed_docc in enumerate(failed):
    if (ind + 1) != len(failed):
        message += str(failed_docc) + ', '
    elif (ind + 1) == len(failed):
        message += str(failed_docc) + '!'

print(message)

plt.figure()
plt.plot(worked, worked_sp)
plt.xlabel('docc')
plt.ylabel('Usp')
plt.title('Spin vertex')

plt.figure()
plt.plot(worked, worked_ch)
plt.xlabel('docc')
plt.ylabel('Uch')
plt.title('Charge vertex')