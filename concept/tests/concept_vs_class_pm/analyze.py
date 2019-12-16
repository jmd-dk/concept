# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Begin analysis
masterprint(f'Analyzing {this_test} data ...')

# Read in the power spectra
spectra = {}
for fname in sorted(glob(this_dir + '/output/powerspec_a=*')):
    if fname.endswith('.png'):
        continue
    a = float(re.search(r'=(.+)', fname).group(1))
    k, P_sim, P_lin = np.loadtxt(fname, usecols=(0, 2, 3), unpack=True)
    spectra[a] = {'k': k, 'P_sim': P_sim, 'P_lin': P_lin}
# Due to deconvolutions performed on the power, the highest k modes
# of the simulation power spectra will be erroneous.
# Truncate the data at some k_max after which the difference between the linear
# and simulation power spectrum is deemed large, using the power spectra
# at a_begin.
rel_tol = 0.1
index = np.where(
    np.abs(spectra[a_begin]['P_sim']/spectra[a_begin]['P_lin'] - 1) > rel_tol
)[0][0]
for spectrum in spectra.values():
    for key, val in spectrum.items():
        spectrum[key] = val[:index]

# Plot the relative error between the simulation and linear power spectrum
# at the beginning and end.
rel_err_begin = np.abs(spectra[a_begin]['P_sim']/spectra[a_begin]['P_lin'] - 1)
rel_err_end   = np.abs(spectra[1      ]['P_sim']/spectra[1      ]['P_lin'] - 1)
plt.semilogx(spectrum['k'], 100*rel_err_begin, label=f'$a = {a_begin}$')
plt.semilogx(spectrum['k'], 100*rel_err_end,   label=f'$a = 1$')
plt.xlabel(f'$k$\, [{unit_length}^{{-1}}]')
plt.ylabel(r'$|P_{\mathrm{sim}}/P_{\mathrm{lin}} - 1|\,[\%]$')
plt.legend()
plt.tight_layout()
fig_file = f'{this_dir}/result.png'
plt.savefig(fig_file)

# We declare the test for successful if the sum of the relative errors
# at the end is less than allowed_err_growth times the corresponding
# value at the beginning.
allowed_err_growth = 2
if np.sum(rel_err_begin) > allowed_err_growth*np.sum(rel_err_end):
    abort(
        f'The results from CO𝘕CEPT disagree with those from CLASS.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Done analyzing
masterprint('done')
