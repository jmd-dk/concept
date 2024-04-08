# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
plt = get_matplotlib().pyplot

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Load power spectrum data
power = {}
for lpt in [2, 3]:
    for dealias in [False, True]:
        power[lpt, dealias] = 0
        for shift in [False, True]:
            k, _power = np.loadtxt(
                glob(
                    f'{this_dir}/output/{lpt}LPT'
                    + '_dealias'*dealias
                    + '_shift'*shift
                    + f'/powerspec*'
                )[0],
                usecols=[0, 2],
                unpack=True,
            )
            power[lpt, dealias] += _power
        power[lpt, dealias] /= 2

# Compute power spectrum ratios
peak_relratios_expected = {False: 3.00/100, True: 2.39/100}
indices_peak = {}
ratios = {}
ratios_peak = {}
for dealias in [False, True]:
    ratios[dealias] = power[3, dealias]/power[2, dealias]
    indices_peak[dealias] = ratios[dealias].argmax()
    ratios_peak[dealias] = ratios[dealias][indices_peak[dealias]]

# Plot
fig, ax = plt.subplots()
for ls, dealias in zip(['-', '--'], [False, True]):
    ax.semilogx(k, (ratios[dealias] - 1)*100, ls, label=f'{dealias = }')
    x = exp(log(k[0]) + (0.85)*(log(k[-1]) - log(k[0])))
    y = peak_relratios_expected[dealias]*100
    ax.semilogx([x, k[-1]], [y]*2, f'k{ls}', lw=1)
    ax.text(x, y, r'expected $\rightarrow$', ha='right', va='center')
ax.set_xlim(k[0], k[-1])
ax.set_xlabel(rf'$k\, [\mathrm{{{unit_length}}}^{{-1}}]$')
ax.set_ylabel(r'$P_{\mathrm{3LPT}}/P_{\mathrm{2LPT}} - 1\, [\%]$')
ax.legend()
fig_file = f'{this_dir}/result.png'
fig.savefig(fig_file, dpi=150)

# Check
abs_tol = 0.001
rel_tol = 0
for dealias in [False, True]:
    dealiased = ('dealiased' if dealias else 'aliased')
    if indices_peak[dealias] != len(k) - 1:
        abort(
            f'The largest value of the {dealiased} power spectrum ratio f'
            f'between 3LPT and 2LPT does not occur at the highest '
            f'k available (the Nyquist frequency). See {fig_file}.'
        )
    if not isclose(
        ratios[dealias][-1] - 1,
        peak_relratios_expected[dealias],
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    ):
        abort(
            f'The {dealiased} 3LPT to 2LPT power spectrum ratio at the '
            f'Nyquist frequency does not match the expected value. '
            f'See {fig_file}.'
        )

# Done analysing
masterprint('done')

