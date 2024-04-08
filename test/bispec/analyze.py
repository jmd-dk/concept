# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
plt = get_matplotlib().pyplot

# Import the Fraction class
from fractions import Fraction

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Check exact number of modes when running without shell antialiasing
for filename in glob(f'{this_dir}/output/*antialiasing=False*'):
    if 'squeezed' in filename or filename.endswith('.png'):
        continue
    k_arr, t_arr, μ_arr, modes = np.loadtxt(filename, usecols=(0, 1, 2, 3), unpack=True)
modes = modes.astype(int)
k_fundamental = 2*π/boxsize
k1_arr = np.round(k_arr/k_fundamental).astype(int)
k2_arr = np.round(t_arr*k1_arr).astype(int)
k3_arr = np.round(sqrt(k1_arr**2 + k2_arr**2 - 2*μ_arr*k1_arr*k2_arr)).astype(int)
def get_modes(k1, k2, k3):
    def get_vecs(l):
        return [
            asarray((i, j, k))
            for i in range(-l, l + 1)
            for j in range(-l, l + 1)
            for k in range(-l, l + 1)
            if i**2 + j**2 + k**2 == l**2
        ]
    modes = 0
    for v1 in get_vecs(k1):
        for v2 in get_vecs(k2):
            for v3 in get_vecs(k3):
                modes += (((v1 + v2 + v3)**2).sum() == 0)
    modes //= 2  # only half of Fourier space available in simulation
    return modes
modes_actual = asarray([
    get_modes(k1, k2, k3)
    for k1, k2, k3, in zip(k1_arr, k2_arr, k3_arr)
])
if (modes != modes_actual).any():
    abort(
        f'The reported integer number of modes for the bispectrum '
        f'measurement carried out without shell antialiasing is incorrect: '
        f'{modes} vs. {modes_actual}.'
    )

# Check approximate number of modes when running with shell antialiasing
k_arr             = {}
modes             = {}
modes_expected    = {}
bpower            = {}
bpower_treelevel  = {}
reduced           = {}
reduced_treelevel = {}
for filename in glob(f'{this_dir}/output/*antialiasing=True*'):
    if 'squeezed' in filename or filename.endswith('.png'):
        continue
    shifted = ('_pi_' in os.path.basename(filename))
    (
        k_arr            [shifted],
        modes            [shifted],
        modes_expected   [shifted],
        bpower           [shifted],
        bpower_treelevel [shifted],
        reduced          [shifted],
        reduced_treelevel[shifted],
    ) = np.loadtxt(filename, usecols=(0, 3, 4, 5, 6, 7, 8), unpack=True)
    with open_file(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'grid size (\d+)', line)
            if match:
                gridsize = int(match.group(1))
                break
nyquist = gridsize//2
if not np.array_equal(k_arr[True], k_arr[False], equal_nan=True):
    abort('Got different k for the paired realisations')
k_arr = k_arr[True]
if not np.array_equal(modes[True], modes[False], equal_nan=True):
    abort('Got different number of modes for the paired realisations')
modes = modes[True]
if not np.array_equal(modes_expected[True], modes_expected[False], equal_nan=True):
    abort(
        'Got different number of expected modes '
        'for the paired realisations'
    )
modes_expected = modes_expected[True]
if not np.array_equal(bpower_treelevel[True], bpower_treelevel[False], equal_nan=True):
    abort(
        'Got different predictions for the tree-level bispectrum '
        'for the paired realisations'
    )
bpower_treelevel = bpower_treelevel[True]
if not np.array_equal(reduced_treelevel[True], reduced_treelevel[False], equal_nan=True):
    abort(
        'Got different predictions for the reduced tree-level bispectrum '
        'for the paired realisations'
    )
reduced_treelevel = reduced_treelevel[True]
if np.array_equal(bpower[True], bpower[False], equal_nan=True):
    abort('Got identical bispectra for the paired realisations')
bpower  = 0.5*(bpower [False] + bpower [True])
reduced = 0.5*(reduced[False] + reduced[True])
fig, axes = plt.subplots(2, sharex=True)
x = k_arr/k_fundamental
axes[0].loglog(x, modes, label='modes')
axes[0].loglog(x, modes_expected, 'k--', lw=1, label='modes expected')
axes[1].semilogx(x, modes/modes_expected - 1)
axes[1].semilogx(x, np.zeros(x.size), ':', lw=1, color='grey')
for i, ax in enumerate(axes):
    ylim = ax.get_ylim()
    ax.plot([2/3*nyquist]*2, ylim, ':', lw=1, color='grey')
    if i == 0:
        ax.text(
            0.97*2/3*nyquist,
            np.prod(ylim)**0.5,
            r'$\frac{2}{3} k_{\mathrm{Nyquist}}$',
            ha='right',
        )
    ax.set_ylim(ylim)
for st in bispec_options['shellthickness'].values():
    for val in st[0].values():
        if 'max' not in val:
            continue
        val = val.replace('k_fundamental', '1').replace('k', '1')
        val = '(' + val.replace('max', '').replace(',', ')/(') + ')'
        x_reliable = eval(val)
mask = (x_reliable < x) & (x < 2/3*nyquist)
axes[1].semilogx(x[mask], (modes/modes_expected - 1)[mask], '--')
axes[1].text(
    x[mask][0],
    (modes/modes_expected - 1)[mask][0] - 0.03,
    r'$\uparrow$ reliable region',
)
axes[0].set_xlim(x[0], x[-1])
axes[1].set_ylim(-0.15, 0.1)
axes[1].set_xlabel(r'$k$ [grid units]')
axes[1].set_ylabel(r'$n_{\mathrm{modes}}/n_{\mathrm{modes}\, \mathrm{expected}} - 1$')
axes[0].legend()
fig.suptitle('equilateral')
fig_file = f'{this_dir}/modes.png'
fig.savefig(fig_file, dpi=150)
reltol = 1e-2
if ((modes/modes_expected - 1)[mask]**2).mean()**0.5 > reltol:
    abort(
        f'The reported number of modes for the bispectrum measurement '
        f'carried out with shell antialiasing is far from the expected '
        f'number of modes. See {fig_file}.'
    )

# Compare realised and tree-level equilateral bispectrum
treelevel_prediction = Fraction(4, 7)
fig, axes = plt.subplots(2, sharex=True)
mask = (x < 2/3*nyquist)
axes[0].loglog(k_arr[mask], bpower[mask], '-', label='realisation')
axes[0].loglog(k_arr[mask], bpower_treelevel[mask], 'k--', lw=1, label='tree-level')
axes[1].semilogx(k_arr[mask], reduced[mask], '-')
axes[1].semilogx(k_arr[mask], reduced_treelevel[mask], 'k--', lw=1)
axes[0].set_xlim(k_arr[mask][0], k_arr[mask][-1])
axes[1].set_xlabel(rf'$k\, [\mathrm{{{unit_length}}}^{{-1}}]$')
axes[0].set_ylabel(rf'$B\, [\mathrm{{{unit_length}}}^{{6}}]$')
axes[1].set_ylabel(rf'$Q$')
axes[0].legend()
fig.suptitle('equilateral')
fig_file = f'{this_dir}/treelevel.png'
fig.savefig(fig_file, dpi=150)
reduced_treelevel_values = [val for val in set(reduced_treelevel) if not np.isnan(val)]
if len(reduced_treelevel_values) > 1:
    abort(
        f'Got multiple values for the reduced tree-level bispectrum, '
        f'though this should be singular (and approximately equal '
        f'to {treelevel_prediction}) for equilateral bispectra. '
        f'See {fig_file}.'
    )
reduced_treelevel_value = reduced_treelevel_values[0]
reltol = 0.02
if not isclose(reduced_treelevel_value, float(treelevel_prediction), rel_tol=reltol):
    abort(
        f'The value of the reduced tree-level bispectrum is found to be '
        f'{reduced_treelevel_value} but ought to be close to '
        f'{treelevel_prediction} for equilateral bispectra. See {fig_file}.'
    )
tol = 0.05
n = 5
if np.abs((reduced[mask][n:-n] - reduced_treelevel[mask][n:-n]).mean()) > tol:
    abort(
        f'The reduced equilateral bispectrum measured from the paired realisations '
        f'does not match the tree-level prediction of {treelevel_prediction}. '
        f'See {fig_file}.'
    )

# Compare realised and analytical squeezed bispectrum
bpower_squeezed = 0
power1 = 0
power2 = 0
power3 = 0
n = 0
for filename in glob(f'{this_dir}/output/*squeezed*'):
    if filename.endswith('.png'):
        continue
    (
        k_arr_squeezed,
        _bpower_squeezed,
        _power1,
        _power2,
        _power3,
    ) = np.loadtxt(filename, usecols=(0, 5, 9, 10, 11), unpack=True)
    bpower_squeezed += _bpower_squeezed
    power1 += _power1
    power2 += _power2
    power3 += _power3
    n += 1
bpower_squeezed /= n
power1 /= n
power2 /= n
power3 /= n
fig, ax = plt.subplots()
ax.loglog(k_arr_squeezed, bpower_squeezed, label='realisation')
fnl = tuple(realization_options['nongaussianity'].values())[0]
bpower_squeezed_analytical = 2*fnl*(power1*power2 + power2*power3 + power3*power1)
ax.loglog(k_arr_squeezed, bpower_squeezed_analytical, 'k--', lw=1, label='analytical')
ax.set_xlabel(rf'$k\, [\mathrm{{{unit_length}}}^{{-1}}]$')
ax.set_ylabel(rf'$B\, [\mathrm{{{unit_length}}}^{{6}}]$')
ax.legend()
fig.suptitle('squeezed')
fig_file = f'{this_dir}/squeezed_nongaussian.png'
fig.savefig(fig_file, dpi=150)
tol = 0.1
n = 8
if np.std(bpower_squeezed[n:]/bpower_squeezed_analytical[n:]) > tol:
    abort(
        f'The squeezed bispectrum measured from the paired non-Gaussian '
        f'realisations is far from the analytical prediction. '
        f'See {fig_file}.'
    )

# Done analysing
masterprint('done')

