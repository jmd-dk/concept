# Imports from the CO𝘕CEPT code
from commons import *
from integration import cosmic_time, init_time
plt = get_matplotlib().pyplot

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# In compiled mode, the C-declared variables from the commons module
# are not imported by the above 'from commons import *' statement.
# In particular, we need the user parameters. We extract these off of
# the user_params dict.
for key, val in user_params.items():
    exec(f'{key} = val')

# Initiate the cosmic time and the scale factor,
# and do the call to CLASS if enable_class_background is True.
init_time()

# Array of scale factor values at which to compute the cosmic time
N_points = 50
scale_factors = np.logspace(np.log10(a_begin), np.log10(1), N_points)

# Compute the cosmic time for each value of the scale factor
cosmic_times = [cosmic_time(a) for a in scale_factors]

# Save the computed cosmic times
compiled = not ast.literal_eval(os.environ['CONCEPT_pure_python'])
mode = f'class={enable_class_background}_compiled={compiled}'
np.savetxt(f'{this_dir}/t_{mode}.dat', cosmic_times, encoding='utf-8')

# If all four data files exist, plot and analyse these
data_filenames = glob(f'{this_dir}/*.dat')
if sum([
    bool(re.search(
        rf'^{this_dir}/t_class=(True|False)_compiled=(True|False)\.dat$',
        fname,
    ))
    for fname in data_filenames
]) == 4:
    masterprint(f'Analysing {this_test} data ...')
    # Load in the data
    all_times = {}
    for filename in data_filenames:
        if re.search('class=True', filename):
            key = 'CLASS'
        else:
            key = 'no CLASS'
        if re.search('compiled=True', filename):
            key += ', compiled'
        else:
            key += ', not compiled'
        all_times[key] = np.loadtxt(filename)
    # Plot the data
    fig_file = f'{this_dir}/result.png'
    fig, ax = plt.subplots(figsize=(16, 12))
    markersize = 50
    for key, times in all_times.items():
        ax.loglog(
            scale_factors, times, '.',
            markersize=markersize,
            label=key,
        )
        markersize -= 10
    ax.set_xlim(a_begin, 1)
    ax.set_xlabel('$a$')
    ax.set_ylabel(rf'$t\,\mathrm{{[{unit_time}]}}$')
    # Using CLASS or not makes a difference at early times
    # due to the inclusion of e.g. radiation and neutrinos.
    # Find the latest time at which this difference is still important.
    rtol = 1e-2
    i = N_points
    something_wrong = False
    for t1, t2 in zip(
        reversed(all_times[   'CLASS, compiled']),
        reversed(all_times['no CLASS, compiled']),
    ):
        i -= 1
        if not np.isclose(t1, t2, rtol=rtol):
            # Time found. Update plot.
            a = scale_factors[i]
            ylim = ax.get_ylim()
            ax.loglog([a, a], ylim, 'k:', zorder=-1)
            ax.text(
                1.1*a,
                0.4*ylim[1],
                (
                    r'$\leftarrow$ $1\%$ disagreement between' + '\n'
                    r'$\leftarrow$ CLASS and no CLASS'
                ),
                fontsize=16,
            )
            ax.set_ylim(ylim)
            # If this time is too late, something is wrong
            a_max_allowed = 0.1
            if a > a_max_allowed:
                something_wrong = True
            break
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(fig_file, dpi=150)
    if something_wrong:
        abort(
            f'A discrepancy in t(a) of 1% between CLASS and the built-in '
            f'Friedmann equation is present as late as a = {a}, '
            f'which is too extreme to be acceptable.\n'
            f'See "{fig_file}" for a visualization.'
        )
    # Whether we are running in compiled mode or not
    # really should not make a big difference.
    rtol = 1e-5
    if not all(
        np.isclose(t1, t2, rtol=rtol)
        for t1, t2 in zip(
            all_times['CLASS, compiled'],
            all_times['CLASS, not compiled'],
        )
    ):
        abort(
            f'The cosmic times computed via interpolation of CLASS data '
            f'are different between compiled and pure Python mode.\n'
            f'See "{fig_file}" for a visualization.'
        )
    rtol = 1e-10
    if not all(
        np.isclose(t1, t2, rtol=rtol)
        for t1, t2 in zip(
            all_times['no CLASS, compiled'],
            all_times['no CLASS, not compiled'],
        )
    ):
        abort(
            f'The cosmic times computed via the simple Friedmann equation '
            f'are different between compiled and pure Python mode.\n'
            f'See "{fig_file}" for a visualization.'
        )
    masterprint('done')
