# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species
plt = get_matplotlib().pyplot

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Read in data from the snapshots
species.allow_similarly_named_components = True
fluids = {'particles simulations': [], 'fluid simulations': []}
a = []
for kind in ('particles', 'fluid'):
    if kind == 'particles':
        regex = f'{this_dir}/output/{kind}/snapshot_a=*_converted*'
    elif kind == 'fluid':
        regex = f'{this_dir}/output/{kind}/snapshot_a=*'
    for fname in sorted(
        glob(regex),
        key=(lambda s: s[(s.index('=') + 1):]),
    ):
        snapshot = load(fname, compare_params=False)
        fluids[f'{kind} simulations'].append(snapshot.components[0])
        if kind == 'particles':
            a.append(snapshot.params['a'])
N_snapshots = len(a)
gridsize = fluids['particles simulations'][0].gridsize
# Sort data chronologically
order = np.argsort(a)
a = [a[o] for o in order]
for kind in ('particles', 'fluid'):
    fluids[f'{kind} simulations'] = [fluids[f'{kind} simulations'][o] for o in order]

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Plot
fig_file = f'{this_dir}/result.png'
fig, axes = plt.subplots(N_snapshots, sharex=True, sharey=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
ϱ = {'particles simulations': [], 'fluid simulations': []}
for kind, markersize, in zip(('particles', 'fluid'), (15, 10)):
    for ax, fluid, a_i in zip(axes, fluids[f'{kind} simulations'], a):
        ϱ[f'{kind} simulations'].append(fluid.ϱ.grid_noghosts[:gridsize, 0, 0])
        ax.plot(
            x,
            ϱ[f'{kind} simulations'][-1],
            '.',
            markersize=markersize,
            alpha=0.7,
            label=(kind.rstrip('s').capitalize() + ' simulation'),
        )
        ax.set_ylabel(
            r'$\varrho$ $\mathrm{{[{}\,m_{{\odot}}\,{}^{{-3}}]}}$'
            .format(
                significant_figures(
                    1/units.m_sun,
                    3,
                    fmt='TeX',
                    incl_zeros=False,
                    force_scientific=False,
                ),
                unit_length,
            )
        )
        ax.set_title(rf'$a={a_i:.3g}$')
axes[ 0].set_xlim(0, boxsize)
axes[-1].set_xlabel(rf'$x\,\mathrm{{[{unit_length}]}}$')
axes[ 0].legend()
fig.tight_layout()
fig.savefig(fig_file, dpi=150)

# Fluid elements in yz-slices should all have the same ϱ and J
tol_fac = 1e-6
for kind in ('particles', 'fluid'):
    for fluid, a_i in zip(fluids[f'{kind} simulations'], a):
        for fluidscalar in fluid.iterate_fluidscalars():
            grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(
                    np.std(yz_slice),
                    0,
                    rel_tol=0,
                    abs_tol=max((tol_fac*np.std(grid), 1e+1*gridsize**2*machine_ϵ))
                ):
                    abort(
                        f'Non-uniformities have emerged at a = {a_i} '
                        f'in yz-slices of fluid scalar variable {fluidscalar} '
                        f'in {kind.rstrip("s")} simulation.\n'
                        f'See "{fig_file}" for a visualization.'
                    )

# Compare ϱ's from the fluid and snapshot simulations
tol_fac = 2e-2
for ϱ_fluid, ϱ_particles, a_i in zip(ϱ['fluid simulations'], ϱ['particles simulations'], a):
    if not isclose(
        np.mean(abs(ϱ_fluid - ϱ_particles)),
        0,
        rel_tol=0,
        abs_tol=(tol_fac*np.std(ϱ_fluid) + machine_ϵ),
    ):
        abort(
            f'Fluid did not gravitate correctly up to a = {a_i}.\n'
            f'See "{fig_file}" for a visualization.'
        )

# Done analysing
masterprint('done')
