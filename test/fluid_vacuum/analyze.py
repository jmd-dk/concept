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
times = []
for kind in ('particles', 'fluid'):
    if kind == 'particles':
        regex = f'{this_dir}/output/{kind}/snapshot_t=*_converted*'
    elif kind == 'fluid':
        regex = f'{this_dir}/output/{kind}/snapshot_t=*'
    for fname in sorted(
        glob(regex),
        key=(lambda s: s[(s.index('=') + 1):]),
    ):
        snapshot = load(fname, compare_params=False)
        fluids[f'{kind} simulations'].append(snapshot.components[0])
        if kind == 'particles':
            times.append(float(re.search(f'snapshot_t=(.*){unit_time}', fname).group(1)))
N_snapshots = len(times)
gridsize = fluids['particles simulations'][0].gridsize
# Sort data chronologically
order = np.argsort(times)
times = [times[o]  for o in order]
for kind in ('particles', 'fluid'):
    fluids[f'{kind} simulations'] = [fluids[f'{kind} simulations'][o] for o in order]
# Use precise times
times = output_times['t']['snapshot']

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Plot
fig_file = f'{this_dir}/result.png'
fig, axes = plt.subplots(N_snapshots, sharex=True, sharey=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
ϱ = {'particles simulations': [], 'fluid simulations': []}
for kind, markersize in zip(('particles', 'fluid'), (15, 10)):
    for ax, fluid, t in zip(axes, fluids[f'{kind} simulations'], times):
        ϱ[f'{kind} simulations'].append(
            fluid.ϱ.grid_noghosts[
                :gridsize,
                :gridsize,
                :gridsize,
            ].mean((1, 2))
        )
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
                ),
                unit_length,
            )
        )
        ax.set_title(rf'$t={t:.3g}\,\mathrm{{{unit_time}}}$')
axes[ 0].set_xlim(0, boxsize)
axes[-1].set_xlabel(rf'$x\,\mathrm{{[{unit_length}]}}$')
axes[ 0].legend()
fig.tight_layout()
fig.savefig(fig_file, dpi=150)

# Fluid elements in yz-slices should all have the same ϱ and J
tol_fac = 1e-6
for kind in ('particles', 'fluid'):
    for fluid, t in zip(fluids[f'{kind} simulations'], times):
        for fluidscalar in fluid.iterate_fluidscalars():
            grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(
                    np.std(yz_slice),
                    0,
                    rel_tol=0,
                    abs_tol=max((tol_fac*np.std(grid), 1e+1*gridsize**2*machine_ϵ)),
                ):
                    abort(
                        f'Non-uniformities have emerged after {t} {unit_time} '
                        f'in yz-slices of fluid scalar variable {fluidscalar} '
                        f'in {kind.rstrip("s")} simulation.\n'
                        f'See "{fig_file}" for a visualization.'
                    )

# Compare ϱ's from the fluid and snapshot simulations
discontinuity_tol = 2
rel_tol = 0.1
for ϱ_fluid, ϱ_particles, t in zip(ϱ['fluid simulations'], ϱ['particles simulations'], times):
    abs_tol = rel_tol*np.std(ϱ_particles)
    slope_left  = ϱ_particles - np.roll(ϱ_particles, -1)
    slope_right = np.roll(ϱ_particles, +1) - ϱ_particles
    discontinuities = abs(slope_right - slope_left)
    discontinuities = [max(d) for d in zip(*[np.roll(discontinuities, r) for r in range(-3, 4)])]
    if not all(
        isclose(
            ϱ_fluid_i,
            ϱ_particles_i,
            rel_tol=0,
            abs_tol=(discontinuity_tol*discontinuity + abs_tol),
        )
        for ϱ_fluid_i, ϱ_particles_i, discontinuity in zip(
            ϱ_fluid, ϱ_particles, discontinuities,
        )
    ):
        abort(
            f'Fluid did not evolve correctly up to t = {t} {unit_time}.\n'
            f'See "{fig_file}" for a visualization.'
        )

# Done analysing
masterprint('done')
