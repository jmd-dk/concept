# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species
plt = get_matplotlib().pyplot

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Read in data from the CO𝘕CEPT snapshots
species.allow_similarly_named_components = True
fluids = []
times = []
for fname in sorted(
    glob(f'{this_dir}/output/snapshot_t=*'),
    key=(lambda s: s[(s.index('=') + 1):]),
):
    snapshot = load(fname, compare_params=False)
    fluids.append(snapshot.components[0])
    times.append(float(re.search(f'snapshot_t=(.*){unit_time}', fname).group(1)))
gridsize = fluids[0].gridsize
N_snapshots = len(fluids)
# Sort data chronologically
order = np.argsort(times)
times  = [times[o]  for o in order]
fluids = [fluids[o] for o in order]
# Use precise times
times = output_times['t']['snapshot']

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Plot
fig_file = f'{this_dir}/result.png'
fig, axes = plt.subplots(N_snapshots, sharex=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
ϱ = []
ϱ_snapshot = []
phases = [-t/(10*units.Gyr)*2*π for t in times]
for ax, fluid, t, phase in zip(axes, fluids, times, phases):
    ϱ_i = asarray([2 + sin(2*π*i/gridsize + phase) for i in range(gridsize)])  # Unitless
    ϱ_i /= sum(ϱ_i)                                                            # Normalize
    ϱ_i *= ρ_mbar*gridsize                                                     # Apply units
    ϱ.append(ϱ_i)
    ϱ_snapshot.append(fluid.ϱ.grid_noghosts[:gridsize, 0, 0])
    ax.plot(x, ϱ[-1], '-', label='Analytical solution')
    ax.plot(x, ϱ_snapshot[-1], '.', markersize=10, alpha=0.7, label='Simulation')
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

# Fluid elements in yz-slices should all have the same ϱ
# and all fluid elements should have the same u = J/ϱ.
tol_fac_ϱ = 1e-6
tol_fac_u = 1e-3
for fluid, t in zip(fluids, times):
    for fluidscalar in fluid.iterate_fluidscalars():
        varnum = fluidscalar.varnum
        grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
        if varnum == 0:
            # ϱ
            ϱ_grid = grid
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(
                    np.std(yz_slice), 0,
                    rel_tol=0,
                    abs_tol=max((tol_fac_ϱ*np.std(grid), 1e+1*gridsize**2*machine_ϵ)),
                ):
                    abort(
                        f'Non-uniformities have emerged at t = {t} {unit_time} '
                        f'in yz-slices of fluid scalar variable {fluidscalar}.\n'
                        f'See "{fig_file}" for a visualization.'
                    )
        elif varnum == 1:
            # J
            u_grid = grid/ϱ_grid
            if not isclose(
                np.std(u_grid),
                0,
                rel_tol=0,
                abs_tol=(tol_fac_u*abs(np.mean(u_grid)) + machine_ϵ),
            ):
                abort(
                    f'Non-uniformities have emerged at t = {t} {unit_time} '
                    f'in fluid scalar variable {fluidscalar}'
                )

# Compare ϱ from the snapshots to the analytical solution
tol_fac = 0.02
for ϱ_i, ϱ_snapshot_i, t in zip(ϱ, ϱ_snapshot, times):
    if not isclose(
        np.mean(abs(ϱ_i - ϱ_snapshot_i)),
        0,
        rel_tol=0,
        abs_tol=(tol_fac*np.std(ϱ_i) + machine_ϵ),
):
        abort(
            f'Fluid did not drift rigidly up to t = {t} {unit_time}.\n'
            f'See "{fig_file}" for a visualization.'
        )

# Done analysing
masterprint('done')
