# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in data from the snapshots
fluids = {'particles simulations': [], 'fluid simulations': []}
times = []
for kind in ('particles', 'fluid'):
    if kind == 'particles':
        regex = '{}/output/{}/snapshot_t=*_converted*'.format(this_dir, kind)
    elif kind == 'fluid':
        regex = '{}/output/{}/snapshot_t=*'.format(this_dir, kind)
    for fname in sorted(glob(regex),
                        key=lambda s: s[(s.index('=') + 1):]):
        snapshot = load(fname, compare_params=False)
        fluids[kind + ' simulations'].append(snapshot.components[0])
        if kind == 'particles':
            times.append(float(re.search('snapshot_t=(.*)' + unit_time, fname).group(1)))
N_snapshots = len(times)
gridsize = fluids['particles simulations'][0].gridsize
# Sort data chronologically
order = np.argsort(times)
times = [times[o]  for o in order]
for kind in ('particles', 'fluid'):
    fluids[kind + ' simulations'] = [fluids[kind + ' simulations'][o] for o in order]
# Use precise times
times = snapshot_times['t']

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, sharey=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
ρ = {'particles simulations': [], 'fluid simulations': []}
for kind, markertype, options in zip(('particles', 'fluid'),
                                     ('ro', 'b*'),
                                     ({'markerfacecolor': 'none', 'markeredgecolor': 'r'}, {}),
                                     ):
    for ax_i, fluid, t in zip(ax, fluids[kind + ' simulations'], times):
        ρ[kind + ' simulations'].append(fluid.ρ.grid_noghosts[:gridsize,
                                                              :gridsize,
                                                              :gridsize].mean((1, 2)))
        ax_i.plot(x, ρ[kind + ' simulations'][-1],
                  markertype,
                  label=(kind.rstrip('s').capitalize() + ' simulation'),
                  **options,
                  )
        ax_i.set_ylabel(r'$\varrho$ $\mathrm{{[{}\,m_{{\odot}}\,{}^{{-3}}]}}$'
                        .format(significant_figures(1/units.m_sun,
                                                    3,
                                                    fmt='tex',
                                                    incl_zeros=False,
                                                    scientific=False,
                                                    ),
                                unit_length)
                        )
        ax_i.set_title(r'$t={:.3g}\,\mathrm{{{}}}$'.format(t, unit_time))
plt.xlim(0, boxsize)
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.xlabel(r'$x\,\mathrm{{[{}]}}$'.format(unit_length))
plt.tight_layout()
plt.savefig(fig_file)

# Fluid elements in yz-slices should all have the same ρ and ρu
tol_fac = 1e-6
for kind in ('particles', 'fluid'):
    for fluid, t in zip(fluids[kind + ' simulations'], times):
        for fluidscalar in fluid.iterate_fluidscalars():
            grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(np.std(yz_slice), 0,
                               rel_tol=0,
                               abs_tol=(tol_fac*np.std(grid) + machine_ϵ)):
                    abort('Non-uniformities have emerged after {} {} '
                          'in yz-slices of fluid scalar variable {} '
                          'in {} simulation.\n'
                          'See "{}" for a visualization.'
                          .format(t, unit_time, fluidscalar, kind.rstrip('s'), fig_file))

# Compare ρ's from the fluid and snapshot simulations
discontinuity_tol = 2
rel_tol = 0.1
for ρ_fluid, ρ_particles, t in zip(ρ['fluid simulations'], ρ['particles simulations'], times):
    abs_tol = rel_tol*np.std(ρ_particles)
    slope_left  = ρ_particles - np.roll(ρ_particles, -1)
    slope_right = np.roll(ρ_particles, +1) - ρ_particles
    discontinuities = np.abs(slope_right - slope_left)
    discontinuities = [max(d) for d in zip(*[np.roll(discontinuities, r) for r in range(-3, 4)])]
    if not all(isclose(ρ_fluid_i, ρ_particles_i,
                       rel_tol=0,
                       abs_tol=(discontinuity_tol*discontinuity + abs_tol), 
                       ) for ρ_fluid_i, ρ_particles_i, discontinuity in zip(ρ_fluid,
                                                                            ρ_particles,
                                                                            discontinuities)):
        abort('Fluid did not evolve correctly up to t = {} {}.\n'
              'See "{}" for a visualization.'
              .format(t, unit_time, fig_file))

# Done analyzing
masterprint('done')

