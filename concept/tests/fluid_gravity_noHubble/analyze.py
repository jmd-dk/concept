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
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
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
ϱ = {'particles simulations': [], 'fluid simulations': []}
for kind, markersize in zip(('particles', 'fluid'), (15, 10)):
    for ax_i, fluid, t in zip(ax, fluids[kind + ' simulations'], times):
        ϱ[kind + ' simulations'].append(fluid.ϱ.grid_noghosts[:gridsize, 0, 0])
        ax_i.plot(x, ϱ[kind + ' simulations'][-1],
                  '.',
                  markersize=markersize,
                  alpha=0.7,
                  label=(kind.rstrip('s').capitalize() + ' simulation'),
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
plt.xlabel(r'$x\,\mathrm{{[{}]}}$'.format(unit_length))
ax[0].legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Fluid elements in yz-slices should all have the same ϱ and J
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

# Compare ϱ's from the fluid and snapshot simulations
tol_fac = 2e-2
for ϱ_fluid, ϱ_particles, t in zip(ϱ['fluid simulations'], ϱ['particles simulations'], times):
    if not isclose(np.mean(abs(ϱ_fluid - ϱ_particles)), 0,
                   rel_tol=0,
                   abs_tol=(tol_fac*np.std(ϱ_fluid) + machine_ϵ)):
        abort('Fluid did not gravitate correctly up to t = {} {}.\n'
              'See "{}" for a visualization.'
              .format(t, unit_time, fig_file))

# Done analyzing
masterprint('done')

