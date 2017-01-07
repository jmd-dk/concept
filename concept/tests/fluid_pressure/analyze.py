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

# Read in data from the CO𝘕CEPT snapshots
fluids = []
times = []
for fname in sorted(glob(this_dir + '/output/snapshot_t=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    fluids.append(snapshot.components[0])
    times.append(float(re.search('snapshot_t=(.*)' + unit_time, fname).group(1)))
gridsize = fluids[0].gridsize
N_snapshots = len(fluids)
# Sort data chronologically
order = np.argsort(times)
times  = [times[o]  for o in order]
fluids = [fluids[o] for o in order]
# Use precise times
times = snapshot_times['t']

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Extract information from the first snapshot
cs = fluids[0].fluidvars['cs']
T = boxsize/cs
t0 = times[0]
ρ0 = fluids[0].fluidvars['ρ'].grid_noghosts[:gridsize, 0, 0]
ρ_max = max(ρ0)
ρ_min = min(ρ0)
ρ_mean = np.mean(ρ0)
ρ0_sin = ρ0 - ρ_mean

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, sharey=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
ρ = []
ρ_snapshot = []
for ax_i, fluid, t in zip(ax, fluids, times):
    ρ.append(ρ_mean + ρ0_sin*cos((t - t0)/T*2*π))
    ρ_snapshot.append(fluid.fluidvars['ρ'].grid_noghosts[:gridsize, 0, 0])
    ax_i.plot([0, boxsize], [ρ_mean]*2, 'k:' )
    ax_i.plot([0, boxsize], [ρ_max ]*2, 'k--')
    ax_i.plot([0, boxsize], [ρ_min ]*2, 'k--')
    ax_i.plot(x, ρ[-1],
              'r', label='Analytical solution')
    ax_i.plot(x, ρ_snapshot[-1],
              'b*', label='Simulation')
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

# Fluid elements in yz-slices should all have the same values
tol_fac = 1e-6
for fluid, t in zip(fluids, times):
    for fluidscalar in fluid.iterate_fluidscalars():
        varnum = fluidscalar.varnum
        grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
        for i in range(gridsize):
            yz_slice = grid[i, :, :]
            if not isclose(np.std(yz_slice), 0,
                           rel_tol=0,
                           abs_tol=(tol_fac*np.std(grid) + machine_ϵ)):
                abort('Non-uniformities have emerged at a = {} '
                      'in yz-slices of fluid scalar variable {}.\n'
                      'See "{}" for a visualization.'
                      .format(t, fluidscalar, fig_file))

# Compare ρ from the snapshots to the analytical solution
abs_tol = 1e-2*np.std(ρ0)
for ρ_i, ρ_snapshot_i, t in zip(ρ, ρ_snapshot, times):
    if not isclose(np.mean(abs(ρ_i - ρ_snapshot_i)), 0,
                   rel_tol=0,
                   abs_tol=abs_tol):
        abort('Fluid evolution differs from the analytical solution at t = {} {}.\n'
              'See "{}" for a visualization.'
              .format(t, unit_time, fig_file))

# Done analyzing
masterprint('done')

