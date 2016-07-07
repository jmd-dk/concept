# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
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

# Standard test imports
import glob, sys, os

# Absolute paths to the directory of this file
this_dir = os.path.dirname(os.path.realpath(__file__))

# Pull in environment variables
for env_var in ('concept_dir', 'this_test'):
    exec('{env_var} = os.environ["{env_var}"]'.format(env_var=env_var))

# Include the concept_dir in the searched paths
sys.path.append(concept_dir)

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load

# Read in data from the CO𝘕CEPT snapshots
fluids = []
times = []
for fname in sorted(glob.glob(this_dir + '/output/snapshot_t=*'),
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

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
δ = []
δ_snapshot = []
phases = [-t/(10*units.Gyr)*2*π for t in times]
for ax_i, fluid, t, phase in zip(ax, fluids, times, phases):
    ρ = asarray([2 + np.sin(2*π*i/gridsize + phase) for i in range(gridsize)])
    δ.append(ρ/np.mean(ρ) - 1)
    δ_snapshot.append(fluid.fluidvars['δ'].grid_noghosts[:gridsize, 0, 0])
    ax_i.plot(x, δ[-1],
              'r', label='Analytical solution')
    ax_i.plot(x, δ_snapshot[-1],
              'b*', label='Simulation')
    ax_i.set_ylabel(r'$\delta$')
    ax_i.set_title(r'$t={:.3g}\,\mathrm'.format(t) + '{' + unit_time + '}$')
    plt.xlim(0, boxsize)
plt.legend(loc='best').get_frame().set_alpha(0.3)
plt.xlabel(r'$x\,\mathrm{[' + unit_length + ']}$')
plt.tight_layout()
plt.savefig(fig_file)

# Fluid elements in yz-slices should all have the same δ
# and all fluid elements should have the same u.
for fluid, t in zip(fluids, times):
    for l, fluidscalar in enumerate(fluid.iterate_fluidvars()):
        grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
        if l == 0:
            # δ
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(np.var(yz_slice), 0, rel_tol=0, abs_tol=1e-9*np.mean(yz_slice**2)):
                    masterwarn('Non-uniformities have emerged after {} {} '
                               'in yz-slices of fluid scalar variable {}.\n'
                               'See "{}" for a visualization.'
                               .format(t, unit_time, fluidscalar, fig_file))
                    sys.exit(1)
        elif l == 1:
            # u
            if not isclose(np.var(grid), 0, rel_tol=0, abs_tol=1e-9*np.mean(grid**2)):
                masterwarn('Non-uniformities have emerged after {} {} '
                           'in fluid scalar variable {}'
                           .format(t, unit_time, fluidscalar))
                sys.exit(1)

# Compare δ to the fluid from the snapshots
tol_fac = 1e-2
for i, t in enumerate(times):
    diff = δ[i] - δ_snapshot[i]
    if not isclose(np.std(diff), 0, rel_tol=0, abs_tol=tol_fac*np.std(δ[i])):
        masterwarn('Fluid did not drift rigidly up to t = {} {}.\n'
                   'See "{}" for a visualization.'
                   .format(t, unit_time, fig_file))
        sys.exit(1)

# Done analyzing
masterprint('done')

