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

# Read in data from the snapshots
fluids = {'particles simulations': [], 'fluid simulations': []}
times = []
for kind in ('particles', 'fluid'):
    if kind == 'particles':
        regex = '{}/output/{}/snapshot_t=*_converted*'.format(this_dir, kind)
    elif kind == 'fluid':
        regex = '{}/output/{}/snapshot_t=*'.format(this_dir, kind)
    for fname in sorted(glob.glob(regex),
                        key=lambda s: s[(s.index('=') + 1):]):
        snapshot = load(fname, compare_params=False)
        fluids[kind + ' simulations'].append(snapshot.components[0])
        if kind == 'particles':
            times.append(float(re.search('snapshot_t=(.*)' + unit_time, fname).group(1)))
N_snapshots = len(times)
gridsize = fluids['particles simulations'][0].gridsize
# Sort data chronologically
order = np.argsort(times)
times  = [times[o]  for o in order]
for kind in ('particles', 'fluid'):
    fluids[kind + ' simulations'] = [fluids[kind + ' simulations'][o] for o in order]
# Use precise times
times = snapshot_times['t']

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
δ = {'particles simulations': [], 'fluid simulations': []}
for kind, markertype, options in zip(('particles', 'fluid'),
                                     ('ro', 'b*'),
                                     ({'markerfacecolor': 'none', 'markeredgecolor': 'r'}, {}),
                                     ):
    for ax_i, fluid, t in zip(ax, fluids[kind + ' simulations'], times):
        δ[kind + ' simulations'].append(fluid.fluidvars['δ'].grid_noghosts[:gridsize, 0, 0])
        ax_i.plot(x, δ[kind + ' simulations'][-1],
                  markertype,
                  label=(kind.rstrip('s').capitalize() + ' simulation'),
                  **options,
                  )
        ax_i.set_ylabel(r'$\delta$')
        ax_i.set_title(r'$t={:.3g}\,\mathrm'.format(t) + '{' + unit_time + '}$')
plt.xlim(0, boxsize)
plt.legend(loc='best').get_frame().set_alpha(0.3)
plt.xlabel(r'$x\,\mathrm{[' + unit_length + ']}$')
plt.tight_layout()
plt.savefig(fig_file)

# Fluid elements in yz-slices should all have the same δ and u
for kind in ('particles', 'fluid'):
    for fluid, t in zip(fluids[kind + ' simulations'], times):
        for l, fluidscalar in enumerate(fluid.iterate_fluidvars()):
            grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
            if l == 0:
                # δ
                for i in range(gridsize):
                    yz_slice = grid[i, :, :]
                    if not isclose(np.var(yz_slice), 0,
                                   rel_tol=0,
                                   abs_tol=(1e-9*np.mean(yz_slice**2) + machine_ϵ)):
                        masterwarn('Non-uniformities have emerged after {} {} '
                                   'in yz-slices of fluid scalar variable {} '
                                   'in {} simulation.\n'
                                   'See "{}" for a visualization.'
                                   .format(t, unit_time, fluidscalar, kind.rstrip('s'), fig_file))
                        sys.exit(1)
            elif l == 1:
                # u
                for i in range(gridsize):
                    yz_slice = grid[i, :, :]
                    if not isclose(np.var(yz_slice), 0,
                                   rel_tol=0,
                                   abs_tol=(1e-9*np.mean(yz_slice**2) + machine_ϵ)):
                        masterwarn('Non-uniformities have emerged after {} {} '
                                   'in fluid scalar variable {} in {} simulation.'
                                   .format(t, unit_time, fluidscalar, kind.rstrip('s')))
                        sys.exit(1)

# Compare δ's from the fluid and snapshot simulations
tol_fac_max = 5e-1
tol_fac_mean = 1e-2
for i, t in enumerate(times):
    if (not all(isclose(δ_fluid, δ_particles, rel_tol=tol_fac_max, abs_tol=0)
                for δ_fluid, δ_particles in zip(δ['fluid simulations'][i],
                                                δ['particles simulations'][i]))
        or np.mean(np.abs([δ_fluid - δ_particles
                           for δ_fluid, δ_particles
                               in zip(δ['fluid simulations'][i],
                                      δ['particles simulations'][i])])) > tol_fac_mean):
        masterwarn('Fluid did not gravitate correctly up to t = {} {}.\n'
                   'See "{}" for a visualization.'
                   .format(t, unit_time, fig_file))
        sys.exit(1)

# Done analyzing
masterprint('done')

