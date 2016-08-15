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
a = []
for kind in ('particles', 'fluid'):
    if kind == 'particles':
        regex = '{}/output/{}/snapshot_a=*_converted*'.format(this_dir, kind)
    elif kind == 'fluid':
        regex = '{}/output/{}/snapshot_a=*'.format(this_dir, kind)
    for fname in sorted(glob.glob(regex),
                        key=lambda s: s[(s.index('=') + 1):]):
        snapshot = load(fname, compare_params=False)
        fluids[kind + ' simulations'].append(snapshot.components[0])
        if kind == 'particles':
            a.append(snapshot.params['a'])
N_snapshots = len(a)
gridsize = fluids['particles simulations'][0].gridsize
# Sort data chronologically
order = np.argsort(a)
a = [a[o] for o in order]
for kind in ('particles', 'fluid'):
    fluids[kind + ' simulations'] = [fluids[kind + ' simulations'][o] for o in order]

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, sharey=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
ϱ = {'particles simulations': [], 'fluid simulations': []}
for kind, markertype, options in zip(('particles', 'fluid'),
                                     ('ro', 'b*'),
                                     ({'markerfacecolor': 'none', 'markeredgecolor': 'r'}, {}),
                                     ):
    for ax_i, fluid, a_i in zip(ax, fluids[kind + ' simulations'], a):
        ϱ[kind + ' simulations'].append(fluid.fluidvars['ϱ'].grid_noghosts[:gridsize, 0, 0])
        ax_i.plot(x, ϱ[kind + ' simulations'][-1],
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
        ax_i.set_title(r'$a={:.3g}$'.format(a_i))
plt.xlim(0, boxsize)
plt.legend(loc='best').get_frame().set_alpha(0.3)
plt.xlabel(r'$x\,\mathrm{{[{}]}}$'.format(unit_length))
plt.tight_layout()
plt.savefig(fig_file)

# Fluid elements in yz-slices should all have the same ϱ and ϱu
tol_fac = 1e-6
for kind in ('particles', 'fluid'):
    for fluid, a_i in zip(fluids[kind + ' simulations'], a):
        for fluidscalar in fluid.iterate_fluidscalars():
            grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(np.std(yz_slice), 0,
                               rel_tol=0,
                               abs_tol=(tol_fac*np.std(grid) + machine_ϵ)):
                    masterwarn('Non-uniformities have emerged at a = {} '
                               'in yz-slices of fluid scalar variable {} '
                               'in {} simulation.\n'
                               'See "{}" for a visualization.'
                               .format(a_i, fluidscalar, kind.rstrip('s'), fig_file))
                    sys.exit(1)

# Compare ϱ's from the fluid and snapshot simulations
tol_fac = 2e-2
for ϱ_fluid, ϱ_particles, a_i in zip(ϱ['fluid simulations'], ϱ['particles simulations'], a):
    if not isclose(np.mean(abs(ϱ_fluid - ϱ_particles)), 0,
                   rel_tol=0,
                   abs_tol=(tol_fac*np.std(ϱ_fluid) + machine_ϵ)):
        masterwarn('Fluid did not gravitate correctly up to a = {}.\n'
                   'See "{}" for a visualization.'
                   .format(a_i, fig_file))
        sys.exit(1)

# Done analyzing
masterprint('done')

