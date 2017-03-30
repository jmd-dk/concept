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

# Read in data from the CO𝘕CEPT snapshots
a = []
x = []
x_std = []
for fname in sorted(glob(this_dir + '/output/snapshot_a=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    posx = snapshot.components[0].posx
    a.append(snapshot.params['a'])
    x.append(np.mean(posx))
    x_std.append(np.std(posx))
N_snapshots = len(a)

# Read in data from the GADGET snapshots
x_gadget = []
x_std_gadget = []
for fname in sorted(glob(this_dir + '/output/snapshot_gadget_*'),
                    key=lambda s: s[(s.index('gadget_') + 7):])[:N_snapshots]:
    components_gadget = load(fname, compare_params=False, only_components=True)[0]
    x_gadget.append(np.mean(components_gadget.posx))
    x_std_gadget.append(np.std(components_gadget.posx))

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Plot
fig_file = this_dir + '/result.png'
plt.text(0.5*max(a), 0.93*boxsize, r'$\uparrow$ End of simulation box $\uparrow$', ha='center')
plt.plot(a, x       , '.', markersize=15, alpha=0.7, label='CO$N$CEPT')
plt.plot(a, x_gadget, '.', markersize=10, alpha=0.7, label='GADGET')
plt.xlabel('$a$')
plt.ylabel(r'$x\,\mathrm{{[{}]}}$'.format(unit_length))
plt.ylim(0, boxsize)
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# There should be no variance on the x positions.
tol = 1e+2*N_snapshots*machine_ϵ
if np.sum(x_std_gadget) > tol:
    abort('Unequal x-positions for the 4 particles in the GADGET snapshots.\n'
          'It is no good to compare the CO𝘕CEPT results to these.')
if np.sum(x_std) > tol:
    abort('Unequal x-positions for the 4 particles in the snapshots.\n'
          'The symmetric initial conditions have produced asymmetrical results!')

# Print out error message for unsuccessful test
tol = 1e-3
if max(np.abs(asarray(x)/asarray(x_gadget) - 1)) > tol:
    abort('The results from CO𝘕CEPT disagree with those from GADGET.\n'
          'See "{}" for a visualization.'.format(fig_file))

# Done analyzing
masterprint('done')

