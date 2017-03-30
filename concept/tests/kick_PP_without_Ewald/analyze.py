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
x0 = []
x0_std = []
x1 = []
x1_std = []
for fname in sorted(glob(this_dir + '/output/snapshot_a=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    posx = snapshot.components[0].posx
    a.append(snapshot.params['a'])
    x0.append(np.mean(posx[:4]))
    x0_std.append(np.std(posx[:4]))
    x1.append(np.mean(posx[4:]))
    x1_std.append(np.std(posx[4:]))
N_snapshots = len(a)

# Read in data from the GADGET snapshots
x0_gadget = []
x0_std_gadget = []
x1_gadget = []
x1_std_gadget = []
for fname in sorted(glob(this_dir + '/output/snapshot_gadget_*'),
                    key=lambda s: s[(s.index('gadget_') + 7):])[:N_snapshots]:
    snapshot = load(fname, compare_params=False)
    posx_gadget = snapshot.components[0].posx[np.argsort(snapshot.ID)]
    x0_gadget.append(np.mean(posx_gadget[:4]))
    x0_std_gadget.append(np.std(posx_gadget[:4]))
    x1_gadget.append(np.mean(posx_gadget[4:]))
    x1_std_gadget.append(np.std(posx_gadget[4:]))

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Plot
fig_file = this_dir + '/result.png'
plt.plot(np.concatenate((a, a)), np.concatenate((x0, x1)),
         '.',
         markersize=15,
         alpha=0.7,
         label='CO$N$CEPT',
         )
plt.plot(np.concatenate((a, a)), np.concatenate((x0_gadget, x1_gadget)),
         '.',
         markersize=10,
         alpha=0.7,
         label='GADGET',
         )
plt.xlabel('$a$')
plt.ylabel(r'$x\,\mathrm{{[{}]}}$'.format(unit_length))
plt.ylim(0, boxsize)
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# There should be no variance on the x positions.
tol = 1e+2*N_snapshots*machine_ϵ
if np.sum(x0_std_gadget) > tol or np.sum(x1_std_gadget) > tol:
    abort('Unequal x-positions for the 2*4 particles in the GADGET snapshots.\n'
          'It is no good to compare the CO𝘕CEPT results to these.')
if np.sum(x0_std) > tol or np.sum(x1_std) > tol:
    abort('Unequal x-positions for the 2*4 particles in the snapshots.\n'
          'The symmetric initial conditions has produced nonsymmetrical results!')

# Printout error message for unsuccessful test
tol = 1e-2
if (   max(np.abs(np.array(x0)/np.array(x0_gadget) - 1)) > tol
    or max(np.abs(np.array(x1)/np.array(x1_gadget) - 1)) > tol):
    abort('The results from CO𝘕CEPT disagree with those from GADGET.\n'
          'See "{}" for a visualization.'.format(fig_file))

# Done analyzing
masterprint('done')

